"""CPU tests for the pre-tokenized maze pretraining data (maze/tokenize_maze_sft.py + TailRL sft_pretrain MazeTokenCache).

The cache must reproduce MazeSFTDataset item for item (ids, labels, attention mask; same kept items, so the same
shuffled batch order, also with forked DataLoader workers) on edge cases: empty / missing / None sequences, no
PATH_START, PATH_START at the last kept position and just past it, truncation, unknown words, odd whitespace, repeated
PATH_START. The loader must refuse a cache built from another source file, sequence cap or tokenizer. Uses the trainer's
own tokenizer construction (create_model_from_scratch) and synthetic maze-like sequences. Needs a patched TailRL checkout
(TAILRL_ROOT, default /tmp/maze/TailRL or /tmp/jw_maze/TailRL):

    TAILRL_ROOT=/tmp/jw_maze/TailRL /mnt/efs/jasonwei/rl-distill/.venv/bin/python -m pytest -q \\
        tests/special_standalone/test_rl_distill_maze_token_cache.py
"""

import json
import os
import random
import sys
from types import SimpleNamespace

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOTS = [os.environ.get("TAILRL_ROOT", ""), "/tmp/maze/TailRL", "/tmp/jw_maze/TailRL"]
SCRIPTS = next((os.path.join(r, "experiments/text_maze/scripts") for r in ROOTS
                if r and os.path.exists(os.path.join(r, "experiments/text_maze/scripts/sft_pretrain.py"))
                and "MazeTokenCache" in open(os.path.join(r, "experiments/text_maze/scripts/sft_pretrain.py")).read()), None)
pytestmark = pytest.mark.skipif(SCRIPTS is None, reason="needs a patched TailRL checkout (TAILRL_ROOT)")
if SCRIPTS:
    sys.path.insert(0, SCRIPTS)
    sys.path.insert(0, os.path.join(REPO, "rl-distill-scripts/maze"))
    import tokenize_maze_sft as tms
    from sft_pretrain import MazeSFTDataset, MazeTokenCache, collate_fn, create_model_from_scratch


def maze_seq(rng, grid=60, moves=20):
    cells = " ".join(rng.choice(["WALL", "PATH", "PATH", "NEWLINE"]) for _ in range(grid))
    path = " ".join(rng.choice(["UP", "DOWN", "LEFT", "RIGHT"]) for _ in range(moves))
    return f"<bos> GRID_START START {cells} GOAL GRID_END PATH_START {path} DONE"


@pytest.fixture(scope="module")
def tok(tmp_path_factory):
    from transformers import AutoTokenizer
    out = str(tmp_path_factory.mktemp("init"))
    path = create_model_from_scratch(SimpleNamespace(output_dir=out, hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                                                     num_key_value_heads=1, intermediate_size=64, max_position_embeddings=128,
                                                     init_seed=0, seed=0))
    return AutoTokenizer.from_pretrained(path)


def edge_items(L):
    rng = random.Random(0)
    base = maze_seq(rng).split()
    p = base.index("PATH_START")
    filler = " ".join(["WALL"] * (L - 1))
    items = [{"sequence": ""}, {}, {"sequence": None}, {"sequence": " ".join(base[:p])},  # dropped
             {"sequence": filler + " PATH_START"},  # PATH_START at L-1: kept, nothing supervised
             {"sequence": filler + " WALL PATH_START UP DONE"},  # PATH_START at L: dropped
             {"sequence": " ".join(base).replace("UP", "ZORK", 1)},  # unknown word
             {"sequence": " ".join(base) + " UP" * (2 * L)},  # truncated
             {"sequence": "   " + "   ".join(base) + "  "},  # odd whitespace
             {"sequence": "PATH_START"},
             {"sequence": " ".join(base).replace("PATH_START", "PATH_START PATH_START", 1)}]
    return items + [{"sequence": maze_seq(rng)} for _ in range(40)]


def build(tmp_path, tok, items, L):
    data = tmp_path / "train.json"
    data.write_text(json.dumps(items))
    seqs = [item.get("sequence") or "" for item in items]
    tokens, offsets, path_start, keep = tms.encode_examples(seqs, tok, L, chunk=7)
    out = str(tmp_path / f"train.tok-L{L}")
    tms.write_cache(out, tokens, offsets, path_start, tms.build_meta(str(data), seqs, tok, L, tokens, offsets, keep))
    return str(data), out


def reference(data, tok, L):
    return MazeSFTDataset(data, tok, L)


@pytest.mark.parametrize("L", [128, 96])
def test_items_equal_mazesftdataset(tmp_path, tok, L):
    data, out = build(tmp_path, tok, edge_items(L), L)
    cache, ref = MazeTokenCache(out, data, tok, L), reference(data, tok, L)
    assert len(cache) == len(ref)
    for i in range(len(ref)):
        a, b = cache[i], ref[i]
        for k in ("input_ids", "labels", "attention_mask"):
            assert torch.equal(a[k], b[k]) and a[k].dtype == b[k].dtype, (i, k)
    sup = [(cache[i]["labels"] != -100).sum().item() for i in range(len(cache))]
    assert 0 in sup  # the PATH_START-at-L-1 item is kept with nothing supervised, as in MazeSFTDataset


@pytest.mark.parametrize("workers", [0, 2])
def test_same_shuffled_batches(tmp_path, tok, workers):
    data, out = build(tmp_path, tok, edge_items(128), 128)
    batches = []
    for ds in (MazeTokenCache(out, data, tok, 128), reference(data, tok, 128)):
        dl = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True, generator=torch.Generator().manual_seed(0),
                                         collate_fn=collate_fn, num_workers=workers)
        batches.append(list(dl))
    assert len(batches[0]) == len(batches[1])
    for a, b in zip(*batches):
        assert all(torch.equal(a[k], b[k]) for k in a)


def test_refuses_mismatched_cache(tmp_path, tok):
    items = edge_items(128)
    data, out = build(tmp_path, tok, items, 128)
    with pytest.raises(AssertionError):
        MazeTokenCache(out, data, tok, 96)  # other sequence cap
    json.dump(items + [{"sequence": "PATH_START UP DONE"}], open(data, "w"))
    with pytest.raises(AssertionError, match="byte"):
        MazeTokenCache(out, data, tok, 128)  # data regenerated, cache not
    json.dump(items, open(data, "w"))
    tok2 = type(tok).from_pretrained(tok.name_or_path)
    tok2.add_tokens(["EXTRA"])
    with pytest.raises(AssertionError, match="vocabulary"):
        MazeTokenCache(out, data, tok2, 128)


def test_builder_rejects_all_dropped(tok):
    with pytest.raises(AssertionError, match="PATH_START"):
        tms.encode_examples(["GRID_START WALL GRID_END", ""], tok, 128)
