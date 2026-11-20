"""Chess against the released pre2post-chess files and code: rows against their tokenizer, the judge against their
reward, the Qwen3 forward against HF, the scripted decoder against their multi-turn rollout, the SFT loss against their
masks. Needs the devbox copies (a built dataset, a released checkpoint, their repo; skipped without them) and
python-chess on PYTHONPATH (/tmp/jw_chess/pylib)."""

import ast
import glob
import importlib.util
import json
import os
import random
import types

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

import sft
from data import Dataset
from model import Decoder, Qwen2, half, judged, load_model
from tasks import ENV, Chess

DATA = os.environ.get("CHESS_DATA", "/tmp/jw_chess/data/chess")
MODEL = os.environ.get("CHESS_MODEL", "/tmp/jw_chess/hf/Chess-SFT-Models/model_20m_16B")
REPO = os.environ.get("CHESS_REPO", "/tmp/jw_chess/pre2post-chess")
pytestmark = pytest.mark.skipif(not all(map(os.path.exists, (DATA, MODEL, REPO))), reason="needs the devbox chess files")


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def d():
    return Dataset(DATA)


@pytest.fixture(scope="module")
def tok():
    os.environ.setdefault("HF_MODULES_CACHE", "/tmp/jw_chess/hf_modules")
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)


@pytest.fixture(scope="module")
def reward():
    os.environ.setdefault("REWARD_MODEL_TYPE", "RULE_BASED")
    rfm = module(os.path.join(REPO, "rl/verl/reward_function_multiturn.py"), "rfm")
    return lambda texts, gts: [r["score"] for r in rfm.compute_score_batch(["x"] * len(texts), texts, gts, [{}] * len(texts))]


@pytest.fixture(scope="module")
def bench(d):
    return pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(d.cfg["task"]["bench"], "test_B*.parquet")))])


def test_words_are_the_released_vocab():
    with open(os.path.join(MODEL, "vocab.json")) as f:
        assert json.load(f) == {w: i for i, w in enumerate(Chess.WORDS)}


def test_benchmark_rows_are_their_eval_inputs(d, tok, bench):
    """test rows = their verl eval: prompts tokenized without <bos>, the 4 over 512 tokens dropped (1480), the solution's
    UCI moves, and each scripted reply = tokenizer(env_reply, add_special_tokens=False)."""
    rows = bench[[len(tok.encode(p, add_special_tokens=False)) <= 512 for p in bench["prompt"]]]
    assert len(rows) == 1480 and d.meta["splits"]["test"]["groups"] == {
        "B1": [0, 308], "B2": [308, 298], "B3": [606, 267], "B4": [873, 287], "B5": [1160, 320]}
    prompts = d.prompts("test")
    st = d.task.parse(prompts)
    sc = st["script"]
    for b, r in enumerate(rows.itertuples()):
        assert prompts[b, :sc["plen"][b]].tolist() == tok.encode(r.prompt, add_special_tokens=False)
        assert (prompts[b, sc["plen"][b]:d.task.slot] == 0).all()
        assert st["gt"][b] == ast.literal_eval(r.reward_model["ground_truth"]) and sc["turns"][b] == len(st["gt"][b])
        replies = [tok(x, add_special_tokens=False).input_ids for x in r.extra_info["env_replies"]]
        assert [sc["env"][b, k, :sc["env_len"][b, k]].tolist() for k in range(len(replies))] == replies
        assert (sc["env_len"][b, len(replies):] == 0).all()


def test_judge_is_their_reward(d, tok, reward):
    """The judge's score = their compute_score_batch on their decode of the same tokens (model and environment): on the
    minimal answers (all correct but the black castling ones), and on 20,000 corrupted answers."""
    rows = d.split("test")
    prompts = torch.from_numpy(rows[:, :d.prompt_len].astype(np.int64))
    st = d.task.parse(prompts)
    gts = [str(g) for g in st["gt"]]
    answers = [[t for t in row[d.prompt_len:].tolist() if t] for row in rows]
    rng = random.Random(0)
    cases, idx = [], []
    for k in range(20000):
        i = rng.randrange(len(answers))
        a = list(answers[i])
        for _ in range(rng.choice([0, 1, 1, 2, 3])):  # random edits: replace / insert / delete a token
            j, op = rng.randrange(len(a) + 1), rng.random()
            t = rng.randrange(0, 85) if rng.random() < 0.7 else rng.choice([82, 83, 84, 0, 1])
            if op < 0.4 and j < len(a):
                a[j] = t
            elif op < 0.7:
                a.insert(j, t)
            elif j < len(a):
                del a[j]
        if rng.random() < 0.3:
            a = [rng.choice([81, 83, 8, 42, 44])] + [rng.randrange(3, 85) for _ in range(rng.randrange(0, 8))] + a
        cases.append(a)
        idx.append(i)
    for batch, ix in ((answers, list(range(len(answers)))), (cases, idx)):
        width = max(map(len, batch))
        resp = torch.tensor([a + [0] * (width - len(a)) for a in batch])
        mine = d.task.judge(resp, st, torch.tensor(ix))["score"].tolist()
        theirs = reward([tok.decode([t % ENV for t in a]) for a in batch], [gts[i] for i in ix])
        assert mine == theirs
        assert [d.task.decode(a) for a in batch[:2000]] == [tok.decode([t % ENV for t in a]) for a in batch[:2000]]
        if batch is answers:  # a black castling solution move (O-O as e8g8 / e8c8) never scores: 3 benchmark puzzles
            fail = [i for i, x in enumerate(mine) if not x]
            assert len(fail) == 3 and all({"e8g8", "e8c8"} & set(st["gt"][i]) and "O-O" in d.task.decode(answers[i]) for i in fail)


def test_trace_rows_are_their_sft_examples(d, tok):
    """train rows, joined: their SFT sequence (encode(pgn)[:-1] + encode(cot)[1:], <eos> kept) and loss mask (labels != -100 after
    their prompt mask and MultiTurnSFTDataset._mask_env_responses)."""
    utils = module(os.path.join(REPO, "sft/training/sft_data_utils.py"), "sft_data_utils")
    vocab = tok.get_vocab()
    me = types.SimpleNamespace(_call_env_id=84, _move_start_ids={vocab[t] for t in ("K", "Q", "R", "B", "N", "P", "O-O", "O-O-O")},
                               _structural_ids={vocab[t] for t in ("<T>", "</T>", "<sep>", "<bos>", "<eos>")}, _equal_id=vocab["="])
    shards = sorted(os.listdir(d.cfg["task"]["traces"]))[:-2]
    with open(os.path.join(d.cfg["task"]["traces"], shards[0])) as f:
        results = json.load(f)["results"][:200]
    x, plen = sft.joined(d.split("train")[:200], d.task.slot, d.prompt_len)
    for r, row, p in zip(results, x, plen):
        lan = tok.custom_tokenizer  # their SFT dataset's tokenizer (LanTokenizerSFT): encode keeps <bos> and <eos>
        prompt = lan.encode(r["pgn"].replace("\n", " ").strip())[:-1]
        seq = prompt + lan.encode(r["cot_by_method"]["trajectory_sep"]["cot_format_no_labels"].strip())[1:]
        ids = torch.tensor(seq)
        labels = ids[1:].clone()
        labels[:len(prompt)] = -100
        utils.MultiTurnSFTDataset._mask_env_responses(me, ids[:-1], labels)
        mine = row[:len(seq)].astype(np.int64)
        assert (mine % ENV).tolist() == seq and (row[len(seq):] == 0).all() and p == len(prompt) + 1
        assert ((mine[1:] < ENV) & (np.arange(1, len(seq)) >= p)).tolist() == (labels != -100).tolist()


def test_qwen3_forward_matches_hf():
    """The functional forward (Qwen3: q/k norms, head_dim 128, YaRN) against HF Qwen3ForCausalLM in fp32, and per-row
    positions against shared ones."""
    from transformers import AutoModelForCausalLM
    c, _, w = load_model(MODEL)
    assert c["rope_scaling"]["type"] == "yarn"
    hf = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32, attn_implementation="eager").cuda().eval()
    m = Qwen2(c)
    ids = torch.randint(3, 85, (3, 900), device="cuda")
    pos = torch.arange(900, device="cuda")
    mine = m.logits(w, m.forward(w, ids, pos, m.causal()))
    with torch.no_grad():
        ref = hf(ids).logits
    assert (mine - ref).abs().max() < 1e-3
    assert torch.equal(m.forward(w, ids, pos.expand(3, -1), m.causal()), m.forward(w, ids, pos, m.causal()))


def test_scripted_decoder_follows_their_rollout(d):
    """Sampled answers of the released 20m model on test prompts: after each sampled <call_env> but the last, the next
    tokens are exactly the script's reply (+ ENV, log-prob 0); a row ends at EOS, at its H-th call, or at the budget /
    context caps; the sampled tokens' log-probs match a teacher-forced forward on the joined sequence (fp16, up to fp16
    noise)."""
    c, _, w = load_model(MODEL)
    m = Qwen2(c)
    prompts = d.prompts("test")[::40].cuda()
    st = d.task.parse(prompts)
    sc = st["script"]
    torch.manual_seed(0)
    for budget, max_ctx in ((2560, 3072), (300, 3072), (2560, 600)):
        tok, logp, _ = Decoder(m, 2624, 8.0, budget=budget, max_ctx=max_ctx).generate(half(w), prompts, 4, sc)
        for i, (row, lp) in enumerate(zip(tok.tolist(), logp)):
            g = i // 4
            plen, H = int(sc["plen"][g]), int(sc["turns"][g])
            replies = [sc["env"][g, k, :sc["env_len"][g, k]].tolist() for k in range(H - 1)]
            j, calls, live = 0, 0, []
            while j < len(row) and row[j]:
                t = row[j]
                assert t < ENV and lp[j] <= 0
                live.append(t)
                j += 1
                if t == 84:
                    calls += 1
                    if calls == H:
                        break
                    assert row[j:j + len(replies[calls - 1])] == [x + ENV for x in replies[calls - 1]]
                    assert (lp[j:j + len(replies[calls - 1])] == 0).all()
                    j += len(replies[calls - 1])
                elif t == 1:
                    break
            assert all(t == 0 for t in row[j:])
            ended = calls == H or live[-1:] == [1] or len(live) == budget or plen + j >= max_ctx
            assert ended, (i, calls, H, len(live), j)
    # teacher-forced log-probs of the first decode's rows
    tok, logp, _ = Decoder(m, 2624, 8.0, budget=2560, max_ctx=3072).generate(half(w), prompts[:8], 2, {k: v[:8] for k, v in sc.items()})
    for i in range(0, 16, 3):
        g, n_tok = i // 2, int((tok[i] != 0).sum())
        seq = torch.cat([prompts[g, :sc["plen"][g]], tok[i, :n_tok] % ENV])[None]
        with torch.no_grad():
            lp = m.logits(half(w), m.forward(half(w), seq, torch.arange(seq.shape[1], device="cuda"), m.causal())).float().log_softmax(-1)
        ref = lp[0, sc["plen"][g] - 1:-1].gather(-1, seq[0, sc["plen"][g]:, None]).squeeze(-1)
        live = tok[i, :n_tok] < ENV
        diff = (logp[i, :n_tok] - ref).abs()[live]  # fp16 noise: fp16 and fp32 forwards differ by ~2e-3 (rare tokens ~0.4)
        assert diff.mean() < 4e-3 and diff.max() < 0.5


def test_released_model_scores_on_the_benchmark(d):
    """The released 20m model's pass@1 on B1 (their protocol) is in the range a CPU re-implementation of their rollout
    measured (0.18-0.20 on the first 200 B1 puzzles)."""
    c, _, w = load_model(MODEL)
    prompts = d.prompts("test")[:200].cuda()
    torch.manual_seed(0)
    out = judged(d.task, Decoder(Qwen2(c), **d.cfg["decode"]), half(w), prompts, d.task.parse(prompts), 8)
    assert 0.13 < out["score"].mean().item() < 0.26 and out["closed_think"].mean() > 0.85


def test_sft_loss_matches_hf(d):
    """The fp32 joined loss (per-row sums and counts of the trained tokens) against HF Qwen3 cross entropy with the
    same mask."""
    from transformers import AutoModelForCausalLM
    t = sft.SFT.__new__(sft.SFT)
    t.s, t.model = {"dtype": "float32"}, Qwen2(load_model(MODEL)[0])
    w = load_model(MODEL)[2]
    x, plen = (torch.from_numpy(v).cuda().long() for v in sft.joined(d.split("train")[:8], d.task.slot, d.prompt_len))
    nll, count = t._loss_joined(w, x, plen)
    hf = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).cuda().eval()
    with torch.no_grad():
        logits = hf(x % ENV).logits[:, :-1].float()
    target = x[:, 1:]
    mask = (torch.arange(1, x.shape[1], device="cuda") >= plen[:, None]) & (target != 0) & (target < ENV)
    ref = (F.cross_entropy(logits.flatten(0, 1), (target % ENV).flatten(), reduction="none").view_as(target) * mask).sum(1)
    assert torch.equal(count, mask.sum(1)) and ((nll - ref).abs() / ref).max() < 1e-4


def test_rl_logprobs_match_the_decoder(d):
    """fastrl's update path on chess rows: train_logprobs with right-padded prompts (plen) on the decoder's tokens (the
    environment's fed as their ids) gives the sampled tokens' log-probs (fp16); the trained mask is the sampled tokens."""
    c, _, w = load_model(MODEL)
    m, wh = Qwen2(c), half(w)
    prompts = d.prompts("rl")[:6].cuda()
    sc = d.task.parse(prompts)["script"]
    torch.manual_seed(1)
    tok, logp, _ = Decoder(m, 2624, 8.0, budget=2560, max_ctx=3072).generate(wh, prompts, 4, sc)
    mask = (tok != 0) & (tok < ENV)
    R = int((tok != 0).sum(1).max())
    with torch.no_grad():
        ref = m.train_logprobs(wh, prompts, tok[:, :R] % ENV, 4, sc["plen"])
    diff = (logp[:, :R] - ref).abs()[mask[:, :R]]
    assert diff.mean() < 4e-3 and diff.max() < 0.5 and (logp[~mask] == 0).all()
