#!/usr/bin/env python3
"""Pre-tokenize a maze SFT training set once, for TailRL sft_pretrain.py --train_cache (MazeTokenCache).

Writes a directory: tokens.npy (uint8, every example's ids concatenated), offsets.npy (int64, n + 1), path_start.npy
(int32, the PATH_START position) and meta.json (max_length, vocab, source file name + byte size, a probe item, counts,
build time). The trainer memory-maps tokens.npy and refuses a cache whose source size, sequence cap, vocabulary or probe
encoding does not match. Examples are exactly MazeSFTDataset's: the JSON items' "sequence" encoded without special
tokens, truncated to --max-length, dropped if empty or without PATH_START; labels (-100 up to and including PATH_START)
are rebuilt from path_start at load time. --verify N re-encodes N random items with the trainer's own
MazeSFTDataset._process_sequence and requires identical ids and labels.

  tokenize_maze_sft.py --data train.json --tokenizer <any run's init_model dir> --max-length 1024 --out train.tok-L1024 \\
      --tailrl <TailRL checkout> [--verify 2000]
The pretraining run-file passes ${DATA_S3}/<train json stem>.tok-L<MAX_LENGTH>/ (synced with the data) to the trainer.
"""
import argparse
import datetime
import json
import os
import random
import shutil
import sys
import time

import numpy as np


def encode_examples(seqs, tok, max_length, chunk=20000):
    """(tokens uint8, offsets int64, path_start int32, kept item indices) for the sequences MazeSFTDataset keeps."""
    ps_id = tok.encode("PATH_START", add_special_tokens=False)
    assert ps_id, "tokenizer has no PATH_START"
    ps_id = ps_id[0]
    assert max(tok.get_vocab().values()) < 256, "uint8 storage needs every token id below 256"
    parts, starts, keep = [], [], []
    for c in range(0, len(seqs), chunk):
        batch = seqs[c:c + chunk]
        for j, ids in enumerate(tok(batch, add_special_tokens=False)["input_ids"]):
            ids = ids[:max_length]
            if not batch[j] or ps_id not in ids:
                continue
            parts.append(np.asarray(ids, dtype=np.uint8))
            starts.append(ids.index(ps_id))
            keep.append(c + j)
    assert parts, f"no example of {len(seqs)} has PATH_START within max_length {max_length}"
    lengths = np.fromiter((len(x) for x in parts), dtype=np.int64, count=len(parts))
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    return np.concatenate(parts), offsets, np.asarray(starts, dtype=np.int32), keep


def write_cache(out, tokens, offsets, path_start, meta):
    """Write the directory atomically enough: a .tmp sibling renamed over any previous cache."""
    tmp = out.rstrip("/") + ".tmp"
    os.makedirs(tmp, exist_ok=True)
    for name, arr in (("tokens", tokens), ("offsets", offsets), ("path_start", path_start)):
        np.save(os.path.join(tmp, f"{name}.npy"), arr)
    with open(os.path.join(tmp, "meta.json"), "w") as f:
        json.dump(meta, f)
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.replace(tmp, out)


def build_meta(data_path, seqs, tok, max_length, tokens, offsets, keep):
    probe = seqs[keep[0]]
    return {"max_length": max_length, "vocab": tok.get_vocab(), "source": os.path.basename(data_path),
            "source_bytes": os.path.getsize(data_path), "n_items": len(seqs), "n_examples": len(keep),
            "probe": {"text": probe, "ids": [int(i) for i in tokens[offsets[0]:offsets[1]]]},
            "built": datetime.datetime.now(datetime.timezone.utc).isoformat()}


def verify(seqs, keep, tokens, offsets, path_start, tok, max_length, tailrl, n):
    """The trainer's own _process_sequence on n random kept items must give identical ids and labels."""
    sys.path.insert(0, os.path.join(tailrl, "experiments/text_maze/scripts"))
    from sft_pretrain import MazeSFTDataset
    ref = MazeSFTDataset.__new__(MazeSFTDataset)  # __init__ would tokenize the whole file; only the per-item method is used
    ref.tokenizer, ref.max_length = tok, max_length
    for k in random.Random(0).sample(range(len(keep)), min(n, len(keep))):
        ex = ref._process_sequence(seqs[keep[k]])
        ids = tokens[offsets[k]:offsets[k + 1]].astype(np.int64)
        labels = ids.copy()
        labels[: path_start[k] + 1] = -100
        assert ex is not None and np.array_equal(ex["input_ids"].numpy(), ids), k
        assert np.array_equal(ex["labels"].numpy(), labels), k


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--max-length", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tailrl", required=True, help="TailRL checkout (for the trainer's own _process_sequence in --verify)")
    ap.add_argument("--verify", type=int, default=2000)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    t0 = time.time()
    with open(args.data) as f:
        seqs = [item.get("sequence") or "" for item in json.load(f)]
    print(f"read {len(seqs)} items in {time.time() - t0:.0f}s", flush=True)
    tokens, offsets, path_start, keep = encode_examples(seqs, tok, args.max_length)
    print(f"{len(keep)} examples kept of {len(seqs)}, {tokens.size / 1e9:.2f}G tokens ({time.time() - t0:.0f}s)", flush=True)
    if args.verify:
        verify(seqs, keep, tokens, offsets, path_start, tok, args.max_length, args.tailrl, args.verify)
        print(f"verify: {min(args.verify, len(keep))} random examples identical to MazeSFTDataset._process_sequence")
    write_cache(args.out, tokens, offsets, path_start, build_meta(args.data, seqs, tok, args.max_length, tokens, offsets, keep))
    size = sum(os.path.getsize(os.path.join(args.out, n)) for n in os.listdir(args.out))
    print(f"wrote {args.out}/ ({size / 1e9:.2f} GB) in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
