#!/usr/bin/env python3
"""Build DeepScaleR RL train/val data in plain verl format (mirrors build_math_rl_data.py).

Source: agentica-org/DeepScaleR-Preview-Dataset (40,315 competition-math problems).
Produces (in $DATA_DIR, default ~/verl/data):
  - deepscaler_rl_train.parquet      (train; NO uid col — verl auto-assigns per-prompt uid at train time)
  - deepscaler_rl_val200.parquet     (200 held-out, seed 42, repeat 1)
  - deepscaler_rl_val200_x16.parquet (same 200 x16, shared per-question uid -> pass@16 / maj@16 / mean@16)

data_source='math' routes to math_verify (see verl/utils/reward_score/__init__.py). The prompt is the
plain problem + the `\\boxed{}` instruction; the unified 12-shot prompt is applied at train/val time via
the chat template (gemma3_it_fewshot_math.jinja), exactly like the DAPO few-shot RL runs.
"""
from __future__ import annotations

import argparse
import os

import datasets
import pandas as pd

BOXED = "Please output the final answer within \\boxed{}."


def _row(idx, question, gold, data_source="math", split="train"):
    q = question.strip()
    if not q.endswith(BOXED):
        q = q + " " + BOXED
    return {
        "data_source": data_source,
        "prompt": [{"content": q, "role": "user"}],
        "reward_model": {"ground_truth": str(gold), "style": "rule"},
        "extra_info": {"index": str(idx), "split": split},
    }


def _write(rows, path, repeat=1, tag="q"):
    base = []
    for i, r in enumerate(rows):
        r = dict(r)
        r["uid"] = f"{tag}-{i}"
        base.append(r)
    df = pd.DataFrame(base)
    if repeat > 1:
        df = pd.concat([df] * repeat, ignore_index=True)
    df.to_parquet(path, index=False)
    print(f"  wrote {path}  ({len(df)} rows, base={len(rows)} x{repeat})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.expanduser("~/verl/data"))
    ap.add_argument("--val-size", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    D = args.data_dir
    os.makedirs(D, exist_ok=True)

    ds = datasets.load_dataset("agentica-org/DeepScaleR-Preview-Dataset", split="train").shuffle(seed=args.seed)
    val = ds.select(range(args.val_size))
    train = ds.select(range(args.val_size, len(ds)))
    print(f"== DeepScaleR split (seed {args.seed}): train={len(train)}  val={len(val)} ==")

    train_rows = [_row(i, r["problem"], r["answer"], split="train") for i, r in enumerate(train)]
    tdf = pd.DataFrame(train_rows)  # no uid col, matches dapo_rl_train.parquet
    tdf.to_parquet(f"{D}/deepscaler_rl_train.parquet", index=False)
    print(f"  wrote {D}/deepscaler_rl_train.parquet  ({len(tdf)} rows)")

    val_rows = [_row(i, r["problem"], r["answer"], split="test") for i, r in enumerate(val)]
    _write(val_rows, f"{D}/deepscaler_rl_val200.parquet", repeat=1, tag="deepscaler_val")
    _write(val_rows, f"{D}/deepscaler_rl_val200_x16.parquet", repeat=16, tag="deepscaler_val")
    print("DONE")


if __name__ == "__main__":
    main()
