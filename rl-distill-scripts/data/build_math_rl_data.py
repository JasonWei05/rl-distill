#!/usr/bin/env python3
"""Build train/val data for the few-shot math RL runs (Gemma 3 1B/4B PT).

Data stays PLAIN verl format (question + `\\boxed{}` instruction + gold). The unified 12-shot
prompt is applied at train/val time via the custom chat template
(`gemma3_it_fewshot_math.jinja`), so it is identical for training rollouts and every validation
set — nothing few-shot is baked into the parquets here.

Produces (in $DATA_DIR, default ~/verl/data):
  - DAPO split:  dapo_rl_train.parquet (17.2k)  +  dapo_rl_val100.parquet (100 held out, seed 42)
  - Val sets (base + repeated for avg@k, mirroring the repeat factors already used in this repo):
      MATH500        x2   GSM8K x1   OlympiadBench x2   MinervaMath x4
      BeyondAIME     x8   AIME2025 x32   AIME2026 x32
Every row's data_source routes to math_verify (see verl/utils/reward_score/__init__.py).
"""

from __future__ import annotations

import argparse
import os

import datasets
import pandas as pd

BOXED = "Please output the final answer within \\boxed{}."


def _row(idx, question, gold, data_source, split="test"):
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
    # Stable per-question uid so the K repeats share it -> verl groups them and computes
    # pass@k (best@k), maj@k, mean@k in validation (random per-row uids would break grouping).
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


def _strip_boxed(q: str) -> str:
    return q


# ---- source loaders -> list of _row dicts (base, unrepeated) --------------------------
def load_hf_math500():
    d = datasets.load_dataset("HuggingFaceH4/MATH-500", split="test")
    return [_row(i, r["problem"], r["answer"], "math500") for i, r in enumerate(d)]


def load_hf_gsm8k():
    d = datasets.load_dataset("openai/gsm8k", "main", split="test")
    return [_row(i, r["question"], r["answer"].split("####")[-1].strip(), "gsm8k") for i, r in enumerate(d)]


def load_hf_beyondaime():
    d = datasets.load_dataset("ByteDance-Seed/BeyondAIME", split="test")
    return [_row(i, r["problem"], r["answer"], "beyondaime") for i, r in enumerate(d)]


def load_existing(path):
    """Reuse an existing verl-format parquet (already question+gold+data_source)."""
    df = pd.read_parquet(path)
    return df.to_dict("records")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.expanduser("~/verl/data"))
    ap.add_argument("--dapo-source", default=os.path.expanduser("~/verl/data/dapo_17k_train.parquet"))
    ap.add_argument("--val-size", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    D = args.data_dir
    os.makedirs(D, exist_ok=True)

    print("== DAPO train/val split ==")
    df = pd.read_parquet(args.dapo_source)
    val = df.sample(n=args.val_size, random_state=args.seed).copy()
    train = df.drop(index=val.index).copy()
    train = train.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    val = val.reset_index(drop=True)
    train.to_parquet(f"{D}/dapo_rl_train.parquet", index=False)   # train: verl auto-assigns per-prompt uid
    print(f"  train={len(train)} -> dapo_rl_train.parquet")
    val_rows = val.to_dict("records")
    _write(val_rows, f"{D}/dapo_rl_val100.parquet", repeat=1, tag="dapo_val")
    _write(val_rows, f"{D}/dapo_rl_val100_x16.parquet", repeat=16, tag="dapo_val")

    print("== val datasets (base + repeats) ==")
    # name -> (loader, repeat)
    plan = {
        "math__math_500":   (load_hf_math500, 2),
        "math__gsm8k_test": (load_hf_gsm8k, 1),
        "math__beyondaime": (load_hf_beyondaime, 8),
        "math__olympiadbench": (lambda: load_existing(f"{D}/math__olympiadbench.parquet"), 2),
        "math__minervamath":   (lambda: load_existing(f"{D}/math__minervamath.parquet"), 4),
        "math__aime2025":  (lambda: load_existing(f"{D}/math__aime2025_30.parquet"), 32),
        "math__aime2026":  (lambda: load_existing(f"{D}/math__aime2026_30.parquet"), 32),
    }
    for name, (loader, rep) in plan.items():
        print(f"- {name} (x{rep})")
        rows = loader()
        _write(rows, f"{D}/{name}.parquet", repeat=1, tag=name)
        if rep > 1:
            _write(rows, f"{D}/{name}_x{rep}.parquet", repeat=rep, tag=name)
    print("DONE")


if __name__ == "__main__":
    main()
