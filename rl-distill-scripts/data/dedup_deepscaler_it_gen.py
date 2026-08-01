#!/usr/bin/env python3
"""Dedupe the DeepScaleR 4B-IT generation dataset (v2 revamp).

Rules (2026-08-01, per Jason):
1. Group questions under MAX-strength normalization (case + whitespace + prose
   punctuation + LaTeX formatting; math operators preserved).
2. Groups whose distinct gold answers are NOT all math_verify-equivalent are
   REMOVED entirely (label conflicts — at most one copy is right, we can't tell
   which without re-judging).
3. Remaining duplicate groups keep ONE copy chosen at random (seeded),
   regardless of pass-rate bucket.

Run with the repo .venv (needs math_verify):
    .venv/bin/python rl-distill-scripts/data/dedup_deepscaler_it_gen.py \
        --inp ~/verl/data/deepscaler_it_gen/deepscaler_it_gen_merged_strict.parquet \
        --out ~/verl/data/deepscaler_it_gen/deepscaler_it_gen_dedup_v2.parquet --seed 42
"""

import argparse
import os
import re
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError

import numpy as np
import pandas as pd

PUNCT = r"[.,;:!?'\"`]"


def normalize_question(s: str) -> str:
    """Max-strength: lowercase, whitespace, prose punctuation, LaTeX formatting."""
    s = re.sub(r"\s+", " ", s.strip().lower())
    s = re.sub(r"\$\$?|\\\(|\\\)|\\\[|\\\]", " ", s)               # math delimiters
    s = re.sub(r"\\left\s*|\\right\s*", "", s)
    s = re.sub(r"\\[dt]frac", r"\\frac", s)
    s = re.sub(r"\\(?:qquad|quad|[,;:!])", " ", s)                  # spacing macros
    s = re.sub(r"\\(?:mathrm|mathbf|mathit|text|textbf|operatorname)\s*\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"([_^])\{(\w)\}", r"\1\2", s)                       # ^{2} -> ^2
    s = re.sub(r"\\cdot\b", "*", s)
    s = re.sub(r"\\times\b", "*", s)
    s = re.sub(PUNCT, "", s)
    return re.sub(r"\s+", " ", s).strip()


def _answers_equiv(a: str, b: str) -> bool:
    """math_verify equivalence of two gold answers (run in a subprocess)."""
    from math_verify.grader import verify
    from math_verify.parser import LatexExtractionConfig, parse

    cfg = (LatexExtractionConfig(),)
    pa = parse("\\boxed{" + a + "}", cfg)
    pb = parse("\\boxed{" + b + "}", cfg)
    if not pa or not pb:
        return False
    return bool(any(verify(x, y) for x in pa for y in pb) or any(verify(y, x) for x in pa for y in pb))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--timeout", type=float, default=15.0)
    args = ap.parse_args()

    df = pd.read_parquet(os.path.expanduser(args.inp))
    df["_key"] = df["problem"].astype(str).map(normalize_question)
    df["_ans"] = df["answer"].astype(str).str.strip()

    import multiprocessing as mp
    pool = ProcessPoolExecutor(max_workers=8, mp_context=mp.get_context("spawn"))

    sizes = df.groupby("_key").size()
    dup_keys = sizes[sizes > 1].index
    print(f"rows={len(df)}  unique questions={len(sizes)}  duplicate groups={len(dup_keys)}")

    drop_keys = []
    checked = equiv_calls = 0
    for k in dup_keys:
        answers = list(dict.fromkeys(df.loc[df["_key"] == k, "_ans"]))  # distinct, ordered
        norm = {re.sub(r"\s+", "", a.lower()) for a in answers}
        if len(norm) <= 1:
            continue  # identical golds
        checked += 1
        ref, ok = answers[0], True
        for other in answers[1:]:
            equiv_calls += 1
            try:
                fut = pool.submit(_answers_equiv, ref, other)
                if not fut.result(timeout=args.timeout):
                    ok = False
                    break
            except (FuturesTimeoutError, Exception):
                ok = False  # conservative: unverifiable => conflicting
                break
        if not ok:
            drop_keys.append(k)
    pool.shutdown(wait=False, cancel_futures=True)

    drop_keys = set(drop_keys)
    n_drop_rows = int(df["_key"].isin(drop_keys).sum())
    print(f"groups with differing gold strings: {checked}; equivalence checks run: {equiv_calls}")
    print(f"CONFLICTING groups removed entirely: {len(drop_keys)} ({n_drop_rows} rows)")

    kept = df[~df["_key"].isin(drop_keys)]
    rng = np.random.RandomState(args.seed)
    # keep one random copy per group, independent of bucket
    kept = kept.sample(frac=1.0, random_state=rng).drop_duplicates("_key", keep="first")
    print(f"after random keep-one-per-group: {len(kept)} unique questions")

    for col, tag in (("n_correct", "lenient"), ("strict_n_correct", "strict")):
        if col in kept.columns:
            vc = kept[col].value_counts()
            counts = {f"{i}/4": int(vc.get(i, 0)) for i in range(5)}
            print(f"{tag} buckets: {counts}")

    out = os.path.expanduser(args.out)
    kept.drop(columns=["_key", "_ans"]).to_parquet(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
