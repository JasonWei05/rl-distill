#!/usr/bin/env python3
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Diagnose the teacher_log_probs distribution. Hypothesis: some 0.0s are genuine
(teacher near-certain), some may be bugs (vLLM logprobs=0 returning placeholder)."""

import sys
from collections import Counter

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

PATH = sys.argv[1] if len(sys.argv) > 1 else "/home/tiger/verl/data/teacher_gen/shard_004.parquet"

df = pd.read_parquet(PATH)
print(f"=== {PATH}: {len(df)} rows ===")
print()

# Flatten all logprobs
all_lp = np.concatenate([np.asarray(x, dtype=np.float64) for x in df["teacher_log_probs"]])
all_ids = np.concatenate([np.asarray(x) for x in df["teacher_token_ids"]])
print(f"total tokens: {len(all_lp):,}")
print()

# Distribution
print("=== overall log-prob stats ===")
print(f"  min={all_lp.min():.4f}  max={all_lp.max():.4f}  mean={all_lp.mean():.4f}  median={np.median(all_lp):.4f}")
print(f"  std={all_lp.std():.4f}  p1={np.percentile(all_lp, 1):.3f}  p99={np.percentile(all_lp, 99):.4f}")
print()

# Exact-zero fraction — the thing the user is worried about
exact_zero = (all_lp == 0.0).sum()
near_zero = (np.abs(all_lp) < 1e-6).sum()
very_small = (np.abs(all_lp) < 1e-4).sum()
small = (np.abs(all_lp) < 1e-2).sum()
print("=== near-zero concentration ===")
print(f"  logprob == 0.0 exactly:  {exact_zero:>10,} ({100 * exact_zero / len(all_lp):.2f}%)")
print(f"  |logprob| < 1e-6:        {near_zero:>10,} ({100 * near_zero / len(all_lp):.2f}%)")
print(f"  |logprob| < 1e-4:        {very_small:>10,} ({100 * very_small / len(all_lp):.2f}%)")
print(f"  |logprob| < 1e-2:        {small:>10,} ({100 * small / len(all_lp):.2f}%)")
print()

# Histogram of buckets
print("=== logprob histogram ===")
bins = [-np.inf, -10, -5, -2, -1, -0.5, -0.1, -0.01, -1e-4, -1e-6, 0, 1e-6]
hist, _ = np.histogram(all_lp, bins=bins)
for lo, hi, n in zip(bins[:-1], bins[1:], hist, strict=False):
    print(f"  [{lo:>8.4g}, {hi:>8.4g}): {n:>10,} ({100 * n / len(all_lp):.2f}%)")
print()

# Are the zeros concentrated in specific token IDs?
print("=== which token IDs are most often zero-logprob? ===")
zero_mask = all_lp == 0.0
if zero_mask.sum() > 0:
    zero_ids = all_ids[zero_mask]
    cnt = Counter(zero_ids.tolist())
    try:
        tok = AutoTokenizer.from_pretrained("/tmp/teacher_27b_step40")
    except Exception:
        tok = None
    for tid, n in cnt.most_common(20):
        txt = tok.decode([int(tid)]) if tok is not None else "?"
        print(f"  token_id={tid:>8}  count={n:>8,}  decoded={txt!r}")
print()

# Look for suspicious RUNS of consecutive zeros per response
print("=== consecutive-zero runs per response ===")
longest_runs = []
max_runs = []
for lp in df["teacher_log_probs"]:
    a = np.asarray(lp)
    if len(a) == 0:
        continue
    is_zero = (a == 0.0).astype(np.int32)
    # find runs
    best = 0
    cur = 0
    for v in is_zero:
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    longest_runs.append(best)
runs = np.asarray(longest_runs)
print(
    f"  per-response longest 0-run: mean={runs.mean():.1f}  median={int(np.median(runs))}  "
    f"p90={np.percentile(runs, 90):.0f}  p99={np.percentile(runs, 99):.0f}  max={runs.max()}"
)
print(f"  rows with run >= 10: {(runs >= 10).sum()}")
print(f"  rows with run >= 50: {(runs >= 50).sum()}")
print()

# Sample: show first 40 tokens of a row with many zeros
sample_idx = int(np.argmax(runs))
print(f"=== row {sample_idx} (longest 0-run = {runs[sample_idx]}) ===")
row = df.iloc[sample_idx]
lps = np.asarray(row["teacher_log_probs"])
ids = np.asarray(row["teacher_token_ids"])
try:
    tok = AutoTokenizer.from_pretrained("/tmp/teacher_27b_step40")
    for i, (tid, lp) in enumerate(zip(ids[:50], lps[:50], strict=False)):
        s = tok.decode([int(tid)])
        print(f"  [{i:>3}] tid={int(tid):>6}  lp={lp:>9.4f}  text={s!r}")
except Exception as e:
    print(f"  (tokenizer load failed: {e})")
