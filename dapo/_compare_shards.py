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

"""Compare shard_004.parquet (small) against another shard to explain the size difference."""

import glob
import os

import pandas as pd

paths = sorted(glob.glob("/home/tiger/verl/data/teacher_gen/shard_*.parquet"))

print(
    f"{'shard':<6} {'rows':>6} {'size MB':>9} {'tok min':>8} {'tok med':>8} {'tok mean':>9} "
    f"{'tok max':>8} {'tot toks M':>11} {'empty':>6} {'trunc@20480':>12}"
)
for p in paths:
    df = pd.read_parquet(p)
    lens = df["teacher_token_ids"].map(len)
    size_mb = os.path.getsize(p) / 1e6
    total_m = lens.sum() / 1e6
    empty = (lens == 0).sum()
    trunc = (lens >= 20400).sum()  # near the 20480 cap
    name = os.path.basename(p).replace("shard_", "").replace(".parquet", "")
    print(
        f"{name:<6} {len(df):>6} {size_mb:>9.1f} {lens.min():>8} {int(lens.median()):>8} "
        f"{lens.mean():>9.0f} {lens.max():>8} {total_m:>11.2f} {empty:>6} {trunc:>12}"
    )

print()
# Look for any truncated or degenerate outputs in shard 4 specifically
df4 = pd.read_parquet("/home/tiger/verl/data/teacher_gen/shard_004.parquet")
lens4 = df4["teacher_token_ids"].map(len)
print(f"=== shard 4 detail (n={len(df4)}) ===")
print(f"  prompt_idx: [{df4['prompt_idx'].min()}, {df4['prompt_idx'].max()}]")
print(f"  {(lens4 < 200).sum()} responses < 200 tokens")
print(f"  {(lens4 < 100).sum()} responses < 100 tokens")
print(f"  {(lens4 >= 19000).sum()} responses >= 19k tokens (likely truncated)")
# How many responses end with a clear 'Final Answer' marker
has_final = df4["messages"].apply(lambda m: "Final Answer" in m[1]["content"] or "\\boxed{" in m[1]["content"]).sum()
print(f"  {has_final} / {len(df4)} have 'Final Answer' or '\\boxed{{' in response")

# Compare prompt lengths across shards (were shard-4 prompts systematically shorter?)
print()
print("=== prompt string length per shard ===")
for p in paths:
    df = pd.read_parquet(p)
    prompt_lens = df["messages"].apply(lambda m: len(m[0]["content"]))
    name = os.path.basename(p).replace("shard_", "").replace(".parquet", "")
    print(
        f"  shard {name}: prompt chars min={prompt_lens.min()}, "
        f"median={int(prompt_lens.median())}, mean={prompt_lens.mean():.0f}, max={prompt_lens.max()}"
    )
