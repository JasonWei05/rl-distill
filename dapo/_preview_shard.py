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

import sys

import pandas as pd

path = sys.argv[1] if len(sys.argv) > 1 else "/home/tiger/verl/data/teacher_gen/shard_004.parquet"
df = pd.read_parquet(path)

print(f"=== {path} ===")
print(f"rows: {len(df)}")
print(f"cols: {list(df.columns)}")
print()

# lengths summary
tok_lens = df["teacher_token_ids"].map(len)
lp_lens = df["teacher_log_probs"].map(len)
print(
    f"teacher_token_ids lengths: min={tok_lens.min()}  mean={tok_lens.mean():.0f}  "
    f"median={int(tok_lens.median())}  max={tok_lens.max()}"
)
print(
    f"teacher_log_probs lengths: min={lp_lens.min()}  mean={lp_lens.mean():.0f}  "
    f"median={int(lp_lens.median())}  max={lp_lens.max()}"
)
print(f"length match (ids == lps): {(tok_lens == lp_lens).all()}")
print()

# unique prompt_idx coverage
print(f"unique prompt_idx: {df['prompt_idx'].nunique()}  (expect ~2175 with 4 responses each = {len(df)})")
print(f"prompt_idx range: [{df['prompt_idx'].min()}, {df['prompt_idx'].max()}]")
print()

# Sample rows
print("=== sample row 0 ===")
row = df.iloc[0]
msgs = row["messages"]
print(f"prompt_idx: {row['prompt_idx']}")
print(f"user[:300]:\n  {msgs[0]['content'][:300]}...")
print()
print(f"assistant[:500]:\n  {msgs[1]['content'][:500]}...")
print()
print(f"assistant[-200:]:\n  ...{msgs[1]['content'][-200:]}")
print()
print(f"n_tokens: {len(row['teacher_token_ids'])}")
print(f"first 10 token_ids: {row['teacher_token_ids'][:10]}")
print(f"first 10 log_probs: {[round(x, 3) for x in row['teacher_log_probs'][:10]]}")
print(f"mean log_prob: {sum(row['teacher_log_probs']) / len(row['teacher_log_probs']):.3f}")
print()

# Confirm 4 responses per prompt
print("=== sample row 1 (same prompt, different response) ===")
row = df.iloc[1]
print(f"prompt_idx: {row['prompt_idx']} (same as row 0? {row['prompt_idx'] == df.iloc[0]['prompt_idx']})")
print(f"assistant[:300]:\n  {row['messages'][1]['content'][:300]}...")
