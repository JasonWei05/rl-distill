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

"""
Create a 50/50 mixed dataset: all of DAPO-math-17k + random 17,398 from nvidia/OpenMathInstruct-2.

Output format matches dapo-math-17k.parquet:
  - data_source: str  ("math" for DAPO, "openmath2" for OpenMathInstruct-2)
  - prompt: np.array([{"content": <problem text>, "role": "user"}])
  - reward_model: {"ground_truth": <answer>, "style": "rule"}
  - extra_info: {"index": <id>, "original_question": <problem>, "split": "train"}

Usage:
  python make_dapo_openmath2_mix.py [--seed 42] [--output /home/tiger/verl/data/dapo_openmath2_mix.parquet]
"""

import argparse
import random

import numpy as np
import pandas as pd
from datasets import load_dataset


def convert_openmath2_row(row, idx):
    """Convert an OpenMathInstruct-2 row to DAPO parquet format."""
    problem = row["problem"]
    answer = row["expected_answer"]
    source = row["problem_source"]

    # Add the boxed instruction suffix if not already present
    if "\\boxed{}" not in problem:
        prompt_text = problem.rstrip() + " Please output the final answer within \\boxed{}."
    else:
        prompt_text = problem

    return {
        "data_source": "math",
        "prompt": np.array([{"content": prompt_text, "role": "user"}], dtype=object),
        "reward_model": {"ground_truth": str(answer), "style": "rule"},
        "extra_info": {
            "index": f"openmath2-{idx}",
            "original_question": problem,
            "split": "train",
            "problem_source": source,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dapo_path", default="/home/tiger/verl/data/dapo-math-17k.parquet")
    parser.add_argument("--output", default="/home/tiger/verl/data/dapo_openmath2_mix.parquet")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max_prompt_tokens",
        type=int,
        default=2000,
        help="Max prompt tokens; filter OpenMathInstruct-2 rows exceeding this",
    )
    parser.add_argument("--tokenizer", default="/home/tiger/verl/models/gemma-3-4b-pt")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    # 1) Load DAPO dataset (all of it)
    dapo_df = pd.read_parquet(args.dapo_path)
    n_dapo = len(dapo_df)
    print(f"Loaded DAPO dataset: {n_dapo} rows from {args.dapo_path}")

    # 2) Stream OpenMathInstruct-2, filter by prompt length, reservoir-sample n_dapo rows
    n_sample = n_dapo
    print(f"Sampling {n_sample} rows from nvidia/OpenMathInstruct-2 (train_1M split, streaming)...")
    print(f"Filtering prompts to <= {args.max_prompt_tokens} tokens")

    ds = load_dataset("nvidia/OpenMathInstruct-2", split="train_1M", streaming=True)

    # Reservoir sampling with length filter
    reservoir = []
    n_seen = 0  # count of rows that pass the filter
    n_skipped = 0
    for i, row in enumerate(ds):
        # Check prompt length
        prompt_text = row["problem"]
        if "\\boxed{}" not in prompt_text:
            prompt_text = prompt_text.rstrip() + " Please output the final answer within \\boxed{}."
        tok_len = len(tok.encode(prompt_text))
        if tok_len > args.max_prompt_tokens:
            n_skipped += 1
            continue

        if n_seen < n_sample:
            reservoir.append(row)
        else:
            j = random.randint(0, n_seen)
            if j < n_sample:
                reservoir[j] = row
        n_seen += 1

        if (i + 1) % 100_000 == 0:
            print(f"  streamed {i + 1}, accepted {n_seen}, skipped {n_skipped}...")

    print(f"  done — streamed {i + 1} total, accepted {n_seen}, skipped {n_skipped}")
    print(f"  reservoir size: {len(reservoir)}")

    # Shuffle the reservoir
    random.shuffle(reservoir)

    # 3) Convert OpenMathInstruct-2 rows to DAPO format
    print("Converting OpenMathInstruct-2 rows to DAPO format...")
    openmath_rows = [convert_openmath2_row(r, idx) for idx, r in enumerate(reservoir)]
    openmath_df = pd.DataFrame(openmath_rows)

    # 4) Concatenate: DAPO first, then OpenMathInstruct-2, then shuffle
    mixed_df = pd.concat([dapo_df, openmath_df], ignore_index=True)
    mixed_df = mixed_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    print(f"\nMixed dataset: {len(mixed_df)} rows")
    print(f"  DAPO:          {n_dapo} ({n_dapo / len(mixed_df) * 100:.1f}%)")
    print(f"  OpenMathInst2: {len(openmath_df)} ({len(openmath_df) / len(mixed_df) * 100:.1f}%)")
    print("  data_source distribution:")
    print(mixed_df["data_source"].value_counts().to_string(header=False))

    # 5) Save
    mixed_df.to_parquet(args.output, index=False)
    print(f"\nSaved to {args.output}")
    print(f"File size: {pd.io.common.file_exists(args.output)}")

    # Quick sanity check
    check = pd.read_parquet(args.output)
    print(f"\nVerification — re-read shape: {check.shape}")
    print(f"Columns: {list(check.columns)}")
    print(f"Sample row 0: data_source={check.iloc[0]['data_source']}")
    print(f"Sample row 0: prompt type={type(check.iloc[0]['prompt'])}")


if __name__ == "__main__":
    main()
