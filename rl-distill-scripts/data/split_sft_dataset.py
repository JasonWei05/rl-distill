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

"""Split JWei05/DAPO-Gemma3-27B-IT-RL-SFT-Data into train/val SFT parquets.

Algorithm (deterministic with --seed):
  1. Shuffle the list of unique prompt_idx values.
  2. First 16,000 prompt_idx values -> train; remainder -> val.
  3. For each train prompt_idx: pick 2 random responses (out of 4) -> 32,000 rows.
  4. For each val prompt_idx:   pick 1 random response  (out of 4) -> ~1,398 rows.
  5. Shuffle row order within each split.

Usage:
    python3 split_sft_dataset.py                       # downloads from HF, writes defaults
    python3 split_sft_dataset.py --input LOCAL.parquet # use local source instead
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=None, help="Local source parquet. If None, downloads from HF.")
    p.add_argument(
        "--repo_id",
        default="JWei05/DAPO-Gemma3-27B-IT-RL-SFT-Data",
        help="HF dataset repo id if --input is not provided.",
    )
    p.add_argument("--filename", default="teacher_27b_step40_n4.parquet")
    p.add_argument("--output_dir", default=None, help="Where to write train/val parquets. Default: ~/verl/data/")
    p.add_argument("--n_train_prompts", type=int, default=16000)
    p.add_argument("--n_val_prompts", type=int, default=None, help="Cap val prompts. Default: use all remaining.")
    p.add_argument("--train_responses_per_prompt", type=int, default=2)
    p.add_argument("--val_responses_per_prompt", type=int, default=1)
    p.add_argument(
        "--output_train",
        default=None,
        help="Output train parquet path. Default: <output_dir>/teacher_27b_step40_sft_train.parquet",
    )
    p.add_argument(
        "--output_val",
        default=None,
        help="Output val parquet path. Default: <output_dir>/teacher_27b_step40_sft_val.parquet",
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.output_dir is None:
        args.output_dir = str(Path.home() / "verl" / "data")
    os.makedirs(args.output_dir, exist_ok=True)

    if args.input is None:
        from huggingface_hub import hf_hub_download

        print(f"downloading {args.repo_id}/{args.filename} ...")
        args.input = hf_hub_download(
            repo_id=args.repo_id,
            filename=args.filename,
            repo_type="dataset",
        )
    print(f"loading {args.input}")
    df = pd.read_parquet(args.input)
    print(f"  {len(df)} rows, columns={list(df.columns)}")

    rng = np.random.default_rng(args.seed)

    # 1. Unique prompt_idx values, shuffled.
    unique = np.array(sorted(df["prompt_idx"].unique()))
    print(f"  {len(unique)} unique prompts")
    assert args.n_train_prompts <= len(unique), (
        f"requested {args.n_train_prompts} train prompts but only {len(unique)} exist"
    )
    rng.shuffle(unique)
    train_ids = set(unique[: args.n_train_prompts].tolist())
    remaining = unique[args.n_train_prompts :]
    if args.n_val_prompts is not None:
        remaining = remaining[: args.n_val_prompts]
    val_ids = set(remaining.tolist())
    print(f"  split: {len(train_ids)} train prompts, {len(val_ids)} val prompts")

    # 2. Sub-sample rows per prompt_idx.
    def sample_per_prompt(df_sub: pd.DataFrame, k: int) -> pd.DataFrame:
        out_rows = []
        for _, group in df_sub.groupby("prompt_idx", sort=False):
            if len(group) <= k:
                out_rows.append(group)
            else:
                picks = rng.choice(len(group), size=k, replace=False)
                out_rows.append(group.iloc[picks])
        res = pd.concat(out_rows, ignore_index=True)
        return res

    train_df = sample_per_prompt(df[df["prompt_idx"].isin(train_ids)], args.train_responses_per_prompt)
    val_df = sample_per_prompt(df[df["prompt_idx"].isin(val_ids)], args.val_responses_per_prompt)

    # 3. Shuffle row order within each split.
    train_df = train_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    val_df = val_df.sample(frac=1.0, random_state=args.seed + 1).reset_index(drop=True)

    train_out = args.output_train or os.path.join(args.output_dir, "teacher_27b_step40_sft_train.parquet")
    val_out = args.output_val or os.path.join(args.output_dir, "teacher_27b_step40_sft_val.parquet")
    os.makedirs(os.path.dirname(train_out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(val_out) or ".", exist_ok=True)
    train_df.to_parquet(train_out, index=False)
    val_df.to_parquet(val_out, index=False)

    print("")
    print(f"train: {len(train_df):>6} rows  ({train_df['prompt_idx'].nunique()} unique prompts)  -> {train_out}")
    print(f"val:   {len(val_df):>6} rows  ({val_df['prompt_idx'].nunique()} unique prompts)  -> {val_out}")


if __name__ == "__main__":
    main()
