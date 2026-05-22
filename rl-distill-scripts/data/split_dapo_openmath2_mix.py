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

"""Split dapo_openmath2_mix.parquet into train/val parquets.

Usage:
    python3 split_dapo_openmath2_mix.py
    python3 split_dapo_openmath2_mix.py --input /path/source.parquet --output_dir /path/out
"""

import argparse
import os
from pathlib import Path

import pandas as pd


def set_split(extra_info, split):
    if isinstance(extra_info, dict):
        out = dict(extra_info)
    else:
        out = {}
    out["split"] = split
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=str(Path.home() / "verl" / "data" / "dapo_openmath2_mix.parquet"))
    p.add_argument("--output_dir", default=str(Path.home() / "verl" / "data"))
    p.add_argument("--val_size", type=int, default=1500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_train", default=None)
    p.add_argument("--output_val", default=None)
    args = p.parse_args()

    df = pd.read_parquet(args.input)
    if args.val_size >= len(df):
        raise ValueError(f"--val_size={args.val_size} must be smaller than dataset size {len(df)}")

    val_df = df.sample(n=args.val_size, random_state=args.seed).copy()
    train_df = df.drop(index=val_df.index).copy()

    train_df = train_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    val_df = val_df.sample(frac=1.0, random_state=args.seed + 1).reset_index(drop=True)

    train_df["extra_info"] = train_df["extra_info"].apply(lambda x: set_split(x, "train"))
    val_df["extra_info"] = val_df["extra_info"].apply(lambda x: set_split(x, "val"))

    train_out = args.output_train or os.path.join(args.output_dir, "dapo_openmath2_mix_train.parquet")
    val_out = args.output_val or os.path.join(args.output_dir, "dapo_openmath2_mix_val.parquet")
    os.makedirs(os.path.dirname(train_out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(val_out) or ".", exist_ok=True)

    train_df.to_parquet(train_out, index=False)
    val_df.to_parquet(val_out, index=False)

    print(f"source: {len(df)} rows -> {args.input}")
    print(f"train:  {len(train_df)} rows -> {train_out}")
    print(f"val:    {len(val_df)} rows -> {val_out}")


if __name__ == "__main__":
    main()
