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

"""Merge teacher generation shard parquets into a single file."""

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Directory with shard_*.parquet files (default: ~/verl/data/teacher_gen)",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output parquet (default: ~/verl/data/teacher_27b_step40_n4.parquet)"
    )
    args = parser.parse_args()

    if args.input_dir is None:
        args.input_dir = str(Path.home() / "verl/data/teacher_gen")
    if args.output is None:
        args.output = str(Path.home() / "verl/data/teacher_27b_step40_n4.parquet")

    shards = sorted(Path(args.input_dir).glob("shard_*.parquet"))
    if not shards:
        print(f"No shard files found in {args.input_dir}")
        return

    dfs = []
    for shard in shards:
        df = pd.read_parquet(shard)
        print(f"  {shard.name}: {len(df)} rows")
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal: {len(merged)} rows")
    print(f"Columns: {list(merged.columns)}")

    # Verify all rows have log probs and token ids
    for col in ["teacher_log_probs", "teacher_token_ids"]:
        missing = merged[col].isna().sum()
        if missing > 0:
            print(f"WARNING: {missing} rows missing {col}")
        empty = sum(1 for x in merged[col] if isinstance(x, list) and len(x) == 0)
        if empty > 0:
            print(f"WARNING: {empty} rows with empty {col}")

    # parquet roundtrip gives numpy arrays (not Python lists) — use hasattr("__len__") instead.
    lengths = [len(x) for x in merged["teacher_token_ids"] if hasattr(x, "__len__")]
    if lengths:
        print(f"Response lengths: min={min(lengths)}, median={sorted(lengths)[len(lengths) // 2]}, max={max(lengths)}")
    else:
        print("Response lengths: (no length-iterable rows — skipping stats)")

    merged.to_parquet(args.output, index=False)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
