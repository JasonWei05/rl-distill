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

"""Prepare a deterministic train/test split for DAPO-Math-17k.

The split mirrors the DAPO-17.4k preparation used in this repo: hold out a
fixed random subset with seed 42 and use the remaining rows for training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _set_split(extra_info: Any, split: str) -> dict[str, Any]:
    out = dict(extra_info) if isinstance(extra_info, dict) else {}
    out["split"] = split
    return out


def _extra_info_index(extra_info: Any, fallback: int) -> str:
    if isinstance(extra_info, dict) and "index" in extra_info:
        return str(extra_info["index"])
    return str(fallback)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Source dapo-math-17k parquet")
    parser.add_argument("--output-train", required=True)
    parser.add_argument("--output-test", required=True)
    parser.add_argument("--heldout-json", default=None)
    parser.add_argument("--heldout-txt", default=None)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source = Path(args.input)
    train_out = Path(args.output_train)
    test_out = Path(args.output_test)

    df = pd.read_parquet(source)
    if args.test_size <= 0:
        raise ValueError("--test-size must be positive")
    if args.test_size >= len(df):
        raise ValueError(f"--test-size={args.test_size} must be smaller than dataset size {len(df)}")

    test_df = df.sample(n=args.test_size, random_state=args.seed).copy()
    train_df = df.drop(index=test_df.index).copy()

    test_source_rows = [int(i) for i in test_df.index.tolist()]
    test_extra_info_indexes = [
        _extra_info_index(extra_info, source_idx)
        for extra_info, source_idx in zip(test_df["extra_info"].tolist(), test_source_rows, strict=False)
    ]

    train_df = train_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    test_df = test_df.sample(frac=1.0, random_state=args.seed + 1).reset_index(drop=True)

    train_df["extra_info"] = train_df["extra_info"].apply(lambda x: _set_split(x, "train"))
    test_df["extra_info"] = test_df["extra_info"].apply(lambda x: _set_split(x, "test"))

    train_out.parent.mkdir(parents=True, exist_ok=True)
    test_out.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(train_out, index=False)
    test_df.to_parquet(test_out, index=False)

    heldout_json = Path(args.heldout_json) if args.heldout_json else test_out.with_suffix(".heldout.json")
    heldout_txt = Path(args.heldout_txt) if args.heldout_txt else test_out.with_suffix(".heldout.txt")
    heldout = {
        "source": str(source),
        "seed": args.seed,
        "test_size": args.test_size,
        "source_row_idx": test_source_rows,
        "extra_info_index": test_extra_info_indexes,
    }
    heldout_json.write_text(json.dumps(heldout, indent=2) + "\n")
    heldout_txt.write_text("\n".join(test_extra_info_indexes) + "\n")

    print(f"source rows: {len(df)} -> {source}")
    print(f"train rows:  {len(train_df)} -> {train_out}")
    print(f"test rows:   {len(test_df)} -> {test_out}")
    print(f"heldout:     {heldout_json}")
    print(f"heldout txt: {heldout_txt}")


if __name__ == "__main__":
    main()
