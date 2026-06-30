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

"""Split DAPO Gemma 3 teacher data by random held-out source rows.

The eval split contains randomly sampled rows from the source parquet. The train
split contains every source row not selected for eval. The sampled source row
indices are written to JSON and text files so the split can be audited.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="JWei05/DAPO-Gemma3-4B-PT-DAPO-17.4k")
    parser.add_argument("--filename", default="data/train.parquet")
    parser.add_argument("--input-parquet", default=None)
    parser.add_argument("--output-dir", default="/tmp/verl/data/dapo_gemma3_4b_pt_teacher_row_split_seed42_eval500")
    parser.add_argument("--train-output", default=None)
    parser.add_argument("--val-output", default=None)
    parser.add_argument("--heldout-output-json", default=None)
    parser.add_argument("--heldout-output-txt", default=None)
    parser.add_argument("--num-val-rows", type=int, default=None)
    parser.add_argument("--num-val-prompts", type=int, default=None, help="Deprecated alias for --num-val-rows")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_source_parquet(args):
    if args.input_parquet:
        input_path = Path(args.input_parquet)
        if input_path.exists():
            return input_path

        input_path.parent.mkdir(parents=True, exist_ok=True)
        print(
            f"[split] downloading datasets/{args.repo_id}/{args.filename} -> {input_path}",
            flush=True,
        )
        downloaded = hf_hub_download(repo_id=args.repo_id, repo_type="dataset", filename=args.filename)
        shutil.copy2(downloaded, input_path)
        return input_path

    print(f"[split] downloading datasets/{args.repo_id}/{args.filename}", flush=True)
    downloaded = hf_hub_download(repo_id=args.repo_id, repo_type="dataset", filename=args.filename)
    return Path(downloaded)


def write_heldout_outputs(json_path, txt_path, metadata):
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
        f.write("\n")

    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with txt_path.open("w", encoding="utf-8") as f:
        for source_row_idx in metadata["heldout_source_row_idx"]:
            f.write(f"{source_row_idx}\n")


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    num_val_rows = args.num_val_rows
    if num_val_rows is None:
        num_val_rows = args.num_val_prompts if args.num_val_prompts is not None else 500

    train_output = Path(args.train_output or output_dir / "train.parquet")
    val_output = Path(args.val_output or output_dir / "validation.parquet")
    heldout_output_json = Path(
        args.heldout_output_json or output_dir / f"heldout_source_row_idx_seed{args.seed}_n{num_val_rows}.json"
    )
    heldout_output_txt = Path(
        args.heldout_output_txt or output_dir / f"heldout_source_row_idx_seed{args.seed}_n{num_val_rows}.txt"
    )

    source_parquet = resolve_source_parquet(args)
    print(f"[split] reading {source_parquet}", flush=True)
    df = pd.read_parquet(source_parquet)

    if num_val_rows <= 0:
        raise ValueError("--num-val-rows must be positive")
    if num_val_rows >= len(df):
        raise ValueError(f"--num-val-rows={num_val_rows} must be smaller than source row count {len(df)}")

    rng = np.random.default_rng(args.seed)
    heldout_source_row_idx = sorted(rng.choice(np.arange(len(df)), size=num_val_rows, replace=False).tolist())
    heldout_set = set(heldout_source_row_idx)
    is_heldout = np.fromiter((idx in heldout_set for idx in range(len(df))), dtype=bool, count=len(df))

    train_df = df.iloc[~is_heldout].copy()
    val_df = df.iloc[is_heldout].copy()
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    if len(val_df) != num_val_rows:
        raise RuntimeError(f"expected {num_val_rows} eval rows, found {len(val_df)}")
    if len(train_df) + len(val_df) != len(df):
        raise RuntimeError("train/eval row counts do not add up to source row count")

    train_output.parent.mkdir(parents=True, exist_ok=True)
    val_output.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(train_output, index=False)
    val_df.to_parquet(val_output, index=False)

    metadata = {
        "source_repo": args.repo_id,
        "source_filename": args.filename,
        "source_parquet": str(source_parquet),
        "split": "random_source_rows",
        "seed": args.seed,
        "num_val_rows": num_val_rows,
        "source_rows": int(len(df)),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "heldout_source_row_idx": heldout_source_row_idx,
    }
    if "prompt_idx" in df.columns:
        metadata["source_unique_prompt_idx"] = int(df["prompt_idx"].nunique())
        metadata["train_unique_prompt_idx"] = int(train_df["prompt_idx"].nunique())
        metadata["val_unique_prompt_idx"] = int(val_df["prompt_idx"].nunique())
        metadata["heldout_prompt_idx_for_rows"] = val_df["prompt_idx"].tolist()
    write_heldout_outputs(heldout_output_json, heldout_output_txt, metadata)

    print(f"[split] source rows: {len(df)}", flush=True)
    print(f"[split] train rows:  {len(train_df)} -> {train_output}", flush=True)
    print(f"[split] eval rows:   {len(val_df)} -> {val_output}", flush=True)
    print(f"[split] heldout:    {heldout_output_json}", flush=True)
    print(f"[split] heldout txt:{heldout_output_txt}", flush=True)


if __name__ == "__main__":
    main()
