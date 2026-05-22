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

"""Upload a teacher-generated distillation parquet to a HF dataset repo."""

import argparse
import os
import tempfile

import pandas as pd
from huggingface_hub import HfApi, create_repo


def size_category(n_rows):
    if n_rows < 1_000:
        return "n<1K"
    if n_rows < 10_000:
        return "1K<n<10K"
    if n_rows < 100_000:
        return "10K<n<100K"
    if n_rows < 1_000_000:
        return "100K<n<1M"
    return "n>1M"


def build_readme(args, n_rows, n_prompts):
    return f"""---
license: gemma
task_categories:
- text-generation
- question-answering
language:
- en
tags:
- math
- distillation
- gemma3
- sft
- rl
size_categories:
- {size_category(n_rows)}
---

# {args.repo_id.split("/")[-1]}

Teacher-generated SFT/distillation data for Gemma 3 math distillation.

## Source

- Teacher: `{args.teacher_repo}`, subfolder `{args.teacher_rev}`
- Prompts: `JWei05/DAPO-OpenMathInstruct2-34k`, train split
- Rows: {n_rows:,}
- Unique prompts: {n_prompts:,}
- Responses per prompt: {args.n}
- Sampling: temperature={args.temperature}, top_p={args.top_p}, top_k={args.top_k}, max_tokens={args.max_tokens}

## Columns

| Column | Description |
|---|---|
| `messages` | User prompt and teacher assistant response in chat-message format |
| `teacher_log_probs` | Teacher log-probability of each sampled response token |
| `teacher_token_ids` | Generated response token IDs, aligned 1:1 with `teacher_log_probs` |
| `prompt_idx` | Row index in the source train split |

## Intended Use

Use this with `rl-distill-scripts/main_distill_offpolicy.py` for forward-KL SFT
distillation into smaller Gemma 3 PT models.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", required=True)
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--filename", default=None)
    parser.add_argument("--teacher_repo", required=True)
    parser.add_argument("--teacher_rev", required=True)
    parser.add_argument("--n", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--max_tokens", type=int, default=20480)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.parquet):
        raise SystemExit(f"missing parquet: {args.parquet}")

    df = pd.read_parquet(args.parquet, columns=["prompt_idx"])
    n_rows = len(df)
    n_prompts = df["prompt_idx"].nunique()

    filename = args.filename or os.path.basename(args.parquet)
    api = HfApi()
    create_repo(repo_id=args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)
    if not args.private:
        try:
            api.update_repo_visibility(repo_id=args.repo_id, repo_type="dataset", private=False)
        except Exception as exc:
            print(f"[upload] visibility update skipped: {exc}")

    print(f"[upload] {args.parquet} -> datasets/{args.repo_id}/{filename}")
    api.upload_file(
        path_or_fileobj=args.parquet,
        path_in_repo=filename,
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message=f"add {filename}",
    )

    readme = build_readme(args, n_rows=n_rows, n_prompts=n_prompts)
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        f.write(readme)
        readme_path = f.name
    api.upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message="add dataset card",
    )
    print(f"[upload] done: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
