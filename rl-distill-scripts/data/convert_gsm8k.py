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

"""Convert openai/gsm8k test split to verl format."""

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from datasets import load_dataset


def convert_gsm8k(output_path: str) -> None:
    print("Loading openai/gsm8k (test split) from HuggingFace...")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    print(f"Loaded {len(ds)} problems")

    records = []
    for idx, item in enumerate(ds):
        answer_text = item["answer"]
        # GSM8K answers end with "#### <number>"
        final_answer = answer_text.split("####")[-1].strip()

        problem_text = item["question"]
        if not problem_text.endswith("Please output the final answer within \\boxed{}."):
            problem_text += " Please output the final answer within \\boxed{}."

        records.append(
            {
                "data_source": "gsm8k",
                "prompt": [{"content": problem_text, "role": "user"}],
                "reward_model": {"ground_truth": final_answer, "style": "rule"},
                "extra_info": {"index": idx, "split": "test"},
            }
        )

    df = pd.DataFrame(records)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving {len(df)} problems to {output_path}...")
    df.to_parquet(output_path, index=False)
    print("Done!")


def main():
    parser = ArgumentParser(description="Convert openai/gsm8k to verl format")
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save converted parquet (default: $HOME/verl/data/math__gsm8k_test.parquet)",
    )
    args = parser.parse_args()

    if args.output_path is None:
        args.output_path = str(Path.home() / "verl" / "data" / "math__gsm8k_test.parquet")

    convert_gsm8k(args.output_path)


if __name__ == "__main__":
    main()
