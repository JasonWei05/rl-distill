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

"""Download JWei05/Nemotron-Cascade-2-SFT-Data-9k-subset and write local parquets.

Adds an empty `teacher_log_probs` column per row so the existing
`DistillSFTDataset` + `forward_kl_loss` path works unchanged. The forward KL loss
is gradient-equivalent to plain cross-entropy when teacher_log_probs contribute
only additively (see forward_kl_loss.py docstring), so an empty/zero teacher
tensor yields CE-SFT with no code changes.

Idempotent: skips if output parquets already exist.
"""

import argparse
import os

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

SPLITS = {
    "train": "data/train-00000-of-00001.parquet",
    "validation": "data/validation-00000-of-00001.parquet",
}


def prep(repo: str, split: str, src_filename: str, out_path: str) -> None:
    if os.path.exists(out_path):
        print(f"[prep_sft_dataset] {out_path} exists; skipping")
        return
    local = hf_hub_download(repo, src_filename, repo_type="dataset")
    table = pq.read_table(local)
    n = table.num_rows
    empty_col = pa.array([[] for _ in range(n)], type=pa.list_(pa.float32()))
    table = table.append_column("teacher_log_probs", empty_col)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pq.write_table(table, out_path)
    print(f"[prep_sft_dataset] {repo} {split}: wrote {n} rows -> {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="JWei05/Nemotron-Cascade-2-SFT-Data-9k-subset")
    p.add_argument("--out_dir", default="/tmp/sft_data")
    args = p.parse_args()

    prep(args.repo, "train", SPLITS["train"], os.path.join(args.out_dir, "train.parquet"))
    prep(args.repo, "val", SPLITS["validation"], os.path.join(args.out_dir, "val.parquet"))


if __name__ == "__main__":
    main()
