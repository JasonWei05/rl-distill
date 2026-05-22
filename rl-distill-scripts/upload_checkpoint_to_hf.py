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

import argparse
import os
import time
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--path-in-repo", default="")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--public", action="store_true")
    parser.add_argument("--commit-message", default="Upload checkpoint")
    parser.add_argument("--retries", type=int, default=5)
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"checkpoint dir does not exist: {checkpoint_dir}")

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    api = HfApi(token=token)
    whoami = api.whoami()
    print(f"[upload_checkpoint_to_hf] authenticated as {whoami.get('name')}")

    api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private and not args.public, exist_ok=True)
    if args.public:
        api.update_repo_visibility(repo_id=args.repo_id, repo_type="model", private=False)
        print(f"[upload_checkpoint_to_hf] set {args.repo_id} visibility to public")
    last_error = None
    for attempt in range(1, args.retries + 1):
        try:
            commit = api.upload_folder(
                repo_id=args.repo_id,
                repo_type="model",
                folder_path=str(checkpoint_dir),
                path_in_repo=args.path_in_repo.strip("/"),
                commit_message=args.commit_message,
            )
            break
        except Exception as exc:
            last_error = exc
            if attempt == args.retries:
                raise
            sleep_s = min(60, 2**attempt)
            print(f"[upload_checkpoint_to_hf] attempt {attempt} failed: {exc!r}; retrying in {sleep_s}s")
            time.sleep(sleep_s)
    else:
        raise RuntimeError("upload failed") from last_error
    print(f"[upload_checkpoint_to_hf] uploaded {checkpoint_dir} -> {args.repo_id}/{args.path_in_repo.strip('/')}")
    print(f"[upload_checkpoint_to_hf] commit: {commit.oid}")


if __name__ == "__main__":
    main()
