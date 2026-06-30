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

"""Download a Hugging Face repo subfolder into a plain local model directory."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download


def _is_weight_or_model_config(path: str) -> bool:
    name = os.path.basename(path)
    if name in {"config.json", "generation_config.json", "model.safetensors.index.json"}:
        return True
    return (
        name.startswith("model-")
        or name.startswith("pytorch_model-")
        or name.endswith(".safetensors")
        or name.endswith(".bin")
    )


def _has_model_files(path: Path) -> bool:
    return path.joinpath("config.json").exists() and (
        path.joinpath("model.safetensors.index.json").exists() or any(path.glob("*.safetensors"))
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--subfolder", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo-type", default="model")
    parser.add_argument("--metadata-repo", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = Path(args.output_dir)
    if _has_model_files(output) and not args.overwrite:
        if args.metadata_repo:
            _patch_metadata(args.metadata_repo, output)
        print(f"model already present: {output}")
        return

    staging = output.parent / f".{output.name}.download"
    if staging.exists():
        shutil.rmtree(staging)
    if output.exists() and args.overwrite:
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        allow_patterns=f"{args.subfolder.rstrip('/')}/*",
        local_dir=str(staging),
        token=os.environ.get("HF_TOKEN"),
    )

    src = staging / args.subfolder.strip("/")
    if not src.joinpath("config.json").exists():
        raise FileNotFoundError(f"missing config.json in downloaded subfolder: {src}")

    if output.exists():
        shutil.rmtree(output)
    shutil.move(str(src), str(output))
    shutil.rmtree(staging, ignore_errors=True)

    if args.metadata_repo:
        _patch_metadata(args.metadata_repo, output)

    print(f"downloaded {args.repo_id}/{args.subfolder} -> {output}")


def _patch_metadata(metadata_repo: str, output: Path) -> None:
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    for filename in api.list_repo_files(repo_id=metadata_repo, repo_type="model"):
        if "/" in filename or _is_weight_or_model_config(filename):
            continue
        target = output / filename
        if target.exists():
            continue
        try:
            hf_hub_download(
                repo_id=metadata_repo,
                repo_type="model",
                filename=filename,
                local_dir=str(output),
                token=os.environ.get("HF_TOKEN"),
            )
            print(f"patched metadata: {filename}")
        except Exception as exc:
            print(f"metadata skip {filename}: {exc}")


if __name__ == "__main__":
    main()
