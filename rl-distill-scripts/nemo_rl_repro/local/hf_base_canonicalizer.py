#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Canonicalize the pinned public Gemma-4 base for exact checkpoint deltas."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file

BASE_MODEL_ID = "google/gemma-4-E4B"
BASE_REVISION = "411aa17b749aa952df1359d2dcea73917a544d9a"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(32 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonicalize_base(
    source_model: Path,
    target_index: Path,
    output_model: Path,
) -> dict[str, int | str]:
    """Save only target checkpoint tensors in its deterministic key layout."""

    index = json.loads(target_index.read_text())
    target_keys = sorted(index["weight_map"])
    output_model.parent.mkdir(parents=True, exist_ok=True)
    temp_model = output_model.with_suffix(output_model.suffix + ".tmp")
    try:
        with safe_open(source_model, framework="pt", device="cpu") as source:
            source_keys = set(source.keys())
            missing = sorted(set(target_keys) - source_keys)
            if missing:
                raise ValueError(f"Pinned base is missing {len(missing)} target tensors")
            tensors = {name: source.get_tensor(name) for name in target_keys}
        save_file(tensors, temp_model, metadata={"format": "pt"})
        del tensors
        temp_model.replace(output_model)
    finally:
        temp_model.unlink(missing_ok=True)

    return {
        "base_model_id": BASE_MODEL_ID,
        "base_revision": BASE_REVISION,
        "canonical_bytes": output_model.stat().st_size,
        "canonical_sha256": sha256_file(output_model),
        "tensor_count": len(target_keys),
    }


def canonicalize_pinned_base(target_index: Path, output_model: Path) -> dict[str, int | str]:
    source_model = Path(
        hf_hub_download(
            repo_id=BASE_MODEL_ID,
            revision=BASE_REVISION,
            filename="model.safetensors",
        )
    )
    return canonicalize_base(source_model, target_index, output_model)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-model", type=Path)
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()

    metadata = (
        canonicalize_base(args.source_model, args.target_index, args.output)
        if args.source_model is not None
        else canonicalize_pinned_base(args.target_index, args.output)
    )
    if args.expected_sha256 and metadata["canonical_sha256"] != args.expected_sha256:
        raise ValueError(f"Canonical base SHA-256 does not match the expected value: {metadata['canonical_sha256']}")
    print(json.dumps(metadata, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
