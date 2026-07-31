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

"""Create a self-contained vLLM-ready Gemma 4 evaluation checkpoint.

NeMo RL's consolidated Gemma 4 export can omit parameters tied through shared
KV layers and can lack ``processor_config.json``. This materializer copies the
checkpoint metadata, adds processor metadata from an immutable base revision,
expands the omitted shared-KV aliases using the existing checkpoint-chain
helper, rebuilds the safetensors index, and records content-bound provenance.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
LOCAL_HELPERS = SCRIPT_DIR.parent / "nemo_rl_repro" / "local"
for import_path in (SCRIPT_DIR, LOCAL_HELPERS):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from gemma4_distill_trace_schema import sha256_file  # noqa: E402
from gemma4_model_identity import inspect_local_hf_model, require_sha256  # noqa: E402
from materialize_hf_checkpoint_chain import (  # noqa: E402
    expand_gemma4_shared_kv_aliases,
    update_single_file_index,
)

DEFAULT_BASE_MODEL = "google/gemma-4-E2B"
DEFAULT_BASE_REVISION = "d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f"
DEFAULT_PROCESSOR_SHA256 = "32bdf45d2ad4cc29a0822ddd157a182de76644f0419a6228d151495256e9813c"
IMMUTABLE_REVISION = re.compile(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}")


def _copy_metadata(source_dir: Path, output_dir: Path) -> list[str]:
    copied = []
    for source in sorted(source_dir.iterdir()):
        if not source.is_file() or source.name in {"model.safetensors", "model.safetensors.index.json"}:
            continue
        destination = output_dir / source.name
        shutil.copy2(source, destination)
        copied.append(source.name)
    return copied


def materialize_eval_checkpoint(
    *,
    source_dir: Path,
    output_dir: Path,
    processor_path: Path,
    processor_sha256: str,
    expected_source_model_sha256: str | None,
    base_model: str,
    base_revision: str,
) -> dict[str, Any]:
    if not IMMUTABLE_REVISION.fullmatch(base_revision):
        raise ValueError("base_revision must be an immutable 40/64-character hexadecimal revision")
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    source_model = source_dir / "model.safetensors"
    source_config = source_dir / "config.json"
    if not source_model.is_file() or not source_config.is_file():
        raise FileNotFoundError(f"source must contain model.safetensors and config.json: {source_dir}")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")
    partial_dir = output_dir.with_name(output_dir.name + ".partial")
    if partial_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing partial directory: {partial_dir}")

    source_model_sha256 = sha256_file(source_model)
    if expected_source_model_sha256 is not None:
        expected_source_model_sha256 = require_sha256(
            expected_source_model_sha256,
            "--expected-source-model-sha256",
        )
        if source_model_sha256 != expected_source_model_sha256:
            raise ValueError(
                f"source model SHA256 mismatch: expected {expected_source_model_sha256}, found {source_model_sha256}"
            )
    processor_sha256 = require_sha256(processor_sha256, "processor SHA256")
    actual_processor_sha256 = sha256_file(processor_path)
    if actual_processor_sha256 != processor_sha256:
        raise ValueError(
            f"processor_config.json SHA256 mismatch: expected {processor_sha256}, found {actual_processor_sha256}"
        )

    partial_dir.mkdir(parents=True)
    try:
        copied_metadata = _copy_metadata(source_dir, partial_dir)
        processor_destination = partial_dir / "processor_config.json"
        if processor_destination.exists() and sha256_file(processor_destination) != processor_sha256:
            raise ValueError("source processor_config.json conflicts with the pinned base processor metadata")
        if not processor_destination.exists():
            shutil.copy2(processor_path, processor_destination)
            copied_metadata.append("processor_config.json")

        output_model = partial_dir / "model.safetensors"
        aliases = expand_gemma4_shared_kv_aliases(source_model, partial_dir / "config.json", output_model)
        index_path = partial_dir / "model.safetensors.index.json"
        index_path.write_text(json.dumps({"metadata": {}, "weight_map": {}}, sort_keys=True) + "\n")
        update_single_file_index(index_path, output_model)

        materialized_model_sha256 = sha256_file(output_model)
        identity = inspect_local_hf_model(partial_dir).manifest()
        manifest = {
            "schema_version": 1,
            "kind": "gemma4_eval_checkpoint_materialization_v1",
            "source": {
                "path": str(source_dir),
                "model_sha256": source_model_sha256,
                "config_sha256": sha256_file(source_config),
            },
            "base_processor": {
                "model": base_model,
                "revision": base_revision,
                "processor_config_sha256": processor_sha256,
            },
            "output": {
                "path": str(output_dir),
                "model_sha256": materialized_model_sha256,
                "model_identity": identity,
                "shared_kv_alias_count": len(aliases),
                "shared_kv_aliases": aliases,
                "copied_metadata_files": sorted(set(copied_metadata)),
            },
        }
        (partial_dir / "materialization_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(partial_dir, output_dir)
    except Exception:
        shutil.rmtree(partial_dir, ignore_errors=True)
        raise
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-source-model-sha256", default=None)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--base-revision", default=DEFAULT_BASE_REVISION)
    parser.add_argument("--processor-sha256", default=DEFAULT_PROCESSOR_SHA256)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not IMMUTABLE_REVISION.fullmatch(args.base_revision):
        raise ValueError("--base-revision must be an immutable 40/64-character hexadecimal revision")

    from huggingface_hub import hf_hub_download

    processor_path = Path(
        hf_hub_download(
            repo_id=args.base_model,
            repo_type="model",
            revision=args.base_revision,
            filename="processor_config.json",
        )
    )
    manifest = materialize_eval_checkpoint(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        processor_path=processor_path,
        processor_sha256=args.processor_sha256,
        expected_source_model_sha256=args.expected_source_model_sha256,
        base_model=args.base_model,
        base_revision=args.base_revision,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
