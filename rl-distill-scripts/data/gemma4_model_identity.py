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

"""Content-bound identities for local or immutable remote Hugging Face models."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from gemma4_distill_trace_schema import hash_json, sha256_file


class ModelIdentityError(ValueError):
    """Raised when a model path cannot be bound to one unambiguous identity."""


@dataclass(frozen=True)
class LocalHFModelIdentity:
    model_identity_sha256: str
    weight_content_sha256: str
    weight_content_kind: str
    payload: dict[str, Any]

    def manifest(self) -> dict[str, Any]:
        return {
            "kind": self.payload["kind"],
            "model_identity_sha256": self.model_identity_sha256,
            "weight_content_sha256": self.weight_content_sha256,
            "weight_content_kind": self.weight_content_kind,
        }


def require_sha256(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ModelIdentityError(f"{field_name} must be a 64-character SHA256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ModelIdentityError(f"{field_name} is not hexadecimal") from error
    return value.lower()


def require_immutable_revision(value: Any, field_name: str = "revision") -> str:
    if not isinstance(value, str) or len(value) not in (40, 64):
        raise ModelIdentityError(f"{field_name} must be an immutable 40/64-character hexadecimal revision")
    try:
        int(value, 16)
    except ValueError as error:
        raise ModelIdentityError(f"{field_name} is not hexadecimal") from error
    return value.lower()


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ModelIdentityError(f"cannot read {description} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ModelIdentityError(f"{description} {path} must contain a JSON object")
    return value


def _identity_allowed_roots(model_root: Path) -> tuple[Path, ...]:
    resolved_root = model_root.resolve()
    allowed = [resolved_root]
    revision = resolved_root.name
    is_immutable_snapshot = False
    if resolved_root.parent.name == "snapshots" and len(revision) in (40, 64):
        try:
            int(revision, 16)
        except ValueError:
            pass
        else:
            is_immutable_snapshot = True
    if is_immutable_snapshot:
        blobs = resolved_root.parent.parent / "blobs"
        if blobs.is_dir():
            allowed.append(blobs.resolve())
    return tuple(allowed)


def _validate_identity_file(model_root: Path, path: Path, field_name: str) -> Path:
    if not path.is_file():
        raise ModelIdentityError(f"{field_name} does not exist: {path}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ModelIdentityError(f"cannot resolve {field_name} {path}: {error}") from error
    if not any(resolved.is_relative_to(allowed_root) for allowed_root in _identity_allowed_roots(model_root)):
        raise ModelIdentityError(f"{field_name} escapes the model directory or HF snapshot blobs: {path}")
    return resolved


def _safe_model_relative_path(model_root: Path, value: Any, field_name: str) -> tuple[str, Path]:
    if not isinstance(value, str) or not value:
        raise ModelIdentityError(f"{field_name} must be a non-empty relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ModelIdentityError(f"{field_name} must stay within the model directory")
    normalized = relative.as_posix()
    path = model_root / relative
    _validate_identity_file(model_root, path, field_name)
    if path.suffix != ".safetensors":
        raise ModelIdentityError(f"{field_name} must reference a .safetensors model shard")
    return normalized, path


def inspect_local_hf_model(model_root: str | Path) -> LocalHFModelIdentity:
    """Hash canonical config/index semantics and the exact safetensors shard set."""

    model_root = Path(model_root).expanduser()
    if not model_root.is_dir():
        raise ModelIdentityError(f"local model must be a directory: {model_root}")
    model_root = model_root.resolve()
    config_path = model_root / "config.json"
    _validate_identity_file(model_root, config_path, "model config")
    config = _load_json(config_path, "model config")
    metadata_files: list[dict[str, Any]] = []
    processor_path = model_root / "processor_config.json"
    is_gemma4 = config.get("model_type") == "gemma4"
    if is_gemma4 and not processor_path.is_file():
        raise ModelIdentityError(f"Gemma 4 model is missing processor_config.json: {model_root}")
    if processor_path.is_file():
        _validate_identity_file(model_root, processor_path, "processor config")
        metadata_files.append(
            {
                "path": processor_path.name,
                "semantic_sha256": hash_json(_load_json(processor_path, "processor config")),
            }
        )

    index_path = model_root / "model.safetensors.index.json"
    index_identity: dict[str, Any] | None = None
    if index_path.is_file():
        _validate_identity_file(model_root, index_path, "model safetensors index")
        weight_index = _load_json(index_path, "model safetensors index")
        weight_map = weight_index.get("weight_map")
        if not isinstance(weight_map, Mapping) or not weight_map:
            raise ModelIdentityError(f"model safetensors index {index_path} has no non-empty weight_map")
        normalized_paths: dict[str, Path] = {}
        for position, listed_path in enumerate(weight_map.values()):
            normalized, shard_path = _safe_model_relative_path(
                model_root,
                listed_path,
                f"model safetensors index weight_map value {position}",
            )
            normalized_paths[normalized] = shard_path
        index_identity = {"path": index_path.name, "semantic_sha256": hash_json(weight_index)}
    else:
        model_path = model_root / "model.safetensors"
        if not model_path.is_file():
            discovered = sorted(path.relative_to(model_root).as_posix() for path in model_root.rglob("*.safetensors"))
            if discovered:
                raise ModelIdentityError(
                    f"local model has safetensors shards but no model.safetensors.index.json: {discovered[:5]}"
                )
            raise ModelIdentityError(f"local model has no model.safetensors: {model_root}")
        _validate_identity_file(model_root, model_path, "model.safetensors")
        normalized_paths = {model_path.name: model_path}

    discovered_paths = {
        path.relative_to(model_root).as_posix() for path in model_root.rglob("*.safetensors") if path.is_file()
    }
    if discovered_paths != set(normalized_paths):
        unindexed = sorted(discovered_paths.difference(normalized_paths))
        missing = sorted(set(normalized_paths).difference(discovered_paths))
        raise ModelIdentityError(
            "safetensors files do not exactly match the selected model weights: "
            f"unindexed={unindexed[:5]}, missing={missing[:5]}"
        )

    weight_files = []
    for relative_path, path in sorted(normalized_paths.items()):
        size_bytes = path.stat().st_size
        if size_bytes <= 0:
            raise ModelIdentityError(f"model shard is empty: {path}")
        weight_files.append({"path": relative_path, "size_bytes": size_bytes, "sha256": sha256_file(path)})
    payload = {
        "kind": "local_hf_safetensors_v1",
        "config": {"path": config_path.name, "semantic_sha256": hash_json(config)},
        "metadata_files": metadata_files,
        "weight_index": index_identity,
        "weight_files": weight_files,
    }
    model_identity_sha256 = hash_json(payload)
    if len(weight_files) == 1 and weight_files[0]["path"] == "model.safetensors":
        weight_content_sha256 = str(weight_files[0]["sha256"])
        weight_content_kind = "single_model_safetensors_sha256"
    else:
        weight_content_sha256 = hash_json({"kind": "hf_safetensors_weight_bundle_v1", "weight_files": weight_files})
        weight_content_kind = "hf_safetensors_weight_bundle_v1"
    return LocalHFModelIdentity(
        model_identity_sha256=model_identity_sha256,
        weight_content_sha256=weight_content_sha256,
        weight_content_kind=weight_content_kind,
        payload=payload,
    )


def remote_model_identity(model: str, revision: str) -> dict[str, str]:
    model = model.strip()
    if not model:
        raise ModelIdentityError("remote model ID must be non-empty")
    revision = require_immutable_revision(revision)
    return {
        "kind": "remote_hf_revision_v1",
        "model": model,
        "revision": revision,
        "model_identity_sha256": hash_json({"model": model, "revision": revision}),
    }


def resolve_model_identity(model: str, revision: str | None = None) -> dict[str, Any]:
    local = Path(model).expanduser()
    if local.exists():
        identity = inspect_local_hf_model(local)
        return {
            **identity.manifest(),
            "resolved_path": str(local.resolve()),
        }
    return remote_model_identity(model, require_immutable_revision(revision, "model revision"))


def generation_teacher_identity(model: str, revision: str | None = None) -> dict[str, Any]:
    """Return the exact teacher object stored in generation run configurations."""

    local = Path(model).expanduser()
    if local.exists():
        identity = inspect_local_hf_model(local)
        return {
            "model": model,
            "revision": None,
            "content_sha256": identity.weight_content_sha256,
            "content_sha256_kind": identity.weight_content_kind,
            "model_identity_sha256": identity.model_identity_sha256,
        }
    remote = remote_model_identity(model, require_immutable_revision(revision, "model revision"))
    return {
        "model": remote["model"],
        "revision": remote["revision"],
        "content_sha256": None,
        "content_sha256_kind": None,
        "model_identity_sha256": remote["model_identity_sha256"],
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--include-payload", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    resolved = resolve_model_identity(args.model, args.revision)
    teacher = generation_teacher_identity(args.model, args.revision)
    output: dict[str, Any] = {
        "model": args.model,
        "resolved_identity": resolved,
        "student_identity_sha256": resolved["model_identity_sha256"],
        "generation_teacher": teacher,
        "teacher_identity_sha256": hash_json(teacher),
    }
    local = Path(args.model).expanduser()
    if args.include_payload and local.exists():
        output["local_identity_payload"] = inspect_local_hf_model(local).payload
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
