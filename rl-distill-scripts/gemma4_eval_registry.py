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

"""Validated registries for the expanded Gemma 4 evaluation matrix."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, TypeVar

IMMUTABLE_REVISION = re.compile(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}")
SHA256 = re.compile(r"[0-9a-f]{64}")
SOURCE_TYPES = frozenset({"hf_snapshot", "hf_subfolder", "s3_hf_export"})
MODEL_CATEGORIES = frozenset({"base", "rl", "distilled"})
DIFFICULTIES = frozenset({"easy", "medium", "hard"})
CORE_MATH_DATASETS = frozenset({"id_easy", "id_medium", "id_hard", "math500", "gsm8k"})
ALL_ID_DATASETS = frozenset({"id_easy", "id_medium", "id_hard"})
OOD_BENCHMARKS = ("mmlu_pro", "gpqa", "mmmlu14k")
ModelT = TypeVar("ModelT")


@dataclass(frozen=True)
class RegisteredModel:
    tag: str
    display_name: str
    category: str
    architecture: str
    trained_on: str | None
    math_datasets: tuple[str, ...]
    source: dict[str, Any]


@dataclass(frozen=True)
class ResolvedModel:
    tag: str
    display_name: str
    category: str
    architecture: str
    trained_on: str | None
    math_datasets: tuple[str, ...]
    model: str
    expected_model_identity_sha256: str
    source: dict[str, Any]


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def load_json_object(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON object {source}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {source}")
    return value


def _require_revision(value: Any, field: str) -> str:
    if not isinstance(value, str) or not IMMUTABLE_REVISION.fullmatch(value):
        raise ValueError(f"{field} must be an immutable 40/64-character hexadecimal revision")
    return value.lower()


def _require_nonempty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _validate_source(source: Any, *, tag: str) -> dict[str, Any]:
    if not isinstance(source, Mapping):
        raise ValueError(f"model {tag!r} source must be an object")
    normalized = dict(source)
    source_type = normalized.get("type")
    if source_type not in SOURCE_TYPES:
        raise ValueError(f"model {tag!r} has unsupported source type {source_type!r}")
    if source_type in {"hf_snapshot", "hf_subfolder"}:
        normalized["repo_id"] = _require_nonempty(normalized.get("repo_id"), f"{tag}.source.repo_id")
        normalized["revision"] = _require_revision(normalized.get("revision"), f"{tag}.source.revision")
        if source_type == "hf_subfolder":
            subfolder = _require_nonempty(normalized.get("subfolder"), f"{tag}.source.subfolder").strip("/")
            if not subfolder or ".." in Path(subfolder).parts:
                raise ValueError(f"{tag}.source.subfolder must stay inside the repository")
            normalized["subfolder"] = subfolder
            normalized["metadata_repo"] = _require_nonempty(
                normalized.get("metadata_repo"), f"{tag}.source.metadata_repo"
            )
            normalized["metadata_revision"] = _require_revision(
                normalized.get("metadata_revision"), f"{tag}.source.metadata_revision"
            )
    else:
        uri = _require_nonempty(normalized.get("uri"), f"{tag}.source.uri")
        completion_uri = _require_nonempty(
            normalized.get("completion_uri"), f"{tag}.source.completion_uri"
        )
        if not uri.startswith("s3://") or not completion_uri.startswith("s3://"):
            raise ValueError(f"model {tag!r} S3 URIs must start with s3://")
        if not uri.endswith("/"):
            raise ValueError(f"model {tag!r} source.uri must end with /")
        normalized["uri"] = uri
        normalized["completion_uri"] = completion_uri
        expected_step = normalized.get("expected_global_step")
        if isinstance(expected_step, bool) or not isinstance(expected_step, int) or expected_step <= 0:
            raise ValueError(f"{tag}.source.expected_global_step must be a positive integer")
        normalized["base_repo"] = _require_nonempty(normalized.get("base_repo"), f"{tag}.source.base_repo")
        normalized["base_revision"] = _require_revision(
            normalized.get("base_revision"), f"{tag}.source.base_revision"
        )
    return normalized


def _validate_common_model(entry: Any, *, index: int) -> tuple[dict[str, Any], tuple[str, ...]]:
    if not isinstance(entry, Mapping):
        raise ValueError(f"model registry entry {index} must be an object")
    value = dict(entry)
    tag = _require_nonempty(value.get("tag"), f"models[{index}].tag")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_]*", tag):
        raise ValueError(f"model tag must contain only lowercase letters, digits, and underscores: {tag!r}")
    value["tag"] = tag
    value["display_name"] = _require_nonempty(value.get("display_name"), f"{tag}.display_name")
    category = value.get("category")
    if category not in MODEL_CATEGORIES:
        raise ValueError(f"model {tag!r} has unsupported category {category!r}")
    value["architecture"] = _require_nonempty(value.get("architecture"), f"{tag}.architecture")
    trained_on = value.get("trained_on")
    if trained_on is not None and trained_on not in DIFFICULTIES:
        raise ValueError(f"model {tag!r} trained_on must be easy, medium, hard, or null")
    if category == "base" and trained_on is not None:
        raise ValueError(f"base model {tag!r} must have trained_on=null")
    if category != "base" and trained_on is None:
        raise ValueError(f"trained model {tag!r} must declare easy or medium lineage")
    datasets = value.get("math_datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError(f"model {tag!r} must have a non-empty math_datasets list")
    names = tuple(str(name) for name in datasets)
    if len(set(names)) != len(names) or set(names) - CORE_MATH_DATASETS:
        raise ValueError(f"model {tag!r} has invalid or duplicate math datasets: {names}")
    required_common = {"math500", "gsm8k"}
    if not required_common.issubset(names):
        raise ValueError(f"model {tag!r} must include MATH500 and GSM8K")
    id_names = set(names) & ALL_ID_DATASETS
    # Base models evaluate every band. RL/distilled models may evaluate just their own band (the
    # legacy registry) or all bands (the distill study: own band = in-distribution, the others
    # report cross-band transfer); either way the own band must be present.
    if category == "base":
        allowed = ({"id_easy", "id_medium"}, set(ALL_ID_DATASETS))
    else:
        allowed = ({f"id_{trained_on}"}, set(ALL_ID_DATASETS))
    expected_ids = allowed[-1]
    if id_names not in allowed:
        raise ValueError(f"model {tag!r} has ID datasets {sorted(id_names)}, expected {sorted(expected_ids)}")
    return value, names


def load_source_registry(path: str | Path) -> tuple[dict[str, Any], list[RegisteredModel]]:
    payload = load_json_object(path)
    if payload.get("schema_version") != 1 or payload.get("protocol") != "gemma4_rl_distill_eval_sources_v1":
        raise ValueError("unsupported Gemma 4 evaluation source registry")
    entries = payload.get("models")
    if not isinstance(entries, list) or not entries:
        raise ValueError("source registry must contain a non-empty models list")
    models = []
    tags: set[str] = set()
    for index, entry in enumerate(entries):
        value, datasets = _validate_common_model(entry, index=index)
        tag = value["tag"]
        if tag in tags:
            raise ValueError(f"duplicate model tag in source registry: {tag}")
        tags.add(tag)
        models.append(
            RegisteredModel(
                tag=tag,
                display_name=value["display_name"],
                category=value["category"],
                architecture=value["architecture"],
                trained_on=value.get("trained_on"),
                math_datasets=datasets,
                source=_validate_source(value.get("source"), tag=tag),
            )
        )
    return payload, models


def load_resolved_registry(path: str | Path) -> tuple[dict[str, Any], list[ResolvedModel]]:
    payload = load_json_object(path)
    if payload.get("schema_version") != 1 or payload.get("protocol") != "gemma4_rl_distill_eval_models_v1":
        raise ValueError("unsupported resolved Gemma 4 evaluation model registry")
    source_sha256 = payload.get("source_registry_sha256")
    if not isinstance(source_sha256, str) or not SHA256.fullmatch(source_sha256):
        raise ValueError("resolved registry has no valid source_registry_sha256")
    entries = payload.get("models")
    if not isinstance(entries, list) or not entries:
        raise ValueError("resolved registry must contain a non-empty models list")
    models = []
    tags: set[str] = set()
    for index, entry in enumerate(entries):
        value, datasets = _validate_common_model(entry, index=index)
        tag = value["tag"]
        if tag in tags:
            raise ValueError(f"duplicate model tag in resolved registry: {tag}")
        tags.add(tag)
        model_path = Path(_require_nonempty(value.get("model"), f"{tag}.model")).expanduser().resolve()
        identity = value.get("expected_model_identity_sha256")
        if not isinstance(identity, str) or not SHA256.fullmatch(identity):
            raise ValueError(f"{tag}.expected_model_identity_sha256 must be a lowercase SHA256 digest")
        models.append(
            ResolvedModel(
                tag=tag,
                display_name=value["display_name"],
                category=value["category"],
                architecture=value["architecture"],
                trained_on=value.get("trained_on"),
                math_datasets=datasets,
                model=str(model_path),
                expected_model_identity_sha256=identity,
                source=_validate_source(value.get("source"), tag=tag),
            )
        )
    return payload, models


def select_models(models: Sequence[ModelT], tags: Sequence[str] | None) -> list[ModelT]:
    if not tags:
        return list(models)
    by_tag = {str(getattr(model, "tag")): model for model in models}
    requested = list(dict.fromkeys(tags))
    missing = [tag for tag in requested if tag not in by_tag]
    if missing:
        raise ValueError(f"unknown model tags: {missing}; available: {sorted(by_tag)}")
    return [by_tag[tag] for tag in requested]


def matrix_request_counts(models: Sequence[RegisteredModel | ResolvedModel]) -> dict[str, Any]:
    per_dataset = {"id_easy": 4_800, "id_medium": 4_800, "id_hard": 4_800, "math500": 8_000, "gsm8k": 10_552}  # bands: 300 q x 16
    math_by_model = {
        model.tag: sum(per_dataset[name] for name in model.math_datasets)
        for model in models
    }
    ood_per_model = 12_032 + 198 + 14_042
    return {
        "math_by_model": math_by_model,
        "math_total_requests": sum(math_by_model.values()),
        "ood_per_model": ood_per_model,
        "ood_total_items": len(models) * ood_per_model,
        "ood_benchmarks": {
            "mmlu_pro": 12_032,
            "gpqa": 198,
            "mmmlu14k": 14_042,
        },
    }
