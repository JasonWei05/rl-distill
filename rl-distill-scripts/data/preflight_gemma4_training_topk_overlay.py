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

"""Fail-closed preflight for Gemma 4 unsharded-HF top-k overlays.

An overlay is not a standalone dataset.  It is accepted only together with
the exact immutable vLLM source bundle it was rescored from.  Successful
output uses the same line-oriented contract as
``preflight_gemma4_topk_distill.py`` so the distillation launcher can consume
either schema without ``eval``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterator, Mapping, Sequence
from itertools import zip_longest
from pathlib import Path
from typing import Any

import gemma4_distill_trace_schema as trace_schema
import numpy as np
import preflight_gemma4_topk_distill as source_preflight
import pyarrow as pa
import pyarrow.parquet as pq
import rescore_gemma4_training_topk as rescorer

ALLOWED_DIRECTIONS = {"e4b_rl100_to_e2b", "e2b_base_to_e4b"}
EXPECTED_TARGET_ENGINE = "hf_bf16_sdpa_full_forward"
EXPECTED_LOADER = "verl class resolver + unsharded transformers.from_pretrained"
EXPECTED_ALIGNMENT = "response token at input index i is scored by hidden/logits at i-1"
EXPECTED_NORMALIZATION = "full-vocabulary logsumexp in FP32 after BF16 LM head and softcap"
EXPECTED_STORAGE = {"token_ids": "int32", "logprobs": "float16"}
EXPECTED_PARITY_COMPARISON = "exact top-k IDs, FP16 top-k logprobs, and FP16 sampled-token logprobs"


class OverlayPreflightError(ValueError):
    """Raised when an unsharded-HF target overlay is unsafe to train from."""


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise OverlayPreflightError(f"cannot read {description} {path}: {error}") from error
    if not isinstance(value, dict):
        raise OverlayPreflightError(f"{description} {path} must contain a JSON object")
    return value


def _require_sha256(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise OverlayPreflightError(f"{field_name} must be a 64-character SHA256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise OverlayPreflightError(f"{field_name} is not hexadecimal") from error
    return value.lower()


def _require_int(value: Any, field_name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise OverlayPreflightError(f"{field_name} must be a JSON integer")
    if minimum is not None and value < minimum:
        raise OverlayPreflightError(f"{field_name} must be at least {minimum}, got {value}")
    return value


def _require_number(value: Any, field_name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise OverlayPreflightError(f"{field_name} must be a JSON number")
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0):
        qualifier = "positive and finite" if positive else "finite"
        raise OverlayPreflightError(f"{field_name} must be {qualifier}")
    return number


def _verify_self_hash(index: Mapping[str, Any], description: str) -> str:
    claimed = _require_sha256(index.get("dataset_index_sha256"), f"{description} dataset_index_sha256")
    unhashed = dict(index)
    del unhashed["dataset_index_sha256"]
    actual = trace_schema.hash_json(unhashed)
    if actual != claimed:
        raise OverlayPreflightError(f"{description} self-hash mismatch: {actual} != {claimed}")
    return claimed


def _resolve_relative(root: Path, value: Any, description: str) -> Path:
    if not isinstance(value, str) or not value:
        raise OverlayPreflightError(f"{description} must be a non-empty relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise OverlayPreflightError(f"{description} must stay within {root}")
    try:
        resolved = (root / relative).resolve(strict=True)
    except OSError as error:
        raise OverlayPreflightError(f"cannot resolve {description} {relative}: {error}") from error
    if not resolved.is_relative_to(root):
        raise OverlayPreflightError(f"{description} escapes {root}")
    return resolved


def _verify_target_identity(target: Any, source_teacher: Mapping[str, Any]) -> str:
    if not isinstance(target, Mapping):
        raise OverlayPreflightError("rescore target_model_identity must be an object")
    target_sha256 = _require_sha256(
        target.get("model_identity_sha256"),
        "rescore target_model_identity.model_identity_sha256",
    )
    source_sha256 = _require_sha256(
        source_teacher.get("model_identity_sha256"),
        "source teacher.model_identity_sha256",
    )
    if target_sha256 != source_sha256:
        raise OverlayPreflightError(
            f"unsharded-HF target is not the exact source teacher: {target_sha256} != {source_sha256}"
        )
    source_content = source_teacher.get("content_sha256")
    if source_content is not None:
        source_content = _require_sha256(source_content, "source teacher.content_sha256")
        target_content = _require_sha256(
            target.get("weight_content_sha256"),
            "rescore target_model_identity.weight_content_sha256",
        )
        if target_content != source_content:
            raise OverlayPreflightError("unsharded-HF target weight content does not match the source teacher")
        if target.get("weight_content_kind") != source_teacher.get("content_sha256_kind"):
            raise OverlayPreflightError("unsharded-HF target weight-content kind does not match the source teacher")
        if target.get("kind") != "local_hf_safetensors_v1":
            raise OverlayPreflightError("a content-bound source teacher requires a local safetensors target identity")
    return target_sha256


def _verify_rescore_config(
    *,
    overlay_root: Path,
    overlay_index: Mapping[str, Any],
    source_index: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], str]:
    config_path = overlay_root / "rescore_config.json"
    if not config_path.is_file():
        raise OverlayPreflightError(f"overlay rescore config does not exist: {config_path}")
    config = _load_json(config_path, "overlay rescore config")
    if config.get("manifest_version") != rescorer.OVERLAY_MANIFEST_VERSION:
        raise OverlayPreflightError("overlay rescore config has an unsupported manifest version")
    if config.get("schema_version") != rescorer.OVERLAY_SCHEMA_VERSION:
        raise OverlayPreflightError("overlay rescore config has an unsupported schema version")
    semantic = config.get("semantic_config")
    if not isinstance(semantic, dict):
        raise OverlayPreflightError("overlay rescore config has no semantic_config object")
    config_sha256 = _require_sha256(config.get("rescore_config_sha256"), "rescore_config_sha256")
    actual_config_sha256 = trace_schema.hash_json(semantic)
    if actual_config_sha256 != config_sha256:
        raise OverlayPreflightError(
            f"overlay rescore semantic hash mismatch: {actual_config_sha256} != {config_sha256}"
        )
    if overlay_index.get("rescore_config_sha256") != config_sha256:
        raise OverlayPreflightError("overlay index and rescore config hashes do not match")
    if not isinstance(config.get("created_at"), str) or not config["created_at"]:
        raise OverlayPreflightError("overlay rescore config created_at must be a non-empty string")

    source_index_sha256 = source_index["dataset_index_sha256"]
    expected_values = {
        "schema_version": rescorer.OVERLAY_SCHEMA_VERSION,
        "source_dataset_index_sha256": source_index_sha256,
        "source_experiment_sha256": source_index["experiment_sha256"],
        "source_direction": source_index["direction"],
        "source_teacher": source_index["teacher"],
        "loader": EXPECTED_LOADER,
        "target_engine": EXPECTED_TARGET_ENGINE,
        "dtype": "bfloat16",
        "attention_implementation": "sdpa",
        "topk_width": rescorer.TOPK_WIDTH,
        "vocab_size": source_index["tokenizer"]["vocab_size"],
        "causal_alignment": EXPECTED_ALIGNMENT,
        "normalization": EXPECTED_NORMALIZATION,
        "storage": EXPECTED_STORAGE,
    }
    for field_name, expected in expected_values.items():
        if semantic.get(field_name) != expected:
            raise OverlayPreflightError(
                f"overlay rescore semantic_config.{field_name} does not match the registered contract"
            )
    _require_int(semantic.get("lm_head_chunk_tokens"), "semantic_config.lm_head_chunk_tokens", minimum=1)
    max_sequence_tokens = _require_int(
        semantic.get("max_sequence_tokens"),
        "semantic_config.max_sequence_tokens",
        minimum=1,
    )
    source_max_tokens = source_index.get("sampling", {}).get("max_model_len")
    if max_sequence_tokens != source_max_tokens:
        raise OverlayPreflightError("rescore max_sequence_tokens does not match the source generation context length")
    softcap = _require_number(
        semantic.get("final_logit_softcapping"),
        "semantic_config.final_logit_softcapping",
        positive=True,
    )
    if softcap != 30.0:
        raise OverlayPreflightError("semantic_config.final_logit_softcapping must be exactly 30.0")
    rescorer_source_sha256 = _require_sha256(
        semantic.get("rescorer_source_sha256"),
        "semantic_config.rescorer_source_sha256",
    )
    current_rescorer_sha256 = trace_schema.sha256_file(Path(rescorer.__file__))
    if rescorer_source_sha256 != current_rescorer_sha256:
        raise OverlayPreflightError(
            "rescore configuration was not produced by the current reviewed rescorer source: "
            f"{rescorer_source_sha256} != {current_rescorer_sha256}"
        )
    if not isinstance(semantic.get("environment_versions"), Mapping):
        raise OverlayPreflightError("semantic_config.environment_versions must be an object")
    target_sha256 = _verify_target_identity(semantic.get("target_model_identity"), source_index["teacher"])

    runtime = config.get("runtime")
    if not isinstance(runtime, Mapping):
        raise OverlayPreflightError("overlay rescore config runtime must be an object")
    runtime_source = runtime.get("source_dataset_index")
    if not isinstance(runtime_source, str) or not runtime_source:
        raise OverlayPreflightError("rescore runtime.source_dataset_index must be a non-empty path")
    # The runtime path is provenance, not identity: an immutable source bundle
    # may be relocated or downloaded from HF.  The explicitly supplied path is
    # selected only by its verified self-hash and per-shard content hashes.
    if not isinstance(runtime.get("model_path"), str) or not runtime["model_path"]:
        raise OverlayPreflightError("rescore runtime.model_path must be a non-empty path")
    return config, semantic, target_sha256


def _verify_parity_receipt(
    *,
    overlay_root: Path,
    config_sha256: str,
    source_index_sha256: str,
    target_model_identity_sha256: str,
) -> None:
    path = overlay_root / rescorer.PARITY_RECEIPT_NAME
    receipt = _load_json(path, "unsharded-HF parity receipt")
    claimed = _require_sha256(receipt.get("parity_receipt_sha256"), "parity receipt self-hash")
    unhashed = dict(receipt)
    del unhashed["parity_receipt_sha256"]
    actual = trace_schema.hash_json(unhashed)
    if actual != claimed:
        raise OverlayPreflightError(f"parity receipt self-hash mismatch: {actual} != {claimed}")
    expected = {
        "schema_version": 1,
        "rescore_config_sha256": config_sha256,
        "source_dataset_index_sha256": source_index_sha256,
        "target_model_identity_sha256": target_model_identity_sha256,
        "target_engine": EXPECTED_TARGET_ENGINE,
        "comparison": EXPECTED_PARITY_COMPARISON,
    }
    for field_name, expected_value in expected.items():
        if receipt.get(field_name) != expected_value:
            raise OverlayPreflightError(f"parity receipt {field_name} does not match the overlay")
    _require_int(receipt.get("checked_rows"), "parity receipt checked_rows", minimum=1)
    _require_int(
        receipt.get("eligibility_response_token_cap"),
        "parity receipt eligibility_response_token_cap",
        minimum=1,
    )
    if not isinstance(receipt.get("passed_at"), str) or not receipt["passed_at"]:
        raise OverlayPreflightError("parity receipt passed_at must be a non-empty string")


def _iter_rows(parquet: pq.ParquetFile, columns: Sequence[str] | None = None) -> Iterator[dict[str, Any]]:
    # One 8K-response row contains more than a million top-k scalar values.
    # Converting several such rows to Python objects at once can exhaust host
    # memory even though the underlying Arrow buffers are compact.
    for batch in parquet.iter_batches(batch_size=1, columns=columns, use_threads=True):
        yield from batch.to_pylist()


def _validate_target_tensors(row: Mapping[str, Any], *, vocab_size: int, row_name: str) -> int:
    input_ids = np.asarray(row["input_ids"], dtype=np.int64)
    response_ids = np.asarray(row["response_token_ids"], dtype=np.int64)
    response_mask = np.asarray(row["response_mask"], dtype=np.int8)
    prompt_length = _require_int(row["prompt_length"], f"{row_name} prompt_length", minimum=1)
    response_length = _require_int(row["response_length"], f"{row_name} response_length", minimum=1)
    if input_ids.ndim != 1 or response_ids.ndim != 1 or response_mask.ndim != 1:
        raise OverlayPreflightError(f"{row_name} token IDs and response mask must be one-dimensional")
    if len(input_ids) != prompt_length + response_length or len(response_mask) != len(input_ids):
        raise OverlayPreflightError(f"{row_name} prompt/response lengths do not match input_ids")
    if len(response_ids) != response_length or not np.array_equal(response_ids, input_ids[prompt_length:]):
        raise OverlayPreflightError(f"{row_name} response_token_ids do not match the response suffix")
    expected_mask = np.concatenate((np.zeros(prompt_length, dtype=np.int8), np.ones(response_length, dtype=np.int8)))
    if not np.array_equal(response_mask, expected_mask):
        raise OverlayPreflightError(f"{row_name} response_mask is not an exact zero-prefix/one-suffix mask")

    ids = np.asarray(row["teacher_topk_token_ids"], dtype=np.int32)
    logprobs = np.asarray(row["teacher_topk_logprobs"], dtype=np.float16)
    sampled_logprobs = np.asarray(row["teacher_sampled_token_logprobs"], dtype=np.float16)
    expected_shape = (response_length, rescorer.TOPK_WIDTH)
    if ids.shape != expected_shape or logprobs.shape != expected_shape:
        raise OverlayPreflightError(
            f"{row_name} top-k target shape mismatch: ids={ids.shape}, logprobs={logprobs.shape}, "
            f"expected={expected_shape}"
        )
    if sampled_logprobs.shape != (response_length,):
        raise OverlayPreflightError(f"{row_name} sampled-token logprobs have the wrong shape")
    ids64 = ids.astype(np.int64, copy=False)
    if ids64.size and (ids64.min() < 0 or ids64.max() >= vocab_size):
        raise OverlayPreflightError(f"{row_name} top-k token ID lies outside the tokenizer vocabulary")
    if ids64.size and np.any(np.diff(np.sort(ids64, axis=1), axis=1) == 0):
        raise OverlayPreflightError(f"{row_name} has a duplicate token ID within a top-k position")
    logprobs32 = logprobs.astype(np.float32)
    sampled32 = sampled_logprobs.astype(np.float32)
    tolerance = rescorer.FP16_MASS_TOLERANCE
    if not np.isfinite(logprobs32).all() or np.any(logprobs32 > 5e-4):
        raise OverlayPreflightError(f"{row_name} has an invalid top-k log probability")
    if np.any(logprobs32[:, 1:] > logprobs32[:, :-1] + 5e-4):
        raise OverlayPreflightError(f"{row_name} top-k log probabilities are not rank sorted")
    masses = np.exp(logprobs32, dtype=np.float64).sum(axis=1)
    if np.any(masses > 1.0 + tolerance):
        raise OverlayPreflightError(f"{row_name} top-k probability mass exceeds one")
    if not np.isfinite(sampled32).all() or np.any(sampled32 > 5e-4):
        raise OverlayPreflightError(f"{row_name} has an invalid sampled-token log probability")
    matches = ids64 == response_ids[:, None]
    matched_rows = np.flatnonzero(matches.any(axis=1))
    if matched_rows.size:
        matched_ranks = matches[matched_rows].argmax(axis=1)
        ranked_values = logprobs32[matched_rows, matched_ranks]
        if np.any(np.abs(ranked_values - sampled32[matched_rows]) > 5e-4):
            raise OverlayPreflightError(f"{row_name} sampled-token logprob disagrees with the ranked target")
    return response_length


def _verify_source_manifest(
    *,
    source_root: Path,
    source_entry: Mapping[str, Any],
    source_parquet: Path,
    source_trace_ids: Sequence[str],
    split_name: str,
    shard_id: int,
) -> tuple[dict[str, Any], str, str]:
    manifest_path = _resolve_relative(
        source_root,
        source_entry.get("manifest_path"),
        f"source {split_name} shard {shard_id} manifest_path",
    )
    manifest = _load_json(manifest_path, "source shard manifest")
    expected = {
        "manifest_version": trace_schema.MANIFEST_VERSION,
        "schema_version": trace_schema.SCHEMA_VERSION,
        "split": split_name,
        "shard_id": shard_id,
        "row_count": len(source_trace_ids),
        "parquet_file": source_parquet.name,
        "parquet_sha256": source_entry["sha256"],
    }
    for field_name, expected_value in expected.items():
        if manifest.get(field_name) != expected_value:
            raise OverlayPreflightError(f"source shard manifest {field_name} mismatch")
    trace_ids_sha256 = trace_schema.hash_json(sorted(source_trace_ids))
    if manifest.get("trace_ids_sha256") != trace_ids_sha256:
        raise OverlayPreflightError("source shard manifest trace-ID hash does not match the source parquet")
    return manifest, trace_schema.sha256_file(manifest_path), trace_ids_sha256


def _verify_overlay_manifest(
    *,
    overlay_entry: Mapping[str, Any],
    manifest_path: Path,
    overlay_parquet: Path,
    source_entry: Mapping[str, Any],
    source_manifest_sha256: str,
    source_trace_ids_sha256: str,
    ordered_trace_ids_sha256: str,
    split_name: str,
    shard_id: int,
    row_count: int,
    response_tokens: int,
    overlay_index: Mapping[str, Any],
    rescore_config_sha256: str,
) -> None:
    manifest = _load_json(manifest_path, "overlay shard manifest")
    parquet_metadata = pq.ParquetFile(overlay_parquet).metadata
    expected = {
        "manifest_version": rescorer.OVERLAY_MANIFEST_VERSION,
        "schema_version": rescorer.OVERLAY_SCHEMA_VERSION,
        "rescore_config_sha256": rescore_config_sha256,
        "source_dataset_index_sha256": overlay_index["source_dataset_index_sha256"],
        "source_experiment_sha256": overlay_index["source_experiment_sha256"],
        "source_parquet_file": source_entry["path"],
        "source_parquet_sha256": source_entry["sha256"],
        "source_manifest_sha256": source_manifest_sha256,
        "source_trace_ids_sha256": source_trace_ids_sha256,
        "split": split_name,
        "shard_id": shard_id,
        "row_count": row_count,
        "ordered_trace_ids_sha256": ordered_trace_ids_sha256,
        "parquet_file": overlay_parquet.name,
        "parquet_sha256": overlay_entry["sha256"],
        "parquet_size_bytes": overlay_parquet.stat().st_size,
        "parquet_row_groups": parquet_metadata.num_row_groups,
    }
    for field_name, expected_value in expected.items():
        if manifest.get(field_name) != expected_value:
            raise OverlayPreflightError(f"overlay shard manifest {field_name} mismatch")
    stats = manifest.get("stats")
    if not isinstance(stats, Mapping):
        raise OverlayPreflightError("overlay shard manifest stats must be an object")
    if _require_int(stats.get("row_count"), "overlay shard stats.row_count", minimum=1) != row_count:
        raise OverlayPreflightError("overlay shard manifest stats.row_count mismatch")
    if (
        _require_int(
            stats.get("response_token_count"),
            "overlay shard stats.response_token_count",
            minimum=1,
        )
        != response_tokens
    ):
        raise OverlayPreflightError("overlay shard manifest response-token count mismatch")
    indexed_expected = {
        "shard_id": shard_id,
        "sha256": manifest["parquet_sha256"],
        "size_bytes": manifest["parquet_size_bytes"],
        "rows": row_count,
        "response_tokens": response_tokens,
        "source_parquet_sha256": source_entry["sha256"],
        "source_manifest_sha256": source_manifest_sha256,
        "source_trace_ids_sha256": source_trace_ids_sha256,
        "ordered_trace_ids_sha256": ordered_trace_ids_sha256,
    }
    for field_name, expected_value in indexed_expected.items():
        if overlay_entry.get(field_name) != expected_value:
            raise OverlayPreflightError(f"overlay index shard {field_name} mismatch")


def _verify_shard_pair(
    *,
    overlay_root: Path,
    source_root: Path,
    overlay_index: Mapping[str, Any],
    source_index: Mapping[str, Any],
    split_name: str,
    shard_id: int,
    overlay_entry: Mapping[str, Any],
    source_entry: Mapping[str, Any],
    semantic: Mapping[str, Any],
    seen_trace_ids: set[str],
) -> tuple[str, int, int]:
    for description, entry in (("source", source_entry), ("overlay", overlay_entry)):
        if not isinstance(entry, Mapping):
            raise OverlayPreflightError(f"{description} {split_name} shard {shard_id} is not an object")
        if _require_int(entry.get("shard_id"), f"{description} shard_id", minimum=0) != shard_id:
            raise OverlayPreflightError(f"{description} {split_name} shard IDs are not contiguous and ordered")
    source_parquet = _resolve_relative(
        source_root,
        source_entry.get("path"),
        f"source {split_name} shard {shard_id} path",
    )
    overlay_parquet = _resolve_relative(
        overlay_root,
        overlay_entry.get("path"),
        f"overlay {split_name} shard {shard_id} path",
    )
    overlay_manifest_path = _resolve_relative(
        overlay_root,
        overlay_entry.get("manifest_path"),
        f"overlay {split_name} shard {shard_id} manifest_path",
    )
    if overlay_manifest_path != overlay_parquet.with_suffix(".manifest.json"):
        raise OverlayPreflightError(
            f"overlay {split_name} shard {shard_id} manifest is not the canonical parquet sibling"
        )
    overlay_manifest = _load_json(overlay_manifest_path, "overlay shard manifest")
    rescoring_timestamp = overlay_manifest.get("created_at")
    if not isinstance(rescoring_timestamp, str) or not rescoring_timestamp:
        raise OverlayPreflightError("overlay shard manifest created_at must be a non-empty string")
    source_sha256 = trace_schema.sha256_file(source_parquet)
    if source_sha256 != _require_sha256(source_entry.get("sha256"), "source shard sha256"):
        raise OverlayPreflightError(f"source {split_name} shard {shard_id} SHA256 mismatch")
    overlay_sha256 = trace_schema.sha256_file(overlay_parquet)
    if overlay_sha256 != _require_sha256(overlay_entry.get("sha256"), "overlay shard sha256"):
        raise OverlayPreflightError(f"overlay {split_name} shard {shard_id} SHA256 mismatch")
    if source_entry.get("size_bytes") != source_parquet.stat().st_size:
        raise OverlayPreflightError(f"source {split_name} shard {shard_id} size mismatch")
    if overlay_entry.get("size_bytes") != overlay_parquet.stat().st_size:
        raise OverlayPreflightError(f"overlay {split_name} shard {shard_id} size mismatch")

    source_file = pq.ParquetFile(source_parquet)
    overlay_file = pq.ParquetFile(overlay_parquet)
    expected_source_schema = trace_schema.trace_arrow_schema()
    if not source_file.schema_arrow.equals(expected_source_schema, check_metadata=False) or (
        source_file.schema_arrow.metadata != expected_source_schema.metadata
    ):
        raise OverlayPreflightError(f"source {split_name} shard {shard_id} has the wrong Arrow schema")
    expected_overlay_schema = rescorer.overlay_schema()
    if not overlay_file.schema_arrow.equals(expected_overlay_schema, check_metadata=False) or (
        overlay_file.schema_arrow.metadata != expected_overlay_schema.metadata
    ):
        raise OverlayPreflightError(f"overlay {split_name} shard {shard_id} has the wrong Arrow schema")
    source_rows = _require_int(source_entry.get("rows"), "source shard rows", minimum=1)
    overlay_rows = _require_int(overlay_entry.get("rows"), "overlay shard rows", minimum=1)
    if source_file.metadata.num_rows != source_rows or overlay_file.metadata.num_rows != overlay_rows:
        raise OverlayPreflightError(f"{split_name} shard {shard_id} parquet/index row-count mismatch")
    if source_rows != overlay_rows:
        raise OverlayPreflightError(f"{split_name} shard {shard_id} source/overlay row counts differ")
    if source_entry.get("row_groups") != source_file.metadata.num_row_groups:
        raise OverlayPreflightError(f"source {split_name} shard {shard_id} row-group count mismatch")

    tokenizer_columns = (
        "tokenizer_model",
        "tokenizer_revision",
        "tokenizer_sha256",
        "tokenizer_vocab_size",
    )
    source_columns = list(dict.fromkeys((*rescorer.COPIED_SOURCE_FIELDS.values(), *tokenizer_columns)))
    source_trace_ids: list[str] = []
    overlay_trace_ids: list[str] = []
    response_tokens = 0
    missing = object()
    target_sha256 = semantic["target_model_identity"]["model_identity_sha256"]
    constants = {
        "schema_version": rescorer.OVERLAY_SCHEMA_VERSION,
        "rescore_config_sha256": overlay_index["rescore_config_sha256"],
        "source_dataset_index_sha256": overlay_index["source_dataset_index_sha256"],
        "source_experiment_sha256": overlay_index["source_experiment_sha256"],
        "source_parquet_sha256": source_entry["sha256"],
        "source_generation_config_sha256": source_index["splits"][split_name]["generation_config_sha256"],
        "direction": source_index["direction"],
        "split": split_name,
        "source_target_engine": "vllm",
        "source_teacher_model": source_index["teacher"]["model"],
        "source_teacher_revision": source_index["teacher"].get("revision"),
        "source_teacher_content_sha256": source_index["teacher"].get("content_sha256"),
        "teacher_topk_rank_order": f"1..{rescorer.TOPK_WIDTH}",
        "target_engine": EXPECTED_TARGET_ENGINE,
        "target_model_identity_sha256": target_sha256,
        "target_dtype": "bfloat16",
        "target_attention_implementation": "sdpa",
        "target_final_logit_softcapping": semantic["final_logit_softcapping"],
        "rescoring_timestamp": rescoring_timestamp,
    }
    source_tokenizer_constants = {
        "tokenizer_model": source_index["tokenizer"]["model"],
        "tokenizer_revision": source_index["tokenizer"].get("revision"),
        "tokenizer_sha256": source_index["tokenizer"]["sha256"],
        "tokenizer_vocab_size": source_index["tokenizer"]["vocab_size"],
    }
    for row_index, (source_row, overlay_row) in enumerate(
        zip_longest(
            _iter_rows(source_file, source_columns),
            _iter_rows(overlay_file),
            fillvalue=missing,
        )
    ):
        if source_row is missing or overlay_row is missing:
            raise OverlayPreflightError(f"{split_name} shard {shard_id} source/overlay row counts differ")
        row_name = f"{split_name} shard {shard_id} row {row_index}"
        if overlay_row.get("row_within_shard") != row_index:
            raise OverlayPreflightError(f"{row_name} row_within_shard is not contiguous")
        for overlay_field, source_field in rescorer.COPIED_SOURCE_FIELDS.items():
            if overlay_row.get(overlay_field) != source_row.get(source_field):
                raise OverlayPreflightError(
                    f"{row_name} copied field {overlay_field} does not match source field {source_field}"
                )
        for field_name, expected_value in constants.items():
            if overlay_row.get(field_name) != expected_value:
                raise OverlayPreflightError(f"{row_name} constant {field_name} mismatch")
        for field_name, expected_value in source_tokenizer_constants.items():
            if source_row.get(field_name) != expected_value:
                raise OverlayPreflightError(f"{row_name} source tokenizer field {field_name} mismatch")
        response_tokens += _validate_target_tensors(
            overlay_row,
            vocab_size=int(source_index["tokenizer"]["vocab_size"]),
            row_name=row_name,
        )
        trace_id = overlay_row.get("trace_id")
        if not isinstance(trace_id, str) or not trace_id:
            raise OverlayPreflightError(f"{row_name} trace_id must be non-empty")
        if trace_id in seen_trace_ids:
            raise OverlayPreflightError(f"overlay repeats trace_id {trace_id!r}")
        seen_trace_ids.add(trace_id)
        source_trace_ids.append(source_row["trace_id"])
        overlay_trace_ids.append(trace_id)
    if source_trace_ids != overlay_trace_ids:
        raise OverlayPreflightError(f"{split_name} shard {shard_id} trace-ID order differs from the source")
    source_manifest, source_manifest_sha256, source_trace_ids_sha256 = _verify_source_manifest(
        source_root=source_root,
        source_entry=source_entry,
        source_parquet=source_parquet,
        source_trace_ids=source_trace_ids,
        split_name=split_name,
        shard_id=shard_id,
    )
    del source_manifest
    ordered_trace_ids_sha256 = trace_schema.hash_json(overlay_trace_ids)
    _verify_overlay_manifest(
        overlay_entry=overlay_entry,
        manifest_path=overlay_manifest_path,
        overlay_parquet=overlay_parquet,
        source_entry=source_entry,
        source_manifest_sha256=source_manifest_sha256,
        source_trace_ids_sha256=source_trace_ids_sha256,
        ordered_trace_ids_sha256=ordered_trace_ids_sha256,
        split_name=split_name,
        shard_id=shard_id,
        row_count=overlay_rows,
        response_tokens=response_tokens,
        overlay_index=overlay_index,
        rescore_config_sha256=overlay_index["rescore_config_sha256"],
    )
    return str(overlay_parquet), overlay_rows, response_tokens


def run_preflight(
    *,
    dataset_index: str | Path,
    source_dataset_index: str | Path,
    student_model: str,
    student_revision: str | None,
    expected_direction: str,
    expected_teacher_identity_sha256: str,
    expected_student_identity_sha256: str,
    local_files_only: bool = False,
    allow_question_overlap: bool = False,
) -> source_preflight.PreflightResult:
    if expected_direction not in ALLOWED_DIRECTIONS:
        raise OverlayPreflightError(f"unsupported expected direction: {expected_direction!r}")
    expected_teacher_identity_sha256 = _require_sha256(
        expected_teacher_identity_sha256,
        "expected_teacher_identity_sha256",
    )
    expected_student_identity_sha256 = _require_sha256(
        expected_student_identity_sha256,
        "expected_student_identity_sha256",
    )
    try:
        overlay_index_path = Path(dataset_index).expanduser().resolve(strict=True)
        source_index_path = Path(source_dataset_index).expanduser().resolve(strict=True)
    except OSError as error:
        raise OverlayPreflightError(f"cannot resolve dataset index: {error}") from error
    overlay_root = overlay_index_path.parent
    source_root = source_index_path.parent
    if (
        overlay_root == source_root
        or overlay_root.is_relative_to(source_root)
        or source_root.is_relative_to(overlay_root)
    ):
        raise OverlayPreflightError("overlay and immutable source roots must be disjoint")

    source_index = _load_json(source_index_path, "source dataset index")
    source_index_sha256 = _verify_self_hash(source_index, "source dataset index")
    if source_index.get("manifest_version") != trace_schema.MANIFEST_VERSION:
        raise OverlayPreflightError("source dataset index has an unsupported manifest version")
    if source_index.get("schema_version") != trace_schema.SCHEMA_VERSION:
        raise OverlayPreflightError("source dataset index is not a vLLM Gemma 4 top-k bundle")
    if source_index.get("direction") != expected_direction:
        raise OverlayPreflightError("source dataset direction does not match the requested experiment")
    source_teacher = source_index.get("teacher")
    if not isinstance(source_teacher, Mapping):
        raise OverlayPreflightError("source dataset teacher must be an object")
    teacher_identity_sha256 = trace_schema.hash_json(source_teacher)
    if teacher_identity_sha256 != expected_teacher_identity_sha256:
        raise OverlayPreflightError("source teacher identity does not match the pinned teacher identity")
    if not isinstance(source_index.get("tokenizer"), Mapping):
        raise OverlayPreflightError("source dataset tokenizer must be an object")

    source_result = source_preflight.run_preflight(
        dataset_index=source_index_path,
        student_model=student_model,
        student_revision=student_revision,
        expected_direction=expected_direction,
        expected_teacher_identity_sha256=expected_teacher_identity_sha256,
        expected_student_identity_sha256=expected_student_identity_sha256,
        local_files_only=local_files_only,
        allow_question_overlap=allow_question_overlap,
    )
    if source_result.dataset_index_sha256 != source_index_sha256:
        raise OverlayPreflightError("source preflight returned an unexpected dataset identity")
    if source_result.direction != expected_direction:
        raise OverlayPreflightError("source preflight returned an unexpected direction")
    if source_result.teacher_identity_sha256 != expected_teacher_identity_sha256:
        raise OverlayPreflightError("source preflight returned an unexpected teacher identity")
    if source_result.student_identity_sha256 != expected_student_identity_sha256:
        raise OverlayPreflightError("source preflight returned an unexpected student identity")
    tokenizer_sha256 = _require_sha256(source_index["tokenizer"].get("sha256"), "source tokenizer.sha256")
    if source_result.student_tokenizer_sha256 != tokenizer_sha256:
        raise OverlayPreflightError("student tokenizer identity does not match the source trace tokenizer")

    overlay_index = _load_json(overlay_index_path, "overlay dataset index")
    overlay_index_sha256 = _verify_self_hash(overlay_index, "overlay dataset index")
    if overlay_index.get("manifest_version") != rescorer.OVERLAY_MANIFEST_VERSION:
        raise OverlayPreflightError("overlay dataset index has an unsupported manifest version")
    if overlay_index.get("schema_version") != rescorer.OVERLAY_SCHEMA_VERSION:
        raise OverlayPreflightError("dataset index is not an unsharded-HF top-k overlay")
    if overlay_index.get("source_dataset_index_sha256") != source_index_sha256:
        raise OverlayPreflightError("overlay is not bound to the explicitly supplied source dataset index")
    if overlay_index.get("source_experiment_sha256") != source_index.get("experiment_sha256"):
        raise OverlayPreflightError("overlay source experiment identity does not match the source dataset")
    if overlay_index.get("direction") != expected_direction:
        raise OverlayPreflightError("overlay direction does not match the requested experiment")
    if _require_int(overlay_index.get("topk_width"), "overlay topk_width", minimum=1) != rescorer.TOPK_WIDTH:
        raise OverlayPreflightError(f"overlay top-k width must be {rescorer.TOPK_WIDTH}")

    config, semantic, target_sha256 = _verify_rescore_config(
        overlay_root=overlay_root,
        overlay_index=overlay_index,
        source_index=source_index,
    )
    if overlay_index.get("target_model_identity") != semantic.get("target_model_identity"):
        raise OverlayPreflightError("overlay index target identity does not match the rescore configuration")
    if overlay_index.get("target_engine") != EXPECTED_TARGET_ENGINE:
        raise OverlayPreflightError("overlay index target engine does not match the registered contract")
    if overlay_index.get("created_at") != config.get("created_at"):
        raise OverlayPreflightError("overlay index created_at does not match the rescore configuration")
    _verify_parity_receipt(
        overlay_root=overlay_root,
        config_sha256=config["rescore_config_sha256"],
        source_index_sha256=source_index_sha256,
        target_model_identity_sha256=target_sha256,
    )

    overlay_splits = overlay_index.get("splits")
    source_splits = source_index.get("splits")
    if not isinstance(overlay_splits, Mapping) or set(overlay_splits) != {"train", "validation"}:
        raise OverlayPreflightError("overlay must contain exactly complete train and validation splits")
    if not isinstance(source_splits, Mapping) or set(source_splits) != {"train", "validation"}:
        raise OverlayPreflightError("source must contain exactly complete train and validation splits")

    files: dict[str, list[str]] = {"train": [], "validation": []}
    seen_paths: set[str] = set()
    seen_trace_ids: set[str] = set()
    total_rows = 0
    total_response_tokens = 0
    for split_name in ("train", "validation"):
        overlay_split = overlay_splits[split_name]
        source_split = source_splits[split_name]
        if not isinstance(overlay_split, Mapping) or not isinstance(source_split, Mapping):
            raise OverlayPreflightError(f"{split_name} split indexes must be objects")
        if overlay_split.get("source_generation_config_sha256") != source_split.get("generation_config_sha256"):
            raise OverlayPreflightError(f"{split_name} generation configuration binding mismatch")
        overlay_shards = overlay_split.get("shards")
        source_shards = source_split.get("shards")
        if not isinstance(overlay_shards, list) or not isinstance(source_shards, list) or not source_shards:
            raise OverlayPreflightError(f"{split_name} source/overlay shards must be non-empty lists")
        if len(overlay_shards) != len(source_shards):
            raise OverlayPreflightError(f"{split_name} overlay does not contain one shard per source shard")
        split_rows = 0
        split_response_tokens = 0
        for shard_id, (overlay_entry, source_entry) in enumerate(zip(overlay_shards, source_shards, strict=True)):
            path, rows, response_tokens = _verify_shard_pair(
                overlay_root=overlay_root,
                source_root=source_root,
                overlay_index=overlay_index,
                source_index=source_index,
                split_name=split_name,
                shard_id=shard_id,
                overlay_entry=overlay_entry,
                source_entry=source_entry,
                semantic=semantic,
                seen_trace_ids=seen_trace_ids,
            )
            if path in seen_paths:
                raise OverlayPreflightError(f"overlay repeats shard path {path}")
            seen_paths.add(path)
            files[split_name].append(path)
            split_rows += rows
            split_response_tokens += response_tokens
        source_rows = _require_int(source_split.get("row_count"), f"source {split_name} row_count", minimum=1)
        source_tokens = _require_int(
            source_split.get("stats", {}).get("response_token_count"),
            f"source {split_name} response_token_count",
            minimum=1,
        )
        indexed_rows = _require_int(overlay_split.get("row_count"), f"overlay {split_name} row_count", minimum=1)
        indexed_tokens = _require_int(
            overlay_split.get("response_token_count"),
            f"overlay {split_name} response_token_count",
            minimum=1,
        )
        if split_rows != source_rows or indexed_rows != source_rows:
            raise OverlayPreflightError(f"{split_name} overlay row total does not exactly cover the source split")
        if split_response_tokens != source_tokens or indexed_tokens != source_tokens:
            raise OverlayPreflightError(
                f"{split_name} overlay response-token total does not exactly cover the source split"
            )
        total_rows += split_rows
        total_response_tokens += split_response_tokens
    if _require_int(overlay_index.get("total_rows"), "overlay total_rows", minimum=1) != total_rows:
        raise OverlayPreflightError("overlay total_rows does not match the verified shard total")
    if (
        _require_int(
            overlay_index.get("total_response_tokens"),
            "overlay total_response_tokens",
            minimum=1,
        )
        != total_response_tokens
    ):
        raise OverlayPreflightError("overlay total_response_tokens does not match the verified shard total")
    if total_rows != source_index.get("total_rows") or total_response_tokens != source_index.get(
        "total_response_tokens"
    ):
        raise OverlayPreflightError("overlay aggregate totals do not exactly cover the source dataset")

    return source_preflight.PreflightResult(
        train_files=tuple(files["train"]),
        validation_files=tuple(files["validation"]),
        topk_width=rescorer.TOPK_WIDTH,
        topk_validation_tolerance=source_preflight.FP16_TOPK_MASS_TOLERANCE,
        dataset_index_sha256=overlay_index_sha256,
        experiment_sha256=source_index["experiment_sha256"],
        direction=expected_direction,
        teacher_identity_sha256=expected_teacher_identity_sha256,
        student_identity_sha256=expected_student_identity_sha256,
        student_tokenizer_sha256=tokenizer_sha256,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-index", required=True)
    parser.add_argument("--source-dataset-index", required=True)
    parser.add_argument("--student-model", required=True)
    parser.add_argument("--student-revision", default=None)
    parser.add_argument("--expected-direction", choices=tuple(sorted(ALLOWED_DIRECTIONS)), required=True)
    parser.add_argument("--expected-teacher-identity-sha256", required=True)
    parser.add_argument("--expected-student-identity-sha256", required=True)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--allow-question-overlap", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run_preflight(
            dataset_index=args.dataset_index,
            source_dataset_index=args.source_dataset_index,
            student_model=args.student_model,
            student_revision=args.student_revision,
            expected_direction=args.expected_direction,
            expected_teacher_identity_sha256=args.expected_teacher_identity_sha256,
            expected_student_identity_sha256=args.expected_student_identity_sha256,
            local_files_only=args.local_files_only,
            allow_question_overlap=args.allow_question_overlap,
        )
    except (OSError, RuntimeError, ValueError, pa.ArrowException, OverlayPreflightError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    sys.stdout.buffer.write(b"\n".join(line.encode("utf-8") for line in result.lines()) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
