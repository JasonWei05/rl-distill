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

"""Audit stored vLLM Gemma 4 top-128 targets against the verl HF forward path.

The audit is diagnostic rather than a bit-equality gate. It selects
deterministic, length-stratified traces from the immutable dataset index and
re-scores stored token sequences with an unsharded Hugging Face forward using
the Transformers class selected by verl, SDPA attention, BF16 compute, the
causal position shift, LM head, and final-logit softcap.

The trace and model trees are read-only. The JSON report must be written
outside both trees. The diagnostic informs whether to derive a separate
unsharded-HF training-shaped target overlay; it is not an FSDP2-equivalence
test and never mutates or replaces the immutable vLLM generation bundle. Torch
and Transformers are imported lazily so provenance, selection, and metric
helpers remain CPU-testable.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from gemma4_distill_trace_schema import (
    SCHEMA_VERSION,
    TOPK_WIDTH,
    TraceValidationError,
    atomic_write_json,
    hash_json,
    sha256_file,
    trace_arrow_schema,
    validate_trace_record,
)
from gemma4_model_identity import inspect_local_hf_model

REPORT_VERSION = 1
SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
SPLITS = ("train", "validation")
INDEX_COLUMNS = (
    "trace_id",
    "split",
    "source_uid",
    "sample_index",
    "prompt_length",
    "response_length",
    "shard_id",
    "row_within_shard",
)
PAYLOAD_COLUMNS = tuple(trace_arrow_schema().names)
SUMMARY_FIELDS = (
    "top1_exact",
    "top1_tie_safe",
    "top10_overlap_fraction",
    "topk_overlap_fraction",
    "stored_support_weighted_abs_logprob_delta",
    "stored_support_logprob_abs_delta",
    "stored_support_probability_l1",
    "stored_support_mass_abs_delta",
    "stored_only_topk_mass",
    "reference_only_topk_mass",
    "stored_support_partial_kl_signed",
    "sampled_token_abs_logprob_delta",
)
DEFAULT_THRESHOLDS = {
    "native_vs_manual_projection_max_abs": 1e-6,
    "top1_tie_safe_mean": 0.99,
    "top10_overlap_fraction_mean": 0.97,
    "topk_overlap_fraction_mean": 0.97,
    "stored_support_weighted_abs_logprob_delta_mean": 0.025,
    "stored_support_probability_l1_mean": 0.025,
    "sampled_token_abs_logprob_delta_p95": 0.12,
    "stored_only_topk_mass_p99": 0.005,
    "reference_only_topk_mass_p99": 0.005,
}


class CrossEngineAuditError(ValueError):
    """Raised when provenance or parity-audit invariants do not hold."""


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CrossEngineAuditError(f"cannot read {description} {path}: {error}") from error
    if not isinstance(value, dict):
        raise CrossEngineAuditError(f"{description} {path} must contain a JSON object")
    return value


def _require_sha256(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise CrossEngineAuditError(f"{field_name} must be a 64-character SHA256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise CrossEngineAuditError(f"{field_name} must be hexadecimal") from error
    return value.lower()


def _resolve_registered_path(index_path: Path, trace_root: Path, value: Any, field_name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise CrossEngineAuditError(f"{field_name} must be a non-empty path")
    path = Path(value)
    if not path.is_absolute():
        path = index_path.parent / path
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise CrossEngineAuditError(f"cannot resolve {field_name} {path}: {error}") from error
    if not resolved.is_relative_to(trace_root):
        raise CrossEngineAuditError(f"{field_name} escapes the trace root: {resolved}")
    return resolved


def validate_output_path(output: Path, *, trace_root: Path, model_root: Path) -> Path:
    output = output.expanduser().resolve()
    for protected_root, description in ((trace_root, "trace"), (model_root, "model")):
        if output == protected_root or output.is_relative_to(protected_root):
            raise CrossEngineAuditError(f"--output must be outside the {description} tree")
    return output


def load_dataset_contract(
    *,
    trace_root: Path,
    dataset_index: Path,
    model_root: Path,
    expected_dataset_index_sha256: str,
    expected_teacher_identity_sha256: str,
) -> dict[str, Any]:
    """Bind one complete index, its registered shards, and the exact local teacher."""

    trace_root = trace_root.expanduser().resolve(strict=True)
    dataset_index = dataset_index.expanduser().resolve(strict=True)
    model_root = model_root.expanduser().resolve(strict=True)
    if not dataset_index.is_relative_to(trace_root):
        raise CrossEngineAuditError("--dataset-index must stay within --trace-root")

    index = _load_json(dataset_index, "dataset index")
    claimed_index_hash = _require_sha256(index.get("dataset_index_sha256"), "dataset_index_sha256")
    expected_dataset_index_sha256 = _require_sha256(
        expected_dataset_index_sha256,
        "--expected-dataset-index-sha256",
    )
    if claimed_index_hash != expected_dataset_index_sha256:
        raise CrossEngineAuditError(
            "dataset index does not match --expected-dataset-index-sha256: "
            f"{claimed_index_hash} != {expected_dataset_index_sha256}"
        )
    unhashed = dict(index)
    del unhashed["dataset_index_sha256"]
    actual_index_hash = hash_json(unhashed)
    if actual_index_hash != claimed_index_hash:
        raise CrossEngineAuditError(f"dataset index self-hash mismatch: {actual_index_hash} != {claimed_index_hash}")
    if index.get("schema_version") != SCHEMA_VERSION:
        raise CrossEngineAuditError(f"unsupported trace schema: {index.get('schema_version')!r}")
    if index.get("topk_width") != TOPK_WIDTH:
        raise CrossEngineAuditError(f"dataset top-k width must be exactly {TOPK_WIDTH}")

    teacher = index.get("teacher")
    if not isinstance(teacher, Mapping):
        raise CrossEngineAuditError("dataset index teacher must be an object")
    expected_teacher_identity_sha256 = _require_sha256(
        expected_teacher_identity_sha256,
        "--expected-teacher-identity-sha256",
    )
    teacher_identity_sha256 = hash_json(teacher)
    if teacher_identity_sha256 != expected_teacher_identity_sha256:
        raise CrossEngineAuditError(
            "registered teacher does not match --expected-teacher-identity-sha256: "
            f"{teacher_identity_sha256} != {expected_teacher_identity_sha256}"
        )
    expected_content_hash = _require_sha256(teacher.get("content_sha256"), "teacher.content_sha256")
    expected_model_identity = _require_sha256(
        teacher.get("model_identity_sha256"),
        "teacher.model_identity_sha256",
    )
    local_identity = inspect_local_hf_model(model_root)
    if local_identity.weight_content_sha256 != expected_content_hash:
        raise CrossEngineAuditError(
            "local teacher weight content does not match the trace index: "
            f"{local_identity.weight_content_sha256} != {expected_content_hash}"
        )
    if local_identity.weight_content_kind != teacher.get("content_sha256_kind"):
        raise CrossEngineAuditError("local teacher content-hash kind does not match the trace index")
    if local_identity.model_identity_sha256 != expected_model_identity:
        raise CrossEngineAuditError(
            "local teacher identity does not match the trace index: "
            f"{local_identity.model_identity_sha256} != {expected_model_identity}"
        )

    direction = index.get("direction")
    if direction not in {"e4b_rl100_to_e2b", "e2b_base_to_e4b"}:
        raise CrossEngineAuditError(f"unsupported distillation direction: {direction!r}")
    splits = index.get("splits")
    if not isinstance(splits, Mapping) or set(splits) != set(SPLITS):
        raise CrossEngineAuditError("dataset index must contain exactly train and validation splits")
    registered_shards: dict[str, list[dict[str, Any]]] = {}
    semantic_configs: dict[str, dict[str, Any]] = {}
    generation_config_sha256s: dict[str, str] = {}
    for split in SPLITS:
        split_index = splits[split]
        if not isinstance(split_index, Mapping):
            raise CrossEngineAuditError(f"splits.{split} must be an object")
        if split_index.get("complete") is not True or split_index.get("missing_shard_ids") != []:
            raise CrossEngineAuditError(f"{split} split is not complete")
        shard_entries = split_index.get("shards")
        parquet_files = split_index.get("parquet_files")
        if not isinstance(shard_entries, list) or not shard_entries:
            raise CrossEngineAuditError(f"{split} split has no registered shards")
        if not isinstance(parquet_files, list):
            raise CrossEngineAuditError(f"{split} split has no parquet_files list")
        run_config_path = _resolve_registered_path(
            dataset_index,
            trace_root,
            split_index.get("run_config_path"),
            f"splits.{split}.run_config_path",
        )
        expected_run_config_sha256 = _require_sha256(
            split_index.get("run_config_sha256"),
            f"splits.{split}.run_config_sha256",
        )
        actual_run_config_sha256 = sha256_file(run_config_path)
        if actual_run_config_sha256 != expected_run_config_sha256:
            raise CrossEngineAuditError(
                f"{split} run-config SHA256 mismatch: {actual_run_config_sha256} != {expected_run_config_sha256}"
            )
        run_config = _load_json(run_config_path, f"{split} run config")
        semantic_config = run_config.get("semantic_config")
        if not isinstance(semantic_config, dict):
            raise CrossEngineAuditError(f"{split} run config has no semantic_config object")
        generation_config_sha256 = _require_sha256(
            run_config.get("generation_config_sha256"),
            f"{split} generation_config_sha256",
        )
        if hash_json(semantic_config) != generation_config_sha256:
            raise CrossEngineAuditError(f"{split} semantic configuration hash mismatch")
        if split_index.get("generation_config_sha256") != generation_config_sha256:
            raise CrossEngineAuditError(f"{split} index/run-config generation hashes differ")
        if semantic_config.get("split") != split or semantic_config.get("direction") != direction:
            raise CrossEngineAuditError(f"{split} run config declares the wrong split/direction")
        semantic_configs[split] = semantic_config
        generation_config_sha256s[split] = generation_config_sha256

        normalized_entries = []
        listed_paths = []
        for shard_position, shard in enumerate(shard_entries):
            if not isinstance(shard, Mapping):
                raise CrossEngineAuditError(f"{split} shard {shard_position} must be an object")
            path_value = shard.get("path")
            path = _resolve_registered_path(
                dataset_index,
                trace_root,
                path_value,
                f"splits.{split}.shards[{shard_position}].path",
            )
            if path.suffix != ".parquet":
                raise CrossEngineAuditError(f"registered shard is not Parquet: {path}")
            normalized_entries.append(
                {
                    "path": path,
                    "path_in_index": str(path_value),
                    "shard_id": int(shard.get("shard_id", -1)),
                    "sha256": _require_sha256(
                        shard.get("sha256"),
                        f"splits.{split}.shards[{shard_position}].sha256",
                    ),
                }
            )
            listed_paths.append(str(path_value))
        if listed_paths != parquet_files:
            raise CrossEngineAuditError(f"{split} shard paths do not match parquet_files")
        registered_shards[split] = normalized_entries

    return {
        "index": index,
        "index_path": dataset_index,
        "index_sha256": claimed_index_hash,
        "index_file_sha256": sha256_file(dataset_index),
        "trace_root": trace_root,
        "model_root": model_root,
        "model_identity": local_identity,
        "teacher_content_sha256": expected_content_hash,
        "teacher_identity_sha256": teacher_identity_sha256,
        "shards": registered_shards,
        "semantic_configs": semantic_configs,
        "generation_config_sha256s": generation_config_sha256s,
    }


def scan_candidates(
    registered_shards: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    """Read compact row metadata from every index-registered shard."""

    candidates: list[dict[str, Any]] = []
    for split in SPLITS:
        for entry in registered_shards[split]:
            path = Path(entry["path"])
            try:
                table = pq.read_table(path, columns=list(INDEX_COLUMNS))
            except Exception as error:
                raise CrossEngineAuditError(f"cannot read audit columns from {path}: {error}") from error
            for row_index, row in enumerate(table.to_pylist()):
                if row.get("split") != split:
                    raise CrossEngineAuditError(f"row split mismatch in {path}:{row_index}")
                if int(row.get("response_length", 0)) <= 0:
                    raise CrossEngineAuditError(f"empty response in {path}:{row_index}")
                row["path"] = str(path)
                row["row_index"] = row_index
                row["registered_shard_sha256"] = entry["sha256"]
                row["path_in_index"] = entry["path_in_index"]
                if int(row["shard_id"]) != int(entry["shard_id"]):
                    raise CrossEngineAuditError(f"shard ID mismatch in {path}:{row_index}")
                if int(row["row_within_shard"]) != row_index:
                    raise CrossEngineAuditError(f"row_within_shard mismatch in {path}:{row_index}")
                candidates.append(row)
    if not candidates:
        raise CrossEngineAuditError("no candidate traces were found")
    return candidates


def verify_registered_shards(registered_shards: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    """Hash every shard used for global selection against the immutable index."""

    expected_by_path: dict[str, str] = {}
    for split in SPLITS:
        for item in registered_shards[split]:
            path = str(item["path"])
            expected = str(item["sha256"])
            prior = expected_by_path.setdefault(path, expected)
            if prior != expected:
                raise CrossEngineAuditError(f"conflicting registered hashes for {path}")
    for path_string, expected in expected_by_path.items():
        actual = sha256_file(path_string)
        if actual != expected:
            raise CrossEngineAuditError(f"registered shard SHA256 mismatch for {path_string}: {actual} != {expected}")


def stratified_selection(candidates: Sequence[Mapping[str, Any]], *, per_split: int) -> list[dict[str, Any]]:
    """Select exact response-length quantiles independently for each split."""

    if per_split <= 0:
        raise CrossEngineAuditError("--traces-per-split must be positive")
    selected: list[dict[str, Any]] = []
    for split in SPLITS:
        pool = sorted(
            (dict(row) for row in candidates if row.get("split") == split),
            key=lambda row: (int(row["response_length"]), str(row["trace_id"])),
        )
        if len(pool) < per_split:
            raise CrossEngineAuditError(f"{split} has only {len(pool)} candidates, fewer than requested {per_split}")
        indices = np.linspace(0, len(pool) - 1, num=per_split, dtype=np.int64).tolist()
        if len(set(indices)) != per_split:
            raise CrossEngineAuditError(f"could not select {per_split} distinct {split} traces")
        selected.extend(pool[index] for index in indices)
    return selected


def _row_group_offsets(parquet_file: pq.ParquetFile) -> list[int]:
    offsets = [0]
    for row_group in range(parquet_file.metadata.num_row_groups):
        offsets.append(offsets[-1] + parquet_file.metadata.row_group(row_group).num_rows)
    return offsets


def _row_group_for_index(offsets: Sequence[int], row_index: int) -> int:
    if row_index < 0 or row_index >= offsets[-1]:
        raise CrossEngineAuditError(f"Parquet row index {row_index} is out of range")
    return int(np.searchsorted(np.asarray(offsets), row_index, side="right") - 1)


def load_selected_rows(selection: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Read only row groups containing selected records, not complete large shards."""

    by_path: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in selection:
        by_path[str(item["path"])].append(item)
    rows: list[dict[str, Any]] = []
    for path_string, items in by_path.items():
        path = Path(path_string)
        parquet_file = pq.ParquetFile(path)
        offsets = _row_group_offsets(parquet_file)
        by_row_group: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for item in items:
            by_row_group[_row_group_for_index(offsets, int(item["row_index"]))].append(item)
        for row_group, group_items in by_row_group.items():
            try:
                table = parquet_file.read_row_group(row_group, columns=list(PAYLOAD_COLUMNS))
            except Exception as error:
                raise CrossEngineAuditError(f"cannot read selected payload from {path}: {error}") from error
            group_start = offsets[row_group]
            for item in group_items:
                local_index = int(item["row_index"]) - group_start
                row = table.slice(local_index, 1).to_pylist()[0]
                if row.get("trace_id") != item.get("trace_id"):
                    raise CrossEngineAuditError(f"selected trace changed at {path}:{item['row_index']}")
                row["path"] = str(path)
                row["path_in_index"] = item["path_in_index"]
                row["registered_shard_sha256"] = item["registered_shard_sha256"]
                row["row_index"] = int(item["row_index"])
                rows.append(row)
    return sorted(rows, key=lambda row: (row["split"], int(row["response_length"]), row["trace_id"]))


def selected_positions(response_length: int, count: int) -> list[int]:
    if response_length <= 0:
        raise CrossEngineAuditError("trace has no response tokens")
    if count <= 0:
        raise CrossEngineAuditError("--positions-per-trace must be positive")
    count = min(count, response_length)
    positions = sorted(set(np.linspace(0, response_length - 1, num=count, dtype=np.int64).tolist()))
    if len(positions) != count:
        raise CrossEngineAuditError(f"could not select {count} distinct response positions")
    return positions


def summary(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "min": None,
            "p01": None,
            "p05": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise CrossEngineAuditError("audit metric contains a non-finite value")
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "min": float(array.min()),
        "p01": float(np.quantile(array, 0.01)),
        "p05": float(np.quantile(array, 0.05)),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(array.max()),
    }


def compare_topk_position(
    *,
    stored_ids: Sequence[int],
    stored_logprobs: Sequence[float],
    reference_top_ids: Sequence[int],
    reference_top_logprobs: Sequence[float],
    reference_logprobs_on_stored: Sequence[float],
    stored_sampled_logprob: float,
    reference_sampled_logprob: float,
    top1_tie_logprob_tolerance: float,
) -> dict[str, float | int]:
    """Compute one position's cross-engine metrics with NumPy."""

    stored_ids_array = np.asarray(stored_ids, dtype=np.int64)
    stored_logprobs_array = np.asarray(stored_logprobs, dtype=np.float64)
    reference_ids_array = np.asarray(reference_top_ids, dtype=np.int64)
    reference_top_logprobs_array = np.asarray(reference_top_logprobs, dtype=np.float64)
    reference_on_stored_array = np.asarray(reference_logprobs_on_stored, dtype=np.float64)
    expected_shape = (TOPK_WIDTH,)
    for field_name, value in (
        ("stored_ids", stored_ids_array),
        ("stored_logprobs", stored_logprobs_array),
        ("reference_top_ids", reference_ids_array),
        ("reference_top_logprobs", reference_top_logprobs_array),
        ("reference_logprobs_on_stored", reference_on_stored_array),
    ):
        if value.shape != expected_shape:
            raise CrossEngineAuditError(f"{field_name} shape must be {expected_shape}, got {value.shape}")
    if len(set(stored_ids_array.tolist())) != TOPK_WIDTH:
        raise CrossEngineAuditError("stored top-k token IDs are not unique")
    if len(set(reference_ids_array.tolist())) != TOPK_WIDTH:
        raise CrossEngineAuditError("reference top-k token IDs are not unique")
    if not all(
        np.isfinite(value).all()
        for value in (
            stored_logprobs_array,
            reference_top_logprobs_array,
            reference_on_stored_array,
        )
    ):
        raise CrossEngineAuditError("top-k log probabilities must be finite")

    stored_probs = np.exp(stored_logprobs_array)
    reference_probs_on_stored = np.exp(reference_on_stored_array)
    absolute_logprob_delta = np.abs(stored_logprobs_array - reference_on_stored_array)
    stored_mass = float(stored_probs.sum())
    reference_mass_on_stored = float(reference_probs_on_stored.sum())
    stored_set = set(int(value) for value in stored_ids_array)
    reference_set = set(int(value) for value in reference_ids_array)
    stored_only_mask = np.asarray(
        [int(token_id) not in reference_set for token_id in stored_ids_array],
        dtype=np.bool_,
    )
    reference_only_mask = np.asarray(
        [int(token_id) not in stored_set for token_id in reference_ids_array],
        dtype=np.bool_,
    )
    top1_exact = int(int(stored_ids_array[0]) == int(reference_ids_array[0]))
    top1_tie_safe = int(
        top1_exact
        or float(reference_top_logprobs_array[0] - reference_on_stored_array[0]) <= top1_tie_logprob_tolerance
    )
    return {
        "top1_exact": top1_exact,
        "top1_tie_safe": top1_tie_safe,
        "top10_overlap_fraction": len(set(stored_ids_array[:10]) & set(reference_ids_array[:10])) / 10.0,
        "topk_overlap_fraction": len(stored_set & reference_set) / float(TOPK_WIDTH),
        "stored_support_weighted_abs_logprob_delta": float(
            np.sum(stored_probs * absolute_logprob_delta) / max(stored_mass, 1e-300)
        ),
        "stored_support_logprob_abs_delta": float(absolute_logprob_delta.mean()),
        "stored_support_probability_l1": float(np.abs(stored_probs - reference_probs_on_stored).sum()),
        "stored_support_mass_abs_delta": abs(stored_mass - reference_mass_on_stored),
        "stored_only_topk_mass": float(stored_probs[stored_only_mask].sum()),
        "reference_only_topk_mass": float(np.exp(reference_top_logprobs_array[reference_only_mask]).sum()),
        "stored_support_partial_kl_signed": float(
            np.sum(stored_probs * (stored_logprobs_array - reference_on_stored_array))
        ),
        "sampled_token_abs_logprob_delta": abs(float(stored_sampled_logprob) - float(reference_sampled_logprob)),
        "stored_topk_mass": stored_mass,
        "reference_mass_on_stored_support": reference_mass_on_stored,
    }


def aggregate(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = {field: summary([float(record[field]) for record in records]) for field in SUMMARY_FIELDS}
    bins: dict[str, list[Mapping[str, Any]]] = {
        "first": [],
        "early": [],
        "middle": [],
        "late": [],
    }
    for record in records:
        fraction = float(record["response_fraction"])
        if int(record["response_position"]) == 0:
            region = "first"
        elif fraction < 0.25:
            region = "early"
        elif fraction < 0.75:
            region = "middle"
        else:
            region = "late"
        bins[region].append(record)
    result["by_response_region"] = {
        region: {
            "count": len(rows),
            "weighted_abs_logprob_delta": summary(
                [float(row["stored_support_weighted_abs_logprob_delta"]) for row in rows]
            ),
            "sampled_token_abs_logprob_delta": summary([float(row["sampled_token_abs_logprob_delta"]) for row in rows]),
            "top1_exact": summary([float(row["top1_exact"]) for row in rows]),
        }
        for region, rows in bins.items()
    }
    return result


def thresholds_from_args(args: argparse.Namespace) -> dict[str, float]:
    thresholds = {
        "native_vs_manual_projection_max_abs": args.max_native_projection_abs,
        "top1_tie_safe_mean": args.min_top1_tie_safe_mean,
        "top10_overlap_fraction_mean": args.min_top10_overlap_mean,
        "topk_overlap_fraction_mean": args.min_topk_overlap_mean,
        "stored_support_weighted_abs_logprob_delta_mean": (args.max_weighted_abs_logprob_delta_mean),
        "stored_support_probability_l1_mean": args.max_support_probability_l1_mean,
        "sampled_token_abs_logprob_delta_p95": args.max_sampled_token_abs_delta_p95,
        "stored_only_topk_mass_p99": args.max_stored_only_topk_mass_p99,
        "reference_only_topk_mass_p99": args.max_reference_only_topk_mass_p99,
    }
    if any(not math.isfinite(value) or value < 0 for value in thresholds.values()):
        raise CrossEngineAuditError("all audit thresholds must be finite and non-negative")
    for field_name in (
        "top1_tie_safe_mean",
        "top10_overlap_fraction_mean",
        "topk_overlap_fraction_mean",
    ):
        if thresholds[field_name] > 1:
            raise CrossEngineAuditError(f"{field_name} cannot exceed one")
    return thresholds


def evaluate_thresholds(
    audit_aggregate: Mapping[str, Any],
    *,
    native_projection_max_abs: float,
    thresholds: Mapping[str, float],
) -> dict[str, Any]:
    """Evaluate explicit calibrated gates and return an auditable decision."""

    observations = {
        "native_vs_manual_projection_max_abs": float(native_projection_max_abs),
        "top1_tie_safe_mean": float(audit_aggregate["top1_tie_safe"]["mean"]),
        "top10_overlap_fraction_mean": float(audit_aggregate["top10_overlap_fraction"]["mean"]),
        "topk_overlap_fraction_mean": float(audit_aggregate["topk_overlap_fraction"]["mean"]),
        "stored_support_weighted_abs_logprob_delta_mean": float(
            audit_aggregate["stored_support_weighted_abs_logprob_delta"]["mean"]
        ),
        "stored_support_probability_l1_mean": float(audit_aggregate["stored_support_probability_l1"]["mean"]),
        "sampled_token_abs_logprob_delta_p95": float(audit_aggregate["sampled_token_abs_logprob_delta"]["p95"]),
        "stored_only_topk_mass_p99": float(audit_aggregate["stored_only_topk_mass"]["p99"]),
        "reference_only_topk_mass_p99": float(audit_aggregate["reference_only_topk_mass"]["p99"]),
    }
    minimum_fields = {
        "top1_tie_safe_mean",
        "top10_overlap_fraction_mean",
        "topk_overlap_fraction_mean",
    }
    checks = {}
    failures = []
    for field_name, observed in observations.items():
        threshold = float(thresholds[field_name])
        comparator = ">=" if field_name in minimum_fields else "<="
        passed = observed >= threshold if comparator == ">=" else observed <= threshold
        checks[field_name] = {
            "observed": observed,
            "comparator": comparator,
            "threshold": threshold,
            "passed": passed,
        }
        if not passed:
            failures.append(f"{field_name}={observed:.12g} does not satisfy {comparator} {threshold:.12g}")
    return {
        "status": "pass" if not failures else "fail",
        "checks": checks,
        "failure_reasons": failures,
    }


def repository_provenance() -> dict[str, Any]:
    def git(*arguments: str) -> str:
        try:
            return subprocess.run(
                ["git", "-C", str(REPO_ROOT), *arguments],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as error:
            raise CrossEngineAuditError(f"cannot resolve repository provenance: {error}") from error

    status = git("status", "--porcelain", "--untracked-files=all")
    return {
        "repository_root": str(REPO_ROOT),
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(status),
        "script_path": str(SCRIPT_PATH.relative_to(REPO_ROOT)),
        "script_sha256": sha256_file(SCRIPT_PATH),
    }


def _validate_trace_row(
    row: Mapping[str, Any],
    *,
    teacher_content_sha256: str,
    generation_config_sha256: str,
    semantic_config: Mapping[str, Any],
) -> tuple[list[int], list[int]]:
    try:
        validate_trace_record(
            row,
            decoder=None,
            expected_config_sha256=generation_config_sha256,
            expected_direction=str(semantic_config["direction"]),
            expected_split=str(semantic_config["split"]),
            expected_shard_id=int(row["shard_id"]),
            expected_row_within_shard=int(row["row_index"]),
            expected_semantic_config=semantic_config,
            max_prompt_tokens=int(semantic_config["sampling"]["max_prompt_tokens"]),
            max_response_tokens=int(semantic_config["sampling"]["max_response_tokens"]),
        )
    except TraceValidationError as error:
        raise CrossEngineAuditError(f"invalid selected trace {row.get('trace_id')}: {error}") from error
    input_ids = [int(value) for value in row["input_ids"]]
    response_mask = [int(value) for value in row["response_mask"]]
    response_ids = [int(value) for value in row["response_token_ids"]]
    sampled_ids = [int(value) for value in row["sampled_token_ids"]]
    prompt_length = int(row["prompt_length"])
    response_length = int(row["response_length"])
    if response_length <= 0:
        raise CrossEngineAuditError(f"empty response for trace {row['trace_id']}")
    if len(input_ids) != prompt_length + response_length:
        raise CrossEngineAuditError(f"length mismatch for trace {row['trace_id']}")
    if response_mask != [0] * prompt_length + [1] * response_length:
        raise CrossEngineAuditError(f"response-mask mismatch for trace {row['trace_id']}")
    if input_ids[prompt_length:] != response_ids or response_ids != sampled_ids:
        raise CrossEngineAuditError(f"response/sample ID mismatch for trace {row['trace_id']}")
    if row.get("teacher_content_sha256") != teacher_content_sha256:
        raise CrossEngineAuditError(f"teacher content mismatch for trace {row['trace_id']}")
    return input_ids, response_ids


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    if args.traces_per_split <= 0:
        raise CrossEngineAuditError("--traces-per-split must be positive")
    if args.positions_per_trace <= 0:
        raise CrossEngineAuditError("--positions-per-trace must be positive")
    if not math.isfinite(args.top1_tie_logprob_tolerance) or args.top1_tie_logprob_tolerance < 0:
        raise CrossEngineAuditError("--top1-tie-logprob-tolerance must be finite and non-negative")
    thresholds = thresholds_from_args(args)

    trace_root = Path(args.trace_root).expanduser().resolve(strict=True)
    model_root = Path(args.model).expanduser().resolve(strict=True)
    dataset_index = Path(args.dataset_index).expanduser() if args.dataset_index else trace_root / "dataset_index.json"
    output = validate_output_path(
        Path(args.output),
        trace_root=trace_root,
        model_root=model_root,
    )
    if output.exists() and not args.overwrite:
        raise CrossEngineAuditError(f"output already exists: {output}; pass --overwrite to replace it")
    contract = load_dataset_contract(
        trace_root=trace_root,
        dataset_index=dataset_index,
        model_root=model_root,
        expected_dataset_index_sha256=args.expected_dataset_index_sha256,
        expected_teacher_identity_sha256=args.expected_teacher_identity_sha256,
    )
    verify_registered_shards(contract["shards"])
    candidates = scan_candidates(contract["shards"])
    selection = stratified_selection(candidates, per_split=args.traces_per_split)
    rows = load_selected_rows(selection)

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import torch
    import transformers
    from transformers import AutoConfig

    from verl.utils.model import get_hf_auto_model_class

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise CrossEngineAuditError("cross-engine BF16 audit requires an available CUDA device")
    config = AutoConfig.from_pretrained(
        model_root,
        attn_implementation=args.attn_implementation,
        local_files_only=True,
    )
    if getattr(config, "model_type", None) != "gemma4":
        raise CrossEngineAuditError(
            f"cross-engine audit requires Gemma 4, found {getattr(config, 'model_type', None)!r}"
        )
    auto_class = get_hf_auto_model_class(config)
    model = auto_class.from_pretrained(
        model_root,
        config=config,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    text_config = model.config.get_text_config()
    softcap = getattr(text_config, "final_logit_softcapping", None)
    if softcap is None or float(softcap) <= 0:
        raise CrossEngineAuditError(f"expected positive final_logit_softcapping, got {softcap}")
    if not math.isclose(
        float(softcap),
        args.expected_final_logit_softcap,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise CrossEngineAuditError(
            "final_logit_softcapping does not match the registered expectation: "
            f"{softcap} != {args.expected_final_logit_softcap}"
        )
    vocab_size = int(text_config.vocab_size)
    if TOPK_WIDTH > vocab_size:
        raise CrossEngineAuditError("top-k width exceeds model vocabulary")
    if int(contract["index"]["tokenizer"]["vocab_size"]) != vocab_size:
        raise CrossEngineAuditError("trace tokenizer vocabulary size does not match the teacher model vocabulary")

    position_records: list[dict[str, Any]] = []
    trace_records: list[dict[str, Any]] = []
    native_projection_max_abs: float | None = None
    torch.cuda.reset_peak_memory_stats(device)

    for trace_number, row in enumerate(rows):
        input_ids_list, response_ids = _validate_trace_row(
            row,
            teacher_content_sha256=contract["teacher_content_sha256"],
            generation_config_sha256=contract["generation_config_sha256s"][row["split"]],
            semantic_config=contract["semantic_configs"][row["split"]],
        )
        prompt_length = int(row["prompt_length"])
        response_length = int(row["response_length"])
        response_positions = selected_positions(response_length, args.positions_per_trace)
        prediction_positions = [prompt_length - 1 + response_position for response_position in response_positions]
        for response_position, prediction_position in zip(
            response_positions,
            prediction_positions,
            strict=True,
        ):
            if input_ids_list[prediction_position + 1] != response_ids[response_position]:
                raise CrossEngineAuditError("causal response-position shift is not aligned")

        input_ids = torch.tensor(input_ids_list, dtype=torch.long, device=device).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        position_ids = torch.arange(
            input_ids.shape[1],
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)
        prediction_index_tensor = torch.tensor(
            prediction_positions,
            dtype=torch.long,
            device=device,
        )

        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            backbone_output = model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                return_dict=True,
            )
            hidden = backbone_output.last_hidden_state[:, prediction_index_tensor, :]
            logits = model.lm_head(hidden).squeeze(0)
            logits = torch.tanh(logits / float(softcap)) * float(softcap)

        logits_fp32 = logits.float()
        log_denominator = torch.logsumexp(logits_fp32, dim=-1, keepdim=True)
        reference_top_logits, reference_top_ids = torch.topk(
            logits_fp32,
            k=TOPK_WIDTH,
            dim=-1,
        )
        reference_top_logprobs = reference_top_logits - log_denominator

        if trace_number == 0:
            with (
                torch.inference_mode(),
                torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                ),
            ):
                native = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    logits_to_keep=prediction_index_tensor,
                    use_cache=False,
                    return_dict=True,
                ).logits.squeeze(0)
            native_projection_max_abs = float((native.float() - logits_fp32).abs().max().item())

        stored_ids_all = row["teacher_topk_token_ids"]
        stored_logprobs_all = row["teacher_topk_logprobs"]
        sampled_ids = [int(value) for value in row["sampled_token_ids"]]
        sampled_logprobs_all = row["sampled_token_logprobs"]
        for local_index, response_position in enumerate(response_positions):
            stored_ids = [int(value) for value in stored_ids_all[response_position]]
            stored_logprobs = [float(value) for value in stored_logprobs_all[response_position]]
            stored_ids_tensor = torch.tensor(
                stored_ids,
                dtype=torch.long,
                device=device,
            )
            reference_on_stored = (
                logits_fp32[local_index].gather(0, stored_ids_tensor) - log_denominator[local_index, 0]
            )
            sampled_id = sampled_ids[response_position]
            reference_sampled_logprob = logits_fp32[local_index, sampled_id] - log_denominator[local_index, 0]
            metrics = compare_topk_position(
                stored_ids=stored_ids,
                stored_logprobs=stored_logprobs,
                reference_top_ids=reference_top_ids[local_index].tolist(),
                reference_top_logprobs=reference_top_logprobs[local_index].tolist(),
                reference_logprobs_on_stored=reference_on_stored.tolist(),
                stored_sampled_logprob=float(sampled_logprobs_all[response_position]),
                reference_sampled_logprob=float(reference_sampled_logprob.item()),
                top1_tie_logprob_tolerance=args.top1_tie_logprob_tolerance,
            )
            position_records.append(
                {
                    "trace_id": row["trace_id"],
                    "split": row["split"],
                    "source_uid": row["source_uid"],
                    "sample_index": int(row["sample_index"]),
                    "prompt_length": prompt_length,
                    "response_length": response_length,
                    "response_position": response_position,
                    "prediction_position": prediction_positions[local_index],
                    "response_fraction": response_position / max(1, response_length - 1),
                    **metrics,
                }
            )
        trace_records.append(
            {
                "trace_id": row["trace_id"],
                "split": row["split"],
                "path": row["path"],
                "path_in_index": row["path_in_index"],
                "registered_shard_sha256": row["registered_shard_sha256"],
                "row_index": int(row["row_index"]),
                "prompt_length": prompt_length,
                "response_length": response_length,
                "positions_scored": response_positions,
                "teacher_content_sha256": row["teacher_content_sha256"],
            }
        )
        del logits_fp32, logits, backbone_output, hidden
        torch.cuda.empty_cache()

    if native_projection_max_abs is None:
        raise CrossEngineAuditError("native/manual projection comparison did not run")
    audit_aggregate = aggregate(position_records)
    gate = evaluate_thresholds(
        audit_aggregate,
        native_projection_max_abs=native_projection_max_abs,
        thresholds=thresholds,
    )
    report = {
        "report_version": REPORT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": gate["status"],
        "gate": gate,
        "contract": {
            "reference": (
                "unsharded Transformers Gemma4ForConditionalGeneration, SDPA, BF16, use_cache=False; "
                "model class selected through verl; not an FSDP2 execution"
            ),
            "causal_alignment": ("stored response j is scored from hidden position prompt_length - 1 + j"),
            "softcap": float(softcap),
            "top_k": TOPK_WIDTH,
            "top1_tie_logprob_tolerance": args.top1_tie_logprob_tolerance,
            "stored_logprob_note": (
                "Parquet values are FP16-quantized vLLM full-vocabulary-normalized log probabilities"
            ),
        },
        "dataset": {
            "trace_root": str(trace_root),
            "index_path": str(contract["index_path"]),
            "dataset_index_sha256": contract["index_sha256"],
            "dataset_index_file_sha256": contract["index_file_sha256"],
            "experiment_sha256": contract["index"].get("experiment_sha256"),
            "direction": contract["index"].get("direction"),
            "teacher": dict(contract["index"]["teacher"]),
            "teacher_identity_sha256": contract["teacher_identity_sha256"],
            "generation_config_sha256s": contract["generation_config_sha256s"],
            "generation_environments": {
                split: contract["semantic_configs"][split]["environment_versions"] for split in SPLITS
            },
            "generation_engines": {split: contract["semantic_configs"][split]["engine"] for split in SPLITS},
        },
        "selection_config": {
            "traces_per_split": args.traces_per_split,
            "positions_per_trace": args.positions_per_trace,
            "selection_scope": "all index-registered shards",
            "all_registered_shard_sha256_verified": True,
        },
        "model": {
            "path": str(model_root),
            "model_identity_sha256": contract["model_identity"].model_identity_sha256,
            "weight_content_sha256": contract["model_identity"].weight_content_sha256,
            "weight_content_kind": contract["model_identity"].weight_content_kind,
            "auto_class": auto_class.__name__,
            "vocab_size": vocab_size,
            "attn_implementation": text_config._attn_implementation,
            "dtype": str(next(model.parameters()).dtype),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "pyarrow": pa.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
        },
        "implementation": repository_provenance(),
        "selection": trace_records,
        "native_vs_manual_projection_max_abs": native_projection_max_abs,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "aggregate": audit_aggregate,
        "interpretation": {
            "kind": "cross_engine_numeric_diagnostic_not_bit_equality_gate",
            "immutable_generation_record": "stored_vllm_token_ids_and_topk_targets",
            "intended_training_targets": "separate_unsharded_hf_bf16_sdpa_rescored_overlay_after_gates",
            "recommendation": (
                "Preserve and consume the exact stored token IDs without re-tokenization. "
                "Keep the vLLM bundle immutable, and derive any unsharded-HF training-shaped targets "
                "as a separately indexed overlay after parity and provenance gates because the "
                "engines are close but not bit-identical. A separate real-FSDP2 audit is required "
                "before claiming equivalence to the distributed training forward."
            ),
        },
        "positions": position_records,
    }
    atomic_write_json(output, report)
    print(
        json.dumps(
            {
                "dataset_index_sha256": contract["index_sha256"],
                "selection_count": len(trace_records),
                "position_count": len(position_records),
                "status": report["status"],
                "native_vs_manual_projection_max_abs": native_projection_max_abs,
                "aggregate": report["aggregate"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    print(f"wrote {output}", flush=True)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--trace-root", required=True)
    parser.add_argument("--dataset-index", required=True)
    parser.add_argument("--expected-dataset-index-sha256", required=True)
    parser.add_argument("--expected-teacher-identity-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--traces-per-split", type=int, default=16)
    parser.add_argument("--positions-per-trace", type=int, default=64)
    parser.add_argument("--attn-implementation", choices=("sdpa",), default="sdpa")
    parser.add_argument("--expected-final-logit-softcap", type=float, default=30.0)
    parser.add_argument("--top1-tie-logprob-tolerance", type=float, default=0.02)
    parser.add_argument(
        "--max-native-projection-abs",
        type=float,
        default=DEFAULT_THRESHOLDS["native_vs_manual_projection_max_abs"],
    )
    parser.add_argument(
        "--min-top1-tie-safe-mean",
        type=float,
        default=DEFAULT_THRESHOLDS["top1_tie_safe_mean"],
    )
    parser.add_argument(
        "--min-top10-overlap-mean",
        type=float,
        default=DEFAULT_THRESHOLDS["top10_overlap_fraction_mean"],
    )
    parser.add_argument(
        "--min-topk-overlap-mean",
        type=float,
        default=DEFAULT_THRESHOLDS["topk_overlap_fraction_mean"],
    )
    parser.add_argument(
        "--max-weighted-abs-logprob-delta-mean",
        type=float,
        default=DEFAULT_THRESHOLDS["stored_support_weighted_abs_logprob_delta_mean"],
    )
    parser.add_argument(
        "--max-support-probability-l1-mean",
        type=float,
        default=DEFAULT_THRESHOLDS["stored_support_probability_l1_mean"],
    )
    parser.add_argument(
        "--max-sampled-token-abs-delta-p95",
        type=float,
        default=DEFAULT_THRESHOLDS["sampled_token_abs_logprob_delta_p95"],
    )
    parser.add_argument(
        "--max-stored-only-topk-mass-p99",
        type=float,
        default=DEFAULT_THRESHOLDS["stored_only_topk_mass_p99"],
    )
    parser.add_argument(
        "--max-reference-only-topk-mass-p99",
        type=float,
        default=DEFAULT_THRESHOLDS["reference_only_topk_mass_p99"],
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        report = run_audit(parse_args(argv))
    except (CrossEngineAuditError, OSError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
