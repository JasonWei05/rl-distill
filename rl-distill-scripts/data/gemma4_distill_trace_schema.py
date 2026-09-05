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

"""Schema and integrity helpers for precomputed Gemma 4 top-k traces.

This module intentionally has no vLLM or transformers import.  Generation and
validation CLIs can therefore be imported and unit-tested on a CPU-only host.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA_VERSION = "gemma4-distill-topk-v1"
MANIFEST_VERSION = 1
TOPK_WIDTH = 128
MASS_HISTOGRAM_BINS = 10_000
FP16_TOPK_MASS_TOLERANCE = 2.5e-3
RESPONSE_TEXT_NORMALIZATION = (
    "tokenizer.decode(response_token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)"
)
QUANTILES = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)


class TraceValidationError(ValueError):
    """Raised when a trace artifact violates the immutable data contract."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def hash_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def derive_sampling_seed(global_seed: int, split: str, source_uid: str, sample_index: int) -> int:
    payload = canonical_json_bytes(
        {
            "global_seed": global_seed,
            "split": split,
            "source_uid": source_uid,
            "sample_index": sample_index,
        }
    )
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31 - 1)
    return seed or 1


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(value))
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)


def trace_arrow_schema(topk_width: int = TOPK_WIDTH) -> pa.Schema:
    topk_ids = pa.list_(pa.list_(pa.int32(), topk_width))
    topk_logprobs = pa.list_(pa.list_(pa.float16(), topk_width))
    schema = pa.schema(
        [
            pa.field("schema_version", pa.string(), nullable=False),
            pa.field("generation_config_sha256", pa.string(), nullable=False),
            pa.field("trace_id", pa.string(), nullable=False),
            pa.field("direction", pa.string(), nullable=False),
            pa.field("split", pa.string(), nullable=False),
            pa.field("source_dataset", pa.string(), nullable=False),
            pa.field("source_dataset_sha256", pa.string(), nullable=False),
            pa.field("source_uid", pa.string(), nullable=False),
            pa.field("source_uid_original", pa.string()),
            pa.field("question_sha256", pa.string(), nullable=False),
            pa.field("prompt_index", pa.int64(), nullable=False),
            pa.field("sample_index", pa.int8(), nullable=False),
            pa.field("question_text", pa.large_string(), nullable=False),
            pa.field("gold_answer", pa.large_string(), nullable=False),
            pa.field("strict_grade", pa.float32(), nullable=False),
            pa.field("strict_correct", pa.bool_(), nullable=False),
            pa.field("strict_prediction", pa.large_string(), nullable=False),
            pa.field("teacher_model", pa.string(), nullable=False),
            pa.field("teacher_revision", pa.string()),
            pa.field("teacher_content_sha256", pa.string()),
            pa.field("tokenizer_model", pa.string(), nullable=False),
            pa.field("tokenizer_revision", pa.string()),
            pa.field("tokenizer_sha256", pa.string(), nullable=False),
            pa.field("tokenizer_vocab_size", pa.int32(), nullable=False),
            pa.field("chat_template_path", pa.string(), nullable=False),
            pa.field("chat_template_sha256", pa.string(), nullable=False),
            pa.field("global_seed", pa.int64(), nullable=False),
            pa.field("sampling_seed", pa.int64(), nullable=False),
            pa.field("sampling_parameters_json", pa.large_string(), nullable=False),
            pa.field("prompt_token_ids", pa.list_(pa.int32()), nullable=False),
            pa.field("response_token_ids", pa.list_(pa.int32()), nullable=False),
            pa.field("input_ids", pa.list_(pa.int32()), nullable=False),
            pa.field("response_mask", pa.list_(pa.int8()), nullable=False),
            pa.field("teacher_topk_token_ids", topk_ids, nullable=False),
            pa.field("teacher_topk_logprobs", topk_logprobs, nullable=False),
            pa.field("sampled_token_ids", pa.list_(pa.int32()), nullable=False),
            pa.field("sampled_token_logprobs", pa.list_(pa.float32()), nullable=False),
            pa.field("teacher_topk_rank_order", pa.string(), nullable=False),
            pa.field("prompt_length", pa.int32(), nullable=False),
            pa.field("response_length", pa.int32(), nullable=False),
            pa.field("finish_reason", pa.string()),
            pa.field("stop_reason", pa.string()),
            pa.field("matched_stop_string", pa.string()),
            pa.field("reached_max_response_tokens", pa.bool_(), nullable=False),
            pa.field("response_text", pa.large_string(), nullable=False),
            pa.field("vllm_response_text", pa.large_string(), nullable=False),
            pa.field("response_text_normalization", pa.string(), nullable=False),
            pa.field("shard_id", pa.int32(), nullable=False),
            pa.field("row_within_shard", pa.int32(), nullable=False),
            pa.field("generation_timestamp", pa.string(), nullable=False),
            pa.field("generator_commit", pa.string(), nullable=False),
            pa.field("generator_source_sha256", pa.string(), nullable=False),
            pa.field("environment_versions_json", pa.large_string(), nullable=False),
        ],
        metadata={b"schema_version": SCHEMA_VERSION.encode("ascii"), b"topk_width": str(topk_width).encode("ascii")},
    )
    return schema


def parquet_manifest_path(parquet_path: str | Path) -> Path:
    return Path(parquet_path).with_suffix(".manifest.json")


def write_parquet_temporary(
    records: Sequence[Mapping[str, Any]],
    destination: str | Path,
    *,
    row_group_size: int,
    compression: str = "zstd",
    compression_level: int = 3,
) -> Path:
    """Write a typed parquet beside ``destination`` without publishing it."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp.parquet", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        table = pa.Table.from_pylist(list(records), schema=trace_arrow_schema())
        pq.write_table(
            table,
            temporary,
            compression=compression,
            compression_level=compression_level,
            row_group_size=row_group_size,
            write_statistics=True,
            data_page_version="2.0",
        )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def publish_parquet_temporary(temporary: str | Path, destination: str | Path) -> None:
    temporary_path = Path(temporary)
    destination_path = Path(destination)
    os.replace(temporary_path, destination_path)
    _fsync_directory(destination_path.parent)


def _as_list(value: Any, field_name: str) -> list[Any]:
    if value is None:
        raise TraceValidationError(f"{field_name} is null")
    if hasattr(value, "as_py"):
        value = value.as_py()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list | tuple):
        raise TraceValidationError(f"{field_name} must be a list, got {type(value).__name__}")
    return list(value)


def _require_sha256(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise TraceValidationError(f"{field_name} must be a 64-character SHA256 hex digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise TraceValidationError(f"{field_name} is not hexadecimal") from error


def _quantile_from_histogram(counts: Sequence[int], quantile: float, scale: float = 1.0) -> float | None:
    total = sum(counts)
    if total == 0:
        return None
    target = quantile * (total - 1)
    seen = 0
    for index, count in enumerate(counts):
        if seen + count > target:
            return index / scale
        seen += count
    return (len(counts) - 1) / scale


@dataclass
class TraceStatistics:
    max_response_tokens: int
    row_count: int = 0
    response_token_count: int = 0
    prompt_token_count: int = 0
    reached_cap_count: int = 0
    strict_correct_count: int = 0
    empty_response_count: int = 0
    response_length_histogram: list[int] = field(init=False)
    topk_mass_histogram: list[int] = field(default_factory=lambda: [0] * (MASS_HISTOGRAM_BINS + 1))

    def __post_init__(self) -> None:
        self.response_length_histogram = [0] * (self.max_response_tokens + 1)

    def update(self, record: Mapping[str, Any], topk_masses: Iterable[float]) -> None:
        response_length = int(record["response_length"])
        if response_length < 0 or response_length > self.max_response_tokens:
            raise TraceValidationError(f"response_length {response_length} is outside [0, {self.max_response_tokens}]")
        self.row_count += 1
        self.response_token_count += response_length
        self.prompt_token_count += int(record["prompt_length"])
        self.reached_cap_count += int(bool(record["reached_max_response_tokens"]))
        self.strict_correct_count += int(bool(record["strict_correct"]))
        self.empty_response_count += int(response_length == 0)
        self.response_length_histogram[response_length] += 1
        for mass in topk_masses:
            bin_index = min(MASS_HISTOGRAM_BINS, max(0, int(round(mass * MASS_HISTOGRAM_BINS))))
            self.topk_mass_histogram[bin_index] += 1

    def to_dict(self) -> dict[str, Any]:
        response_quantiles = {str(q): _quantile_from_histogram(self.response_length_histogram, q) for q in QUANTILES}
        mass_quantiles = {
            str(q): _quantile_from_histogram(self.topk_mass_histogram, q, MASS_HISTOGRAM_BINS) for q in QUANTILES
        }
        return {
            "row_count": self.row_count,
            "prompt_token_count": self.prompt_token_count,
            "response_token_count": self.response_token_count,
            "reached_max_response_tokens_count": self.reached_cap_count,
            "strict_correct_count": self.strict_correct_count,
            "empty_response_count": self.empty_response_count,
            "response_length_quantiles": response_quantiles,
            "teacher_topk_mass_quantiles": mass_quantiles,
            "teacher_topk_mass_quantile_resolution": 1.0 / MASS_HISTOGRAM_BINS,
        }


@dataclass
class ShardValidationResult:
    stats: dict[str, Any]
    trace_ids: set[str]
    source_samples: list[tuple[str, str, int]]
    source_uid_to_question: dict[str, str]


def validate_trace_record(
    record: Mapping[str, Any],
    *,
    decoder: Callable[[Sequence[int]], str] | None,
    expected_config_sha256: str | None = None,
    expected_direction: str | None = None,
    expected_split: str | None = None,
    expected_shard_id: int | None = None,
    expected_row_within_shard: int | None = None,
    expected_semantic_config: Mapping[str, Any] | None = None,
    expected_topk_width: int = TOPK_WIDTH,
    max_prompt_tokens: int = 4096,
    max_response_tokens: int = 8192,
    logprob_tolerance: float = 5e-4,
) -> list[float]:
    """Validate one materialized row and return its per-position top-k mass."""

    required = set(trace_arrow_schema(expected_topk_width).names)
    missing = sorted(required.difference(record))
    if missing:
        raise TraceValidationError(f"missing required columns: {missing}")
    if record["schema_version"] != SCHEMA_VERSION:
        raise TraceValidationError(f"unexpected schema_version: {record['schema_version']!r}")
    if expected_config_sha256 is not None and record["generation_config_sha256"] != expected_config_sha256:
        raise TraceValidationError("generation_config_sha256 does not match the run configuration")
    if expected_direction is not None and record["direction"] != expected_direction:
        raise TraceValidationError(f"direction mismatch: {record['direction']!r} != {expected_direction!r}")
    if expected_split is not None and record["split"] != expected_split:
        raise TraceValidationError(f"split mismatch: {record['split']!r} != {expected_split!r}")
    if expected_shard_id is not None and int(record["shard_id"]) != expected_shard_id:
        raise TraceValidationError(f"shard_id mismatch: {record['shard_id']} != {expected_shard_id}")
    if expected_row_within_shard is not None and int(record["row_within_shard"]) != expected_row_within_shard:
        raise TraceValidationError(
            f"row_within_shard mismatch: {record['row_within_shard']} != {expected_row_within_shard}"
        )

    question_sha256 = sha256_text(str(record["question_text"]))
    if record["question_sha256"] != question_sha256:
        raise TraceValidationError("question_sha256 does not match question_text")
    sample_index = int(record["sample_index"])
    expected_sampling_seed = derive_sampling_seed(
        int(record["global_seed"]), str(record["split"]), str(record["source_uid"]), sample_index
    )
    if int(record["sampling_seed"]) != expected_sampling_seed:
        raise TraceValidationError("sampling_seed does not match the deterministic seed contract")
    expected_trace_id = hash_json(
        {
            "generation_config_sha256": record["generation_config_sha256"],
            "source_uid": record["source_uid"],
            "question_sha256": record["question_sha256"],
            "sample_index": sample_index,
        }
    )
    if record["trace_id"] != expected_trace_id:
        raise TraceValidationError("trace_id does not match its configuration/question/sample identity")
    strict_grade = float(record["strict_grade"])
    if not math.isfinite(strict_grade) or not 0.0 <= strict_grade <= 1.0:
        raise TraceValidationError("strict_grade must be finite and lie in [0, 1]")
    if bool(record["strict_correct"]) != (strict_grade > 0.5):
        raise TraceValidationError("strict_correct is inconsistent with strict_grade")

    try:
        sampling_parameters = json.loads(record["sampling_parameters_json"])
        environment_versions = json.loads(record["environment_versions_json"])
    except (TypeError, json.JSONDecodeError) as error:
        raise TraceValidationError("row JSON provenance fields are malformed") from error
    if not isinstance(sampling_parameters, dict) or not isinstance(environment_versions, dict):
        raise TraceValidationError("row JSON provenance fields must contain objects")

    if expected_semantic_config is not None:
        expected_fields = {
            "direction": expected_semantic_config["direction"],
            "source_dataset": expected_semantic_config["source_dataset"],
            "source_dataset_sha256": expected_semantic_config["source_dataset_sha256"],
            "teacher_model": expected_semantic_config["teacher"]["model"],
            "teacher_revision": expected_semantic_config["teacher"]["revision"],
            "teacher_content_sha256": expected_semantic_config["teacher"]["content_sha256"],
            "tokenizer_model": expected_semantic_config["tokenizer"]["model"],
            "tokenizer_revision": expected_semantic_config["tokenizer"]["revision"],
            "tokenizer_sha256": expected_semantic_config["tokenizer"]["sha256"],
            "tokenizer_vocab_size": expected_semantic_config["tokenizer"]["vocab_size"],
            "chat_template_path": expected_semantic_config["chat_template"]["path"],
            "chat_template_sha256": expected_semantic_config["chat_template"]["sha256"],
            "global_seed": expected_semantic_config["global_seed"],
            # ``generator_commit`` is provenance only: the generator identity is the hashed source
            # (``generator_source_sha256``), and a resumed collection legitimately carries the commit
            # that was checked out when each shard was produced.
            "generator_source_sha256": expected_semantic_config["generator"]["source_sha256"],
        }
        for field_name, expected_value in expected_fields.items():
            if record[field_name] != expected_value:
                raise TraceValidationError(f"{field_name} does not match the run configuration")
        if sampling_parameters != expected_semantic_config["sampling"]:
            raise TraceValidationError("sampling_parameters_json does not match the run configuration")
        if environment_versions != expected_semantic_config["environment_versions"]:
            raise TraceValidationError("environment_versions_json does not match the run configuration")

    for field_name in (
        "generation_config_sha256",
        "trace_id",
        "source_dataset_sha256",
        "question_sha256",
        "tokenizer_sha256",
        "chat_template_sha256",
        "generator_source_sha256",
    ):
        _require_sha256(record[field_name], field_name)
    if not record["teacher_revision"] and not record["teacher_content_sha256"]:
        raise TraceValidationError("teacher_revision or teacher_content_sha256 must identify the immutable teacher")
    if record["teacher_content_sha256"]:
        _require_sha256(record["teacher_content_sha256"], "teacher_content_sha256")

    prompt_ids = [int(value) for value in _as_list(record["prompt_token_ids"], "prompt_token_ids")]
    response_ids = [int(value) for value in _as_list(record["response_token_ids"], "response_token_ids")]
    input_ids = [int(value) for value in _as_list(record["input_ids"], "input_ids")]
    response_mask = [int(value) for value in _as_list(record["response_mask"], "response_mask")]
    sampled_ids = [int(value) for value in _as_list(record["sampled_token_ids"], "sampled_token_ids")]
    sampled_logprobs = [float(value) for value in _as_list(record["sampled_token_logprobs"], "sampled_token_logprobs")]
    topk_ids = _as_list(record["teacher_topk_token_ids"], "teacher_topk_token_ids")
    topk_logprobs = _as_list(record["teacher_topk_logprobs"], "teacher_topk_logprobs")

    prompt_length = int(record["prompt_length"])
    response_length = int(record["response_length"])
    if prompt_length != len(prompt_ids) or response_length != len(response_ids):
        raise TraceValidationError("stored prompt/response lengths do not match their token arrays")
    if prompt_length <= 0 or prompt_length > max_prompt_tokens:
        raise TraceValidationError(f"prompt length {prompt_length} is outside [1, {max_prompt_tokens}]")
    if response_length > max_response_tokens:
        raise TraceValidationError(f"response length {response_length} exceeds {max_response_tokens}")
    if input_ids != prompt_ids + response_ids:
        raise TraceValidationError("input_ids is not prompt_token_ids + response_token_ids")
    expected_mask = [0] * prompt_length + [1] * response_length
    if response_mask != expected_mask:
        raise TraceValidationError("response_mask is not an exact zero-prefix/one-suffix mask")
    if len(input_ids) != len(response_mask):
        raise TraceValidationError("input_ids and response_mask lengths differ")
    if sampled_ids != response_ids:
        raise TraceValidationError("sampled_token_ids must exactly equal response_token_ids")
    if len(sampled_logprobs) != response_length:
        raise TraceValidationError("sampled_token_logprobs length does not match response length")
    if len(topk_ids) != response_length or len(topk_logprobs) != response_length:
        raise TraceValidationError("top-k outer dimensions do not match response length")
    if record["teacher_topk_rank_order"] != f"1..{expected_topk_width}":
        raise TraceValidationError("teacher_topk_rank_order does not declare the fixed rank range")

    vocab_size = int(record["tokenizer_vocab_size"])
    if vocab_size <= 0:
        raise TraceValidationError("tokenizer_vocab_size must be positive")
    for field_name, values in (("prompt_token_ids", prompt_ids), ("response_token_ids", response_ids)):
        if any(token_id < 0 or token_id >= vocab_size for token_id in values):
            raise TraceValidationError(f"{field_name} contains an ID outside [0, {vocab_size})")

    masses: list[float] = []
    for position, (position_ids_raw, position_logprobs_raw, sampled_id, sampled_logprob) in enumerate(
        zip(topk_ids, topk_logprobs, sampled_ids, sampled_logprobs, strict=True)
    ):
        position_ids = [int(value) for value in _as_list(position_ids_raw, "teacher_topk_token_ids[position]")]
        position_logprobs = [
            float(value) for value in _as_list(position_logprobs_raw, "teacher_topk_logprobs[position]")
        ]
        if len(position_ids) != expected_topk_width or len(position_logprobs) != expected_topk_width:
            raise TraceValidationError(
                f"position {position} does not contain exactly {expected_topk_width} ranked targets"
            )
        if len(set(position_ids)) != expected_topk_width:
            raise TraceValidationError(f"position {position} has duplicate top-k vocabulary IDs")
        if any(token_id < 0 or token_id >= vocab_size for token_id in position_ids):
            raise TraceValidationError(f"position {position} has a top-k ID outside [0, {vocab_size})")
        if not math.isfinite(sampled_logprob):
            raise TraceValidationError(f"position {position} sampled-token log probability is not finite")
        if any(not math.isfinite(value) for value in position_logprobs):
            raise TraceValidationError(f"position {position} has a non-finite top-k log probability")
        if any(value > logprob_tolerance for value in position_logprobs):
            raise TraceValidationError(f"position {position} has a positive top-k log probability")
        if any(
            later > earlier + logprob_tolerance
            for earlier, later in zip(position_logprobs, position_logprobs[1:], strict=False)
        ):
            raise TraceValidationError(f"position {position} top-k log probabilities are not rank-sorted")
        mass = math.fsum(math.exp(value) for value in position_logprobs)
        # Independently rounded FP16 log probabilities can reconstruct a mass
        # slightly above one (about 0.2% for a near-uniform 103-way support).
        # This bound still rejects materially unnormalized targets without
        # rejecting a valid full-vocabulary distribution solely due to storage.
        mass_tolerance = max(logprob_tolerance, FP16_TOPK_MASS_TOLERANCE)
        if mass > 1.0 + mass_tolerance:
            raise TraceValidationError(f"position {position} top-k probability mass exceeds one: {mass}")
        masses.append(mass)
        if sampled_id in position_ids:
            ranked_logprob = position_logprobs[position_ids.index(sampled_id)]
            sampled_logprob_fp16 = struct.unpack("<e", struct.pack("<e", sampled_logprob))[0]
            if abs(ranked_logprob - sampled_logprob_fp16) > logprob_tolerance:
                raise TraceValidationError(
                    f"position {position} sampled-token log probability disagrees with its top-k value"
                )

    if record["response_text_normalization"] != RESPONSE_TEXT_NORMALIZATION:
        raise TraceValidationError("unknown response text normalization contract")
    if decoder is not None:
        decoded = decoder(response_ids)
        if decoded != record["response_text"]:
            raise TraceValidationError("stored response_text does not exactly decode from response_token_ids")
    if bool(record["reached_max_response_tokens"]) != (
        response_length == max_response_tokens and record["finish_reason"] == "length"
    ):
        raise TraceValidationError("reached_max_response_tokens is inconsistent with response length/finish reason")
    return masses


def validate_parquet_shard(
    path: str | Path,
    *,
    decoder: Callable[[Sequence[int]], str] | None,
    expected_config_sha256: str,
    expected_direction: str,
    expected_split: str,
    expected_shard_id: int,
    max_prompt_tokens: int,
    max_response_tokens: int,
    expected_semantic_config: Mapping[str, Any] | None = None,
    external_stats: TraceStatistics | None = None,
) -> ShardValidationResult:
    parquet_path = Path(path)
    parquet_file = pq.ParquetFile(parquet_path)
    actual_schema = parquet_file.schema_arrow
    expected_schema = trace_arrow_schema()
    if not actual_schema.equals(expected_schema, check_metadata=False):
        raise TraceValidationError(f"unexpected parquet schema in {parquet_path}:\n{actual_schema}")

    stats = TraceStatistics(max_response_tokens=max_response_tokens)
    trace_ids: set[str] = set()
    source_samples: list[tuple[str, str, int]] = []
    source_uid_to_question: dict[str, str] = {}
    row_index = 0
    for batch in parquet_file.iter_batches(batch_size=8):
        for record in batch.to_pylist():
            try:
                masses = validate_trace_record(
                    record,
                    decoder=decoder,
                    expected_config_sha256=expected_config_sha256,
                    expected_direction=expected_direction,
                    expected_split=expected_split,
                    expected_shard_id=expected_shard_id,
                    expected_row_within_shard=row_index,
                    expected_semantic_config=expected_semantic_config,
                    max_prompt_tokens=max_prompt_tokens,
                    max_response_tokens=max_response_tokens,
                )
            except TraceValidationError as error:
                raise TraceValidationError(f"{parquet_path} row {row_index}: {error}") from error
            trace_id = str(record["trace_id"])
            if trace_id in trace_ids:
                raise TraceValidationError(f"{parquet_path} repeats trace_id {trace_id}")
            trace_ids.add(trace_id)
            source_uid = str(record["source_uid"])
            question_hash = str(record["question_sha256"])
            prior_question = source_uid_to_question.setdefault(source_uid, question_hash)
            if prior_question != question_hash:
                raise TraceValidationError(f"source_uid {source_uid!r} maps to multiple questions")
            source_samples.append((source_uid, question_hash, int(record["sample_index"])))
            stats.update(record, masses)
            if external_stats is not None:
                external_stats.update(record, masses)
            row_index += 1
    return ShardValidationResult(
        stats=stats.to_dict(),
        trace_ids=trace_ids,
        source_samples=source_samples,
        source_uid_to_question=source_uid_to_question,
    )


def validate_shard_bundle(
    parquet_path: str | Path,
    *,
    run_config: Mapping[str, Any],
    decoder: Callable[[Sequence[int]], str] | None,
    external_stats: TraceStatistics | None = None,
) -> tuple[dict[str, Any], ShardValidationResult]:
    parquet_path = Path(parquet_path)
    manifest_path = parquet_manifest_path(parquet_path)
    if not parquet_path.is_file() or not manifest_path.is_file():
        raise TraceValidationError(f"missing parquet/manifest pair for {parquet_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TraceValidationError(f"cannot read {manifest_path}: {error}") from error
    if manifest.get("manifest_version") != MANIFEST_VERSION:
        raise TraceValidationError(f"unsupported manifest version in {manifest_path}")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise TraceValidationError(f"schema version mismatch in {manifest_path}")
    if manifest.get("generation_config_sha256") != run_config["generation_config_sha256"]:
        raise TraceValidationError(f"configuration hash mismatch in {manifest_path}")
    if manifest.get("parquet_file") != parquet_path.name:
        raise TraceValidationError(f"parquet filename mismatch in {manifest_path}")
    actual_sha256 = sha256_file(parquet_path)
    if manifest.get("parquet_sha256") != actual_sha256:
        raise TraceValidationError(f"SHA256 mismatch for {parquet_path}")
    if int(manifest.get("parquet_size_bytes", -1)) != parquet_path.stat().st_size:
        raise TraceValidationError(f"size mismatch for {parquet_path}")

    semantic = run_config["semantic_config"]
    shard_id = int(manifest["shard_id"])
    result = validate_parquet_shard(
        parquet_path,
        decoder=decoder,
        expected_config_sha256=run_config["generation_config_sha256"],
        expected_direction=semantic["direction"],
        expected_split=semantic["split"],
        expected_shard_id=shard_id,
        max_prompt_tokens=int(semantic["sampling"]["max_prompt_tokens"]),
        max_response_tokens=int(semantic["sampling"]["max_response_tokens"]),
        expected_semantic_config=semantic,
        external_stats=external_stats,
    )
    if result.stats != manifest.get("stats"):
        raise TraceValidationError(f"recorded statistics do not match {parquet_path}")
    if result.stats["row_count"] != int(manifest.get("row_count", -1)):
        raise TraceValidationError(f"row count mismatch for {parquet_path}")
    parquet_metadata = pq.ParquetFile(parquet_path).metadata
    if parquet_metadata.num_row_groups != int(manifest.get("parquet_row_groups", -1)):
        raise TraceValidationError(f"row-group count mismatch for {parquet_path}")
    return manifest, result


def tokenizer_fingerprint(tokenizer: Any) -> tuple[str, int]:
    """Hash vocabulary IDs and behavior-relevant tokenizer configuration."""

    vocabulary = tokenizer.get_vocab()
    if not vocabulary:
        raise ValueError("tokenizer.get_vocab() returned an empty vocabulary")
    entries = sorted(((int(token_id), str(token)) for token, token_id in vocabulary.items()))
    vocabulary_size = max(token_id for token_id, _ in entries) + 1
    digest = hashlib.sha256()
    for token_id, token in entries:
        digest.update(str(token_id).encode("ascii"))
        digest.update(b"\0")
        digest.update(token.encode("utf-8"))
        digest.update(b"\0")

    def jsonable(value: Any) -> Any:
        if value is None or isinstance(value, str | int | float | bool):
            return value
        if isinstance(value, Mapping):
            return {str(key): jsonable(item) for key, item in value.items()}
        if isinstance(value, list | tuple):
            return [jsonable(item) for item in value]
        return str(value)

    # Do not hash the Python class name: vLLM wraps an otherwise identical HF
    # tokenizer in a dynamically named Cached* subclass.
    config = {
        "bos_token_id": getattr(tokenizer, "bos_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "unk_token_id": getattr(tokenizer, "unk_token_id", None),
        "special_tokens_map": jsonable(getattr(tokenizer, "special_tokens_map", {})),
        "model_max_length": getattr(tokenizer, "model_max_length", None),
        # vLLM deliberately changes ``truncation_side`` on its cached wrapper.
        # Trace generation never asks the tokenizer to pad or truncate (prompt
        # overflow fails closed), so wrapper-local side settings are not part
        # of the token-ID/decode identity we need to preserve.
        "vocabulary_size": vocabulary_size,
    }
    digest.update(canonical_json_bytes(config))
    return digest.hexdigest(), vocabulary_size


def normalized_decode(tokenizer: Any, token_ids: Sequence[int]) -> str:
    try:
        return tokenizer.decode(list(token_ids), skip_special_tokens=False, clean_up_tokenization_spaces=False)
    except TypeError:
        return tokenizer.decode(list(token_ids), skip_special_tokens=False)
