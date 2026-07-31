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

"""Fail-closed preflight for Gemma 4 precomputed top-k distillation.

The successful output is deliberately line-oriented so a shell launcher can
parse it without ``eval``.  ``*_FILES_HYDRA`` values are compact JSON arrays,
which are also valid Hydra list literals when passed as one argv element.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from gemma4_distill_trace_schema import (
    FP16_TOPK_MASS_TOLERANCE,
    MANIFEST_VERSION,
    SCHEMA_VERSION,
    TOPK_WIDTH,
    hash_json,
    sha256_file,
    sha256_text,
    tokenizer_fingerprint,
)
from gemma4_model_identity import ModelIdentityError, inspect_local_hf_model, remote_model_identity

EXPECTED_QUESTIONS = {"train": 9723, "validation": 200}
EXPECTED_SAMPLES_PER_QUESTION = 5
EXPECTED_GLOBAL_SEED = 42
EXPECTED_CHAT_TEMPLATE = Path(__file__).with_name("gemma3_it_fewshot_math.jinja")
EXPECTED_SAMPLING = {
    "temperature": 1.0,
    "top_p": 1.0,
    "sampling_top_k": -1,
    "max_prompt_tokens": 4096,
    "max_response_tokens": 8192,
    "max_model_len": 12288,
    "stop": ["<end_of_turn>", "<start_of_turn>"],
    "include_stop_str_in_output": False,
    "skip_special_tokens": False,
    "logprobs": TOPK_WIDTH,
}


class PreflightError(ValueError):
    """Raised when an artifact is unsafe to pass to the trainer."""


@dataclass(frozen=True)
class PreflightResult:
    train_files: tuple[str, ...]
    validation_files: tuple[str, ...]
    topk_width: int
    topk_validation_tolerance: float
    dataset_index_sha256: str
    experiment_sha256: str
    direction: str
    teacher_identity_sha256: str
    student_identity_sha256: str
    student_tokenizer_sha256: str

    def lines(self) -> list[str]:
        compact = {"ensure_ascii": False, "separators": (",", ":")}
        return [
            f"TRAIN_FILES_HYDRA={json.dumps(list(self.train_files), **compact)}",
            f"VAL_FILES_HYDRA={json.dumps(list(self.validation_files), **compact)}",
            f"TOPK_WIDTH={self.topk_width}",
            f"TOPK_VALIDATION_TOLERANCE={self.topk_validation_tolerance:.12g}",
            f"DATASET_INDEX_SHA256={self.dataset_index_sha256}",
            f"GENERATION_EXPERIMENT_SHA256={self.experiment_sha256}",
            f"DIRECTION={self.direction}",
            f"TEACHER_IDENTITY_SHA256={self.teacher_identity_sha256}",
            f"STUDENT_IDENTITY_SHA256={self.student_identity_sha256}",
            f"STUDENT_TOKENIZER_SHA256={self.student_tokenizer_sha256}",
        ]


@dataclass(frozen=True)
class _VerifiedSplit:
    files: tuple[str, ...]
    uid_questions: dict[str, str]


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PreflightError(f"cannot read {description} {path}: {error}") from error
    if not isinstance(value, dict):
        raise PreflightError(f"{description} {path} must contain a JSON object")
    return value


def _require_sha256(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise PreflightError(f"{field_name} must be a 64-character SHA256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise PreflightError(f"{field_name} is not hexadecimal") from error
    return value


def _require_immutable_revision(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or len(value) not in (40, 64):
        raise PreflightError(f"{field_name} must be an immutable 40/64-character hexadecimal revision")
    try:
        int(value, 16)
    except ValueError as error:
        raise PreflightError(f"{field_name} is not hexadecimal") from error
    return value


def _require_int(value: Any, field_name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PreflightError(f"{field_name} must be a JSON integer")
    if minimum is not None and value < minimum:
        raise PreflightError(f"{field_name} must be at least {minimum}, got {value}")
    return value


def _require_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise PreflightError(f"{field_name} must be a JSON number")
    number = float(value)
    if not math.isfinite(number):
        raise PreflightError(f"{field_name} must be finite")
    return number


def _verify_index_self_hash(index: Mapping[str, Any]) -> str:
    claimed = _require_sha256(index.get("dataset_index_sha256"), "dataset_index_sha256")
    unhashed = dict(index)
    del unhashed["dataset_index_sha256"]
    actual = hash_json(unhashed)
    if actual != claimed:
        raise PreflightError(f"dataset index self-hash mismatch: {actual} != {claimed}")
    return claimed


def _resolve_index_path(index_path: Path, listed_path: Any, field_name: str) -> Path:
    if not isinstance(listed_path, str) or not listed_path:
        raise PreflightError(f"{field_name} must be a non-empty path string")
    path = Path(listed_path)
    if not path.is_absolute():
        path = index_path.parent / path
    return path.resolve()


def _common_generation_config(semantic: Mapping[str, Any]) -> dict[str, Any]:
    required = (
        "schema_version",
        "direction",
        "topk_width",
        "global_seed",
        "teacher",
        "tokenizer",
        "chat_template",
        "sampling",
        "engine",
        "generator",
        "environment_versions",
    )
    missing = [key for key in required if key not in semantic]
    if missing:
        raise PreflightError(f"run semantic configuration is missing fields: {missing}")
    return {key: semantic[key] for key in required}


def _experiment_hash_candidates(
    common: Mapping[str, Any],
    samples_per_question: Mapping[str, int],
) -> set[str]:
    """Accept current indexes and the original equal-sample-count index identity."""

    candidates = {hash_json(common)}
    sample_counts = set(samples_per_question.values())
    if len(sample_counts) == 1:
        legacy_common = dict(common)
        legacy_common["samples_per_question"] = next(iter(sample_counts))
        candidates.add(hash_json(legacy_common))
    return candidates


def _normalize_split_counts(value: int | Mapping[str, int], *, field_name: str) -> dict[str, int]:
    if isinstance(value, bool):
        raise PreflightError(f"{field_name} must contain positive integers")
    if isinstance(value, int):
        values = {split: value for split in ("train", "validation")}
    elif isinstance(value, Mapping):
        if set(value) != {"train", "validation"}:
            raise PreflightError(f"{field_name} must contain exactly train and validation")
        values = dict(value)
    else:
        raise PreflightError(f"{field_name} must be an integer or split mapping")
    for split, count in values.items():
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise PreflightError(f"{field_name}.{split} must be a positive integer")
    return values


def _verify_run_config(
    *,
    index_path: Path,
    split_name: str,
    split_index: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    run_config_path = _resolve_index_path(index_path, split_index.get("run_config_path"), "run_config_path")
    if not run_config_path.is_file():
        raise PreflightError(f"{split_name} run config does not exist: {run_config_path}")
    expected_file_hash = _require_sha256(split_index.get("run_config_sha256"), f"splits.{split_name}.run_config_sha256")
    actual_file_hash = sha256_file(run_config_path)
    if actual_file_hash != expected_file_hash:
        raise PreflightError(
            f"{split_name} run config SHA256 mismatch for {run_config_path}: {actual_file_hash} != {expected_file_hash}"
        )
    run_config = _load_json(run_config_path, f"{split_name} run config")
    if run_config.get("manifest_version") != MANIFEST_VERSION or run_config.get("schema_version") != SCHEMA_VERSION:
        raise PreflightError(f"{split_name} run config has an unsupported manifest/schema version")
    semantic = run_config.get("semantic_config")
    if not isinstance(semantic, dict):
        raise PreflightError(f"{split_name} run config has no semantic_config object")
    generation_hash = _require_sha256(
        run_config.get("generation_config_sha256"), f"{split_name} generation_config_sha256"
    )
    actual_generation_hash = hash_json(semantic)
    if actual_generation_hash != generation_hash:
        raise PreflightError(
            f"{split_name} semantic configuration hash mismatch: {actual_generation_hash} != {generation_hash}"
        )
    if split_index.get("generation_config_sha256") != generation_hash:
        raise PreflightError(f"{split_name} index/run-config generation hashes do not match")
    if semantic.get("split") != split_name:
        raise PreflightError(f"{split_name} run config declares split {semantic.get('split')!r}")
    return run_config, semantic, run_config_path


def _verify_generation_contract(common: Mapping[str, Any]) -> None:
    generator = common.get("generator")
    if not isinstance(generator, Mapping):
        raise PreflightError("run config generator must be an object")
    _require_immutable_revision(generator.get("commit"), "run config generator.commit")
    _require_sha256(generator.get("source_sha256"), "run config generator.source_sha256")
    if generator.get("repository_dirty") is not False:
        raise PreflightError("production traces must be generated from a clean repository")
    if _require_int(common.get("global_seed"), "run config global_seed") != EXPECTED_GLOBAL_SEED:
        raise PreflightError(f"run config global_seed must be exactly {EXPECTED_GLOBAL_SEED}")
    chat_template = common.get("chat_template")
    if not isinstance(chat_template, Mapping):
        raise PreflightError("run config chat_template must be an object")
    expected_template_sha256 = sha256_file(EXPECTED_CHAT_TEMPLATE)
    actual_template_sha256 = _require_sha256(chat_template.get("sha256"), "run config chat_template.sha256")
    if actual_template_sha256 != expected_template_sha256:
        raise PreflightError(
            "run config chat template does not match data/gemma3_it_fewshot_math.jinja: "
            f"{actual_template_sha256} != {expected_template_sha256}"
        )
    sampling = common.get("sampling")
    if not isinstance(sampling, Mapping):
        raise PreflightError("run config sampling must be an object")
    for field_name in ("temperature", "top_p"):
        actual = _require_number(sampling.get(field_name), f"run config sampling.{field_name}")
        if actual != EXPECTED_SAMPLING[field_name]:
            raise PreflightError(
                f"run config sampling.{field_name} must be exactly {EXPECTED_SAMPLING[field_name]}, got {actual}"
            )
    for field_name in (
        "sampling_top_k",
        "max_prompt_tokens",
        "max_response_tokens",
        "max_model_len",
        "logprobs",
    ):
        actual = _require_int(sampling.get(field_name), f"run config sampling.{field_name}")
        if actual != EXPECTED_SAMPLING[field_name]:
            raise PreflightError(
                f"run config sampling.{field_name} must be exactly {EXPECTED_SAMPLING[field_name]}, got {actual}"
            )
    for field_name in ("include_stop_str_in_output", "skip_special_tokens"):
        if sampling.get(field_name) is not EXPECTED_SAMPLING[field_name]:
            raise PreflightError(f"run config sampling.{field_name} must be exactly {EXPECTED_SAMPLING[field_name]}")
    if sampling.get("stop") != EXPECTED_SAMPLING["stop"]:
        raise PreflightError(f"run config sampling.stop must be exactly {EXPECTED_SAMPLING['stop']!r}")


def _verify_shards(
    *,
    index_path: Path,
    split_name: str,
    split_index: Mapping[str, Any],
    expected_question_count: int,
    expected_samples_per_question: int,
) -> _VerifiedSplit:
    if split_index.get("complete") is not True:
        raise PreflightError(f"{split_name} split is not marked complete")
    if split_index.get("missing_shard_ids") not in ([], ()):  # JSON uses [], tests may use tuples.
        raise PreflightError(f"{split_name} split lists missing shard IDs")
    shards = split_index.get("shards")
    if not isinstance(shards, list) or not shards:
        raise PreflightError(f"{split_name} split must contain at least one shard")
    indexed_files = split_index.get("parquet_files")
    if not isinstance(indexed_files, list) or len(indexed_files) != len(shards):
        raise PreflightError(f"{split_name} parquet_files does not align with shards")

    resolved_files: list[str] = []
    seen_paths: set[Path] = set()
    seen_ids: set[int] = set()
    shard_rows = 0
    source_samples: dict[str, set[int]] = {}
    uid_questions: dict[str, str] = {}
    for position, shard in enumerate(shards):
        if not isinstance(shard, Mapping):
            raise PreflightError(f"{split_name} shard {position} is not an object")
        shard_id = _require_int(shard.get("shard_id"), f"{split_name} shard {position} shard_id", minimum=0)
        if shard_id in seen_ids:
            raise PreflightError(f"{split_name} repeats shard_id {shard_id}")
        seen_ids.add(shard_id)
        if shard_id != position:
            raise PreflightError(f"{split_name} shard IDs must be contiguous and ordered; got {shard_id} at {position}")

        path = _resolve_index_path(index_path, shard.get("path"), f"{split_name} shard {shard_id} path")
        indexed_path = _resolve_index_path(
            index_path, indexed_files[position], f"{split_name} parquet_files[{position}]"
        )
        if indexed_path != path:
            raise PreflightError(f"{split_name} parquet_files order/path disagrees with shard {shard_id}")
        if path in seen_paths:
            raise PreflightError(f"{split_name} repeats shard path {path}")
        seen_paths.add(path)
        if not path.is_file():
            raise PreflightError(f"{split_name} shard does not exist: {path}")

        expected_size = _require_int(shard.get("size_bytes"), f"{split_name} shard {shard_id} size_bytes", minimum=1)
        rows = _require_int(shard.get("rows"), f"{split_name} shard {shard_id} rows", minimum=1)
        row_groups = _require_int(shard.get("row_groups"), f"{split_name} shard {shard_id} row_groups", minimum=1)
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            raise PreflightError(
                f"{split_name} shard {shard_id} size mismatch for {path}: {actual_size} != {expected_size}"
            )
        expected_sha256 = _require_sha256(shard.get("sha256"), f"{split_name} shard {shard_id} sha256")
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise PreflightError(
                f"{split_name} shard {shard_id} SHA256 mismatch for {path}: {actual_sha256} != {expected_sha256}"
            )
        stats = shard.get("stats")
        if not isinstance(stats, Mapping):
            raise PreflightError(f"{split_name} shard {shard_id} stats must be an object")
        if _require_int(stats.get("row_count"), f"{split_name} shard {shard_id} stats.row_count", minimum=1) != rows:
            raise PreflightError(f"{split_name} shard {shard_id} stats.row_count does not match rows")
        if _require_int(
            stats.get("empty_response_count"),
            f"{split_name} shard {shard_id} stats.empty_response_count",
            minimum=0,
        ):
            raise PreflightError(f"{split_name} shard {shard_id} contains empty responses")
        declared_shard_response_tokens = _require_int(
            stats.get("response_token_count"),
            f"{split_name} shard {shard_id} stats.response_token_count",
            minimum=rows,
        )
        try:
            with pq.ParquetFile(path) as parquet:
                metadata = parquet.metadata
                if metadata.num_rows != rows:
                    raise PreflightError(
                        f"{split_name} shard {shard_id} Parquet rows {metadata.num_rows} do not match {rows}"
                    )
                if metadata.num_row_groups != row_groups:
                    raise PreflightError(
                        f"{split_name} shard {shard_id} Parquet row groups {metadata.num_row_groups} "
                        f"do not match {row_groups}"
                    )
                sample_table = parquet.read(
                    columns=[
                        "source_uid",
                        "question_sha256",
                        "question_text",
                        "sample_index",
                        "response_length",
                    ],
                    use_threads=True,
                )
                sample_columns = sample_table.to_pydict()
        except PreflightError:
            raise
        except (OSError, TypeError, ValueError, pa.ArrowException) as error:
            raise PreflightError(f"cannot inspect {split_name} shard {shard_id} {path}: {error}") from error
        if sample_table.num_rows != rows:
            raise PreflightError(f"{split_name} shard {shard_id} sample-column scan returned the wrong row count")
        actual_shard_response_tokens = 0
        for row_index, (source_uid, question_sha256, question_text, sample_index, response_length) in enumerate(
            zip(
                sample_columns["source_uid"],
                sample_columns["question_sha256"],
                sample_columns["question_text"],
                sample_columns["sample_index"],
                sample_columns["response_length"],
                strict=True,
            )
        ):
            row_name = f"{split_name} shard {shard_id} row {row_index}"
            if not isinstance(source_uid, str) or not source_uid:
                raise PreflightError(f"{row_name} source_uid must be a non-empty string")
            question_sha256 = _require_sha256(question_sha256, f"{row_name} question_sha256")
            if not isinstance(question_text, str) or not question_text:
                raise PreflightError(f"{row_name} question_text must be a non-empty string")
            actual_question_sha256 = sha256_text(question_text)
            if actual_question_sha256 != question_sha256:
                raise PreflightError(
                    f"{row_name} question_sha256 does not match question_text: "
                    f"{question_sha256} != {actual_question_sha256}"
                )
            sample_index = _require_int(sample_index, f"{row_name} sample_index", minimum=0)
            response_length = _require_int(response_length, f"{row_name} response_length", minimum=1)
            actual_shard_response_tokens += response_length
            prior_question = uid_questions.setdefault(source_uid, question_sha256)
            if prior_question != question_sha256:
                raise PreflightError(f"{split_name} source UID {source_uid!r} maps to multiple questions")
            samples = source_samples.setdefault(source_uid, set())
            if sample_index in samples:
                raise PreflightError(f"{split_name} source UID {source_uid!r} repeats sample_index {sample_index}")
            samples.add(sample_index)
        if actual_shard_response_tokens != declared_shard_response_tokens:
            raise PreflightError(
                f"{split_name} shard {shard_id} actual response-token sum {actual_shard_response_tokens} "
                f"does not match stats {declared_shard_response_tokens}"
            )
        shard_rows += rows
        resolved_files.append(str(path))

    split_rows = _require_int(split_index.get("row_count"), f"{split_name} row_count", minimum=1)
    split_stats = split_index.get("stats")
    if shard_rows != split_rows:
        raise PreflightError(f"{split_name} shard rows sum to {shard_rows}, index declares {split_rows}")
    if not isinstance(split_stats, Mapping):
        raise PreflightError(f"{split_name} aggregate stats must be an object")
    if _require_int(split_stats.get("row_count"), f"{split_name} stats.row_count", minimum=1) != split_rows:
        raise PreflightError(f"{split_name} aggregate stats.row_count does not match row_count")
    if (
        _require_int(split_stats.get("empty_response_count"), f"{split_name} stats.empty_response_count", minimum=0)
        != 0
    ):
        raise PreflightError(f"{split_name} contains empty responses and is not trainer-consumable")
    shard_response_tokens = sum(shard["stats"]["response_token_count"] for shard in shards)
    split_response_tokens = _require_int(
        split_stats.get("response_token_count"), f"{split_name} stats.response_token_count", minimum=split_rows
    )
    if shard_response_tokens != split_response_tokens:
        raise PreflightError(f"{split_name} shard response-token counts do not match aggregate stats")
    if len(uid_questions) != expected_question_count:
        raise PreflightError(
            f"{split_name} contains {len(uid_questions)} source UIDs, expected exactly {expected_question_count}"
        )
    expected_sample_indices = set(range(expected_samples_per_question))
    for source_uid, sample_indices in source_samples.items():
        if sample_indices != expected_sample_indices:
            raise PreflightError(
                f"{split_name} source UID {source_uid!r} has sample indices {sorted(sample_indices)}, "
                f"expected {sorted(expected_sample_indices)}"
            )
    return _VerifiedSplit(files=tuple(resolved_files), uid_questions=uid_questions)


def _default_tokenizer_loader(model: str, revision: str | None, local_files_only: bool) -> Any:
    from transformers import AutoTokenizer

    kwargs: dict[str, Any] = {"trust_remote_code": True, "local_files_only": local_files_only}
    if revision:
        kwargs["revision"] = revision
    return AutoTokenizer.from_pretrained(model, **kwargs)


def _normalize_student_model(student_model: str) -> str:
    if not isinstance(student_model, str) or not student_model.strip():
        raise PreflightError("student_model must be a non-empty model ID or local directory")
    return student_model.strip()


def _student_identity_sha256(student_model: str, student_revision: str | None) -> str:
    student_model = _normalize_student_model(student_model)
    local_student = Path(student_model).expanduser()
    if local_student.exists():
        try:
            return inspect_local_hf_model(local_student).model_identity_sha256
        except ModelIdentityError as error:
            raise PreflightError(str(error)) from error
    revision = _require_immutable_revision(student_revision, "student_revision")
    return remote_model_identity(student_model, revision)["model_identity_sha256"]


def _verify_student_tokenizer(
    *,
    student_model: str,
    student_revision: str | None,
    local_files_only: bool,
    expected: Mapping[str, Any],
    tokenizer_loader: Callable[[str, str | None, bool], Any],
) -> str:
    if not isinstance(expected, Mapping):
        raise PreflightError("index tokenizer must be an object")
    local_student = Path(student_model).expanduser()
    if not local_student.exists() and not student_revision:
        raise PreflightError("a remote --student-model requires an immutable --student-revision")
    if not local_student.exists():
        _require_immutable_revision(student_revision, "student_revision")
    tokenizer = tokenizer_loader(student_model, student_revision, local_files_only)
    actual_sha256, actual_vocab_size = tokenizer_fingerprint(tokenizer)
    expected_sha256 = _require_sha256(expected.get("sha256"), "index tokenizer.sha256")
    expected_vocab_size = _require_int(expected.get("vocab_size"), "index tokenizer.vocab_size", minimum=1)
    if actual_sha256 != expected_sha256 or actual_vocab_size != expected_vocab_size:
        raise PreflightError(
            "student tokenizer does not match trace tokenizer: "
            f"sha256 {actual_sha256} / vocab {actual_vocab_size}, expected "
            f"{expected_sha256} / {expected_vocab_size}"
        )
    return actual_sha256


def run_preflight(
    *,
    dataset_index: str | Path,
    student_model: str,
    student_revision: str | None,
    expected_direction: str,
    expected_teacher_identity_sha256: str,
    expected_student_identity_sha256: str,
    local_files_only: bool = False,
    allow_question_overlap: bool = False,
    expected_questions: Mapping[str, int] | None = None,
    expected_samples_per_question: int | Mapping[str, int] = EXPECTED_SAMPLES_PER_QUESTION,
    tokenizer_loader: Callable[[str, str | None, bool], Any] = _default_tokenizer_loader,
) -> PreflightResult:
    student_model = _normalize_student_model(student_model)
    allowed_directions = {"e4b_rl100_to_e2b", "e2b_base_to_e4b"}
    if expected_direction not in allowed_directions:
        raise PreflightError(f"unsupported expected direction: {expected_direction!r}")
    expected_teacher_identity_sha256 = _require_sha256(
        expected_teacher_identity_sha256, "expected_teacher_identity_sha256"
    )
    expected_student_identity_sha256 = _require_sha256(
        expected_student_identity_sha256, "expected_student_identity_sha256"
    )
    expected_question_counts = dict(expected_questions or EXPECTED_QUESTIONS)
    if set(expected_question_counts) != {"train", "validation"}:
        raise PreflightError("expected question counts must contain exactly train and validation")
    for split_name, count in expected_question_counts.items():
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise PreflightError(f"expected {split_name} question count must be a positive integer")
    expected_sample_counts = _normalize_split_counts(
        expected_samples_per_question, field_name="expected samples_per_question"
    )
    index_path = Path(dataset_index).resolve()
    index = _load_json(index_path, "dataset index")
    index_sha256 = _verify_index_self_hash(index)
    if index.get("manifest_version") != MANIFEST_VERSION or index.get("schema_version") != SCHEMA_VERSION:
        raise PreflightError("dataset index has an unsupported manifest/schema version")
    if index.get("decode_check_performed") is not True:
        raise PreflightError("dataset index was created without the required tokenizer decode check")
    if _require_int(index.get("topk_width"), "dataset index topk_width", minimum=1) != TOPK_WIDTH:
        raise PreflightError(f"dataset index top-k width must be {TOPK_WIDTH}")
    if index.get("direction") != expected_direction:
        raise PreflightError(
            f"dataset direction {index.get('direction')!r} does not match expected direction {expected_direction!r}"
        )
    index_teacher = index.get("teacher")
    if not isinstance(index_teacher, Mapping):
        raise PreflightError("dataset index teacher must be an object")
    teacher_identity_sha256 = hash_json(index_teacher)
    if teacher_identity_sha256 != expected_teacher_identity_sha256:
        raise PreflightError(
            "teacher identity does not match the expected identity: "
            f"{teacher_identity_sha256} != {expected_teacher_identity_sha256}"
        )
    tolerance = _require_number(
        index.get("recommended_training_topk_validation_tolerance"),
        "recommended_training_topk_validation_tolerance",
    )
    if tolerance != FP16_TOPK_MASS_TOLERANCE:
        raise PreflightError(
            f"recommended top-k tolerance must equal the schema constant {FP16_TOPK_MASS_TOLERANCE}, got {tolerance}"
        )
    overlap_count = _require_int(
        index.get("cross_split_question_text_overlap_count"),
        "cross_split_question_text_overlap_count",
        minimum=0,
    )
    overlap_hashes = index.get("cross_split_question_text_overlap_sha256s")
    if not isinstance(overlap_hashes, list) or len(overlap_hashes) != overlap_count:
        raise PreflightError("cross-split overlap count does not match its SHA256 list")
    for position, overlap_hash in enumerate(overlap_hashes):
        _require_sha256(overlap_hash, f"cross_split_question_text_overlap_sha256s[{position}]")
    if overlap_hashes != sorted(set(overlap_hashes)):
        raise PreflightError("cross-split overlap SHA256 list must be unique and sorted")
    if overlap_count and not allow_question_overlap:
        raise PreflightError(
            f"dataset index reports {overlap_count} train/validation question-text overlaps; "
            "repair the split or explicitly pass --allow-question-overlap"
        )
    student_identity_sha256 = _student_identity_sha256(student_model, student_revision or None)
    if student_identity_sha256 != expected_student_identity_sha256:
        raise PreflightError(
            "student identity does not match the expected identity: "
            f"{student_identity_sha256} != {expected_student_identity_sha256}"
        )

    splits = index.get("splits")
    if not isinstance(splits, Mapping) or set(splits) != {"train", "validation"}:
        raise PreflightError("dataset index must contain exactly complete train and validation splits")
    common_configs: dict[str, dict[str, Any]] = {}
    verified_splits: dict[str, _VerifiedSplit] = {}
    split_rows_total = 0
    response_tokens_total = 0
    for split_name in ("train", "validation"):
        split_index = splits[split_name]
        if not isinstance(split_index, Mapping):
            raise PreflightError(f"splits.{split_name} must be an object")
        _, semantic, _ = _verify_run_config(
            index_path=index_path,
            split_name=split_name,
            split_index=split_index,
        )
        common_configs[split_name] = _common_generation_config(semantic)
        question_count = _require_int(split_index.get("question_count"), f"{split_name} question_count", minimum=1)
        expected_question_count = expected_question_counts[split_name]
        if question_count != expected_question_count:
            raise PreflightError(
                f"{split_name} question_count must be exactly {expected_question_count}, got {question_count}"
            )
        unique_question_count = _require_int(
            semantic.get("unique_question_count"),
            f"{split_name} semantic_config.unique_question_count",
            minimum=1,
        )
        if question_count != unique_question_count:
            raise PreflightError(f"{split_name} question count does not match its complete run config")
        samples_per_question = _require_int(
            semantic.get("samples_per_question"),
            f"{split_name} semantic_config.samples_per_question",
            minimum=1,
        )
        expected_split_samples = expected_sample_counts[split_name]
        if samples_per_question != expected_split_samples:
            raise PreflightError(
                f"{split_name} samples_per_question must be exactly {expected_split_samples}, "
                f"got {samples_per_question}"
            )
        verified_splits[split_name] = _verify_shards(
            index_path=index_path,
            split_name=split_name,
            split_index=split_index,
            expected_question_count=expected_question_count,
            expected_samples_per_question=expected_split_samples,
        )
        split_rows_total += _require_int(split_index.get("row_count"), f"{split_name} row_count", minimum=1)
        stats = split_index["stats"]
        response_tokens_total += _require_int(
            stats.get("response_token_count"), f"{split_name} stats.response_token_count", minimum=1
        )
        if split_index.get("source_dataset") != semantic.get("source_dataset"):
            raise PreflightError(f"{split_name} source_dataset does not match its run config")
        if split_index.get("source_dataset_sha256") != semantic.get("source_dataset_sha256"):
            raise PreflightError(f"{split_name} source dataset SHA256 does not match its run config")
        total_shards = _require_int(
            semantic.get("total_shards"), f"{split_name} semantic_config.total_shards", minimum=1
        )
        if total_shards != len(split_index["shards"]):
            raise PreflightError(f"{split_name} shard count does not match its complete run config")
        expected_rows = question_count * samples_per_question
        if split_index["row_count"] != expected_rows:
            raise PreflightError(f"{split_name} row count does not equal questions x samples-per-question")

    if common_configs["train"] != common_configs["validation"]:
        raise PreflightError("train and validation use mixed generation/teacher configurations")
    experiment_sha256 = _require_sha256(index.get("experiment_sha256"), "experiment_sha256")
    experiment_hash_candidates = _experiment_hash_candidates(common_configs["train"], expected_sample_counts)
    if experiment_sha256 not in experiment_hash_candidates:
        raise PreflightError(
            "common generation identity mismatch: "
            f"{experiment_sha256} is not one of {sorted(experiment_hash_candidates)}"
        )

    common = common_configs["train"]
    if (
        common["schema_version"] != SCHEMA_VERSION
        or _require_int(common["topk_width"], "run config topk_width", minimum=1) != TOPK_WIDTH
    ):
        raise PreflightError("run configs do not match the required schema/top-k width")
    direction = common["direction"]
    if direction not in allowed_directions:
        raise PreflightError(f"unsupported generation direction: {direction!r}")
    if direction != expected_direction:
        raise PreflightError(
            f"run config direction {direction!r} does not match expected direction {expected_direction!r}"
        )
    if index.get("direction") != direction:
        raise PreflightError("dataset index direction does not match the run configurations")
    index_sample_counts = _normalize_split_counts(
        index.get("samples_per_question"), field_name="dataset index samples_per_question"
    )
    if index_sample_counts != expected_sample_counts:
        raise PreflightError("dataset index samples_per_question does not match the expected split sample counts")
    for field_name in ("teacher", "tokenizer", "chat_template", "sampling"):
        if not isinstance(common[field_name], Mapping):
            raise PreflightError(f"run config {field_name} must be an object")
        if index.get(field_name) != common[field_name]:
            raise PreflightError(f"dataset index {field_name} does not match the run configurations")
    _verify_generation_contract(common)
    teacher = common["teacher"]
    if not isinstance(teacher, Mapping) or not (teacher.get("revision") or teacher.get("content_sha256")):
        raise PreflightError("teacher identity lacks both immutable revision and content SHA256")
    _require_sha256(teacher.get("model_identity_sha256"), "teacher.model_identity_sha256")
    if teacher.get("content_sha256") is not None:
        _require_sha256(teacher["content_sha256"], "teacher.content_sha256")
        if not isinstance(teacher.get("content_sha256_kind"), str) or not teacher["content_sha256_kind"]:
            raise PreflightError("local teacher identity requires content_sha256_kind")
    else:
        _require_immutable_revision(teacher.get("revision"), "teacher.revision")
        if teacher.get("content_sha256_kind") is not None:
            raise PreflightError("remote teacher content_sha256_kind must be null")
    if hash_json(teacher) != teacher_identity_sha256:
        raise PreflightError("run config teacher identity does not match the verified dataset teacher identity")

    if _require_int(index.get("total_rows"), "dataset index total_rows", minimum=1) != split_rows_total:
        raise PreflightError("dataset index total_rows does not equal the split row total")
    if (
        _require_int(index.get("total_response_tokens"), "dataset index total_response_tokens", minimum=1)
        != response_tokens_total
    ):
        raise PreflightError("dataset index total_response_tokens does not equal the split token total")
    if set(verified_splits["train"].files).intersection(verified_splits["validation"].files):
        raise PreflightError("train and validation reference at least one identical shard path")
    train_uid_questions = verified_splits["train"].uid_questions
    validation_uid_questions = verified_splits["validation"].uid_questions
    uid_overlap = set(train_uid_questions).intersection(validation_uid_questions)
    if uid_overlap:
        raise PreflightError(f"train/validation source UIDs overlap; first: {sorted(uid_overlap)[:5]}")
    actual_question_overlap = set(train_uid_questions.values()).intersection(validation_uid_questions.values())
    if actual_question_overlap != set(overlap_hashes):
        raise PreflightError("indexed train/validation question-text overlap hashes do not match the Parquet data")
    student_tokenizer_sha256 = _verify_student_tokenizer(
        student_model=student_model,
        student_revision=student_revision or None,
        local_files_only=local_files_only,
        expected=common["tokenizer"],
        tokenizer_loader=tokenizer_loader,
    )
    return PreflightResult(
        train_files=verified_splits["train"].files,
        validation_files=verified_splits["validation"].files,
        topk_width=TOPK_WIDTH,
        topk_validation_tolerance=tolerance,
        dataset_index_sha256=index_sha256,
        experiment_sha256=experiment_sha256,
        direction=direction,
        teacher_identity_sha256=teacher_identity_sha256,
        student_identity_sha256=student_identity_sha256,
        student_tokenizer_sha256=student_tokenizer_sha256,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-index", required=True)
    parser.add_argument("--student-model", required=True)
    parser.add_argument("--student-revision", default=None)
    parser.add_argument("--expected-direction", choices=("e4b_rl100_to_e2b", "e2b_base_to_e4b"), required=True)
    parser.add_argument("--expected-teacher-identity-sha256", required=True)
    parser.add_argument("--expected-student-identity-sha256", required=True)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--allow-question-overlap", action="store_true")
    parser.add_argument("--expected-train-questions", type=int, default=EXPECTED_QUESTIONS["train"])
    parser.add_argument("--expected-validation-questions", type=int, default=EXPECTED_QUESTIONS["validation"])
    parser.add_argument("--expected-train-samples-per-question", type=int, default=EXPECTED_SAMPLES_PER_QUESTION)
    parser.add_argument("--expected-validation-samples-per-question", type=int, default=EXPECTED_SAMPLES_PER_QUESTION)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run_preflight(
            dataset_index=args.dataset_index,
            student_model=args.student_model,
            student_revision=args.student_revision,
            expected_direction=args.expected_direction,
            expected_teacher_identity_sha256=args.expected_teacher_identity_sha256,
            expected_student_identity_sha256=args.expected_student_identity_sha256,
            local_files_only=args.local_files_only,
            allow_question_overlap=args.allow_question_overlap,
            expected_questions={
                "train": args.expected_train_questions,
                "validation": args.expected_validation_questions,
            },
            expected_samples_per_question={
                "train": args.expected_train_samples_per_question,
                "validation": args.expected_validation_samples_per_question,
            },
        )
    except (OSError, RuntimeError, ValueError, PreflightError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    sys.stdout.buffer.write(b"\n".join(line.encode("utf-8") for line in result.lines()) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
