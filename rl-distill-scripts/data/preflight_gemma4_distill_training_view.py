#!/usr/bin/env python3
"""Fail-closed preflight for a derived Gemma-4 trace training view."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import preflight_gemma4_topk_distill as source_preflight
import pyarrow as pa
import pyarrow.parquet as pq
from build_gemma4_distill_training_view import (
    SELECTION_SCHEMA_VERSION,
    VIEW_SCHEMA_VERSION,
)
from gemma4_distill_trace_schema import (
    FP16_TOPK_MASS_TOLERANCE,
    SCHEMA_VERSION,
    TOPK_WIDTH,
    derive_sampling_seed,
    hash_json,
    sha256_file,
    sha256_text,
    trace_arrow_schema,
)

# Direction labels recorded by the trace generator, one per (teacher, band). The suffix
# names the trace collection, not the student: a teacher's trace set is reused for every
# student (e4b base and e2b base); the student is verified separately by identity SHA.
ALLOWED_DIRECTIONS = {
    "e4b_easy_to_e2b",
    "e4b_medium_to_e2b",
    "e4b_hard_to_e2b",
    "12b_easy_to_e2b",
    "12b_medium_to_e2b",
    "12b_hard_to_e2b",
    "26b_easy_to_e2b",
    "26b_medium_to_e2b",
    "26b_hard_to_e2b",
    "e2b_easy_to_e2b",
    "e2b_medium_to_e2b",
    "e2b_hard_to_e2b",
}


class TrainingViewPreflightError(ValueError):
    """Raised when a derived training view cannot be trusted by the trainer."""


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TrainingViewPreflightError(f"cannot read {description} {path}: {error}") from error
    if not isinstance(value, dict):
        raise TrainingViewPreflightError(f"{description} {path} must contain a JSON object")
    return value


def _self_hash(value: Mapping[str, Any], field: str, description: str) -> str:
    claimed = value.get(field)
    if not isinstance(claimed, str) or len(claimed) != 64:
        raise TrainingViewPreflightError(f"{description} {field} is malformed")
    unhashed = dict(value)
    del unhashed[field]
    actual = hash_json(unhashed)
    if actual != claimed:
        raise TrainingViewPreflightError(f"{description} self-hash mismatch: {actual} != {claimed}")
    return claimed


def _resolve(root: Path, relative: Any, description: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise TrainingViewPreflightError(f"{description} must be a non-empty relative path")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise TrainingViewPreflightError(f"unsafe {description}: {relative!r}")
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise TrainingViewPreflightError(f"{description} escapes the view root: {relative}") from error
    return resolved


def _positive_int(value: Any, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TrainingViewPreflightError(f"{description} must be a positive integer")
    return value


def _verify_generation(index: Mapping[str, Any], expected_direction: str) -> dict[str, Any]:
    semantic = index.get("generation_semantic_config")
    if not isinstance(semantic, dict):
        raise TrainingViewPreflightError("generation_semantic_config is missing")
    source = index.get("source_trace")
    if not isinstance(source, Mapping):
        raise TrainingViewPreflightError("source_trace is missing")
    if hash_json(semantic) != source.get("generation_config_sha256"):
        raise TrainingViewPreflightError("embedded generation semantic config hash is invalid")
    if semantic.get("schema_version") != SCHEMA_VERSION or semantic.get("topk_width") != TOPK_WIDTH:
        raise TrainingViewPreflightError("source generation schema/top-k width is invalid")
    direction = semantic.get("direction")
    if direction not in ALLOWED_DIRECTIONS or direction != expected_direction:
        raise TrainingViewPreflightError(
            f"source direction {direction!r} does not match expected direction {expected_direction!r}"
        )
    if index.get("direction") != direction:
        raise TrainingViewPreflightError("view direction does not match the source generation direction")
    for field in ("teacher", "tokenizer", "chat_template", "sampling"):
        if index.get(field) != semantic.get(field):
            raise TrainingViewPreflightError(f"view {field} does not match source generation provenance")
    sampling = semantic.get("sampling")
    expected_sampling = source_preflight.EXPECTED_SAMPLING
    if not isinstance(sampling, Mapping):
        raise TrainingViewPreflightError("source sampling contract is missing")
    for key, expected in expected_sampling.items():
        actual = sampling.get(key)
        if isinstance(expected, float):
            if not isinstance(actual, int | float) or not math.isclose(float(actual), expected):
                raise TrainingViewPreflightError(f"sampling.{key} does not match the production contract")
        elif actual != expected:
            raise TrainingViewPreflightError(f"sampling.{key} does not match the production contract")
    return semantic


def _verify_validation_run_config(root: Path, index: Mapping[str, Any], train_semantic: Mapping[str, Any]) -> str:
    """Check the copied validation-split run config and return its generation identity SHA256.

    A view whose validation rows come from the bundle's validation split carries a second generation
    identity (the split and sample count differ), so it must be pinned in the index and share every
    teacher/tokenizer/template/sampling field with the train generation.
    """
    source = index.get("source_trace")
    if not isinstance(source, Mapping):
        raise TrainingViewPreflightError("source_trace is missing")
    expected_sha256 = source.get("validation_generation_config_sha256")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise TrainingViewPreflightError("source_trace.validation_generation_config_sha256 is missing")
    config_path = _resolve(root, index.get("source_validation_run_config_path"), "source validation run config path")
    if sha256_file(config_path) != source.get("validation_run_config_sha256"):
        raise TrainingViewPreflightError("copied source validation run configuration SHA256 mismatch")
    config = _load_json(config_path, "copied source validation run configuration")
    semantic = config.get("semantic_config")
    if (
        not isinstance(semantic, Mapping)
        or config.get("generation_config_sha256") != expected_sha256
        or hash_json(semantic) != expected_sha256
    ):
        raise TrainingViewPreflightError("copied source validation run configuration has the wrong generation identity")
    if semantic.get("split") != "validation":
        raise TrainingViewPreflightError(
            "copied source validation run configuration does not describe the validation split"
        )
    for key in (
        "schema_version",
        "topk_width",
        "direction",
        "global_seed",
        "teacher",
        "tokenizer",
        "chat_template",
        "sampling",
    ):
        if semantic.get(key) != train_semantic.get(key):
            raise TrainingViewPreflightError(f"validation split generation differs from the train generation in {key}")
    return expected_sha256


def _verify_view_receipts(root: Path, index: Mapping[str, Any], index_sha256: str) -> None:
    source = index.get("source_trace")
    if not isinstance(source, Mapping):
        raise TrainingViewPreflightError("source_trace is missing")
    completion_path = root / "COMPLETE.json"
    completion = _load_json(completion_path, "training view completion receipt")
    if completion.get("dataset_index_sha256") != index_sha256:
        raise TrainingViewPreflightError("training view completion receipt does not bind the dataset index")
    expected_completion = {
        "source_dataset_index_sha256": source.get("dataset_index_sha256"),
        "train_questions": index["splits"]["train"]["question_count"],
        "validation_questions": index["splits"]["validation"]["question_count"],
        "total_rows": index["total_rows"],
        "total_response_tokens": index["total_response_tokens"],
    }
    for field, expected in expected_completion.items():
        if completion.get(field) != expected:
            raise TrainingViewPreflightError(
                f"training view completion receipt field {field!r} does not match the dataset index"
            )

    source_config_path = _resolve(root, index.get("source_run_config_path"), "source run config path")
    if sha256_file(source_config_path) != source.get("run_config_sha256"):
        raise TrainingViewPreflightError("copied source run configuration SHA256 mismatch")
    source_config = _load_json(source_config_path, "copied source run configuration")
    if source_config.get("generation_config_sha256") != source.get("generation_config_sha256"):
        raise TrainingViewPreflightError("copied source run configuration has the wrong generation identity")
    semantic = source_config.get("semantic_config")
    if not isinstance(semantic, dict) or hash_json(semantic) != source.get("generation_config_sha256"):
        raise TrainingViewPreflightError("copied source run configuration semantic hash is invalid")
    if semantic != index.get("generation_semantic_config"):
        raise TrainingViewPreflightError("embedded generation config differs from the copied source run config")


def _verify_split(
    *,
    root: Path,
    split_name: str,
    split: Mapping[str, Any],
    expected_uids: set[str],
    expected_sample_indices: set[int],
    expected_generation_config_sha256: str,
    expected_direction: str,
    expected_global_seed: int,
    expected_source_split: str = "train",
) -> tuple[list[str], int]:
    if split.get("complete") is not True:
        raise TrainingViewPreflightError(f"{split_name} split is not complete")
    question_count = _positive_int(split.get("question_count"), f"{split_name}.question_count")
    if question_count != len(expected_uids):
        raise TrainingViewPreflightError(f"{split_name} question count does not match the selection manifest")
    shards = split.get("shards")
    files = split.get("parquet_files")
    if not isinstance(shards, list) or not shards or not isinstance(files, list) or len(files) != len(shards):
        raise TrainingViewPreflightError(f"{split_name} shard/file lists are malformed")
    seen: dict[str, set[int]] = {}
    resolved_files: list[str] = []
    row_total = 0
    response_token_total = 0
    expected_schema = trace_arrow_schema()
    for position, entry in enumerate(shards):
        if not isinstance(entry, Mapping) or entry.get("shard_id") != position:
            raise TrainingViewPreflightError(f"{split_name} shard IDs must be contiguous")
        if files[position] != entry.get("path"):
            raise TrainingViewPreflightError(f"{split_name} shard/file roster mismatch at {position}")
        path = _resolve(root, entry.get("path"), f"{split_name} shard path")
        if not path.is_file():
            raise TrainingViewPreflightError(f"missing {split_name} shard: {path}")
        if path.stat().st_size != entry.get("size_bytes") or sha256_file(path) != entry.get("sha256"):
            raise TrainingViewPreflightError(f"{split_name} shard size/SHA mismatch: {path}")
        parquet = pq.ParquetFile(path)
        if not parquet.schema_arrow.equals(expected_schema, check_metadata=False):
            raise TrainingViewPreflightError(f"{split_name} shard trace schema mismatch: {path}")
        rows = _positive_int(entry.get("rows"), f"{split_name} shard rows")
        row_groups = _positive_int(entry.get("row_groups"), f"{split_name} shard row_groups")
        if parquet.metadata.num_rows != rows or parquet.metadata.num_row_groups != row_groups:
            raise TrainingViewPreflightError(f"{split_name} shard parquet metadata mismatch: {path}")
        table = parquet.read(
            columns=[
                "generation_config_sha256",
                "trace_id",
                "direction",
                "split",
                "source_uid",
                "question_sha256",
                "question_text",
                "sample_index",
                "global_seed",
                "sampling_seed",
                "response_length",
            ]
        )
        response_tokens = 0
        values = table.to_pydict()
        for (
            generation_config_sha256,
            trace_id,
            direction,
            source_split,
            uid,
            question_sha256,
            question_text,
            sample_index,
            global_seed,
            sampling_seed,
            response_length,
        ) in zip(
            values["generation_config_sha256"],
            values["trace_id"],
            values["direction"],
            values["split"],
            values["source_uid"],
            values["question_sha256"],
            values["question_text"],
            values["sample_index"],
            values["global_seed"],
            values["sampling_seed"],
            values["response_length"],
            strict=True,
        ):
            if generation_config_sha256 != expected_generation_config_sha256:
                raise TrainingViewPreflightError(f"{split_name} contains a mixed generation identity")
            if direction != expected_direction or source_split != expected_source_split:
                raise TrainingViewPreflightError(
                    f"{split_name} rewrote immutable direction/split provenance for UID {uid!r}"
                )
            if uid not in expected_uids:
                raise TrainingViewPreflightError(f"{split_name} contains unselected source UID {uid!r}")
            if sha256_text(str(question_text)) != question_sha256:
                raise TrainingViewPreflightError(f"{split_name} question hash mismatch for UID {uid!r}")
            sample_index = int(sample_index)
            if sample_index not in expected_sample_indices:
                raise TrainingViewPreflightError(
                    f"{split_name} UID {uid!r} contains unexpected sample index {sample_index}"
                )
            if int(global_seed) != expected_global_seed:
                raise TrainingViewPreflightError(f"{split_name} changed global_seed for UID {uid!r}")
            if int(sampling_seed) != derive_sampling_seed(
                expected_global_seed, expected_source_split, str(uid), sample_index
            ):
                raise TrainingViewPreflightError(f"{split_name} changed sampling_seed for UID {uid!r}")
            expected_trace_id = hash_json(
                {
                    "generation_config_sha256": expected_generation_config_sha256,
                    "source_uid": uid,
                    "question_sha256": question_sha256,
                    "sample_index": sample_index,
                }
            )
            if trace_id != expected_trace_id:
                raise TrainingViewPreflightError(f"{split_name} trace identity mismatch for UID {uid!r}")
            samples = seen.setdefault(str(uid), set())
            if sample_index in samples:
                raise TrainingViewPreflightError(f"{split_name} repeats UID/sample {uid!r}/{sample_index}")
            samples.add(sample_index)
            response_tokens += int(response_length)
        stats = entry.get("stats")
        if not isinstance(stats, Mapping) or stats.get("row_count") != rows:
            raise TrainingViewPreflightError(f"{split_name} shard stats are malformed: {path}")
        if stats.get("empty_response_count") != 0 or stats.get("response_token_count") != response_tokens:
            raise TrainingViewPreflightError(f"{split_name} shard response stats mismatch: {path}")
        row_total += rows
        response_token_total += response_tokens
        resolved_files.append(str(path))
    if set(seen) != expected_uids or any(samples != expected_sample_indices for samples in seen.values()):
        raise TrainingViewPreflightError(f"{split_name} does not contain the exact selected UID/sample product")
    expected_rows = len(expected_uids) * len(expected_sample_indices)
    if row_total != expected_rows or split.get("row_count") != expected_rows:
        raise TrainingViewPreflightError(f"{split_name} row count does not equal questions x samples")
    stats = split.get("stats")
    if not isinstance(stats, Mapping) or stats.get("row_count") != row_total:
        raise TrainingViewPreflightError(f"{split_name} aggregate stats are malformed")
    if stats.get("empty_response_count") != 0 or stats.get("response_token_count") != response_token_total:
        raise TrainingViewPreflightError(f"{split_name} aggregate response stats mismatch")
    return resolved_files, response_token_total


def run_preflight(
    *,
    dataset_index: str | Path,
    student_model: str,
    expected_direction: str,
    expected_teacher_identity_sha256: str,
    expected_student_identity_sha256: str,
    local_files_only: bool,
    expected_train_questions: int,
    expected_validation_questions: int,
    expected_train_samples_per_question: int,
    expected_validation_samples_per_question: int,
) -> source_preflight.PreflightResult:
    index_path = Path(dataset_index).resolve(strict=True)
    root = index_path.parent
    index = _load_json(index_path, "training view index")
    if index.get("schema_version") != VIEW_SCHEMA_VERSION:
        raise TrainingViewPreflightError(f"unsupported training view schema: {index.get('schema_version')!r}")
    index_sha256 = _self_hash(index, "dataset_index_sha256", "training view index")
    semantic = _verify_generation(index, expected_direction)

    splits = index.get("splits")
    if not isinstance(splits, Mapping) or set(splits) != {"train", "validation"}:
        raise TrainingViewPreflightError("training view must contain train and validation splits")
    _verify_view_receipts(root, index, index_sha256)

    selection_path = _resolve(root, index.get("selection_manifest_path"), "selection manifest path")
    if sha256_file(selection_path) != index.get("selection_manifest_sha256"):
        raise TrainingViewPreflightError("selection manifest SHA256 mismatch")
    selection = _load_json(selection_path, "selection manifest")
    if selection.get("schema_version") != SELECTION_SCHEMA_VERSION:
        raise TrainingViewPreflightError("selection manifest schema is invalid")
    _self_hash(selection, "selection_sha256", "selection manifest")
    if selection.get("selection_algorithm") != "numpy-randomstate-permutation-v1" or selection.get("seed") != 42:
        raise TrainingViewPreflightError("selection manifest does not use the required deterministic seed/algorithm")
    train_uids = selection.get("train_source_uids")
    validation_uids = selection.get("validation_source_uids")
    if not isinstance(train_uids, list) or not isinstance(validation_uids, list):
        raise TrainingViewPreflightError("selection UID rosters are missing")
    train_uid_set = set(train_uids)
    validation_uid_set = set(validation_uids)
    if len(train_uid_set) != len(train_uids) or len(validation_uid_set) != len(validation_uids):
        raise TrainingViewPreflightError("selection manifest repeats source UIDs")
    if train_uid_set.intersection(validation_uid_set):
        raise TrainingViewPreflightError("selection manifest train/validation UIDs overlap")
    if len(train_uid_set) != expected_train_questions or len(validation_uid_set) != expected_validation_questions:
        raise TrainingViewPreflightError("selection question counts do not match the requested run")
    if selection.get("train_question_count") != len(train_uid_set) or selection.get("validation_question_count") != len(
        validation_uid_set
    ):
        raise TrainingViewPreflightError("selection manifest count fields do not match its UID rosters")
    validation_source_split = selection.get("validation_source_split", "train")
    if validation_source_split not in ("train", "validation"):
        raise TrainingViewPreflightError("selection manifest has an invalid validation_source_split")
    if index["source_trace"].get("validation_split", "train") != validation_source_split:
        raise TrainingViewPreflightError("view source_trace.validation_split disagrees with the selection manifest")
    source_question_count = selection.get("source_question_count")
    unused_question_count = selection.get("unused_question_count")
    if not isinstance(source_question_count, int) or not isinstance(unused_question_count, int):
        raise TrainingViewPreflightError("selection source/unused question counts are inconsistent")
    if validation_source_split == "train":
        if source_question_count != len(train_uid_set) + len(validation_uid_set) + unused_question_count:
            raise TrainingViewPreflightError("selection source/unused question counts are inconsistent")
        validation_generation_config_sha256 = None
    else:
        if source_question_count != len(train_uid_set) + unused_question_count:
            raise TrainingViewPreflightError("selection source/unused question counts are inconsistent")
        validation_source_count = selection.get("validation_source_question_count")
        validation_unused = selection.get("validation_unused_question_count")
        if (
            not isinstance(validation_source_count, int)
            or not isinstance(validation_unused, int)
            or validation_source_count != len(validation_uid_set) + validation_unused
        ):
            raise TrainingViewPreflightError("selection validation-source/unused question counts are inconsistent")
        validation_generation_config_sha256 = _verify_validation_run_config(root, index, semantic)
    if selection.get("train_samples_per_question") != expected_train_samples_per_question:
        raise TrainingViewPreflightError("selection train sample count does not match the requested run")
    if expected_validation_samples_per_question != 1 or selection.get("validation_samples_per_question") != 1:
        raise TrainingViewPreflightError("training views require exactly one validation sample per question")
    validation_sample_index = selection.get("validation_sample_index")
    if not isinstance(validation_sample_index, int) or validation_sample_index < 0:
        raise TrainingViewPreflightError("validation sample index is malformed")

    generation_config_sha256 = str(index["source_trace"]["generation_config_sha256"])
    global_seed = semantic.get("global_seed")
    if not isinstance(global_seed, int) or global_seed != 42:
        raise TrainingViewPreflightError("source generation global_seed must be 42")
    train_files, train_tokens = _verify_split(
        root=root,
        split_name="train",
        split=splits["train"],
        expected_uids=train_uid_set,
        expected_sample_indices=set(range(expected_train_samples_per_question)),
        expected_generation_config_sha256=generation_config_sha256,
        expected_direction=expected_direction,
        expected_global_seed=global_seed,
    )
    validation_files, validation_tokens = _verify_split(
        root=root,
        split_name="validation",
        split=splits["validation"],
        expected_uids=validation_uid_set,
        expected_sample_indices={validation_sample_index},
        expected_generation_config_sha256=validation_generation_config_sha256 or generation_config_sha256,
        expected_direction=expected_direction,
        expected_global_seed=global_seed,
        expected_source_split=validation_source_split,
    )
    expected_rows = expected_train_questions * expected_train_samples_per_question + expected_validation_questions
    if index.get("total_rows") != expected_rows:
        raise TrainingViewPreflightError("training view total row count is invalid")
    if index.get("total_response_tokens") != train_tokens + validation_tokens:
        raise TrainingViewPreflightError("training view total response-token count is invalid")
    if index.get("samples_per_question") != {
        "train": expected_train_samples_per_question,
        "validation": 1,
    }:
        raise TrainingViewPreflightError("training view split sample-count contract is invalid")

    teacher = semantic.get("teacher")
    if not isinstance(teacher, Mapping) or hash_json(teacher) != expected_teacher_identity_sha256:
        raise TrainingViewPreflightError("teacher identity does not match the requested run")
    student_identity = source_preflight._student_identity_sha256(student_model, None)
    if student_identity != expected_student_identity_sha256:
        raise TrainingViewPreflightError(
            f"student identity mismatch: {student_identity} != {expected_student_identity_sha256}"
        )
    student_tokenizer_sha256 = source_preflight._verify_student_tokenizer(
        student_model=student_model,
        student_revision=None,
        local_files_only=local_files_only,
        expected=semantic["tokenizer"],
        tokenizer_loader=source_preflight._default_tokenizer_loader,
    )
    return source_preflight.PreflightResult(
        train_files=tuple(train_files),
        validation_files=tuple(validation_files),
        topk_width=TOPK_WIDTH,
        topk_validation_tolerance=FP16_TOPK_MASS_TOLERANCE,
        dataset_index_sha256=index_sha256,
        experiment_sha256=str(index["source_trace"]["generation_config_sha256"]),
        direction=expected_direction,
        teacher_identity_sha256=expected_teacher_identity_sha256,
        student_identity_sha256=expected_student_identity_sha256,
        student_tokenizer_sha256=student_tokenizer_sha256,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-index", required=True)
    parser.add_argument("--student-model", required=True)
    parser.add_argument("--expected-direction", choices=sorted(ALLOWED_DIRECTIONS), required=True)
    parser.add_argument("--expected-teacher-identity-sha256", required=True)
    parser.add_argument("--expected-student-identity-sha256", required=True)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--expected-train-questions", type=int, default=8_000)
    parser.add_argument("--expected-validation-questions", type=int, default=500)
    parser.add_argument("--expected-train-samples-per-question", type=int, default=8)
    parser.add_argument("--expected-validation-samples-per-question", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run_preflight(
            dataset_index=args.dataset_index,
            student_model=args.student_model,
            expected_direction=args.expected_direction,
            expected_teacher_identity_sha256=args.expected_teacher_identity_sha256,
            expected_student_identity_sha256=args.expected_student_identity_sha256,
            local_files_only=args.local_files_only,
            expected_train_questions=args.expected_train_questions,
            expected_validation_questions=args.expected_validation_questions,
            expected_train_samples_per_question=args.expected_train_samples_per_question,
            expected_validation_samples_per_question=args.expected_validation_samples_per_question,
        )
    except (OSError, TrainingViewPreflightError, source_preflight.PreflightError, pa.ArrowException) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print("\n".join(result.lines()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
