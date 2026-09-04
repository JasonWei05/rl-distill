#!/usr/bin/env python3
"""Build a deterministic logical train/validation view of collected traces.

The source trace bundle is immutable and records the split and sampling seed
used during generation.  A training view therefore copies selected rows
without rewriting those provenance fields.  Logical train/validation
membership lives in this view's signed index and selection manifest.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from gemma4_distill_trace_schema import (
    SCHEMA_VERSION,
    TOPK_WIDTH,
    atomic_write_json,
    hash_json,
    sha256_file,
    trace_arrow_schema,
)
from gemma4_trace_s3 import TraceS3Mirror

VIEW_SCHEMA_VERSION = "gemma4-distill-training-view-v1"
SELECTION_SCHEMA_VERSION = "gemma4-distill-training-selection-v1"
DEFAULT_TRAIN_QUESTIONS = 8_000
DEFAULT_VALIDATION_QUESTIONS = 500
DEFAULT_TRAIN_SAMPLES = 8
DEFAULT_VALIDATION_SAMPLE_INDEX = 0


class TrainingViewError(ValueError):
    """Raised when a source bundle or derived view violates its contract."""


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TrainingViewError(f"cannot read {description} {path}: {error}") from error
    if not isinstance(value, dict):
        raise TrainingViewError(f"{description} {path} must contain a JSON object")
    return value


def _verify_self_hash(value: Mapping[str, Any], field: str, description: str) -> str:
    claimed = value.get(field)
    if not isinstance(claimed, str) or len(claimed) != 64:
        raise TrainingViewError(f"{description} {field} is missing or malformed")
    unhashed = dict(value)
    del unhashed[field]
    actual = hash_json(unhashed)
    if actual != claimed:
        raise TrainingViewError(f"{description} self-hash mismatch: {actual} != {claimed}")
    return claimed


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve(strict=True)
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise TrainingViewError(f"path escapes output root {root}: {path}") from error


VALIDATION_SOURCES = ("train", "validation")


def _source_parquet(source_root: Path, split: str = "train") -> Path:
    """Return the prepared roster parquet for ``split``.

    Collection bundles keep the RL data-prep outputs under ``source/``: one ``*_train.parquet`` and
    one ``*_val*.parquet``.  A lone parquet (older canaries) is accepted for either split.
    """
    candidates = sorted((source_root / "source").glob("*.parquet"))
    if len(candidates) == 1:
        return candidates[0]
    if split == "train":
        matches = [path for path in candidates if path.stem.endswith("_train")]
    else:
        matches = [path for path in candidates if "_val" in path.stem]
    if len(matches) != 1:
        raise TrainingViewError(f"expected exactly one {split} roster parquet under source/, found {matches}")
    return matches[0]


def _resolve_source_path(source_root: Path, relative: Any, description: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise TrainingViewError(f"{description} must be a non-empty relative path")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise TrainingViewError(f"unsafe {description}: {relative!r}")
    resolved = (source_root / path).resolve()
    try:
        resolved.relative_to(source_root.resolve())
    except ValueError as error:
        raise TrainingViewError(f"{description} escapes source root: {relative}") from error
    return resolved


def _validated_source_shards(
    source_root: Path,
    source_split: Mapping[str, Any],
    *,
    expected_questions: int,
    expected_samples_per_question: int,
    roster_split: str = "train",
) -> tuple[list[Path], list[str]]:
    """Verify the source shard roster and return traced UIDs in source-roster order.

    Small collection canaries deliberately generate only a prefix/subset of a
    larger prepared prompt parquet.  The dataset index records the number of
    traced questions, while the source parquet records the complete prepared
    roster.  Derive the eligible roster from the immutable trace shards and
    then order those UIDs by the prepared source parquet.  Production bundles
    still fail closed unless this is the exact expected question/sample
    product.
    """

    roster_table = pq.read_table(_source_parquet(source_root, roster_split), columns=["uid"])
    roster_uids = roster_table.column("uid").to_pylist()
    if len(set(roster_uids)) != len(roster_uids):
        raise TrainingViewError(f"source roster repeats UIDs: rows={len(roster_uids)} unique={len(set(roster_uids))}")
    if not all(isinstance(uid, str) and uid for uid in roster_uids):
        raise TrainingViewError("source roster contains an empty or non-string UID")

    shards = source_split.get("shards")
    files = source_split.get("parquet_files")
    if not isinstance(shards, list) or not shards:
        raise TrainingViewError("source split has no shard roster")
    if not isinstance(files, list) or len(files) != len(shards):
        raise TrainingViewError("source split parquet_files does not align with its shards")

    expected_schema = trace_arrow_schema()
    paths: list[Path] = []
    traced_samples: dict[str, set[int]] = {}
    for position, entry in enumerate(shards):
        if not isinstance(entry, Mapping) or int(entry.get("shard_id", -1)) != position:
            raise TrainingViewError(f"source shard IDs are not contiguous at position {position}")
        if files[position] != entry.get("path"):
            raise TrainingViewError(f"source parquet_files disagrees with shard {position}")
        source_path = _resolve_source_path(source_root, entry.get("path"), f"source shard {position} path")
        if not source_path.is_file():
            raise TrainingViewError(f"source shard is missing: {source_path}")
        if source_path.stat().st_size != int(entry.get("size_bytes", -1)):
            raise TrainingViewError(f"source shard size mismatch: {source_path}")
        if sha256_file(source_path) != entry.get("sha256"):
            raise TrainingViewError(f"source shard SHA256 mismatch: {source_path}")
        parquet = pq.ParquetFile(source_path)
        if not parquet.schema_arrow.equals(expected_schema, check_metadata=False):
            raise TrainingViewError(f"source shard schema mismatch: {source_path}")
        if parquet.metadata.num_rows != int(entry.get("rows", -1)):
            raise TrainingViewError(f"source shard row-count mismatch: {source_path}")
        if parquet.metadata.num_row_groups != int(entry.get("row_groups", -1)):
            raise TrainingViewError(f"source shard row-group-count mismatch: {source_path}")
        table = parquet.read(columns=["source_uid", "sample_index"])
        for uid, sample_index in zip(
            table.column("source_uid").to_pylist(),
            table.column("sample_index").to_pylist(),
            strict=True,
        ):
            if not isinstance(uid, str) or not uid:
                raise TrainingViewError(f"source shard contains an invalid source_uid: {source_path}")
            samples = traced_samples.setdefault(uid, set())
            sample_index = int(sample_index)
            if sample_index in samples:
                raise TrainingViewError(f"source dataset repeats UID/sample {uid!r}/{sample_index}")
            samples.add(sample_index)
        paths.append(source_path)

    expected_samples = set(range(expected_samples_per_question))
    if len(traced_samples) != expected_questions:
        raise TrainingViewError(
            f"source trace shards contain {len(traced_samples)} unique UIDs; expected {expected_questions}"
        )
    if any(samples != expected_samples for samples in traced_samples.values()):
        raise TrainingViewError("source trace shards do not contain the exact expected sample product")
    unknown = set(traced_samples).difference(roster_uids)
    if unknown:
        raise TrainingViewError(f"source trace UIDs are absent from the prepared roster: {sorted(unknown)[:3]}")
    traced_uids = [uid for uid in roster_uids if uid in traced_samples]
    if len(traced_uids) != expected_questions:
        raise TrainingViewError("could not order every traced UID using the prepared source roster")
    return paths, traced_uids


def build_selection(
    source_uids: Sequence[str],
    *,
    seed: int,
    train_questions: int,
    validation_questions: int,
    train_samples_per_question: int,
    validation_sample_index: int,
    validation_source_uids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Deterministically select train/validation questions.

    ``validation_source_uids=None`` (validation source ``train``) carves the validation questions
    out of the train roster.  Otherwise (validation source ``validation``) every requested train
    question comes from the train roster and the validation questions are a deterministic subset
    of the bundle's own validation split, so the two never compete for the same questions.
    """
    if train_questions <= 0 or validation_questions <= 0:
        raise TrainingViewError("train and validation question counts must be positive")
    if train_samples_per_question <= 0 or validation_sample_index < 0:
        raise TrainingViewError("sample counts and indices are invalid")
    permutation = np.random.RandomState(seed).permutation(len(source_uids)).tolist()
    if validation_source_uids is None:
        if train_questions + validation_questions > len(source_uids):
            raise TrainingViewError("requested train plus validation questions exceed the source roster")
        train_indices = permutation[:train_questions]
        validation_indices = permutation[train_questions : train_questions + validation_questions]
        unused_indices = permutation[train_questions + validation_questions :]
        validation_uids = [source_uids[index] for index in validation_indices]
        validation_extra: dict[str, Any] = {"validation_source_split": "train"}
    else:
        if train_questions > len(source_uids):
            raise TrainingViewError("requested train questions exceed the source roster")
        if validation_questions > len(validation_source_uids):
            raise TrainingViewError("requested validation questions exceed the validation roster")
        train_indices = permutation[:train_questions]
        unused_indices = permutation[train_questions:]
        validation_permutation = np.random.RandomState(seed).permutation(len(validation_source_uids)).tolist()
        validation_indices = validation_permutation[:validation_questions]
        validation_unused = validation_permutation[validation_questions:]
        validation_uids = [validation_source_uids[index] for index in validation_indices]
        validation_extra = {
            "validation_source_split": "validation",
            "validation_source_question_count": len(validation_source_uids),
            "validation_source_roster_sha256": hash_json(list(validation_source_uids)),
            "validation_unused_question_count": len(validation_unused),
            "validation_unused_source_uids_sha256": hash_json(
                [validation_source_uids[index] for index in validation_unused]
            ),
        }
    manifest: dict[str, Any] = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "selection_algorithm": "numpy-randomstate-permutation-v1",
        "seed": seed,
        "source_question_count": len(source_uids),
        "source_roster_sha256": hash_json(list(source_uids)),
        "train_question_count": train_questions,
        "validation_question_count": validation_questions,
        "unused_question_count": len(unused_indices),
        "train_samples_per_question": train_samples_per_question,
        "validation_samples_per_question": 1,
        "validation_sample_index": validation_sample_index,
        "train_source_uids": [source_uids[index] for index in train_indices],
        "validation_source_uids": validation_uids,
        "unused_source_uids_sha256": hash_json([source_uids[index] for index in unused_indices]),
        **validation_extra,
    }
    manifest["selection_sha256"] = hash_json(manifest)
    return manifest


@dataclass
class _SplitWriter:
    split: str
    output_root: Path
    rows_per_shard: int
    row_group_rows: int
    mirror: TraceS3Mirror | None
    buffered: list[pa.Table] = field(default_factory=list)
    buffered_rows: int = 0
    shard_entries: list[dict[str, Any]] = field(default_factory=list)
    row_count: int = 0
    response_token_count: int = 0
    prompt_token_count: int = 0
    empty_response_count: int = 0

    def append(self, table: pa.Table) -> None:
        if table.num_rows == 0:
            return
        self.buffered.append(table)
        self.buffered_rows += table.num_rows
        while self.buffered_rows >= self.rows_per_shard:
            combined = pa.concat_tables(self.buffered)
            self._write(combined.slice(0, self.rows_per_shard))
            remainder = combined.slice(self.rows_per_shard)
            self.buffered = [remainder] if remainder.num_rows else []
            self.buffered_rows = remainder.num_rows

    def finish(self) -> None:
        if self.buffered_rows:
            self._write(pa.concat_tables(self.buffered))
        self.buffered = []
        self.buffered_rows = 0

    def _write(self, table: pa.Table) -> None:
        shard_id = len(self.shard_entries)
        directory = self.output_root / self.split
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / f"traces-{self.split}-{shard_id:06d}.parquet"
        temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
        pq.write_table(
            table,
            temporary,
            compression="zstd",
            compression_level=3,
            row_group_size=self.row_group_rows,
            write_statistics=True,
            data_page_version="2.0",
        )
        os.replace(temporary, destination)
        parquet = pq.ParquetFile(destination)
        response_tokens = sum(int(value) for value in table.column("response_length").to_pylist())
        prompt_tokens = sum(int(value) for value in table.column("prompt_length").to_pylist())
        rows = table.num_rows
        entry = {
            "shard_id": shard_id,
            "path": _relative(destination, self.output_root),
            "sha256": sha256_file(destination),
            "size_bytes": destination.stat().st_size,
            "rows": rows,
            "row_groups": parquet.metadata.num_row_groups,
            "stats": {
                "row_count": rows,
                "response_token_count": response_tokens,
                "prompt_token_count": prompt_tokens,
                "empty_response_count": 0,
            },
        }
        self.shard_entries.append(entry)
        self.row_count += rows
        self.response_token_count += response_tokens
        self.prompt_token_count += prompt_tokens
        if self.mirror is not None:
            self.mirror.upload_file(destination, root=self.output_root, sha256=entry["sha256"])

    def index(self) -> dict[str, Any]:
        return {
            "question_count": 0,
            "row_count": self.row_count,
            "complete": True,
            "parquet_files": [entry["path"] for entry in self.shard_entries],
            "shards": self.shard_entries,
            "stats": {
                "row_count": self.row_count,
                "response_token_count": self.response_token_count,
                "prompt_token_count": self.prompt_token_count,
                "empty_response_count": self.empty_response_count,
            },
        }


def _filter_rows(table: pa.Table, uid_values: pa.Array, sample_indices: set[int]) -> pa.Table:
    uid_mask = pc.is_in(table.column("source_uid"), value_set=uid_values)
    sample_mask = pc.is_in(
        table.column("sample_index"),
        value_set=pa.array(sorted(sample_indices), type=table.schema.field("sample_index").type),
    )
    return table.filter(pc.and_(uid_mask, sample_mask))


def build_training_view(
    *,
    source_root: Path,
    output_root: Path,
    source_s3_uri: str,
    output_s3_uri: str | None,
    seed: int,
    train_questions: int,
    validation_questions: int,
    train_samples_per_question: int,
    validation_sample_index: int,
    expected_source_questions: int,
    expected_source_samples_per_question: int,
    rows_per_shard: int,
    row_group_rows: int,
    validation_source: str = "train",
) -> dict[str, Any]:
    if validation_source not in VALIDATION_SOURCES:
        raise TrainingViewError(f"validation_source must be one of {VALIDATION_SOURCES}, got {validation_source!r}")
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    index_path = source_root / "dataset_index.json"
    complete_path = source_root / "COMPLETE.json"
    if not index_path.is_file() or not complete_path.is_file():
        raise TrainingViewError("source root must contain dataset_index.json and COMPLETE.json")
    source_index = _load_json(index_path, "source dataset index")
    source_index_sha256 = _verify_self_hash(source_index, "dataset_index_sha256", "source dataset index")
    source_complete = _load_json(complete_path, "source completion receipt")
    if source_complete.get("dataset_index_sha256") != source_index_sha256:
        raise TrainingViewError("source completion receipt does not bind the dataset index")
    splits = source_index.get("splits")
    required_splits = {"train", "validation"} if validation_source == "validation" else {"train"}
    if (
        not isinstance(splits, Mapping)
        or not required_splits.issubset(splits)
        or not set(splits).issubset({"train", "validation"})
    ):
        raise TrainingViewError(f"source dataset must contain the {sorted(required_splits)} split(s)")
    source_split = splits["train"]
    source_questions = int(source_split.get("question_count", -1))
    source_rows = int(source_split.get("row_count", -1))
    source_samples = source_rows // source_questions if source_questions > 0 else -1
    if (
        source_questions != expected_source_questions
        or source_samples != expected_source_samples_per_question
        or source_rows != expected_source_questions * expected_source_samples_per_question
    ):
        raise TrainingViewError(
            "source dataset has the wrong question/sample product: "
            f"got {source_questions} x {source_samples}, expected "
            f"{expected_source_questions} x {expected_source_samples_per_question}"
        )
    run_config_path = source_root / str(source_split["run_config_path"])
    run_config = _load_json(run_config_path, "source run config")
    semantic = run_config.get("semantic_config")
    if not isinstance(semantic, dict) or hash_json(semantic) != run_config.get("generation_config_sha256"):
        raise TrainingViewError("source run configuration is malformed or has a bad semantic hash")
    if semantic.get("schema_version") != SCHEMA_VERSION or semantic.get("topk_width") != TOPK_WIDTH:
        raise TrainingViewError("source run configuration has the wrong trace schema or top-k width")

    source_shards, source_uids = _validated_source_shards(
        source_root,
        source_split,
        expected_questions=source_questions,
        expected_samples_per_question=expected_source_samples_per_question,
    )
    validation_shards: list[Path] = source_shards
    validation_roster: list[str] | None = None
    validation_run_config: dict[str, Any] | None = None
    validation_run_config_path: Path | None = None
    if validation_source == "validation":
        validation_split = splits["validation"]
        validation_questions_available = int(validation_split.get("question_count", -1))
        validation_rows = int(validation_split.get("row_count", -1))
        if validation_questions_available <= 0 or validation_rows != validation_questions_available:
            raise TrainingViewError(
                "source validation split must hold exactly one trace per question: "
                f"questions={validation_questions_available} rows={validation_rows}"
            )
        validation_run_config_path = source_root / str(validation_split["run_config_path"])
        validation_run_config = _load_json(validation_run_config_path, "source validation run config")
        validation_semantic = validation_run_config.get("semantic_config")
        if not isinstance(validation_semantic, dict) or hash_json(validation_semantic) != validation_run_config.get(
            "generation_config_sha256"
        ):
            raise TrainingViewError("source validation run configuration is malformed or has a bad semantic hash")
        if validation_semantic.get("split") != "validation":
            raise TrainingViewError("source validation run configuration does not describe the validation split")
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
            if validation_semantic.get(key) != semantic.get(key):
                raise TrainingViewError(f"source validation split was generated under a different {key}")
        validation_shards, validation_roster = _validated_source_shards(
            source_root,
            validation_split,
            expected_questions=validation_questions_available,
            expected_samples_per_question=1,
            roster_split="validation",
        )
        if validation_sample_index != 0:
            raise TrainingViewError("the validation split holds sample index 0 only")
    selection = build_selection(
        source_uids,
        seed=seed,
        train_questions=train_questions,
        validation_questions=validation_questions,
        train_samples_per_question=train_samples_per_question,
        validation_sample_index=validation_sample_index,
        validation_source_uids=validation_roster,
    )
    train_uids = set(selection["train_source_uids"])
    validation_uids = set(selection["validation_source_uids"])
    if train_uids.intersection(validation_uids):
        raise TrainingViewError("deterministic selection produced train/validation overlap")

    output_root.mkdir(parents=True, exist_ok=True)
    selection_path = output_root / "selection.json"
    atomic_write_json(selection_path, selection)
    copied_run_config_path = output_root / "source_run_config.json"
    atomic_write_json(copied_run_config_path, run_config)
    copied_validation_run_config_path: Path | None = None
    if validation_run_config is not None:
        copied_validation_run_config_path = output_root / "source_validation_run_config.json"
        atomic_write_json(copied_validation_run_config_path, validation_run_config)
    mirror = TraceS3Mirror(output_s3_uri) if output_s3_uri else None
    train_writer = _SplitWriter("train", output_root, rows_per_shard, row_group_rows, mirror)
    validation_writer = _SplitWriter("validation", output_root, rows_per_shard, row_group_rows, mirror)
    train_uid_values = pa.array(sorted(train_uids), type=pa.string())
    validation_uid_values = pa.array(sorted(validation_uids), type=pa.string())
    seen_train: dict[str, set[int]] = {}
    seen_validation: dict[str, set[int]] = {}

    def _record(filtered: pa.Table, seen: dict[str, set[int]]) -> None:
        columns = filtered.select(["source_uid", "sample_index"]).to_pydict()
        for uid, sample_index in zip(columns["source_uid"], columns["sample_index"], strict=True):
            samples = seen.setdefault(str(uid), set())
            if int(sample_index) in samples:
                raise TrainingViewError(f"duplicate selected trace for UID {uid} sample {sample_index}")
            samples.add(int(sample_index))

    same_shards = validation_source == "train"
    for source_path in source_shards:
        table = pq.read_table(source_path)
        train_table = _filter_rows(table, train_uid_values, set(range(train_samples_per_question)))
        _record(train_table, seen_train)
        train_writer.append(train_table)
        if same_shards:
            validation_table = _filter_rows(table, validation_uid_values, {validation_sample_index})
            _record(validation_table, seen_validation)
            validation_writer.append(validation_table)
    if not same_shards:
        for source_path in validation_shards:
            table = pq.read_table(source_path)
            validation_table = _filter_rows(table, validation_uid_values, {validation_sample_index})
            _record(validation_table, seen_validation)
            validation_writer.append(validation_table)

    train_writer.finish()
    validation_writer.finish()
    expected_train_samples = set(range(train_samples_per_question))
    if set(seen_train) != train_uids or any(samples != expected_train_samples for samples in seen_train.values()):
        raise TrainingViewError("the materialized train split does not contain every selected UID/sample")
    if set(seen_validation) != validation_uids or any(
        samples != {validation_sample_index} for samples in seen_validation.values()
    ):
        raise TrainingViewError("the materialized validation split does not contain exactly one selected trace per UID")

    train_index = train_writer.index()
    validation_index = validation_writer.index()
    train_index["question_count"] = train_questions
    validation_index["question_count"] = validation_questions
    view: dict[str, Any] = {
        "schema_version": VIEW_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "source_trace": {
            "s3_uri": source_s3_uri.rstrip("/"),
            "dataset_index_sha256": source_index_sha256,
            "generation_config_sha256": run_config["generation_config_sha256"],
            "run_config_sha256": sha256_file(copied_run_config_path),
            "trace_spec": source_complete.get("trace_spec"),
            "validation_split": validation_source,
            **(
                {
                    "validation_generation_config_sha256": validation_run_config["generation_config_sha256"],
                    "validation_run_config_sha256": sha256_file(copied_validation_run_config_path),
                }
                if validation_run_config is not None and copied_validation_run_config_path is not None
                else {}
            ),
        },
        "generation_semantic_config": semantic,
        "direction": semantic["direction"],
        "topk_width": TOPK_WIDTH,
        "teacher": semantic["teacher"],
        "tokenizer": semantic["tokenizer"],
        "chat_template": semantic["chat_template"],
        "sampling": semantic["sampling"],
        "selection_manifest_path": selection_path.name,
        "selection_manifest_sha256": sha256_file(selection_path),
        "source_run_config_path": copied_run_config_path.name,
        **(
            {"source_validation_run_config_path": copied_validation_run_config_path.name}
            if copied_validation_run_config_path is not None
            else {}
        ),
        "samples_per_question": {"train": train_samples_per_question, "validation": 1},
        "total_rows": train_writer.row_count + validation_writer.row_count,
        "total_response_tokens": train_writer.response_token_count + validation_writer.response_token_count,
        "splits": {"train": train_index, "validation": validation_index},
    }
    view["dataset_index_sha256"] = hash_json(view)
    view_path = output_root / "dataset_index.json"
    atomic_write_json(view_path, view)
    completion = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "s3_uri": output_s3_uri.rstrip("/") if output_s3_uri else None,
        "dataset_index_sha256": view["dataset_index_sha256"],
        "source_dataset_index_sha256": source_index_sha256,
        "train_questions": train_questions,
        "validation_questions": validation_questions,
        "total_rows": view["total_rows"],
        "total_response_tokens": view["total_response_tokens"],
    }
    complete_output = output_root / "COMPLETE.json"
    atomic_write_json(complete_output, completion)

    if mirror is not None:
        for path in (selection_path, copied_run_config_path, view_path):
            mirror.upload_file(path, root=output_root)
        mirror.upload_file(complete_output, root=output_root)
    print(
        "GEMMA4_TRAINING_VIEW_COMPLETE "
        f"train_rows={train_writer.row_count} validation_rows={validation_writer.row_count} "
        f"index={view['dataset_index_sha256']}",
        flush=True,
    )
    return view


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-s3-uri", required=True)
    parser.add_argument("--output-s3-uri")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-questions", type=int, default=DEFAULT_TRAIN_QUESTIONS)
    parser.add_argument("--validation-questions", type=int, default=DEFAULT_VALIDATION_QUESTIONS)
    parser.add_argument("--train-samples-per-question", type=int, default=DEFAULT_TRAIN_SAMPLES)
    parser.add_argument("--validation-sample-index", type=int, default=DEFAULT_VALIDATION_SAMPLE_INDEX)
    parser.add_argument(
        "--validation-source",
        choices=VALIDATION_SOURCES,
        default="train",
        help="train: carve validation questions out of the train roster (default); "
        "validation: use the bundle's own validation split (1 trace per question) and train on every train question.",
    )
    parser.add_argument("--expected-source-questions", type=int, default=10_000)
    parser.add_argument("--expected-source-samples-per-question", type=int, default=8)
    parser.add_argument("--rows-per-shard", type=int, default=64)
    parser.add_argument("--row-group-rows", type=int, default=2)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        build_training_view(
            source_root=args.source_root,
            output_root=args.output_root,
            source_s3_uri=args.source_s3_uri,
            output_s3_uri=args.output_s3_uri,
            seed=args.seed,
            train_questions=args.train_questions,
            validation_questions=args.validation_questions,
            train_samples_per_question=args.train_samples_per_question,
            validation_sample_index=args.validation_sample_index,
            expected_source_questions=args.expected_source_questions,
            expected_source_samples_per_question=args.expected_source_samples_per_question,
            rows_per_shard=args.rows_per_shard,
            row_group_rows=args.row_group_rows,
            validation_source=args.validation_source,
        )
    except (OSError, TrainingViewError, ValueError, pa.ArrowException) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
