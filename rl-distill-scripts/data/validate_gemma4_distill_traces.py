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

"""Validate Gemma 4 trace shards and emit a trainer-consumable index."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gemma4_distill_trace_schema import (
    FP16_TOPK_MASS_TOLERANCE,
    MANIFEST_VERSION,
    SCHEMA_VERSION,
    TOPK_WIDTH,
    TraceStatistics,
    TraceValidationError,
    atomic_write_json,
    hash_json,
    normalized_decode,
    parquet_manifest_path,
    sha256_file,
    tokenizer_fingerprint,
    validate_shard_bundle,
)

EXPECTED_QUESTIONS = {"train": 9723, "validation": 200}
EXPECTED_SAMPLES_PER_QUESTION = 5


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def load_run_config(split_dir: Path) -> dict[str, Any]:
    path = split_dir / "run_config.json"
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TraceValidationError(f"cannot read {path}: {error}") from error
    if config.get("manifest_version") != MANIFEST_VERSION or config.get("schema_version") != SCHEMA_VERSION:
        raise TraceValidationError(f"unsupported run configuration in {path}")
    semantic = config.get("semantic_config")
    if not isinstance(semantic, dict):
        raise TraceValidationError(f"semantic_config is missing from {path}")
    if hash_json(semantic) != config.get("generation_config_sha256"):
        raise TraceValidationError(f"semantic configuration hash mismatch in {path}")
    if semantic.get("schema_version") != SCHEMA_VERSION or semantic.get("topk_width") != TOPK_WIDTH:
        raise TraceValidationError(f"schema/top-k contract mismatch in {path}")
    return config


def _load_tokenizer(
    run_configs: Sequence[Mapping[str, Any]],
    *,
    model_override: str | None,
    revision_override: str | None,
    local_files_only: bool,
) -> Any:
    tokenizer_configs = [config["semantic_config"]["tokenizer"] for config in run_configs]
    fingerprints = {
        (config["model"], config.get("revision"), config["sha256"], int(config["vocab_size"]))
        for config in tokenizer_configs
    }
    if len(fingerprints) != 1:
        raise TraceValidationError("split directories do not use one identical tokenizer")
    expected = tokenizer_configs[0]
    model = model_override or expected["model"]
    revision = revision_override if revision_override is not None else expected.get("revision")
    from transformers import AutoTokenizer

    kwargs: dict[str, Any] = {"trust_remote_code": True, "local_files_only": local_files_only}
    if revision and not Path(model).exists():
        kwargs["revision"] = revision
    tokenizer = AutoTokenizer.from_pretrained(model, **kwargs)
    actual_sha256, actual_vocab_size = tokenizer_fingerprint(tokenizer)
    if actual_sha256 != expected["sha256"] or actual_vocab_size != int(expected["vocab_size"]):
        raise TraceValidationError(
            "loaded tokenizer does not match trace provenance: "
            f"sha256 {actual_sha256} / vocab {actual_vocab_size}, expected "
            f"{expected['sha256']} / {expected['vocab_size']}"
        )
    return tokenizer


def _common_experiment_config(semantic: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": semantic["schema_version"],
        "direction": semantic["direction"],
        "samples_per_question": semantic["samples_per_question"],
        "topk_width": semantic["topk_width"],
        "global_seed": semantic["global_seed"],
        "teacher": semantic["teacher"],
        "tokenizer": semantic["tokenizer"],
        "chat_template": semantic["chat_template"],
        "sampling": semantic["sampling"],
        "engine": semantic["engine"],
        "generator": semantic["generator"],
        "environment_versions": semantic["environment_versions"],
    }


def _relative_path(path: Path, parent: Path) -> str:
    """Return a location-independent path relative to the dataset index."""

    return os.path.relpath(path.resolve(), parent.resolve())


def validate_dataset(
    split_dirs: Mapping[str, str | Path],
    *,
    output_index: str | Path,
    decoder: Callable[[Sequence[int]], str] | None,
    expected_questions: Mapping[str, int | None],
    expected_samples_per_question: int,
    allow_incomplete: bool = False,
    allow_empty_responses: bool = False,
    fail_on_question_overlap: bool = False,
) -> dict[str, Any]:
    if not split_dirs:
        raise TraceValidationError("at least one split directory is required")
    unknown_splits = sorted(set(split_dirs).difference({"train", "validation"}))
    if unknown_splits:
        raise TraceValidationError(f"unsupported split names: {unknown_splits}")

    normalized_dirs = {split: Path(path) for split, path in split_dirs.items()}
    run_configs = {split: load_run_config(path) for split, path in normalized_dirs.items()}
    common_configs = {
        _split: _common_experiment_config(config["semantic_config"]) for _split, config in run_configs.items()
    }
    common_hashes = {hash_json(config) for config in common_configs.values()}
    if len(common_hashes) != 1:
        raise TraceValidationError("split directories do not share one teacher/tokenizer/sampling experiment contract")
    experiment_sha256 = next(iter(common_hashes))

    all_trace_ids: set[str] = set()
    split_uid_questions: dict[str, dict[str, str]] = {}
    split_question_hashes: dict[str, set[str]] = {}
    index_parent = Path(output_index).parent
    split_indexes: dict[str, Any] = {}
    total_rows = 0
    total_response_tokens = 0

    for split, split_dir in sorted(normalized_dirs.items()):
        run_config = run_configs[split]
        semantic = run_config["semantic_config"]
        if semantic["split"] != split:
            raise TraceValidationError(f"{split_dir} declares split {semantic['split']!r}, expected {split!r}")
        total_shards = int(semantic["total_shards"])
        expected_ids = set(range(total_shards))
        discovered: dict[int, Path] = {}
        for parquet_path in sorted(split_dir.glob(f"traces-{split}-*.parquet")):
            try:
                shard_id = int(parquet_path.stem.rsplit("-", 1)[1])
            except (IndexError, ValueError) as error:
                raise TraceValidationError(f"cannot parse shard ID from {parquet_path.name}") from error
            if shard_id in discovered:
                raise TraceValidationError(f"duplicate shard ID {shard_id} in {split_dir}")
            discovered[shard_id] = parquet_path
        missing = sorted(expected_ids.difference(discovered))
        extra = sorted(set(discovered).difference(expected_ids))
        if extra:
            raise TraceValidationError(f"{split_dir} contains unexpected shard IDs: {extra}")
        if missing and not allow_incomplete:
            raise TraceValidationError(f"{split_dir} is missing shard IDs: {missing[:20]}")

        max_response_tokens = int(semantic["sampling"]["max_response_tokens"])
        aggregate_stats = TraceStatistics(max_response_tokens=max_response_tokens)
        source_samples: defaultdict[str, set[int]] = defaultdict(set)
        uid_to_question: dict[str, str] = {}
        question_hashes: set[str] = set()
        shard_entries: list[dict[str, Any]] = []
        for shard_id, parquet_path in sorted(discovered.items()):
            manifest, validation = validate_shard_bundle(
                parquet_path,
                run_config=run_config,
                decoder=decoder,
                external_stats=aggregate_stats,
            )
            if int(manifest["shard_id"]) != shard_id:
                raise TraceValidationError(f"{parquet_path} manifest declares the wrong shard ID")
            duplicates = all_trace_ids.intersection(validation.trace_ids)
            if duplicates:
                raise TraceValidationError(f"duplicate trace IDs across shards: {sorted(duplicates)[:3]}")
            all_trace_ids.update(validation.trace_ids)
            for source_uid, question_hash, sample_index in validation.source_samples:
                prior_question = uid_to_question.setdefault(source_uid, question_hash)
                if prior_question != question_hash:
                    raise TraceValidationError(f"source UID {source_uid!r} maps to multiple questions across shards")
                if sample_index in source_samples[source_uid]:
                    raise TraceValidationError(f"duplicate sample index {sample_index} for source UID {source_uid!r}")
                source_samples[source_uid].add(sample_index)
                question_hashes.add(question_hash)
            shard_entries.append(
                {
                    "shard_id": shard_id,
                    "path": _relative_path(parquet_path, index_parent),
                    "manifest_path": _relative_path(parquet_manifest_path(parquet_path), index_parent),
                    "sha256": manifest["parquet_sha256"],
                    "size_bytes": manifest["parquet_size_bytes"],
                    "rows": manifest["row_count"],
                    "row_groups": manifest["parquet_row_groups"],
                    "stats": manifest["stats"],
                }
            )

        expected_sample_indices = set(range(expected_samples_per_question))
        bad_samples = {
            uid: sorted(indices) for uid, indices in source_samples.items() if indices != expected_sample_indices
        }
        if bad_samples and not allow_incomplete:
            first_items = list(bad_samples.items())[:3]
            raise TraceValidationError(
                f"{split} questions do not have exact sample indices 0..{expected_samples_per_question - 1}: "
                f"{first_items}"
            )
        run_samples = int(semantic["samples_per_question"])
        if run_samples != expected_samples_per_question:
            raise TraceValidationError(
                f"{split} run declares {run_samples} samples/question, expected {expected_samples_per_question}"
            )
        question_count = len(source_samples)
        declared_questions = int(semantic["unique_question_count"])
        if not allow_incomplete and question_count != declared_questions:
            raise TraceValidationError(
                f"{split} has {question_count} questions across shards, run config declares {declared_questions}"
            )
        required_question_count = expected_questions.get(split)
        if not allow_incomplete and required_question_count is not None and question_count != required_question_count:
            raise TraceValidationError(f"{split} has {question_count} questions, expected {required_question_count}")
        stats = aggregate_stats.to_dict()
        if stats["empty_response_count"] and not allow_empty_responses:
            raise TraceValidationError(
                f"{split} contains {stats['empty_response_count']} empty responses; the current distillation "
                "dataset rejects zero-response-token rows. Regenerate/filter them, or use "
                "--allow-empty-responses only for a non-training archival index."
            )
        expected_rows = question_count * expected_samples_per_question
        if not allow_incomplete and stats["row_count"] != expected_rows:
            raise TraceValidationError(f"{split} has {stats['row_count']} rows, expected {expected_rows}")
        split_uid_questions[split] = uid_to_question
        split_question_hashes[split] = question_hashes
        total_rows += stats["row_count"]
        total_response_tokens += stats["response_token_count"]
        split_indexes[split] = {
            "source_dataset": semantic["source_dataset"],
            "source_dataset_sha256": semantic["source_dataset_sha256"],
            "generation_config_sha256": run_config["generation_config_sha256"],
            "run_config_path": _relative_path(split_dir / "run_config.json", index_parent),
            "run_config_sha256": sha256_file(split_dir / "run_config.json"),
            "question_count": question_count,
            "row_count": stats["row_count"],
            "complete": not missing,
            "missing_shard_ids": missing,
            "stats": stats,
            "parquet_files": [entry["path"] for entry in shard_entries],
            "shards": shard_entries,
        }

    question_overlap: set[str] = set()
    if "train" in split_uid_questions and "validation" in split_uid_questions:
        uid_overlap = set(split_uid_questions["train"]).intersection(split_uid_questions["validation"])
        if uid_overlap:
            raise TraceValidationError(f"train/validation source UIDs overlap: {sorted(uid_overlap)[:5]}")
        question_overlap = split_question_hashes["train"].intersection(split_question_hashes["validation"])
        if question_overlap and fail_on_question_overlap:
            raise TraceValidationError(
                f"train/validation question text overlaps ({len(question_overlap)} hashes); first: "
                f"{sorted(question_overlap)[:3]}"
            )

    first_semantic = next(iter(run_configs.values()))["semantic_config"]
    index: dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "experiment_sha256": experiment_sha256,
        "direction": first_semantic["direction"],
        "topk_width": TOPK_WIDTH,
        "recommended_training_topk_validation_tolerance": FP16_TOPK_MASS_TOLERANCE,
        "samples_per_question": expected_samples_per_question,
        "decode_check_performed": decoder is not None,
        "teacher": first_semantic["teacher"],
        "tokenizer": first_semantic["tokenizer"],
        "chat_template": first_semantic["chat_template"],
        "sampling": first_semantic["sampling"],
        "total_rows": total_rows,
        "total_response_tokens": total_response_tokens,
        "cross_split_question_text_overlap_count": len(question_overlap),
        "cross_split_question_text_overlap_sha256s": sorted(question_overlap),
        "splits": split_indexes,
    }
    index["dataset_index_sha256"] = hash_json(index)
    atomic_write_json(output_index, index)
    return index


def _parse_split_dir(value: str) -> tuple[str, str]:
    split, separator, path = value.partition("=")
    if not separator or split not in {"train", "validation"} or not path:
        raise argparse.ArgumentTypeError("--split-dir must be train=PATH or validation=PATH")
    return split, path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", action="append", type=_parse_split_dir, required=True)
    parser.add_argument("--output-index", required=True)
    parser.add_argument("--tokenizer-model", default=None)
    parser.add_argument("--tokenizer-revision", default=None)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skip-decode-check", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--allow-empty-responses", action="store_true")
    parser.add_argument("--fail-on-question-overlap", action="store_true")
    parser.add_argument("--expected-train-questions", type=int, default=EXPECTED_QUESTIONS["train"])
    parser.add_argument("--expected-validation-questions", type=int, default=EXPECTED_QUESTIONS["validation"])
    parser.add_argument("--expected-samples-per-question", type=int, default=EXPECTED_SAMPLES_PER_QUESTION)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    split_dirs = dict(args.split_dir)
    if len(split_dirs) != len(args.split_dir):
        print("ERROR: each split may be specified only once", file=sys.stderr)
        return 2
    try:
        run_configs = [load_run_config(Path(path)) for path in split_dirs.values()]
        tokenizer = None
        if not args.skip_decode_check:
            tokenizer = _load_tokenizer(
                run_configs,
                model_override=args.tokenizer_model,
                revision_override=args.tokenizer_revision,
                local_files_only=args.local_files_only,
            )
        decoder = None if tokenizer is None else lambda ids: normalized_decode(tokenizer, ids)
        index = validate_dataset(
            split_dirs,
            output_index=args.output_index,
            decoder=decoder,
            expected_questions={
                "train": args.expected_train_questions,
                "validation": args.expected_validation_questions,
            },
            expected_samples_per_question=args.expected_samples_per_question,
            allow_incomplete=args.allow_incomplete,
            allow_empty_responses=args.allow_empty_responses,
            fail_on_question_overlap=args.fail_on_question_overlap,
        )
    except (OSError, RuntimeError, ValueError, TraceValidationError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(
        f"VALID: {index['total_rows']} rows, {index['total_response_tokens']} response tokens -> {args.output_index}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
