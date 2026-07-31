#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Fail closed unless a real Gemma 4 FSDP2 train/backward audit matches production."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from gemma4_model_identity import inspect_local_hf_model

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
EXPECTED_REPORT_VERSION = 2
EXPECTED_SCHEMA = "gemma4-hf-bf16-sdpa-topk-overlay-v1"
REQUIRED_SOURCE_PATHS = {
    "rl-distill-scripts/data/audit_gemma4_fsdp2_training_topk.py",
    "rl-distill-scripts/data/audit_gemma4_cross_engine_topk.py",
    "rl-distill-scripts/data/gemma4_model_identity.py",
    "rl-distill-scripts/full_vocab_kl_loss.py",
    "verl/utils/dataset/dataset_utils.py",
    "verl/utils/fsdp_utils.py",
    "verl/workers/engine/utils.py",
    "verl/workers/engine/fsdp/transformer_impl.py",
    "verl/workers/engine_workers.py",
}


class TrainingAuditReceiptError(ValueError):
    """Raised when an audit receipt cannot authorize production training."""


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TrainingAuditReceiptError(f"cannot read {description} {path}: {error}") from error
    if not isinstance(value, dict):
        raise TrainingAuditReceiptError(f"{description} {path} must contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TrainingAuditReceiptError(f"{field} must be an object")
    return value


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TrainingAuditReceiptError(f"{field} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise TrainingAuditReceiptError(f"{field} must be finite")
    return result


def _expect_equal(actual: Any, expected: Any, field: str) -> None:
    if actual != expected:
        raise TrainingAuditReceiptError(f"{field} is {actual!r}; expected {expected!r}")


def _git(repo_root: Path, *args: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except subprocess.CalledProcessError as error:
        raise TrainingAuditReceiptError(f"git {' '.join(args)} failed: {error.stderr.strip()}") from error


def verify_receipt(
    *,
    receipt_path: Path,
    dataset_index_path: Path,
    student_model_path: Path,
    expected_world_size: int,
    repo_root: Path = REPO_ROOT,
    verify_repository: bool = True,
) -> dict[str, Any]:
    receipt_path = receipt_path.expanduser().resolve(strict=True)
    dataset_index_path = dataset_index_path.expanduser().resolve(strict=True)
    student_model_path = student_model_path.expanduser().resolve(strict=True)
    report = _load_json(receipt_path, "training-engine audit receipt")
    index = _load_json(dataset_index_path, "overlay dataset index")

    _expect_equal(index.get("schema_version"), EXPECTED_SCHEMA, "dataset schema_version")
    _expect_equal(report.get("report_version"), EXPECTED_REPORT_VERSION, "audit report_version")
    _expect_equal(report.get("status"), "pass", "audit status")
    gate = _mapping(report.get("gate"), "audit gate")
    _expect_equal(gate.get("status"), "pass", "audit gate status")
    checks = _mapping(gate.get("checks"), "audit gate checks")
    if not checks or any(
        _mapping(check, f"audit gate check {name}").get("passed") is not True for name, check in checks.items()
    ):
        raise TrainingAuditReceiptError("every registered audit gate check must pass")

    contract = _mapping(report.get("contract"), "audit contract")
    expected_contract = {
        "execution_mode": "train",
        "gradient_checkpointing": True,
        "forward_path": "compact_hidden",
        "fsdp_wrap": True,
        "backward_exercised": True,
        "model_dtype": "fp32",
        "fsdp_param_dtype": "bf16",
        "fsdp_reduce_dtype": "fp32",
        "fsdp_buffer_dtype": "fp32",
        "fsdp_cast_forward_inputs": True,
        "train_seed": 42,
        "train_batch_size": 128,
        "micro_batch_size_per_gpu": 2,
        "max_padded_tokens_per_microbatch": 5120,
        "kl_chunk_size": 4096,
        "max_length": 12288,
        "top_k": 128,
    }
    for field, expected in expected_contract.items():
        _expect_equal(contract.get(field), expected, f"audit contract {field}")

    dataset = _mapping(report.get("dataset"), "audit dataset")
    _expect_equal(Path(str(dataset.get("index_path"))).resolve(), dataset_index_path, "audit dataset index_path")
    _expect_equal(dataset.get("dataset_index_sha256"), index.get("dataset_index_sha256"), "audit dataset hash")
    _expect_equal(
        dataset.get("source_dataset_index_sha256"),
        index.get("source_dataset_index_sha256"),
        "audit source dataset hash",
    )
    target_identity = _mapping(index.get("target_model_identity"), "overlay target_model_identity").get(
        "model_identity_sha256"
    )
    _expect_equal(dataset.get("target_model_identity_sha256"), target_identity, "audit target model identity")
    _expect_equal(dataset.get("split"), "validation", "audit dataset split")

    model = _mapping(report.get("model"), "audit model")
    _expect_equal(model.get("model_identity_sha256"), target_identity, "audit model identity")
    _expect_equal(model.get("dtype_load"), "fp32", "audit model dtype_load")
    _expect_equal(model.get("autocast"), "bfloat16", "audit model autocast")
    _expect_equal(model.get("attention_implementation"), "sdpa", "audit model attention implementation")
    student_model = _mapping(report.get("student_model"), "audit student_model")
    _expect_equal(Path(str(student_model.get("path"))).resolve(), student_model_path, "audit student model path")
    current_student_identity = inspect_local_hf_model(student_model_path).model_identity_sha256
    _expect_equal(
        student_model.get("model_identity_sha256"),
        current_student_identity,
        "audit student model identity",
    )
    _expect_equal(student_model.get("dtype_load"), "fp32", "audit student model dtype_load")
    _expect_equal(student_model.get("autocast"), "bfloat16", "audit student model autocast")
    _expect_equal(student_model.get("attention_implementation"), "sdpa", "audit student attention implementation")

    selection = _mapping(report.get("selection"), "audit selection")
    _expect_equal(selection.get("world_size"), expected_world_size, "audit world_size")
    if int(selection.get("trace_count", 0)) < 16 or int(selection.get("position_count", 0)) < 511:
        raise TrainingAuditReceiptError("audit must cover at least 16 traces and 511 response positions")
    production_batches = selection.get("production_first_train_batch")
    if not isinstance(production_batches, list) or len(production_batches) != expected_world_size:
        raise TrainingAuditReceiptError("audit must record one production first-batch layout per rank")
    global_indices: list[int] = []
    microbatch_counts: list[int] = []
    train_split = _mapping(_mapping(index.get("splits"), "overlay splits").get("train"), "overlay train split")
    train_row_count = int(train_split.get("row_count", 0))
    if train_row_count <= 0:
        raise TrainingAuditReceiptError("overlay train row_count must be positive")
    from torch.utils.data import DistributedSampler

    for expected_rank, batch in enumerate(production_batches):
        batch = _mapping(batch, f"production batch rank {expected_rank}")
        _expect_equal(batch.get("rank"), expected_rank, f"production batch rank {expected_rank} identity")
        rank_indices = batch.get("global_indices")
        microbatches = batch.get("microbatches")
        if not isinstance(rank_indices, list) or len(rank_indices) != 16:
            raise TrainingAuditReceiptError(f"production batch rank {expected_rank} must contain 16 rows")
        if not isinstance(microbatches, list) or not microbatches:
            raise TrainingAuditReceiptError(f"production batch rank {expected_rank} has no microbatch layout")
        sampler = DistributedSampler(
            range(train_row_count),
            num_replicas=expected_world_size,
            rank=expected_rank,
            shuffle=True,
            seed=42,
            drop_last=True,
        )
        sampler.set_epoch(0)
        _expect_equal(rank_indices, list(sampler)[:16], f"production batch rank {expected_rank} sampler indices")
        global_indices.extend(int(index) for index in rank_indices)
        microbatch_counts.append(int(batch.get("microbatch_count", -1)))
        rank_sequence_lengths = batch.get("sequence_lengths")
        if not isinstance(rank_sequence_lengths, list) or len(rank_sequence_lengths) != 16:
            raise TrainingAuditReceiptError(f"production batch rank {expected_rank} must record 16 sequence lengths")
        covered_local_indices: list[int] = []
        for microbatch_index, microbatch in enumerate(microbatches):
            microbatch = _mapping(microbatch, f"production rank {expected_rank} microbatch {microbatch_index}")
            indices = microbatch.get("indices")
            lengths = microbatch.get("sequence_lengths")
            if not isinstance(indices, list) or not isinstance(lengths, list) or len(indices) != len(lengths):
                raise TrainingAuditReceiptError("production microbatch indices and lengths must be matching lists")
            if not 1 <= len(indices) <= 2:
                raise TrainingAuditReceiptError("production microbatch size must be one or two")
            if any(int(length) <= 0 for length in lengths):
                raise TrainingAuditReceiptError("production sequence lengths must be positive")
            expected_lengths = [int(rank_sequence_lengths[int(index)]) for index in indices]
            _expect_equal([int(length) for length in lengths], expected_lengths, "production microbatch lengths")
            padded_tokens = max(expected_lengths) * len(expected_lengths)
            _expect_equal(microbatch.get("padded_tokens"), padded_tokens, "production microbatch padded_tokens")
            if len(indices) > 1 and padded_tokens > 5120:
                raise TrainingAuditReceiptError("production multi-sequence microbatch exceeds padded-token ceiling")
            covered_local_indices.extend(int(index) for index in indices)
        if sorted(covered_local_indices) != list(range(16)):
            raise TrainingAuditReceiptError(f"production batch rank {expected_rank} does not cover each local row once")
    if len(set(microbatch_counts)) != 1:
        raise TrainingAuditReceiptError("production microbatch counts differ across ranks")
    if len(global_indices) != 128 or len(set(global_indices)) != 128:
        raise TrainingAuditReceiptError("production first train batch must contain 128 distinct dataset rows")

    aggregate = _mapping(report.get("aggregate"), "audit aggregate")
    exact = _mapping(report.get("exact_serialization"), "audit exact_serialization")
    independent_thresholds = (
        ("top-k overlap", _mapping(aggregate.get("topk_overlap_fraction"), "topk overlap").get("mean"), 0.995, ">="),
        (
            "weighted logprob drift",
            _mapping(aggregate.get("stored_support_weighted_abs_logprob_delta"), "weighted drift").get("mean"),
            0.003,
            "<=",
        ),
        (
            "sampled-token drift p95",
            _mapping(aggregate.get("sampled_token_abs_logprob_delta"), "sampled-token drift").get("p95"),
            0.01,
            "<=",
        ),
        (
            "support probability L1",
            _mapping(aggregate.get("stored_support_probability_l1"), "support probability L1").get("mean"),
            0.003,
            "<=",
        ),
        (
            "ordered top-k exact fraction",
            _mapping(exact.get("ordered_topk_exact"), "ordered top-k exact fraction").get("mean"),
            1.0,
            "==",
        ),
    )
    for name, raw_observed, threshold, operator in independent_thresholds:
        observed = _finite_number(raw_observed, name)
        passed = (
            observed >= threshold
            if operator == ">="
            else observed <= threshold
            if operator == "<="
            else observed == threshold
        )
        if not passed:
            raise TrainingAuditReceiptError(f"{name} is {observed}; required {operator} {threshold}")

    backward = _mapping(report.get("backward"), "audit backward")
    grad_norm = _finite_number(backward.get("total_norm"), "audit backward total_norm")
    if grad_norm > 50.0:
        raise TrainingAuditReceiptError(f"audit backward total_norm is {grad_norm}; required <= 50.0")

    implementation = _mapping(report.get("implementation"), "audit implementation")
    _expect_equal(implementation.get("dirty"), False, "audit implementation dirty")
    source_sha256s = _mapping(implementation.get("source_sha256s"), "audit source_sha256s")
    _expect_equal(set(source_sha256s), REQUIRED_SOURCE_PATHS, "audit source paths")
    for relative_path in sorted(REQUIRED_SOURCE_PATHS):
        _expect_equal(
            source_sha256s.get(relative_path), _sha256(repo_root / relative_path), f"audit source hash {relative_path}"
        )

    if verify_repository:
        current_commit = _git(repo_root, "rev-parse", "HEAD")
        _expect_equal(implementation.get("commit"), current_commit, "audit implementation commit")
        dirty = _git(repo_root, "status", "--porcelain", "--untracked-files=all")
        if dirty:
            raise TrainingAuditReceiptError("production repository has uncommitted or untracked changes")

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--dataset-index", type=Path, required=True)
    parser.add_argument("--student-model", type=Path, required=True)
    parser.add_argument("--expected-world-size", type=int, required=True)
    args = parser.parse_args()
    if args.expected_world_size <= 0:
        parser.error("--expected-world-size must be positive")
    return args


def main() -> int:
    args = parse_args()
    try:
        report = verify_receipt(
            receipt_path=args.receipt,
            dataset_index_path=args.dataset_index,
            student_model_path=args.student_model,
            expected_world_size=args.expected_world_size,
        )
    except (OSError, TrainingAuditReceiptError, ValueError) as error:
        print(f"training-engine audit rejected: {error}", file=sys.stderr)
        return 2
    print(f"TRAINING_ENGINE_AUDIT_RECEIPT_SHA256={_sha256(args.receipt.resolve())}")
    print(f"TRAINING_ENGINE_AUDIT_COMMIT={report['implementation']['commit']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
