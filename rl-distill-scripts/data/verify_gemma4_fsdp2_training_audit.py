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

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
EXPECTED_REPORT_VERSION = 1
EXPECTED_SCHEMA = "gemma4-hf-bf16-sdpa-topk-overlay-v1"
REQUIRED_SOURCE_PATHS = {
    "rl-distill-scripts/data/audit_gemma4_fsdp2_training_topk.py",
    "rl-distill-scripts/data/audit_gemma4_cross_engine_topk.py",
    "rl-distill-scripts/full_vocab_kl_loss.py",
    "verl/utils/fsdp_utils.py",
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
    expected_world_size: int,
    repo_root: Path = REPO_ROOT,
    verify_repository: bool = True,
) -> dict[str, Any]:
    receipt_path = receipt_path.expanduser().resolve(strict=True)
    dataset_index_path = dataset_index_path.expanduser().resolve(strict=True)
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

    selection = _mapping(report.get("selection"), "audit selection")
    _expect_equal(selection.get("world_size"), expected_world_size, "audit world_size")
    if int(selection.get("trace_count", 0)) < 16 or int(selection.get("position_count", 0)) < 511:
        raise TrainingAuditReceiptError("audit must cover at least 16 traces and 511 response positions")

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
    if grad_norm > 10.0:
        raise TrainingAuditReceiptError(f"audit backward total_norm is {grad_norm}; required <= 10.0")

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
