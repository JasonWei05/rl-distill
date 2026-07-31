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

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATA_DIR))

import verify_gemma4_fsdp2_training_audit as verifier  # noqa: E402


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, Path, dict]:
    identity = "a" * 64
    dataset_index = tmp_path / "dataset_index.json"
    index = {
        "schema_version": verifier.EXPECTED_SCHEMA,
        "dataset_index_sha256": "b" * 64,
        "source_dataset_index_sha256": "c" * 64,
        "target_model_identity": {"model_identity_sha256": identity},
    }
    _write_json(dataset_index, index)
    source_sha256s = {
        relative: verifier._sha256(verifier.REPO_ROOT / relative) for relative in verifier.REQUIRED_SOURCE_PATHS
    }
    report = {
        "report_version": verifier.EXPECTED_REPORT_VERSION,
        "status": "pass",
        "gate": {"status": "pass", "checks": {"all": {"passed": True}}},
        "contract": {
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
        },
        "dataset": {
            "index_path": str(dataset_index),
            "dataset_index_sha256": index["dataset_index_sha256"],
            "source_dataset_index_sha256": index["source_dataset_index_sha256"],
            "target_model_identity_sha256": identity,
            "split": "validation",
        },
        "model": {
            "model_identity_sha256": identity,
            "dtype_load": "fp32",
            "autocast": "bfloat16",
            "attention_implementation": "sdpa",
        },
        "selection": {"world_size": 8, "trace_count": 16, "position_count": 511},
        "aggregate": {
            "topk_overlap_fraction": {"mean": 1.0},
            "stored_support_weighted_abs_logprob_delta": {"mean": 0.0002},
            "sampled_token_abs_logprob_delta": {"p95": 0.0009},
            "stored_support_probability_l1": {"mean": 0.0002},
        },
        "exact_serialization": {"ordered_topk_exact": {"mean": 1.0}},
        "backward": {"total_norm": 1.2},
        "implementation": {
            "commit": "d" * 40,
            "dirty": False,
            "source_sha256s": source_sha256s,
        },
    }
    receipt = tmp_path / "audit.json"
    _write_json(receipt, report)
    return receipt, dataset_index, report


def test_verifier_accepts_exact_production_contract(tmp_path: Path) -> None:
    receipt, dataset_index, _ = _fixture(tmp_path)

    verified = verifier.verify_receipt(
        receipt_path=receipt,
        dataset_index_path=dataset_index,
        expected_world_size=8,
        verify_repository=False,
    )

    assert verified["status"] == "pass"


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (lambda report: report["contract"].update(fsdp_param_dtype="fp32"), "fsdp_param_dtype"),
        (lambda report: report["contract"].update(gradient_checkpointing=False), "gradient_checkpointing"),
        (lambda report: report["selection"].update(position_count=510), "16 traces and 511"),
        (
            lambda report: report["aggregate"]["stored_support_weighted_abs_logprob_delta"].update(mean=0.01),
            "weighted logprob drift",
        ),
        (lambda report: report["implementation"].update(dirty=True), "implementation dirty"),
    ],
)
def test_verifier_rejects_mismatched_or_weak_receipts(tmp_path: Path, mutator, match: str) -> None:
    receipt, dataset_index, original = _fixture(tmp_path)
    report = copy.deepcopy(original)
    mutator(report)
    _write_json(receipt, report)

    with pytest.raises(verifier.TrainingAuditReceiptError, match=match):
        verifier.verify_receipt(
            receipt_path=receipt,
            dataset_index_path=dataset_index,
            expected_world_size=8,
            verify_repository=False,
        )
