from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPOSITORY_ROOT / "rl-distill-scripts/supervise_gemma4_topk_distill.py"
SPEC = importlib.util.spec_from_file_location("gemma4_distill_supervisor", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
supervisor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(supervisor)


def test_parse_metric_line_extracts_training_metrics():
    metrics = supervisor.parse_metric_line(
        "step:2 - full_vocab_kl/mean:0.156 - train/loss:0.156 - train/grad_norm:10.85 - train/lr:4e-08"
    )

    assert metrics == {
        "step": 2.0,
        "full_vocab_kl/mean": 0.156,
        "train/loss": 0.156,
        "train/grad_norm": 10.85,
        "train/lr": 4e-08,
    }


def test_numerical_anomaly_reason_accepts_verified_bf16_forward_batch():
    assert (
        supervisor.numerical_anomaly_reason(
            {
                "train/loss": 0.156,
                "train/grad_norm": 10.85,
                "full_vocab_kl/teacher_mass": 0.915,
                "full_vocab_kl/student_mass": 0.906,
            },
            max_grad_norm=100.0,
        )
        is None
    )


@pytest.mark.parametrize("grad_norm", [148.63, float("nan"), float("inf")])
def test_numerical_anomaly_reason_rejects_bad_gradient_norm(grad_norm):
    assert "grad_norm" in supervisor.numerical_anomaly_reason(
        {"train/loss": 0.156, "train/grad_norm": grad_norm},
        max_grad_norm=100.0,
    )


def test_latest_checkpoint_step_requires_materialized_directory(tmp_path: Path):
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("10\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="missing directory"):
        supervisor.latest_checkpoint_step(tmp_path)

    (tmp_path / "global_step_10").mkdir()
    assert supervisor.latest_checkpoint_step(tmp_path) == 10


def test_completion_receipt_requires_matching_run_and_upload_contract(tmp_path: Path):
    receipt = {
        "checkpoint_step": 750,
        "wandb_run_id": "run1234",
        "hf_push_enabled": True,
        "hf_repo": "test/repo",
    }
    (tmp_path / "run_complete.json").write_text(json.dumps(receipt), encoding="utf-8")

    assert (
        supervisor.completion_receipt_error(
            tmp_path,
            expected_step=750,
            wandb_run_id="run1234",
            hf_push=True,
            hf_repo="test/repo",
        )
        is None
    )
    assert "wandb_run_id" in supervisor.completion_receipt_error(
        tmp_path,
        expected_step=750,
        wandb_run_id="different",
        hf_push=True,
        hf_repo="test/repo",
    )


def test_completion_receipt_fails_closed_when_missing(tmp_path: Path):
    assert "missing completion receipt" in supervisor.completion_receipt_error(
        tmp_path,
        expected_step=750,
        wandb_run_id="run1234",
        hf_push=False,
        hf_repo="unused",
    )
