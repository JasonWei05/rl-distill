from __future__ import annotations

import importlib.util
import json
import subprocess
from argparse import Namespace
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
        "cudnn_sdpa": "1",
        "eval_cudnn_sdpa": "0",
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


def test_child_environment_pins_verified_gemma4_batching_contract(tmp_path: Path):
    args = Namespace(
        model_path=tmp_path / "model",
        dataset_index=tmp_path / "overlay.json",
        source_dataset_index=tmp_path / "source.json",
        training_engine_audit_receipt=tmp_path / "audit.json",
        max_steps=3,
        save_freq=3,
        test_freq=3,
        project="test-project",
        experiment_name="test-experiment",
        checkpoint_dir=tmp_path / "checkpoints",
        max_checkpoints_to_keep=4,
        hf_push=False,
        hf_repo="unused",
        max_grad_norm=50.0,
        cudnn_sdpa=0,
        eval_cudnn_sdpa=0,
        grad_diagnostics=True,
        supervisor_dir=tmp_path / "supervisor",
        gpus=8,
        entity="test-entity",
    )

    environment = supervisor.build_child_environment(args, "run123")

    assert environment["MICRO_BATCH_SIZE_PER_GPU"] == "1"
    assert environment["MAX_PADDED_TOKENS_PER_MICROBATCH"] == "4096"
    assert environment["VERL_GEMMA4_CUDNN_SDPA"] == "0"
    assert environment["VERL_GEMMA4_EVAL_CUDNN_SDPA"] == "0"
    assert environment["FSDP_CAST_FORWARD_INPUTS"] == "true"
    assert environment["VERL_MAX_PRECLIP_GRAD_NORM"] == "50.0"


def test_default_receipt_is_the_three_batch_production_audit():
    assert supervisor.DEFAULT_TRAINING_ENGINE_AUDIT_RECEIPT.name == (
        "gemma4-e2b-overlay-vs-fsdp2-production-three-batch.json"
    )


def test_dry_run_executes_full_launcher_validation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    args = Namespace(
        model_path=tmp_path / "model",
        dataset_index=tmp_path / "overlay.json",
        source_dataset_index=tmp_path / "source.json",
        training_engine_audit_receipt=tmp_path / "audit.json",
        max_steps=3,
        save_freq=3,
        test_freq=1,
        project="test-project",
        experiment_name="test-experiment",
        checkpoint_dir=tmp_path / "checkpoints",
        max_checkpoints_to_keep=4,
        hf_push=False,
        hf_repo="unused",
        max_grad_norm=50.0,
        cudnn_sdpa=1,
        eval_cudnn_sdpa=0,
        grad_diagnostics=False,
        supervisor_dir=tmp_path / "supervisor",
        gpus=8,
        entity="test-entity",
        dry_run=True,
    )
    observed: dict[str, object] = {}

    def fake_run(command, *, cwd, env, check):
        observed.update(command=command, cwd=cwd, validate_only=env.get("VALIDATE_ONLY"), check=check)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(supervisor, "load_dotenv", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(supervisor, "parse_args", lambda: args)
    monkeypatch.setattr(supervisor, "validate_environment", lambda _args: None)
    monkeypatch.setattr(supervisor.subprocess, "run", fake_run)

    assert supervisor.main() == 0
    assert observed == {
        "command": ["bash", str(supervisor.LAUNCHER)],
        "cwd": supervisor.PROJECT_ROOT,
        "validate_only": "true",
        "check": False,
    }
