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

"""Durably supervise the pinned E2B-to-E4B Gemma 4 distillation run.

Infrastructure failures resume from the latest complete local checkpoint.
Numerical failures stop fail-closed so they are diagnosed rather than replayed.
The child keeps one W&B run ID across restart attempts, and the trainer itself
waits for each requested Hugging Face upload before a successful exit.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import secrets
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = PROJECT_ROOT / "rl-distill-scripts/gemma4_topk_distill_fsdp2.sh"

DEFAULT_MODEL = Path(
    "/home/ubuntu/.cache/huggingface/models--google--gemma-4-E4B/snapshots/411aa17b749aa952df1359d2dcea73917a544d9a"
)
DEFAULT_DATASET_INDEX = Path("/tmp/verl/datasets/gemma4-e2b-base-topk128-hf-overlay-v128-seed42/dataset_index.json")
DEFAULT_SOURCE_INDEX = Path(
    "/tmp/verl/datasets/gemma4-e2b-base-topk128-traces-e32aaa02681a-val128-seed42/dataset_index.json"
)
DEFAULT_TRAINING_ENGINE_AUDIT_RECEIPT = Path("/tmp/verl/audits/gemma4-e2b-overlay-vs-fsdp2-production.json")
TEACHER_IDENTITY_SHA256 = "2d48d343709dcae087d6ff2def9f09d2950ca66dc2183a8bee38850c4ddbbb36"
STUDENT_IDENTITY_SHA256 = "acdc0d2bcb8f676593b5387807da1cd1b84a9e26fa279db4a86f54a211055b2d"
NUMERICAL_FAILURE_MARKERS = (
    "FloatingPointError",
    "non-finite training loss detected before backward",
    "exceeds fail-closed threshold",
    "gradient norm is non-finite",
)
METRIC_KEY = re.compile(r"^[A-Za-z0-9_./()%-]+$")

STOP_REQUESTED = False
TRAIN_PROCESS: subprocess.Popen[bytes] | None = None


def log(message: str) -> None:
    print(f"[{datetime.now(UTC).isoformat(timespec='seconds')}] {message}", flush=True)


def request_stop(_signum: int, _frame: Any) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    log("Stop requested")


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def pid_is_alive(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def terminate_process_group(process: subprocess.Popen[bytes] | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=60)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def latest_checkpoint_step(checkpoint_dir: Path) -> int:
    tracker = checkpoint_dir / "latest_checkpointed_iteration.txt"
    if not tracker.is_file():
        return 0
    try:
        step = int(tracker.read_text(encoding="utf-8").strip())
    except ValueError as error:
        raise RuntimeError(f"invalid checkpoint tracker: {tracker}") from error
    checkpoint = checkpoint_dir / f"global_step_{step}"
    if not checkpoint.is_dir():
        raise RuntimeError(f"checkpoint tracker points to a missing directory: {checkpoint}")
    return step


def completion_receipt_error(
    checkpoint_dir: Path,
    *,
    expected_step: int,
    wandb_run_id: str,
    hf_push: bool,
    hf_repo: str,
) -> str | None:
    """Return why the trainer's post-upload completion receipt is invalid."""
    receipt_path = checkpoint_dir / "run_complete.json"
    try:
        receipt = read_json(receipt_path)
    except (OSError, json.JSONDecodeError) as error:
        return f"invalid completion receipt {receipt_path}: {error}"
    if not receipt:
        return f"missing completion receipt: {receipt_path}"

    expected = {
        "checkpoint_step": expected_step,
        "wandb_run_id": wandb_run_id,
        "hf_push_enabled": hf_push,
        "hf_repo": hf_repo if hf_push else None,
    }
    for key, expected_value in expected.items():
        if receipt.get(key) != expected_value:
            return f"completion receipt field {key!r} is {receipt.get(key)!r}; expected {expected_value!r}"
    return None


def parse_metric_line(line: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for item in line.strip().split(" - "):
        if ":" not in item:
            continue
        key, raw_value = item.split(":", 1)
        if not METRIC_KEY.fullmatch(key):
            continue
        try:
            values[key] = float(raw_value)
        except ValueError:
            continue
    return values


def numerical_anomaly_reason(metrics: dict[str, float], *, max_grad_norm: float) -> str | None:
    for key in ("train/loss", "val/loss", "full_vocab_kl/mean"):
        value = metrics.get(key)
        if value is not None and not math.isfinite(value):
            return f"{key} is non-finite: {value}"

    train_loss = metrics.get("train/loss")
    if train_loss is not None and abs(train_loss) > 5.0:
        return f"train/loss magnitude exceeds 5.0: {train_loss}"

    grad_norm = metrics.get("train/grad_norm")
    if grad_norm is not None:
        if not math.isfinite(grad_norm):
            return f"train/grad_norm is non-finite: {grad_norm}"
        if grad_norm > max_grad_norm:
            return f"train/grad_norm exceeds {max_grad_norm}: {grad_norm}"

    for key in (
        "full_vocab_kl/teacher_mass",
        "full_vocab_kl/student_mass",
        "val/full_vocab_kl/teacher_mass",
        "val/full_vocab_kl/student_mass",
    ):
        value = metrics.get(key)
        if value is None:
            continue
        if not math.isfinite(value) or value < -0.01 or value > 1.01:
            return f"{key} left the expected probability-mass range: {value}"
    return None


def scan_new_log(path: Path, offset: int) -> tuple[int, list[str]]:
    if not path.is_file():
        return offset, []
    with path.open(encoding="utf-8", errors="replace") as handle:
        handle.seek(offset)
        lines = handle.readlines()
        return handle.tell(), lines


def validate_environment(args: argparse.Namespace) -> None:
    required_paths = [
        LAUNCHER,
        args.model_path,
        args.dataset_index,
        args.source_dataset_index,
        args.training_engine_audit_receipt,
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing required production paths: {missing}")
    if args.gpus != 8:
        raise ValueError("the verified production contract requires exactly 8 GPUs")
    if args.max_steps <= 0 or args.save_freq <= 0 or args.test_freq <= 0:
        raise ValueError("max-steps, save-freq, and test-freq must be positive")
    if not math.isfinite(args.max_grad_norm) or args.max_grad_norm <= 0:
        raise ValueError("max-grad-norm must be finite and positive")
    for token_name in ("WANDB_API_KEY", "HF_TOKEN"):
        if not os.environ.get(token_name):
            raise RuntimeError(f"{token_name} is not set in .env or the environment")
    available = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if len([line for line in available if line.strip()]) < args.gpus:
        raise RuntimeError(f"requested {args.gpus} GPUs but only found {len(available)}")


def build_child_environment(args: argparse.Namespace, run_id: str) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "MODEL_PATH": str(args.model_path),
            "DATASET_INDEX": str(args.dataset_index),
            "SOURCE_DATASET_INDEX": str(args.source_dataset_index),
            "TRAINING_ENGINE_AUDIT_RECEIPT": str(args.training_engine_audit_receipt),
            "DISTILL_DIRECTION": "e2b_base_to_e4b",
            "EXPECTED_TEACHER_IDENTITY_SHA256": TEACHER_IDENTITY_SHA256,
            "EXPECTED_STUDENT_IDENTITY_SHA256": STUDENT_IDENTITY_SHA256,
            "EXPECTED_TRAIN_QUESTIONS": "9723",
            "EXPECTED_VALIDATION_QUESTIONS": "128",
            "EXPECTED_TRAIN_SAMPLES_PER_QUESTION": "5",
            "EXPECTED_VALIDATION_SAMPLES_PER_QUESTION": "1",
            "MICRO_BATCH_SIZE_PER_GPU": "2",
            "MAX_PADDED_TOKENS_PER_MICROBATCH": "5120",
            "FULL_VOCAB_KL_CHUNK_SIZE": "4096",
            "TRAIN_BATCH_SIZE": "128",
            "LR": "2e-6",
            "LR_WARMUP_STEPS": "100",
            "LR_SCHEDULER_TYPE": "linear",
            "MIN_LR_RATIO": "0.1",
            "TOTAL_EPOCHS": "2",
            "TOTAL_TRAINING_STEPS": str(args.max_steps),
            "SAVE_FREQ": str(args.save_freq),
            "TEST_FREQ": str(args.test_freq),
            "VAL_BEFORE_TRAIN": "true",
            "PROJECT_NAME": args.project,
            "EXP_NAME": args.experiment_name,
            "CKPTS_DIR": str(args.checkpoint_dir),
            "MAX_CKPT_TO_KEEP": str(args.max_checkpoints_to_keep),
            "HF_PUSH_ENABLE": "true" if args.hf_push else "false",
            "HF_PUSH_REPO": args.hf_repo,
            "HF_PUSH_PRIVATE": "true",
            "HF_PUSH_MAX_TO_KEEP": "8",
            "FSDP_PARAM_DTYPE": "bf16",
            "FSDP_REDUCE_DTYPE": "fp32",
            "FSDP_BUFFER_DTYPE": "fp32",
            "FSDP_CAST_FORWARD_INPUTS": "true",
            "VERL_GEMMA4_CUDNN_SDPA": "1",
            "VERL_FAIL_ON_NONFINITE_LOSS": "1",
            "VERL_FAIL_ON_NONFINITE_GRAD": "1",
            "VERL_MAX_PRECLIP_GRAD_NORM": str(args.max_grad_norm),
            "VERL_FSDP2_GRAD_DIAGNOSTICS": "1" if args.grad_diagnostics else "0",
            "VERL_FSDP2_GRAD_DIAGNOSTICS_TOPK": "20",
            "VERL_FSDP2_GRAD_DIAGNOSTICS_PATH": str(args.supervisor_dir / "latest_grad_diagnostics.json"),
            "NPROC_PER_NODE": str(args.gpus),
            "WANDB_ENTITY": args.entity,
            "WANDB_RUN_ID": run_id,
            "WANDB_RESUME": "allow",
        }
    )
    environment.pop("WANDB_RESUME_FROM", None)
    environment.pop("WANDB_FORK_FROM", None)
    return environment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--supervisor-dir", type=Path)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--dataset-index", type=Path, default=DEFAULT_DATASET_INDEX)
    parser.add_argument("--source-dataset-index", type=Path, default=DEFAULT_SOURCE_INDEX)
    parser.add_argument(
        "--training-engine-audit-receipt",
        type=Path,
        default=DEFAULT_TRAINING_ENGINE_AUDIT_RECEIPT,
    )
    parser.add_argument("--max-steps", type=int, default=750)
    parser.add_argument("--save-freq", type=int, default=250)
    parser.add_argument("--test-freq", type=int, default=10)
    parser.add_argument("--max-checkpoints-to-keep", type=int, default=4)
    parser.add_argument("--max-grad-norm", type=float, default=50.0)
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--project", default="gemma4-distill-vs-rl")
    parser.add_argument("--entity", default="rl-distill")
    parser.add_argument("--hf-repo", default="JWei05/gemma4-e2b-base-to-e4b-topk128-distill")
    parser.add_argument("--hf-push", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grad-diagnostics", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    # The trainer allows up to one hour for pending HF uploads during final
    # shutdown. Keep the outer stall gate longer so it does not kill a healthy
    # upload before the trainer can report its own fail-closed timeout.
    parser.add_argument("--stall-timeout-seconds", type=float, default=4500.0)
    parser.add_argument("--restart-backoff-seconds", type=float, default=30.0)
    parser.add_argument("--max-restarts-at-same-checkpoint", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.checkpoint_dir = args.checkpoint_dir.resolve()
    args.supervisor_dir = (args.supervisor_dir or Path("/tmp/verl/supervisor") / args.experiment_name).resolve()
    args.model_path = args.model_path.resolve()
    args.dataset_index = args.dataset_index.resolve()
    args.source_dataset_index = args.source_dataset_index.resolve()
    args.training_engine_audit_receipt = args.training_engine_audit_receipt.resolve()
    return args


def main() -> int:
    global TRAIN_PROCESS

    load_dotenv(PROJECT_ROOT / ".env")
    args = parse_args()
    validate_environment(args)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.supervisor_dir.mkdir(parents=True, exist_ok=True)

    state_path = args.supervisor_dir / "state.json"
    previous_state = read_json(state_path)
    previous_pid = previous_state.get("training_pid")
    if pid_is_alive(previous_pid):
        raise RuntimeError(f"training pid {previous_pid} is still alive; refusing a duplicate launch")

    run_id_path = args.checkpoint_dir / ".wandb_run_id"
    if run_id_path.is_file():
        run_id = run_id_path.read_text(encoding="utf-8").strip()
    else:
        if latest_checkpoint_step(args.checkpoint_dir) > 0:
            raise RuntimeError("checkpoint exists without .wandb_run_id; refusing to split resumed metrics")
        run_id = secrets.token_hex(4)
        run_id_path.write_text(run_id + "\n", encoding="utf-8")

    child_environment = build_child_environment(args, run_id)
    state = {
        "status": "validated",
        "supervisor_pid": os.getpid(),
        "experiment_name": args.experiment_name,
        "checkpoint_dir": str(args.checkpoint_dir),
        "wandb_run_id": run_id,
        "wandb_path": f"{args.entity}/{args.project}/{run_id}",
        "hf_push": args.hf_push,
        "hf_repo": args.hf_repo if args.hf_push else None,
        "max_steps": args.max_steps,
        "updated_at": datetime.now(UTC).isoformat(),
    }
    atomic_write_json(state_path, state)
    log(f"W&B run: {state['wandb_path']}")
    log(f"Checkpoint directory: {args.checkpoint_dir}")
    if args.dry_run:
        log("Dry run passed")
        return 0

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    attempt = int(previous_state.get("attempt", 0))
    failure_counts: dict[int, int] = {}

    try:
        while not STOP_REQUESTED:
            checkpoint_step = latest_checkpoint_step(args.checkpoint_dir)
            if checkpoint_step >= args.max_steps:
                receipt_error = completion_receipt_error(
                    args.checkpoint_dir,
                    expected_step=args.max_steps,
                    wandb_run_id=run_id,
                    hf_push=args.hf_push,
                    hf_repo=args.hf_repo,
                )
                if receipt_error is not None:
                    state.update(
                        {
                            "status": "completion_unverified",
                            "failure": receipt_error,
                            "latest_checkpoint_step": checkpoint_step,
                            "updated_at": datetime.now(UTC).isoformat(),
                        }
                    )
                    atomic_write_json(state_path, state)
                    log(f"COMPLETION GATE FAILURE: {receipt_error}")
                    return 4
                state.update(
                    {
                        "status": "complete",
                        "latest_checkpoint_step": checkpoint_step,
                        "updated_at": datetime.now(UTC).isoformat(),
                    }
                )
                atomic_write_json(state_path, state)
                log(f"Run complete at checkpoint step {checkpoint_step}")
                return 0

            attempt += 1
            attempt_log = args.supervisor_dir / f"train_attempt_{attempt:03d}.log"
            log_handle = attempt_log.open("ab", buffering=0)
            TRAIN_PROCESS = subprocess.Popen(
                ["bash", str(LAUNCHER)],
                cwd=PROJECT_ROOT,
                env=child_environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            log(f"Started attempt={attempt} pid={TRAIN_PROCESS.pid} from checkpoint_step={checkpoint_step}")
            state.update(
                {
                    "status": "training",
                    "attempt": attempt,
                    "training_pid": TRAIN_PROCESS.pid,
                    "attempt_log": str(attempt_log),
                    "latest_checkpoint_step": checkpoint_step,
                    "updated_at": datetime.now(UTC).isoformat(),
                }
            )
            atomic_write_json(state_path, state)

            offset = 0
            last_progress = time.monotonic()
            anomaly: str | None = None
            latest_metrics: dict[str, float] = {}
            while not STOP_REQUESTED and TRAIN_PROCESS.poll() is None:
                offset, lines = scan_new_log(attempt_log, offset)
                for line in lines:
                    for marker in NUMERICAL_FAILURE_MARKERS:
                        if marker in line:
                            anomaly = line.strip()[-1000:]
                            break
                    metrics = parse_metric_line(line)
                    if metrics:
                        latest_metrics.update(metrics)
                        if "step" in metrics or "train/loss" in metrics or "val/loss" in metrics:
                            last_progress = time.monotonic()
                        reason = numerical_anomaly_reason(metrics, max_grad_norm=args.max_grad_norm)
                        if reason is not None:
                            anomaly = reason
                    if anomaly is not None:
                        break

                if latest_metrics:
                    state["latest_metrics"] = latest_metrics
                    state["updated_at"] = datetime.now(UTC).isoformat()
                    atomic_write_json(state_path, state)
                if anomaly is not None:
                    log(f"NUMERICAL GATE FAILURE: {anomaly}")
                    terminate_process_group(TRAIN_PROCESS)
                    break
                if time.monotonic() - last_progress > args.stall_timeout_seconds:
                    log(f"No metric progress for {args.stall_timeout_seconds:.0f}s; treating as infrastructure stall")
                    terminate_process_group(TRAIN_PROCESS)
                    break
                time.sleep(args.poll_seconds)

            if STOP_REQUESTED:
                break

            return_code = TRAIN_PROCESS.wait()
            TRAIN_PROCESS = None
            log_handle.close()
            offset, final_lines = scan_new_log(attempt_log, offset)
            for line in final_lines:
                for marker in NUMERICAL_FAILURE_MARKERS:
                    if marker in line:
                        anomaly = line.strip()[-1000:]
                        break
                metrics = parse_metric_line(line)
                if metrics:
                    latest_metrics.update(metrics)
                    reason = numerical_anomaly_reason(metrics, max_grad_norm=args.max_grad_norm)
                    if reason is not None:
                        anomaly = reason
                if anomaly is not None:
                    break
            if latest_metrics:
                state["latest_metrics"] = latest_metrics
            checkpoint_after = latest_checkpoint_step(args.checkpoint_dir)

            if anomaly is not None:
                state.update(
                    {
                        "status": "numerical_anomaly",
                        "failure": anomaly,
                        "last_return_code": return_code,
                        "latest_checkpoint_step": checkpoint_after,
                        "updated_at": datetime.now(UTC).isoformat(),
                    }
                )
                atomic_write_json(state_path, state)
                return 2

            if checkpoint_after >= args.max_steps:
                receipt_error = completion_receipt_error(
                    args.checkpoint_dir,
                    expected_step=args.max_steps,
                    wandb_run_id=run_id,
                    hf_push=args.hf_push,
                    hf_repo=args.hf_repo,
                )
                if return_code == 0 and receipt_error is None:
                    log("Trainer exited successfully after final checkpoint and required uploads")
                    continue
                failure = receipt_error or f"trainer exited rc={return_code} without verified completion"
                state.update(
                    {
                        "status": "completion_unverified",
                        "failure": failure,
                        "last_return_code": return_code,
                        "latest_checkpoint_step": checkpoint_after,
                        "updated_at": datetime.now(UTC).isoformat(),
                    }
                )
                atomic_write_json(state_path, state)
                log(f"COMPLETION GATE FAILURE: {failure}")
                return 4

            failure_counts[checkpoint_after] = failure_counts.get(checkpoint_after, 0) + 1
            failures_here = failure_counts[checkpoint_after]
            state.update(
                {
                    "status": "restarting_after_infrastructure_failure",
                    "last_return_code": return_code,
                    "latest_checkpoint_step": checkpoint_after,
                    "failures_at_checkpoint": failures_here,
                    "updated_at": datetime.now(UTC).isoformat(),
                }
            )
            atomic_write_json(state_path, state)
            log(
                f"Attempt={attempt} exited rc={return_code}; latest checkpoint={checkpoint_after}; "
                f"restart count at this checkpoint={failures_here}"
            )
            if failures_here > args.max_restarts_at_same_checkpoint:
                state["status"] = "repeated_infrastructure_failure"
                state["updated_at"] = datetime.now(UTC).isoformat()
                atomic_write_json(state_path, state)
                return 3
            time.sleep(args.restart_backoff_seconds)
    finally:
        terminate_process_group(TRAIN_PROCESS)

    state.update({"status": "stopped", "updated_at": datetime.now(UTC).isoformat()})
    atomic_write_json(state_path, state)
    return 130


if __name__ == "__main__":
    sys.exit(main())
