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

"""Supervise the local 8-GPU Gemma-4 E4B 500-step replication.

The supervisor gives the training process a stable W&B identity, resumes from
the newest finalized NeMo-RL checkpoint after a crash, keeps the private HF
checkpoint uploader alive, and records enough state for an external monitor to
audit the run without attaching to the training process.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import secrets
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

import wandb
from dotenv import load_dotenv
from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal.datastore import DataStore

STOP_REQUESTED = False
TRAIN_PROCESS: subprocess.Popen[bytes] | None = None
UPLOAD_PROCESS: subprocess.Popen[bytes] | None = None


def log(message: str) -> None:
    print(
        f"[{datetime.now(UTC).isoformat(timespec='seconds')}] {message}",
        flush=True,
    )


def request_stop(_signum: int, _frame: Any) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    log("Stop requested")


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as state_file:
        json.dump(payload, state_file, indent=2, sort_keys=True)
        state_file.write("\n")
    os.replace(tmp_path, path)


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open() as input_file:
        return json.load(input_file)


def latest_checkpoint_step(checkpoint_dir: Path) -> int:
    latest_step = 0
    for step_dir in checkpoint_dir.glob("step_*"):
        try:
            step = int(step_dir.name.removeprefix("step_"))
        except ValueError:
            continue
        if not (step_dir / "training_info.json").is_file():
            continue
        latest_step = max(latest_step, step)
    return latest_step


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
        process.wait(timeout=30)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def validate_environment(repo_root: Path, driver_python: Path, gpus: int) -> None:
    worker_venv_root = Path(os.environ.get("NEMO_RL_VENV_DIR", "/tmp/nemo-rl-worker-venvs"))
    policy_worker_python = (
        worker_venv_root / "nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2" / "bin/python"
    )
    required_paths = [
        driver_python,
        policy_worker_python,
        repo_root / "third_party/nemo-rl/nemo_rl/distributed/model_utils.py",
        repo_root / "third_party/nemo-rl/nemo_rl/models/automodel/train.py",
        repo_root / "rl-distill-scripts/nemo_rl_repro/config/dapo_gemma4_e4b_pt_repro.yaml",
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required production paths: {missing}")

    model_utils = required_paths[2].read_text()
    automodel_train = required_paths[3].read_text()
    if "class ChunkedLocalLogprob" not in model_utils:
        raise RuntimeError("Batch-aware local-logprob rematerialization patch is missing")
    if "get_local_logprob_seq_chunk_size" not in automodel_train:
        raise RuntimeError("Batch-aware inference logprob chunking patch is missing")
    if 'chunk_size=self.cfg.get("logprob_chunk_size", None)' not in automodel_train:
        raise RuntimeError("Training loss does not forward logprob_chunk_size")

    for python_path, label in (
        (driver_python, "driver"),
        (policy_worker_python, "policy worker"),
    ):
        version = subprocess.run(
            [str(python_path), "-c", "import transformers; print(transformers.__version__)"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if version != "5.5.4":
            raise RuntimeError(f"Expected tested transformers 5.5.4 in {label} venv, got {version}")

    gpu_query = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    )
    available_gpus = len([line for line in gpu_query.stdout.splitlines() if line.strip()])
    if available_gpus < gpus:
        raise RuntimeError(f"Requested {gpus} GPUs, but only found {available_gpus}")

    for token_name in ("WANDB_API_KEY", "HF_TOKEN"):
        if not os.environ.get(token_name):
            raise RuntimeError(f"{token_name} is not set in the environment or .env")


def start_uploader(
    *,
    driver_python: Path,
    uploader_script: Path,
    checkpoint_dir: Path,
    repo_id: str,
    uploader_log: Path,
    env: dict[str, str],
) -> tuple[subprocess.Popen[bytes], TextIO]:
    uploader_log.parent.mkdir(parents=True, exist_ok=True)
    log_file = uploader_log.open("a")
    command = [
        str(driver_python),
        str(uploader_script),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--repo-id",
        repo_id,
        "--interval",
        "20",
        "--poll-seconds",
        "30",
    ]
    process = subprocess.Popen(
        command,
        cwd=checkpoint_dir.parent,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log(f"Started HF uploader pid={process.pid}")
    return process, log_file


def get_wandb_health(
    *,
    api: wandb.Api,
    entity: str,
    project: str,
    run_id: str,
) -> dict[str, Any]:
    run = api.run(f"{entity}/{project}/{run_id}")
    summary = dict(run.summary)
    return {
        "state": run.state,
        "step": summary.get("_step"),
        "probs_ratio": summary.get("train/probs_ratio"),
        "probs_ratio_clamped": summary.get("train/probs_ratio_clamped"),
        "grad_norm": summary.get("train/grad_norm"),
        "validation_accuracy": summary.get("validation/accuracy"),
    }


def _history_record_values(record: wandb_internal_pb2.Record) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for item in record.history.item:
        key = ".".join(item.nested_key) if item.nested_key else item.key
        try:
            values[key] = json.loads(item.value_json)
        except (json.JSONDecodeError, TypeError):
            values[key] = item.value_json
    return values


def get_local_wandb_health(*, logs_dir: Path, run_id: str) -> dict[str, Any] | None:
    """Read the newest complete metrics directly from the live W&B stream.

    W&B's public API can lag the locally-written history by several training
    steps. The local binary stream is append-only, so scanning complete
    records gives the metric gate an immediate view without modifying the run.
    """

    candidates = list(logs_dir.glob(f"exp_*/wandb/wandb/run-*-{run_id}/run-{run_id}.wandb"))
    if not candidates:
        return None
    wandb_path = max(candidates, key=lambda path: path.stat().st_mtime_ns)

    datastore = DataStore()
    datastore.open_for_scan(str(wandb_path))
    latest: dict[str, Any] = {}
    try:
        while True:
            try:
                data = datastore.scan_data()
            except Exception:
                # The writer may currently be appending the final record. All
                # prior complete records remain valid for health monitoring.
                break
            if data is None:
                break
            record = wandb_internal_pb2.Record()
            record.ParseFromString(data)
            if record.WhichOneof("record_type") == "history":
                latest.update(_history_record_values(record))
    finally:
        datastore.close()

    if not latest:
        return None
    return {
        "state": "running",
        "source": "local_wandb_stream",
        "step": latest.get("_step"),
        "probs_ratio": latest.get("train/probs_ratio"),
        "probs_ratio_clamped": latest.get("train/probs_ratio_clamped"),
        "grad_norm": latest.get("train/grad_norm"),
        "validation_accuracy": latest.get("validation/accuracy"),
    }


def health_is_bad(health: dict[str, Any]) -> str | None:
    for name in ("probs_ratio", "probs_ratio_clamped"):
        value = health.get(name)
        if value is None:
            continue
        if not math.isfinite(float(value)):
            return f"{name} is non-finite: {value}"
        if abs(float(value) - 1.0) > 0.02:
            return f"{name} left the [0.98, 1.02] gate: {value}"

    grad_norm = health.get("grad_norm")
    if grad_norm is not None and not math.isfinite(float(grad_norm)):
        return f"grad_norm is non-finite: {grad_norm}"
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument(
        "--run-name",
        default=("nemorl-dapo-gemma4-e4b-pt-DeepScaleR-4of4strict-seed42-8k-local8g-full500-fixed-v2-20260730"),
    )
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--hf-repo-id", default="JWei05/nemorl-dapo-gemma4-e4b-pt-500step")
    parser.add_argument("--entity", default="rl-distill")
    parser.add_argument("--project", default="DAPO")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--health-poll-seconds", type=int, default=120)
    parser.add_argument("--restart-backoff-seconds", type=int, default=60)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    global TRAIN_PROCESS, UPLOAD_PROCESS

    repo_root = Path(__file__).resolve().parents[3]
    load_dotenv(repo_root / ".env")
    args = parse_args()

    driver_venv = Path(os.environ.get("NEMO_RL_DRIVER_VENV", "/tmp/nemo-rl-venv"))
    driver_python = driver_venv / "bin/python"
    checkpoint_dir = (args.checkpoint_dir or Path("/tmp/verl/ckpts") / args.run_name).resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = Path("/tmp/verl/logs") / args.run_name
    supervisor_dir = Path("/tmp/verl/supervisor") / args.run_name
    logs_dir.mkdir(parents=True, exist_ok=True)
    supervisor_dir.mkdir(parents=True, exist_ok=True)

    validate_environment(repo_root, driver_python, args.gpus)

    run_id_file = checkpoint_dir / ".wandb_run_id"
    if run_id_file.is_file():
        run_id = run_id_file.read_text().strip()
    else:
        if latest_checkpoint_step(checkpoint_dir) > 0:
            raise RuntimeError(
                "Checkpoint directory is non-empty but has no .wandb_run_id; "
                "refusing to split resumed metrics into a new W&B run"
            )
        run_id = secrets.token_hex(4)
        run_id_file.write_text(run_id + "\n")

    state_path = supervisor_dir / "state.json"
    existing_state = read_json(state_path)
    existing_training_pid = existing_state.get("training_pid")
    if pid_is_alive(existing_training_pid):
        raise RuntimeError(
            f"Training pid {existing_training_pid} from the existing supervisor "
            "state is still alive; refusing to launch a duplicate run"
        )
    uploader_script = Path(__file__).with_name("hf_checkpoint_uploader.py")
    config_path = repo_root / "rl-distill-scripts/nemo_rl_repro/config/dapo_gemma4_e4b_pt_repro.yaml"
    launcher_path = repo_root / "rl-distill-scripts/nemo_rl_repro/run_grpo_repro.py"
    chat_template = repo_root / "rl-distill-scripts/data/gemma3_it_fewshot_math.jinja"

    child_env = dict(os.environ)
    child_env.pop("WANDB_RESUME_FROM", None)
    child_env.pop("WANDB_FORK_FROM", None)
    child_env.update(
        {
            "NEMO_RL_ROOT": str(repo_root / "third_party/nemo-rl"),
            "REPRO_DIR": str(repo_root / "rl-distill-scripts/nemo_rl_repro"),
            "NEMO_RL_FORCE_LOCAL_RAY": "1",
            "VLLM_USE_DEEP_GEMM": "0",
            "HF_HOME": child_env.get("HF_HOME", "/tmp/hf-home"),
            "NRL_JOB_START_EPOCH": child_env.get("NRL_JOB_START_EPOCH", str(int(time.time()))),
            "WANDB_RUN_ID": run_id,
            "WANDB_RESUME": "allow",
        }
    )

    train_command = [
        str(driver_python),
        str(launcher_path),
        "--config",
        str(config_path),
        f"cluster.gpus_per_node={args.gpus}",
        f"grpo.max_num_steps={args.max_steps}",
        "checkpointing.save_period=20",
        "checkpointing.save_consolidated=true",
        f"policy.tokenizer.chat_template={chat_template}",
        f"logger.wandb.name={args.run_name}",
        f"logger.log_dir={logs_dir}",
        f"checkpointing.checkpoint_dir={checkpoint_dir}",
    ]

    state = {
        "run_name": args.run_name,
        "wandb_run_id": run_id,
        "wandb_path": f"{args.entity}/{args.project}/{run_id}",
        "hf_repo_id": args.hf_repo_id,
        "checkpoint_dir": str(checkpoint_dir),
        "max_steps": args.max_steps,
        "supervisor_pid": os.getpid(),
        "status": "validated",
        "updated_at": datetime.now(UTC).isoformat(),
    }
    atomic_write_json(state_path, state)
    log(f"W&B run: {state['wandb_path']}")
    log(f"Checkpoint directory: {checkpoint_dir}")
    log(f"Private HF repository: {args.hf_repo_id}")

    if args.dry_run:
        log("Dry run passed; production command validated")
        return 0

    uploader_log_handle: TextIO | None = None
    train_log_handle: TextIO | None = None
    api = wandb.Api(timeout=30)
    attempt = int(existing_state.get("attempt", 0))
    last_health_poll = 0.0
    last_health_step: int | None = None

    try:
        while not STOP_REQUESTED:
            if UPLOAD_PROCESS is None or UPLOAD_PROCESS.poll() is not None:
                if UPLOAD_PROCESS is not None:
                    log(f"HF uploader exited rc={UPLOAD_PROCESS.returncode}; restarting")
                if uploader_log_handle is not None:
                    uploader_log_handle.close()
                UPLOAD_PROCESS, uploader_log_handle = start_uploader(
                    driver_python=driver_python,
                    uploader_script=uploader_script,
                    checkpoint_dir=checkpoint_dir,
                    repo_id=args.hf_repo_id,
                    uploader_log=supervisor_dir / "hf_uploader.log",
                    env=child_env,
                )

            current_checkpoint = latest_checkpoint_step(checkpoint_dir)
            if current_checkpoint >= args.max_steps:
                uploaded_steps = {
                    int(step) for step in read_json(checkpoint_dir / ".hf_upload_state.json").get("uploaded_steps", [])
                }
                if args.max_steps in uploaded_steps:
                    state.update(
                        {
                            "status": "complete",
                            "latest_checkpoint_step": current_checkpoint,
                            "updated_at": datetime.now(UTC).isoformat(),
                        }
                    )
                    atomic_write_json(state_path, state)
                    log("Step 500 checkpoint and HF upload are complete")
                    return 0
                state.update(
                    {
                        "status": "waiting_for_final_hf_upload",
                        "latest_checkpoint_step": current_checkpoint,
                        "updated_at": datetime.now(UTC).isoformat(),
                    }
                )
                atomic_write_json(state_path, state)
                time.sleep(args.poll_seconds)
                continue

            attempt += 1
            attempt_log = supervisor_dir / f"train_attempt_{attempt:03d}.log"
            train_log_handle = attempt_log.open("a")
            attempt_env = dict(child_env)
            if attempt > 1:
                # W&B's resume_from/rewind API is private preview and fails on
                # normal accounts. Standard resume keeps crash recovery usable;
                # the durable checkpoint step in state.json identifies the
                # replay boundary if an interrupted tail must be disregarded.
                attempt_env["WANDB_RESUME"] = "allow"
            TRAIN_PROCESS = subprocess.Popen(
                train_command,
                cwd=repo_root,
                env=attempt_env,
                stdout=train_log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            log(f"Started training attempt={attempt} pid={TRAIN_PROCESS.pid} from checkpoint_step={current_checkpoint}")
            state.update(
                {
                    "status": "training",
                    "attempt": attempt,
                    "training_pid": TRAIN_PROCESS.pid,
                    "latest_checkpoint_step": current_checkpoint,
                    "attempt_log": str(attempt_log),
                    "updated_at": datetime.now(UTC).isoformat(),
                }
            )
            atomic_write_json(state_path, state)

            metric_failure: str | None = None
            while not STOP_REQUESTED and TRAIN_PROCESS.poll() is None:
                if UPLOAD_PROCESS.poll() is not None:
                    log(f"HF uploader exited rc={UPLOAD_PROCESS.returncode}")
                    if uploader_log_handle is not None:
                        uploader_log_handle.close()
                    UPLOAD_PROCESS, uploader_log_handle = start_uploader(
                        driver_python=driver_python,
                        uploader_script=uploader_script,
                        checkpoint_dir=checkpoint_dir,
                        repo_id=args.hf_repo_id,
                        uploader_log=supervisor_dir / "hf_uploader.log",
                        env=child_env,
                    )

                now = time.monotonic()
                if now - last_health_poll >= args.health_poll_seconds:
                    last_health_poll = now
                    try:
                        health = get_local_wandb_health(logs_dir=logs_dir, run_id=run_id)
                        if health is None:
                            health = get_wandb_health(
                                api=api,
                                entity=args.entity,
                                project=args.project,
                                run_id=run_id,
                            )
                            health["source"] = "wandb_api"
                    except Exception as error:
                        log(f"W&B health poll unavailable: {type(error).__name__}: {error}")
                    else:
                        health_step = health.get("step")
                        state["wandb_health"] = health
                        state["updated_at"] = datetime.now(UTC).isoformat()
                        atomic_write_json(state_path, state)
                        if health_step != last_health_step:
                            log(f"W&B health: {health}")
                            last_health_step = health_step
                        metric_failure = health_is_bad(health)
                        if metric_failure is not None:
                            log(f"METRIC GATE FAILURE: {metric_failure}")
                            terminate_process_group(TRAIN_PROCESS)
                            break
                time.sleep(args.poll_seconds)

            if STOP_REQUESTED:
                break

            return_code = TRAIN_PROCESS.wait()
            TRAIN_PROCESS = None
            if train_log_handle is not None:
                train_log_handle.close()
                train_log_handle = None
            current_checkpoint = latest_checkpoint_step(checkpoint_dir)

            if metric_failure is not None:
                state.update(
                    {
                        "status": "metric_gate_failed",
                        "failure": metric_failure,
                        "latest_checkpoint_step": current_checkpoint,
                        "updated_at": datetime.now(UTC).isoformat(),
                    }
                )
                atomic_write_json(state_path, state)
                return 2

            if return_code == 0 and current_checkpoint >= args.max_steps:
                log("Training reached the requested final checkpoint")
                continue

            log(f"Training attempt={attempt} exited rc={return_code}; latest checkpoint={current_checkpoint}")
            state.update(
                {
                    "status": "restarting_after_failure",
                    "last_return_code": return_code,
                    "latest_checkpoint_step": current_checkpoint,
                    "updated_at": datetime.now(UTC).isoformat(),
                }
            )
            atomic_write_json(state_path, state)
            subprocess.run(
                [str(driver_venv / "bin/ray"), "stop", "--force"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(args.restart_backoff_seconds)
    finally:
        terminate_process_group(TRAIN_PROCESS)
        terminate_process_group(UPLOAD_PROCESS)
        subprocess.run(
            [str(driver_venv / "bin/ray"), "stop", "--force"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if train_log_handle is not None:
            train_log_handle.close()
        if uploader_log_handle is not None:
            uploader_log_handle.close()

    state.update(
        {
            "status": "stopped",
            "updated_at": datetime.now(UTC).isoformat(),
        }
    )
    atomic_write_json(state_path, state)
    return 130


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    sys.exit(main())
