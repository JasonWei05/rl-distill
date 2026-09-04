#!/usr/bin/env python3
"""Greedily schedule all registered Gemma 4 evaluations on one eight-GPU node."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from gemma4_eval_registry import canonical_json_sha256, load_source_registry  # noqa: E402

TAG_PATTERN = re.compile(r"^[a-z0-9_]+$")
COMPLETE_PROTOCOL = "gemma4_rl_distill_base_evals_v2"


@dataclass(frozen=True)
class EvalTask:
    tag: str
    display_name: str
    architecture: str
    gpus: int


@dataclass
class RunningTask:
    task: EvalTask
    gpu_ids: tuple[int, ...]
    attempt: int
    process: subprocess.Popen[str]
    log_path: Path
    log_handle: Any
    started_at_utc: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_tasks(source_registry: Path) -> tuple[dict[str, Any], list[EvalTask]]:
    payload, models = load_source_registry(source_registry)
    tasks = []
    for model in models:
        if not TAG_PATTERN.fullmatch(model.tag):
            raise ValueError(f"unsafe model tag: {model.tag!r}")
        gpus = 2 if model.architecture == "gemma-4-12B" else 1
        tasks.append(EvalTask(model.tag, model.display_name, model.architecture, gpus))
    if len(tasks) != 15:
        raise ValueError(f"packed evaluation requires exactly 15 models, found {len(tasks)}")
    # 12B models take 2 GPUs, everything else 1; any mix is allowed (the distill study has none).
    return payload, tasks


def task_queue(tasks: Sequence[EvalTask]) -> list[EvalTask]:
    return [task for _, task in sorted(enumerate(tasks), key=lambda item: (-item[1].gpus, item[0]))]


def initial_packing(tasks: Sequence[EvalTask], gpu_ids: Sequence[int]) -> list[tuple[str, tuple[int, ...]]]:
    free = list(gpu_ids)
    pending = task_queue(tasks)
    packed: list[tuple[str, tuple[int, ...]]] = []
    while pending:
        fit = next((index for index, task in enumerate(pending) if task.gpus <= len(free)), None)
        if fit is None:
            break
        task = pending.pop(fit)
        assigned = tuple(free[: task.gpus])
        del free[: task.gpus]
        packed.append((task.tag, assigned))
    return packed


def packed_gpu_environment(gpu_ids: Sequence[int]) -> dict[str, str]:
    values = tuple(gpu_ids)
    if not values or len(set(values)) != len(values) or any(gpu not in range(8) for gpu in values):
        raise ValueError(f"invalid packed physical GPU IDs: {values}")
    resolved = ",".join(str(gpu) for gpu in values)
    return {"CUDA_VISIBLE_DEVICES": resolved, "PACKED_PHYSICAL_GPU_IDS": resolved}


def run_capture(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, check=False)


def remote_completion(result_s3_root: str, tag: str) -> tuple[bool, str]:
    uri = f"{result_s3_root.rstrip('/')}/{tag}/RUN_COMPLETE.json"
    completed = run_capture(["aws", "s3", "cp", uri, "-", "--no-progress"])
    if completed.returncode != 0:
        return False, "missing"
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return False, "invalid_json"
    if payload.get("protocol") != COMPLETE_PROTOCOL:
        return False, "wrong_protocol"
    if payload.get("model_tag") != tag:
        return False, "wrong_model_tag"
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        return False, "empty_file_manifest"
    return True, "valid"


def upload_file(path: Path, uri: str, *, required: bool = False) -> bool:
    completed = run_capture(["aws", "s3", "cp", str(path), uri, "--only-show-errors"])
    if completed.returncode == 0:
        return True
    print(f"WARNING: upload failed: {path} -> {uri}", flush=True)
    if required:
        raise RuntimeError(f"required upload failed: {uri}")
    return False


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-registry",
        type=Path,
        default=SCRIPTS_DIR / "config/gemma4_rl_distill_eval_sources.json",
    )
    parser.add_argument(
        "--model-runner",
        type=Path,
        default=Path(__file__).with_name("run_gemma4_rl_distill_eval_one_model.sh"),
    )
    parser.add_argument("--work-root", type=Path, default=Path("/tmp/gemma4-rl-distill-eval-packed"))
    parser.add_argument(
        "--result-s3-root",
        default="s3://scale-ml/genai/rl-distill/gemma4-distill-study-evals-v1",
    )
    parser.add_argument("--gpu-ids", nargs="+", type=int, default=list(range(8)))
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--start-stagger-seconds", type=int, default=20)
    parser.add_argument("--retry-delay-seconds", type=int, default=60)
    parser.add_argument("--poll-seconds", type=int, default=10)
    parser.add_argument("--heartbeat-seconds", type=int, default=300)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if sorted(args.gpu_ids) != list(range(8)):
        raise ValueError("packed production evaluation requires physical GPUs 0 through 7 exactly once")
    if args.max_attempts <= 0:
        raise ValueError("--max-attempts must be positive")
    for name in ("start_stagger_seconds", "retry_delay_seconds", "poll_seconds", "heartbeat_seconds"):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be nonnegative")
    if not args.result_s3_root.startswith("s3://"):
        raise ValueError("--result-s3-root must be an S3 URI")

    source_payload, tasks = load_tasks(args.source_registry)
    queue = task_queue(tasks)
    packing = initial_packing(tasks, args.gpu_ids)
    plan = {
        "schema_version": 1,
        "protocol": "gemma4_rl_distill_packed_plan_v1",
        "source_registry_sha256": canonical_json_sha256(source_payload),
        "gpu_ids": args.gpu_ids,
        "tasks": [asdict(task) for task in queue],
        "initial_packing": [{"tag": tag, "gpu_ids": list(gpus)} for tag, gpus in packing],
        "result_s3_root": args.result_s3_root,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    args.model_runner = args.model_runner.resolve()
    if not args.model_runner.is_file():
        raise FileNotFoundError(args.model_runner)
    work_root = args.work_root.resolve()
    log_root = work_root / "logs"
    state_path = work_root / "packed_state.json"
    log_root.mkdir(parents=True, exist_ok=True)
    shared_data_root = Path(os.environ.get("SHARED_DATA_ROOT", work_root / "shared/data")).resolve()
    shared_mmmlu_root = Path(os.environ.get("SHARED_MMMLU_ROOT", work_root / "shared/mmmlu14k_tasks")).resolve()
    for required in (shared_data_root / "math_eval_manifest.json", shared_mmmlu_root / "manifest.json"):
        if not required.is_file():
            raise FileNotFoundError(f"packed shared asset is missing: {required}")

    packed_root = f"{args.result_s3_root.rstrip('/')}/_packed"
    attempts = {task.tag: 0 for task in tasks}
    ready_at = {task.tag: 0.0 for task in tasks}
    state: dict[str, Any] = {
        **plan,
        "protocol": "gemma4_rl_distill_packed_state_v1",
        "started_at_utc": utc_now(),
        "status": "running",
        "tasks": {task.tag: {**asdict(task), "status": "pending", "attempts": 0} for task in tasks},
    }

    def persist_state(*, required: bool = False) -> None:
        state["updated_at_utc"] = utc_now()
        write_json(state_path, state)
        upload_file(state_path, f"{packed_root}/packed_state.json", required=required)

    manifest_path = work_root / "packed_plan.json"
    write_json(manifest_path, plan)
    upload_file(manifest_path, f"{packed_root}/packed_plan.json", required=True)

    pending: list[EvalTask] = []
    for task in queue:
        complete, reason = remote_completion(args.result_s3_root, task.tag)
        if complete:
            state["tasks"][task.tag].update(
                {"status": "skipped_remote_complete", "completion_reason": reason, "finished_at_utc": utc_now()}
            )
            print(f"SKIP remote-complete model={task.tag}", flush=True)
        else:
            pending.append(task)
    persist_state(required=True)

    free_gpus = sorted(args.gpu_ids)
    running: dict[str, RunningTask] = {}
    permanently_failed: list[str] = []
    stop_requested = False

    def stop_children(signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True
        print(f"PACKED scheduler received signal={signum}; terminating child process groups", flush=True)
        for item in list(running.values()):
            try:
                os.killpg(item.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    signal.signal(signal.SIGTERM, stop_children)
    signal.signal(signal.SIGINT, stop_children)

    def launch(task: EvalTask) -> None:
        nonlocal free_gpus
        gpu_ids = tuple(free_gpus[: task.gpus])
        del free_gpus[: task.gpus]
        attempts[task.tag] += 1
        attempt = attempts[task.tag]
        log_path = log_root / f"{task.tag}.attempt_{attempt:02d}.log"
        log_handle = log_path.open("a", encoding="utf-8")
        env = os.environ.copy()
        env.update(
            {
                "MODEL_TAG": task.tag,
                "GPU_COUNT": str(task.gpus),
                **packed_gpu_environment(gpu_ids),
                "RESULT_S3_ROOT": args.result_s3_root,
                "MODEL_WORK_ROOT": str(work_root / "runs" / task.tag),
                "MODEL_ROOT_OVERRIDE": str(work_root / "models" / task.tag),
                "SHARED_DATA_ROOT": str(shared_data_root),
                "SHARED_MMMLU_ROOT": str(shared_mmmlu_root),
                "PREPARE_SHARED_ASSETS": "false",
                "HF_HOME": str(work_root / "shared/hf_cache"),
                "XDG_CACHE_HOME": str(work_root / "shared/cache"),
                "VLLM_CACHE_ROOT": str(work_root / "caches/vllm" / task.tag),
                "TRITON_CACHE_DIR": str(work_root / "caches/triton" / task.tag),
                "TORCHINDUCTOR_CACHE_DIR": str(work_root / "caches/torchinductor" / task.tag),
                "VLLM_ATTENTION_BACKEND": "TRITON_ATTN",
                "VLLM_USE_FLASHINFER_SAMPLER": "0",
                "VLLM_USE_DEEP_GEMM": "0",
                "TOKENIZERS_PARALLELISM": "false",
            }
        )
        started = utc_now()
        log_handle.write(
            f"[{started}] PACKED_START model={task.tag} attempt={attempt} "
            f"physical_gpus={','.join(str(gpu) for gpu in gpu_ids)}\n"
        )
        log_handle.flush()
        process = subprocess.Popen(
            ["bash", str(args.model_runner)],
            cwd=SCRIPTS_DIR.parent,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        running[task.tag] = RunningTask(task, gpu_ids, attempt, process, log_path, log_handle, started)
        state["tasks"][task.tag].update(
            {
                "status": "running",
                "attempts": attempt,
                "gpu_ids": list(gpu_ids),
                "pid": process.pid,
                "started_at_utc": started,
            }
        )
        print(
            f"PACKED_LAUNCH model={task.tag} attempt={attempt} "
            f"physical_gpus={','.join(str(gpu) for gpu in gpu_ids)} pid={process.pid}",
            flush=True,
        )
        persist_state()

    def terminate_task_group(item: RunningTask) -> None:
        """Remove any vLLM descendants before returning physical GPUs to the pool."""
        try:
            os.killpg(item.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                os.killpg(item.process.pid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.2)
        try:
            os.killpg(item.process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def upload_attempt_log(item: RunningTask) -> None:
        upload_file(
            item.log_path,
            f"{packed_root}/logs/{item.task.tag}/attempt_{item.attempt:02d}.log",
        )

    def cleanup_success(task: EvalTask) -> None:
        for path in (work_root / "models" / task.tag, work_root / "runs" / task.tag):
            if path.is_dir() and path.is_relative_to(work_root):
                shutil.rmtree(path)

    last_heartbeat = 0.0
    while pending or running:
        now = time.monotonic()
        for tag, item in list(running.items()):
            returncode = item.process.poll()
            if returncode is None:
                continue
            terminate_task_group(item)
            item.log_handle.close()
            del running[tag]
            free_gpus.extend(item.gpu_ids)
            free_gpus.sort()
            upload_attempt_log(item)
            complete, reason = remote_completion(args.result_s3_root, tag)
            if complete:
                state["tasks"][tag].update(
                    {
                        "status": "complete",
                        "returncode": returncode,
                        "completion_reason": reason,
                        "finished_at_utc": utc_now(),
                        "gpu_ids": [],
                        "pid": None,
                    }
                )
                cleanup_success(item.task)
                print(f"PACKED_COMPLETE model={tag} attempt={item.attempt} rc={returncode}", flush=True)
            elif attempts[tag] < args.max_attempts and not stop_requested:
                ready_at[tag] = time.monotonic() + args.retry_delay_seconds
                pending.append(item.task)
                state["tasks"][tag].update(
                    {
                        "status": "retry_wait",
                        "returncode": returncode,
                        "completion_reason": reason,
                        "next_attempt": attempts[tag] + 1,
                        "gpu_ids": [],
                        "pid": None,
                    }
                )
                print(
                    f"PACKED_RETRY model={tag} completed_attempt={item.attempt} rc={returncode} reason={reason}",
                    flush=True,
                )
            else:
                permanently_failed.append(tag)
                failure = {
                    "model_tag": tag,
                    "attempts": attempts[tag],
                    "returncode": returncode,
                    "completion_reason": reason,
                    "failed_at_utc": utc_now(),
                }
                failure_path = work_root / "failures" / f"{tag}.json"
                write_json(failure_path, failure)
                upload_file(failure_path, f"{packed_root}/failures/{tag}.json")
                state["tasks"][tag].update({"status": "failed", **failure, "gpu_ids": [], "pid": None})
                print(f"PACKED_FAILED model={tag} attempts={attempts[tag]} reason={reason}", flush=True)
            persist_state()

        if stop_requested:
            break

        launched_any = False
        while True:
            now = time.monotonic()
            fit_index = next(
                (
                    index
                    for index, task in enumerate(pending)
                    if ready_at[task.tag] <= now and task.gpus <= len(free_gpus)
                ),
                None,
            )
            if fit_index is None:
                break
            task = pending.pop(fit_index)
            launch(task)
            launched_any = True
            if args.start_stagger_seconds:
                time.sleep(args.start_stagger_seconds)

        now = time.monotonic()
        if now - last_heartbeat >= args.heartbeat_seconds:
            summary = {status: 0 for status in ("pending", "running", "retry_wait", "complete", "failed")}
            for record in state["tasks"].values():
                status = record["status"]
                summary[status] = summary.get(status, 0) + 1
            print(
                f"PACKED_HEARTBEAT time={utc_now()} free_gpus={','.join(map(str, free_gpus)) or 'none'} "
                f"pending={len(pending)} running={len(running)} states={json.dumps(summary, sort_keys=True)}",
                flush=True,
            )
            persist_state()
            last_heartbeat = now

        if pending or running:
            time.sleep(0 if launched_any else args.poll_seconds)

    if stop_requested:
        for item in running.values():
            try:
                item.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(item.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            item.log_handle.close()
            upload_attempt_log(item)
        state["status"] = "interrupted"
        state["finished_at_utc"] = utc_now()
        persist_state()
        return 130

    final_missing = []
    for task in tasks:
        complete, reason = remote_completion(args.result_s3_root, task.tag)
        if not complete:
            final_missing.append({"tag": task.tag, "reason": reason})

    final_payload = {
        "schema_version": 1,
        "protocol": "gemma4_rl_distill_packed_complete_v1",
        "completed_at_utc": utc_now(),
        "source_registry_sha256": canonical_json_sha256(source_payload),
        "result_s3_root": args.result_s3_root,
        "model_tags": [task.tag for task in tasks],
        "attempts": attempts,
        "permanently_failed": permanently_failed,
        "missing_remote_completions": final_missing,
    }
    final_path = work_root / ("PACKED_RUN_COMPLETE.json" if not final_missing else "PACKED_RUN_FAILED.json")
    write_json(final_path, final_payload)
    state["status"] = "complete" if not final_missing and not permanently_failed else "failed"
    state["finished_at_utc"] = utc_now()
    state["permanently_failed"] = permanently_failed
    state["missing_remote_completions"] = final_missing
    persist_state(required=True)
    if final_missing or permanently_failed:
        upload_file(final_path, f"{packed_root}/PACKED_RUN_FAILED.json", required=True)
        print(
            f"GEMMA4_PACKED_EVAL_FAILED permanent={permanently_failed} missing={final_missing}",
            flush=True,
        )
        return 1
    upload_file(final_path, f"{packed_root}/PACKED_RUN_COMPLETE.json", required=True)
    print("GEMMA4_PACKED_EVAL_DONE models=15 gpus=8", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
