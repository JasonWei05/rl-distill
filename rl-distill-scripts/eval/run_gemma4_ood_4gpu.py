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

"""Queue the registered OOD matrix behind math and run it on four GPUs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

from gemma4_eval_registry import load_resolved_registry, select_models
from run_gemma4_three_model_evals import (
    DEFAULT_BASE_MODEL,
    DEFAULT_DISTILLED_MODEL,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_RL_MODEL,
    ModelSpec,
    resolve_models,
)

SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARKS = ("mmlu_pro", "gpqa", "mmmlu14k")
ESTIMATED_WORK = {"mmlu_pro": 50_000, "gpqa": 2_000, "mmmlu14k": 14_042}
ARTIFACT_EXPECTATIONS = {
    "mmlu_pro": {"result_key": "mmlu_pro", "effective_samples": 12_032, "logged_sample_rows": 12_032},
    "gpqa": {
        "result_key": "gpqa_diamond_cot_n_shot",
        "effective_samples": 198,
        "logged_sample_rows": 396,
    },
    "mmmlu14k": {
        "result_key": "gemma4_mmmlu14k",
        "effective_samples": 14_042,
        "logged_sample_rows": 14_042,
    },
}


@dataclass(frozen=True)
class OODTask:
    task_id: str
    model: ModelSpec
    benchmark: str
    estimated_work: int
    output_dir: Path
    log_path: Path
    completion_path: Path
    command: tuple[str, ...]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: Any, lock: threading.RLock | None = None) -> None:
    if lock is not None:
        lock.acquire()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".partial")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if lock is not None:
            lock.release()


def _wait_for_math(math_complete: Path, math_state: Path, poll_seconds: int) -> None:
    while not math_complete.is_file():
        if math_state.is_file():
            state = json.loads(math_state.read_text(encoding="utf-8"))
            if state.get("status") == "failed":
                raise RuntimeError(f"math failed; refusing to start OOD: {math_state}")
        time.sleep(poll_seconds)
    completion = json.loads(math_complete.read_text(encoding="utf-8"))
    if completion.get("protocol") not in {
        "gemma4_three_model_math_4gpu_v1",
        "gemma4_registered_math_4gpu_v1",
    }:
        raise ValueError(f"unexpected math completion protocol in {math_complete}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_task_artifacts(task: OODTask) -> dict[str, Any]:
    expectation = ARTIFACT_EXPECTATIONS[task.benchmark]
    wrapper_manifest_path = task.output_dir / "ood_eval_manifest.json"
    try:
        wrapper_manifest = json.loads(wrapper_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read OOD wrapper manifest {wrapper_manifest_path}: {error}") from error
    wrapper_config = wrapper_manifest.get("config", {})
    wrapper_identity = wrapper_manifest.get("model_identity", {})
    if wrapper_config.get("model") != task.model.model:
        raise ValueError(f"{task.task_id} wrapper manifest belongs to a different model")
    if wrapper_config.get("benchmarks") != [task.benchmark]:
        raise ValueError(f"{task.task_id} wrapper manifest has an unexpected benchmark selection")
    if wrapper_identity.get("model_identity_sha256") != task.model.expected_identity_sha256:
        raise ValueError(f"{task.task_id} wrapper manifest has an unexpected model identity")
    result_files = list(task.output_dir.rglob("results_*.json"))
    if not result_files:
        raise ValueError(f"{task.task_id} produced no lm-eval result JSON")
    result_path = max(result_files, key=lambda path: (path.stat().st_mtime_ns, str(path)))
    timestamp = result_path.stem.removeprefix("results_")
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read lm-eval result {result_path}: {error}") from error
    result_key = expectation["result_key"]
    if result_key not in result.get("results", {}):
        raise ValueError(f"{result_path} has no expected result key {result_key!r}")
    n_samples = result.get("n-samples")
    if not isinstance(n_samples, dict) or not n_samples:
        raise ValueError(f"{result_path} has no per-task n-samples records")
    try:
        effective_samples = sum(int(value["effective"]) for value in n_samples.values())
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{result_path} has malformed n-samples records") from error
    if effective_samples != expectation["effective_samples"]:
        raise ValueError(
            f"{task.task_id} effective sample count mismatch: "
            f"expected {expectation['effective_samples']}, found {effective_samples}"
        )

    sample_files = sorted(task.output_dir.rglob(f"samples_*_{timestamp}.jsonl"))
    if not sample_files:
        raise ValueError(f"{task.task_id} produced no logged samples for result timestamp {timestamp}")
    sample_manifest = []
    logged_rows = 0
    for path in sample_files:
        with path.open(encoding="utf-8") as source:
            rows = 0
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    raise ValueError(f"{path} contains a blank sample row at line {line_number}")
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path} contains invalid JSON at line {line_number}") from error
                if not isinstance(sample, dict):
                    raise ValueError(f"{path} sample row {line_number} is not a JSON object")
                rows += 1
        logged_rows += rows
        sample_manifest.append(
            {
                "path": str(path.relative_to(task.output_dir)),
                "rows": rows,
                "size": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    if logged_rows != expectation["logged_sample_rows"]:
        raise ValueError(
            f"{task.task_id} logged sample count mismatch: "
            f"expected {expectation['logged_sample_rows']}, found {logged_rows}"
        )
    sample_manifest_sha256 = hashlib.sha256(
        json.dumps(sample_manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "result_path": str(result_path.relative_to(task.output_dir)),
        "result_sha256": _sha256_file(result_path),
        "result_key": result_key,
        "effective_samples": effective_samples,
        "sample_file_count": len(sample_files),
        "logged_sample_rows": logged_rows,
        "sample_manifest_sha256": sample_manifest_sha256,
    }


def _task_complete(task: OODTask) -> bool:
    if not task.completion_path.is_file():
        return False
    try:
        payload = json.loads(task.completion_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    try:
        artifacts = _validate_task_artifacts(task)
    except (OSError, ValueError):
        return False
    return (
        payload.get("task_id") == task.task_id
        and payload.get("model_identity_sha256") == task.model.expected_identity_sha256
        and payload.get("benchmark") == task.benchmark
        and payload.get("artifacts") == artifacts
    )


def _run_workers(
    tasks: Sequence[OODTask],
    *,
    gpus: Sequence[str],
    state_path: Path,
    resume: bool,
) -> None:
    task_queue: queue.Queue[OODTask] = queue.Queue()
    for task in sorted(tasks, key=lambda item: (-item.estimated_work, item.task_id)):
        task_queue.put(task)
    lock = threading.RLock()
    failures: list[tuple[str, int | None]] = []
    state: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "gemma4_three_model_ood_4gpu_v1",
        "started_at_utc": _utc_now(),
        "status": "running",
        "gpus": list(gpus),
        "tasks": {task.task_id: {"status": "pending"} for task in tasks},
    }
    _atomic_json(state_path, state, lock)

    def worker(gpu: str) -> None:
        while True:
            try:
                task = task_queue.get_nowait()
            except queue.Empty:
                return
            completed: subprocess.CompletedProcess[str] | None = None
            try:
                if resume and _task_complete(task):
                    with lock:
                        state["tasks"][task.task_id] = {
                            "status": "skipped_complete",
                            "gpu": gpu,
                            "finished_at_utc": _utc_now(),
                        }
                    _atomic_json(state_path, state, lock)
                    continue

                task.output_dir.mkdir(parents=True, exist_ok=True)
                task.completion_path.unlink(missing_ok=True)
                task.log_path.parent.mkdir(parents=True, exist_ok=True)
                with lock:
                    state["tasks"][task.task_id] = {
                        "status": "running",
                        "gpu": gpu,
                        "started_at_utc": _utc_now(),
                        "log": str(task.log_path),
                    }
                _atomic_json(state_path, state, lock)
                environment = os.environ.copy()
                environment["CUDA_VISIBLE_DEVICES"] = gpu
                environment["TOKENIZERS_PARALLELISM"] = "false"
                with task.log_path.open("a", encoding="utf-8") as log:
                    log.write(f"[{_utc_now()}] CUDA_VISIBLE_DEVICES={gpu} {shlex.join(task.command)}\n")
                    log.flush()
                    completed = subprocess.run(
                        task.command,
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        text=True,
                        check=False,
                    )
                if completed.returncode != 0:
                    raise RuntimeError(f"subprocess exited with status {completed.returncode}")
                artifacts = _validate_task_artifacts(task)
                _atomic_json(
                    task.completion_path,
                    {
                        "schema_version": 1,
                        "protocol": "gemma4_three_model_ood_task_v1",
                        "completed_at_utc": _utc_now(),
                        "task_id": task.task_id,
                        "benchmark": task.benchmark,
                        "model": task.model.model,
                        "model_identity_sha256": task.model.expected_identity_sha256,
                        "output_dir": str(task.output_dir),
                        "artifacts": artifacts,
                    },
                )
                if not _task_complete(task):
                    raise RuntimeError("completion artifact verification failed")
                with lock:
                    state["tasks"][task.task_id] = {
                        "status": "complete",
                        "gpu": gpu,
                        "finished_at_utc": _utc_now(),
                        "returncode": completed.returncode,
                        "log": str(task.log_path),
                    }
                _atomic_json(state_path, state, lock)
            except Exception as error:  # noqa: BLE001 - worker failures must reach the scheduler state
                returncode = completed.returncode if completed is not None else None
                with lock:
                    state["tasks"][task.task_id] = {
                        "status": "failed",
                        "gpu": gpu,
                        "finished_at_utc": _utc_now(),
                        "returncode": returncode,
                        "error": f"{type(error).__name__}: {error}",
                        "log": str(task.log_path),
                    }
                    failures.append((task.task_id, returncode))
                _atomic_json(state_path, state, lock)
            finally:
                task_queue.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,), name=f"gpu-{gpu}") for gpu in gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    state["finished_at_utc"] = _utc_now()
    state["status"] = "failed" if failures else "complete"
    _atomic_json(state_path, state, lock)
    if failures:
        raise RuntimeError(f"OOD tasks failed: {failures}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs="+", default=["0", "1", "2", "3"])
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--resolved-model-registry", type=Path, default=None)
    parser.add_argument("--benchmarks", nargs="+", choices=BENCHMARKS, default=list(BENCHMARKS))
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--distilled-model", type=Path, default=DEFAULT_DISTILLED_MODEL)
    parser.add_argument("--rl-model", type=Path, default=DEFAULT_RL_MODEL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--lm-eval-executable", default="/tmp/.venv-gemma4-e2e/bin/lm_eval")
    parser.add_argument("--mmmlu-task-dir", type=Path, default=None)
    parser.add_argument("--mmmlu-manifest", type=Path, default=None)
    parser.add_argument("--skip-harness-git-check", action="store_true")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--kv-cache-memory-gib", type=float, default=None)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--no-wait-for-math", action="store_true")
    parser.add_argument(
        "--math-optional",
        action="store_true",
        help="Do not depend on the math phase at all (no wait, no completion marker required). Used when the "
        "OOD suite runs on a machine that never ran the math family (EVAL_PHASES=ood).",
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if len(set(args.gpus)) != len(args.gpus) or not args.gpus:
        raise ValueError("--gpus must contain unique GPU identifiers")
    if args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be positive")
    registry_mode = args.resolved_model_registry is not None
    if registry_mode:
        _, registered_models = load_resolved_registry(args.resolved_model_registry)
        resolved = select_models(registered_models, args.models)
        models = [
            ModelSpec(model.tag, model.model, model.expected_model_identity_sha256)
            for model in resolved
        ]
    else:
        requested_models = args.models or ["base_e2b", "distilled_e2b_step750", "rl_e2b_step125"]
        invalid_models = set(requested_models) - {"base_e2b", "distilled_e2b_step750", "rl_e2b_step125"}
        if invalid_models:
            raise ValueError(f"unknown legacy model tags: {sorted(invalid_models)}")
        models = resolve_models(
            SimpleNamespace(
                models=requested_models,
                base_model=args.base_model,
                distilled_model=args.distilled_model,
                rl_model=args.rl_model,
            )
        )
    output_root = args.output_root.resolve()
    tasks = []
    for model in models:
        for benchmark in args.benchmarks:
            task_id = f"{model.tag}__{benchmark}"
            output_dir = output_root / model.tag / "ood" / benchmark
            command = [
                args.python_executable,
                str(SCRIPT_DIR / "eval_gemma4_ood.py"),
                "--model",
                model.model,
                "--expected-model-identity-sha256",
                model.expected_identity_sha256,
                "--output-dir",
                str(output_dir),
                "--profile",
                "gemma4-report",
                "--benchmarks",
                benchmark,
                "--gpqa-task",
                "gpqa_diamond_cot_n_shot",
                "--lm-eval-executable",
                args.lm_eval_executable,
                "--tensor-parallel-size",
                "1",
                "--gpu-memory-utilization",
                str(args.gpu_memory_utilization),
                "--max-model-len",
                "8192",
            ]
            if args.kv_cache_memory_gib is not None:
                command.extend(["--kv-cache-memory-gib", str(args.kv_cache_memory_gib)])
            if args.mmmlu_task_dir is not None:
                command.extend(["--mmmlu-task-dir", str(args.mmmlu_task_dir.resolve())])
            if args.mmmlu_manifest is not None:
                command.extend(["--mmmlu-manifest", str(args.mmmlu_manifest.resolve())])
            if args.skip_harness_git_check:
                command.append("--skip-harness-git-check")
            tasks.append(
                OODTask(
                    task_id=task_id,
                    model=model,
                    benchmark=benchmark,
                    estimated_work=ESTIMATED_WORK[benchmark],
                    output_dir=output_dir,
                    log_path=output_root / "logs" / "ood" / f"{task_id}.log",
                    completion_path=output_dir / "complete.json",
                    command=tuple(command),
                )
            )

    math_complete = output_root / "math_4gpu_complete.json"
    math_state = output_root / "math_4gpu_state.json"
    plan = {
        "schema_version": 1,
        "protocol": "gemma4_registered_ood_4gpu_v1" if registry_mode else "gemma4_three_model_ood_4gpu_v1",
        "created_at_utc": _utc_now(),
        "mode": "execute" if args.execute else "dry_run",
        "gpus": args.gpus,
        "models": [asdict(model) for model in models],
        "math_dependency": {"complete": str(math_complete), "state": str(math_state)},
        "tasks": [
            {
                "task_id": task.task_id,
                "model": task.model.tag,
                "benchmark": task.benchmark,
                "estimated_work": task.estimated_work,
                "output_dir": str(task.output_dir),
                "log_path": str(task.log_path),
                "command": list(task.command),
            }
            for task in tasks
        ],
    }
    _atomic_json(output_root / "ood_4gpu_plan.json", plan)
    print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
    if not args.execute:
        return 0
    if args.math_optional:
        print(f"OOD: math phase not required (--math-optional); not waiting for {math_complete}", flush=True)
    elif not args.no_wait_for_math:
        _wait_for_math(math_complete, math_state, args.poll_seconds)
    elif not math_complete.is_file():
        raise FileNotFoundError(f"math completion marker is missing: {math_complete}")

    state_path = output_root / "ood_4gpu_state.json"
    _run_workers(tasks, gpus=args.gpus, state_path=state_path, resume=not args.no_resume)
    completion = {
        "schema_version": 1,
        "protocol": "gemma4_registered_ood_4gpu_v1" if registry_mode else "gemma4_three_model_ood_4gpu_v1",
        "completed_at_utc": _utc_now(),
        "state": str(state_path),
        "task_completions": [str(task.completion_path) for task in tasks],
    }
    _atomic_json(output_root / "ood_4gpu_complete.json", completion)
    print(json.dumps(completion, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
