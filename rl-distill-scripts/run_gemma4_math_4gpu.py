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

"""Run and merge the registered Gemma 4 math matrix on four independent GPUs."""

from __future__ import annotations

import argparse
import json
import os
import queue
import shlex
import subprocess
import sys
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import pandas as pd
from eval_math_passk import PREDICTIVE_ENTROPY_KIND
from gemma4_eval_metrics import aggregate_math_traces
from gemma4_eval_registry import load_resolved_registry, select_models
from run_gemma4_three_model_evals import (
    DEFAULT_BASE_MODEL,
    DEFAULT_DATA_MANIFEST,
    DEFAULT_DISTILLED_MODEL,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_RL_MODEL,
    ModelSpec,
    _load_json,
    build_math_command,
    resolve_models,
    select_named_math_datasets,
    select_math_datasets,
)

CLEAN_ID_NAME = "id_validation_clean"
FULL_ID_NAME = "id_validation_full"
MATH_DATASET_NAMES = (
    FULL_ID_NAME,
    CLEAN_ID_NAME,
    "math500",
    "gsm8k",
    "olympiadbench",
    "minervamath",
    "aime2025",
    "aime2026",
)


@dataclass(frozen=True)
class MathTask:
    task_id: str
    model: ModelSpec
    datasets: tuple[dict[str, Any], ...]
    total_requests: int
    metrics_path: Path
    trace_dir: Path
    log_path: Path
    command: tuple[str, ...]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _replace_arg(command: list[str], flag: str, value: str) -> None:
    command[command.index(flag) + 1] = value


def partition_datasets(datasets: Sequence[dict[str, Any]], shard_count: int) -> list[tuple[dict[str, Any], ...]]:
    """Greedily balance indivisible datasets by registered request count."""

    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    if not datasets:
        raise ValueError("at least one dataset is required")
    bins: list[list[dict[str, Any]]] = [[] for _ in range(min(shard_count, len(datasets)))]
    totals = [0] * len(bins)
    for entry in sorted(datasets, key=lambda item: (-int(item["total_requests"]), str(item["name"]))):
        target = min(range(len(bins)), key=lambda index: (totals[index], index))
        bins[target].append(entry)
        totals[target] += int(entry["total_requests"])
    return [tuple(entries) for entries in bins]


def _task_complete(task: MathTask) -> bool:
    if not task.metrics_path.is_file():
        return False
    try:
        payload = _load_json(task.metrics_path)
    except ValueError:
        return False
    config = payload.get("config", {})
    model_identity = config.get("model_identity", {}) if isinstance(config, dict) else {}
    if payload.get("model") != task.model.model:
        return False
    if model_identity.get("model_identity_sha256") != task.model.expected_identity_sha256:
        return False
    expected_names = {str(entry["name"]) for entry in task.datasets}
    if set(payload.get("results", {})) != expected_names:
        return False
    for entry in task.datasets:
        trace = task.trace_dir / f"{task.model.tag}__{entry['name']}.jsonl"
        if not trace.is_file():
            return False
        with trace.open(encoding="utf-8") as source:
            if sum(1 for _ in source) != int(entry["total_requests"]):
                return False
    return True


def _write_state(path: Path, state: Mapping[str, Any], lock: threading.Lock) -> None:
    with lock:
        temporary = path.with_suffix(path.suffix + ".partial")
        temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)


def _run_workers(
    tasks: Sequence[MathTask],
    *,
    gpus: Sequence[str],
    state_path: Path,
    resume: bool,
) -> None:
    task_queue: queue.Queue[MathTask] = queue.Queue()
    for task in sorted(tasks, key=lambda item: (-item.total_requests, item.task_id)):
        task_queue.put(task)

    state_lock = threading.RLock()
    state: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "gemma4_three_model_math_4gpu_v1",
        "started_at_utc": _utc_now(),
        "status": "running",
        "gpus": list(gpus),
        "tasks": {task.task_id: {"status": "pending"} for task in tasks},
    }
    _write_state(state_path, state, state_lock)
    failures: list[tuple[str, int | None]] = []

    def worker(gpu: str) -> None:
        while True:
            try:
                task = task_queue.get_nowait()
            except queue.Empty:
                return
            try:
                if resume and _task_complete(task):
                    with state_lock:
                        state["tasks"][task.task_id] = {
                            "status": "skipped_complete",
                            "gpu": gpu,
                            "finished_at_utc": _utc_now(),
                        }
                    _write_state(state_path, state, state_lock)
                    continue

                task.metrics_path.parent.mkdir(parents=True, exist_ok=True)
                task.trace_dir.mkdir(parents=True, exist_ok=True)
                task.log_path.parent.mkdir(parents=True, exist_ok=True)
                with state_lock:
                    state["tasks"][task.task_id] = {
                        "status": "running",
                        "gpu": gpu,
                        "started_at_utc": _utc_now(),
                        "log": str(task.log_path),
                    }
                _write_state(state_path, state, state_lock)

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
                valid = completed.returncode == 0 and _task_complete(task)
                with state_lock:
                    state["tasks"][task.task_id] = {
                        "status": "complete" if valid else "failed",
                        "gpu": gpu,
                        "finished_at_utc": _utc_now(),
                        "returncode": completed.returncode,
                        "log": str(task.log_path),
                    }
                    if not valid:
                        failures.append((task.task_id, completed.returncode))
                _write_state(state_path, state, state_lock)
            except Exception as error:  # noqa: BLE001 - worker failures must reach the scheduler state
                with state_lock:
                    state["tasks"][task.task_id] = {
                        "status": "failed",
                        "gpu": gpu,
                        "finished_at_utc": _utc_now(),
                        "error": f"{type(error).__name__}: {error}",
                        "log": str(task.log_path),
                    }
                    failures.append((task.task_id, None))
                _write_state(state_path, state, state_lock)
            finally:
                task_queue.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,), name=f"gpu-{gpu}") for gpu in gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    state["finished_at_utc"] = _utc_now()
    state["status"] = "failed" if failures else "complete"
    _write_state(state_path, state, state_lock)
    if failures:
        raise RuntimeError(f"math evaluation tasks failed: {failures}")


def derive_clean_id_result(
    *,
    model: ModelSpec,
    full_entry: Mapping[str, Any],
    clean_entry: Mapping[str, Any],
    trace_dir: Path,
) -> dict[str, Any]:
    """Filter the clean ID subset from the already-generated full ID traces."""

    clean_frame = pd.read_parquet(str(clean_entry["output_path"]), columns=["uid"])
    clean_uids = {str(uid) for uid in clean_frame["uid"]}
    if len(clean_uids) != int(clean_entry["unique_questions"]):
        raise ValueError("clean ID parquet does not contain the registered number of unique UIDs")

    source_path = trace_dir / f"{model.tag}__{full_entry['name']}.jsonl"
    metric_traces = []
    seen_uids: set[str] = set()
    sample_indices_by_uid: dict[str, set[int]] = {}
    with source_path.open(encoding="utf-8") as source:
        for line in source:
            trace = json.loads(line)
            uid = str(trace["uid"])
            if uid not in clean_uids:
                continue
            sample_index = int(trace["sample_index"])
            indices = sample_indices_by_uid.setdefault(uid, set())
            if sample_index in indices:
                raise ValueError(f"full ID traces contain duplicate sample index {sample_index} for UID {uid!r}")
            indices.add(sample_index)
            seen_uids.add(uid)
            metric_traces.append(
                {
                    "uid": uid,
                    "sample_index": sample_index,
                    "acc": trace["acc"],
                    "answer_class": trace["answer_class"],
                    "answer_class_method": trace.get("answer_class_method"),
                    "sequence_entropy": trace["sequence_entropy"],
                    "token_entropy_sum": trace["token_entropy_sum"],
                    "token_entropy_count": trace["token_entropy_count"],
                }
            )
    if seen_uids != clean_uids:
        raise ValueError(f"full ID traces are missing {len(clean_uids - seen_uids)} clean-set UIDs")
    if len(metric_traces) != int(clean_entry["total_requests"]):
        raise ValueError(
            f"clean ID trace count mismatch: expected {clean_entry['total_requests']}, found {len(metric_traces)}"
        )
    samples = int(clean_entry["samples_per_question"])
    expected_indices = set(range(samples))
    malformed = sorted(uid for uid, indices in sample_indices_by_uid.items() if indices != expected_indices)
    if malformed:
        raise ValueError(f"clean ID UIDs do not have exact sample indices 0..{samples - 1}: {malformed[:8]}")
    aggregation = aggregate_math_traces(
        metric_traces,
        k_values=[samples],
        expected_samples_per_question=samples,
        subset_strategy="full_only",
        seed=0,
        prediction_field="answer_class",
        predictive_entropy_kind=PREDICTIVE_ENTROPY_KIND,
    )
    full_metrics = aggregation["by_k"][str(samples)]
    return {
        "k": samples,
        "n_questions": aggregation["n_questions"],
        "mean@k": round(100 * full_metrics["mean_at_k"], 2),
        "pass@k": round(100 * full_metrics["pass_at_k"], 2),
        "maj@k": round(100 * full_metrics["maj_at_k"], 2),
        "derived_from": str(full_entry["name"]),
        "trace_source": str(source_path),
        "trace_filter": "UID membership in the registered clean validation parquet; no duplicate trace file written",
        **aggregation,
    }


def merge_model_results(
    *,
    model: ModelSpec,
    tasks: Sequence[MathTask],
    full_entry: Mapping[str, Any] | None,
    clean_entry: Mapping[str, Any] | None,
    output_root: Path,
    data_manifest_path: Path,
) -> Path:
    results: dict[str, Any] = {}
    source_configs = []
    model_tasks = [task for task in tasks if task.model.tag == model.tag]
    for task in model_tasks:
        payload = _load_json(task.metrics_path)
        source_configs.append(payload.get("config"))
        for name, result in payload["results"].items():
            if name in results:
                raise ValueError(f"duplicate merged result for {model.tag}/{name}")
            results[name] = result

    trace_dir = output_root / model.tag / "math" / "traces"
    if clean_entry is not None:
        if full_entry is None:
            raise ValueError("clean ID derivation requires the full ID dataset")
        results[CLEAN_ID_NAME] = derive_clean_id_result(
            model=model,
            full_entry=full_entry,
            clean_entry=clean_entry,
            trace_dir=trace_dir,
        )
    output_path = output_root / model.tag / "math" / "metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "tag": model.tag,
                "model": model.model,
                "config": {
                    "protocol": "gemma4_three_model_math_4gpu_v1",
                    "data_manifest": str(data_manifest_path),
                    "model_identity_sha256": model.expected_identity_sha256,
                    "clean_id_derived_without_regeneration": clean_entry is not None,
                    "source_shard_configs": source_configs,
                },
                "results": results,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return output_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs="+", default=["0", "1", "2", "3"])
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--resolved-model-registry", type=Path, default=None)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--distilled-model", type=Path, default=DEFAULT_DISTILLED_MODEL)
    parser.add_argument("--rl-model", type=Path, default=DEFAULT_RL_MODEL)
    parser.add_argument("--data-manifest", type=Path, default=DEFAULT_DATA_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--request-batch-size", type=int, default=8)
    parser.add_argument("--questions-per-batch", type=int, default=1)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if len(set(args.gpus)) != len(args.gpus) or not args.gpus:
        raise ValueError("--gpus must contain unique GPU identifiers")
    data_manifest_path = args.data_manifest.resolve()
    data_manifest = _load_json(data_manifest_path)
    registry_mode = args.resolved_model_registry is not None
    if registry_mode:
        _, registered_models = load_resolved_registry(args.resolved_model_registry)
        resolved = select_models(registered_models, args.models)
        models = [
            ModelSpec(model.tag, model.model, model.expected_model_identity_sha256)
            for model in resolved
        ]
        dataset_names_by_model = {model.tag: list(model.math_datasets) for model in resolved}
        if args.datasets:
            requested = set(args.datasets)
            dataset_names_by_model = {
                tag: [name for name in names if name in requested]
                for tag, names in dataset_names_by_model.items()
            }
            empty = [tag for tag, names in dataset_names_by_model.items() if not names]
            if empty:
                raise ValueError(f"--datasets removes every registered dataset for models: {empty}")
    else:
        requested_models = args.models or ["base_e2b", "distilled_e2b_step750", "rl_e2b_step125"]
        invalid_models = set(requested_models) - {"base_e2b", "distilled_e2b_step750", "rl_e2b_step125"}
        if invalid_models:
            raise ValueError(f"unknown legacy model tags: {sorted(invalid_models)}")
        selected_names = args.datasets or list(MATH_DATASET_NAMES)
        invalid_datasets = set(selected_names) - set(MATH_DATASET_NAMES)
        if invalid_datasets:
            raise ValueError(f"unknown legacy dataset names: {sorted(invalid_datasets)}")
        models = resolve_models(
            SimpleNamespace(
                models=requested_models,
                base_model=args.base_model,
                distilled_model=args.distilled_model,
                rl_model=args.rl_model,
            )
        )
        dataset_names_by_model = {model.tag: list(selected_names) for model in models}
    output_root = args.output_root.resolve()
    tasks: list[MathTask] = []
    full_entries_by_model: dict[str, dict[str, Any] | None] = {}
    clean_entries_by_model: dict[str, dict[str, Any] | None] = {}
    for model in models:
        selected = select_named_math_datasets(data_manifest, dataset_names_by_model[model.tag])
        by_name = {str(entry["name"]): entry for entry in selected}
        full_entry = by_name.get(FULL_ID_NAME)
        clean_entry = by_name.get(CLEAN_ID_NAME)
        if clean_entry is not None and full_entry is None:
            raise ValueError("selecting id_validation_clean also requires id_validation_full")
        full_entries_by_model[model.tag] = full_entry
        clean_entries_by_model[model.tag] = clean_entry
        generation_datasets = [entry for entry in selected if entry["name"] != CLEAN_ID_NAME]
        for shard_index, shard in enumerate(partition_datasets(generation_datasets, len(args.gpus))):
            task_id = f"{model.tag}__shard_{shard_index:02d}"
            task_root = output_root / "_math_tasks" / task_id
            metrics_path = task_root / "metrics.json"
            trace_dir = output_root / model.tag / "math" / "traces"
            log_path = output_root / "logs" / "math" / f"{task_id}.log"
            command = build_math_command(
                model=model,
                data_manifest_path=data_manifest_path,
                data_manifest=data_manifest,
                datasets=shard,
                output_root=task_root,
                python_executable=args.python_executable,
                tensor_parallel_size=1,
                gpu_memory_utilization=args.gpu_memory_utilization,
                request_batch_size=args.request_batch_size,
                questions_per_batch=args.questions_per_batch,
            )
            command.extend(["--subset_strategy", "monte_carlo", "--monte_carlo_resamples", "4096"])
            _replace_arg(command, "--out", str(metrics_path))
            _replace_arg(command, "--trace_dir", str(trace_dir))
            tasks.append(
                MathTask(
                    task_id=task_id,
                    model=model,
                    datasets=shard,
                    total_requests=sum(int(entry["total_requests"]) for entry in shard),
                    metrics_path=metrics_path,
                    trace_dir=trace_dir,
                    log_path=log_path,
                    command=tuple(command),
                )
            )

    output_root.mkdir(parents=True, exist_ok=True)
    plan = {
        "schema_version": 1,
        "protocol": "gemma4_registered_math_4gpu_v1" if registry_mode else "gemma4_three_model_math_4gpu_v1",
        "created_at_utc": _utc_now(),
        "mode": "execute" if args.execute else "dry_run",
        "gpus": args.gpus,
        "models": [asdict(model) for model in models],
        "data_manifest": str(data_manifest_path),
        "selected_datasets_by_model": dataset_names_by_model,
        "clean_id_derivation_by_model": {
            model.tag: (
                {
                    "source": FULL_ID_NAME,
                    "derived": CLEAN_ID_NAME,
                    "avoided_generations": int(clean_entries_by_model[model.tag]["total_requests"]),
                }
                if clean_entries_by_model[model.tag] is not None
                else None
            )
            for model in models
        },
        "tasks": [
            {
                "task_id": task.task_id,
                "model": task.model.tag,
                "datasets": [entry["name"] for entry in task.datasets],
                "total_requests": task.total_requests,
                "metrics_path": str(task.metrics_path),
                "trace_dir": str(task.trace_dir),
                "log_path": str(task.log_path),
                "command": list(task.command),
            }
            for task in tasks
        ],
    }
    plan_path = output_root / "math_4gpu_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
    if not args.execute:
        return 0

    state_path = output_root / "math_4gpu_state.json"
    _run_workers(tasks, gpus=args.gpus, state_path=state_path, resume=not args.no_resume)
    merged = [
        str(
            merge_model_results(
                model=model,
                tasks=tasks,
                full_entry=full_entries_by_model[model.tag],
                clean_entry=clean_entries_by_model[model.tag],
                output_root=output_root,
                data_manifest_path=data_manifest_path,
            )
        )
        for model in models
    ]
    completion = {
        "schema_version": 1,
        "protocol": "gemma4_registered_math_4gpu_v1" if registry_mode else "gemma4_three_model_math_4gpu_v1",
        "completed_at_utc": _utc_now(),
        "state": str(state_path),
        "merged_metrics": merged,
    }
    (output_root / "math_4gpu_complete.json").write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(completion, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
