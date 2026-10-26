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

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

SCRIPTS_DIR = Path(__file__).parents[2] / "rl-distill-scripts" / "eval"
DATA_DIR = SCRIPTS_DIR.parent / "data"
for import_path in (SCRIPTS_DIR, DATA_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import run_gemma4_math_4gpu as math_scheduler  # noqa: E402
import run_gemma4_ood_4gpu as ood_scheduler  # noqa: E402
from eval_gemma4_ood import OODEvalConfig, build_ood_commands  # noqa: E402
from eval_math_passk import load_dataset_protocol_manifest  # noqa: E402
from prepare_gemma4_three_model_eval_data import convert_hf_rows, samples_per_question  # noqa: E402
from run_gemma4_math_4gpu import derive_clean_id_result, partition_datasets  # noqa: E402
from run_gemma4_three_model_evals import (  # noqa: E402
    ModelSpec,
    build_math_command,
    select_math_datasets,
)


@pytest.mark.parametrize(
    ("questions", "samples", "requests"),
    [
        (200, 16, 3_200),
        (500, 8, 4_000),
        (1_319, 2, 2_638),
        (674, 4, 2_696),
        (272, 8, 2_176),
        (30, 128, 3_840),
    ],
)
def test_power_of_two_repetition_rule_is_strictly_greater_than_2000(
    questions: int,
    samples: int,
    requests: int,
) -> None:
    assert samples_per_question(questions) == samples
    assert questions * samples == requests
    assert requests > 2_000
    if samples > 1:
        assert questions * (samples // 2) <= 2_000


def _dataset_entry(tmp_path: Path, name: str, unique: int, samples: int) -> dict[str, object]:
    path = tmp_path / f"{name}.parquet"
    path.write_bytes(name.encode())
    return {
        "name": name,
        "role": "test",
        "source": {},
        "unique_questions": unique,
        "samples_per_question": samples,
        "total_requests": unique * samples,
        "output_path": str(path.resolve()),
        "output_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def test_dataset_manifest_and_launcher_preserve_variable_sample_counts(tmp_path: Path) -> None:
    names = (
        "id_validation_full",
        "id_validation_clean",
        "math500",
        "gsm8k",
        "olympiadbench",
        "minervamath",
        "aime2025",
        "aime2026",
    )
    entries = [
        _dataset_entry(tmp_path, name, 30 if name.startswith("aime") else 200, 128 if name.startswith("aime") else 16)
        for name in names
    ]
    template = tmp_path / "template.jinja"
    template.write_text("template")
    manifest = {
        "schema_version": 1,
        "protocol": "gemma4_three_model_math_eval_v1",
        "repetition_rule": {
            "threshold": 2_000,
            "comparison": "strictly_greater_than",
            "allowed_factors": "powers_of_two",
            "implementation": "smallest power of two k such that unique_questions * k > threshold",
        },
        "chat_template": {"path": str(template), "sha256": hashlib.sha256(b"template").hexdigest()},
        "sampling": {
            "temperature": 1.0,
            "top_k": -1,
            "top_p": 1.0,
            "max_response_tokens": 8192,
            "max_prompt_tokens": 4096,
            "max_model_len": 12288,
            "predictive_topk_width": 128,
        },
        "datasets": entries,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    loaded = load_dataset_protocol_manifest(manifest_path)
    assert loaded[str((tmp_path / "aime2025.parquet").resolve())]["samples_per_question"] == 128
    selected = select_math_datasets(manifest, id_validation="clean")
    assert selected[0]["name"] == "id_validation_clean"
    assert "id_validation_full" not in {entry["name"] for entry in selected}

    command = build_math_command(
        model=ModelSpec("base_e2b", "/models/base", "a" * 64),
        data_manifest_path=manifest_path,
        data_manifest=manifest,
        datasets=selected,
        output_root=tmp_path / "results",
        python_executable="/venv/bin/python",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        request_batch_size=8,
    )
    assert command[command.index("--dataset_manifest") + 1] == str(manifest_path)
    assert command[command.index("--max_tokens") + 1] == "8192"
    assert command[command.index("--top_k") + 1] == "-1"
    assert command[command.index("--predictive_topk_width") + 1] == "128"
    assert command[command.index("--expected_model_identity_sha256") + 1] == "a" * 64


def test_dataset_manifest_rejects_nonminimal_repetition_and_sampling_drift(tmp_path: Path) -> None:
    entry = _dataset_entry(tmp_path, "math500", 500, 16)
    manifest = {
        "schema_version": 1,
        "protocol": "gemma4_three_model_math_eval_v1",
        "repetition_rule": {
            "threshold": 2_000,
            "comparison": "strictly_greater_than",
            "allowed_factors": "powers_of_two",
        },
        "sampling": {
            "temperature": 1.0,
            "top_k": -1,
            "top_p": 1.0,
            "max_response_tokens": 8192,
            "max_prompt_tokens": 4096,
            "max_model_len": 12288,
            "predictive_topk_width": 128,
        },
        "datasets": [entry],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="smallest registered power of two"):
        load_dataset_protocol_manifest(path)

    entry["samples_per_question"] = 8
    entry["total_requests"] = 4_000
    manifest["sampling"]["temperature"] = 0.7
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="registered Gemma 4 sampling settings"):
        load_dataset_protocol_manifest(path)


def test_gemma4_report_ood_profile_uses_native_five_shot_tasks(tmp_path: Path) -> None:
    config = OODEvalConfig(
        model="/models/base",
        model_revision=None,
        output_dir=str(tmp_path),
        profile="gemma4-report",
        gpqa_task="gpqa_diamond_cot_n_shot",
    )
    commands = build_ood_commands(config, lm_eval_executable="/venv/bin/lm_eval")
    assert [command[command.index("--tasks") + 1] for command in commands] == [
        "mmlu_pro",
        "gpqa_diamond_cot_n_shot",
        "gemma4_mmmlu14k",
    ]
    assert all(command[command.index("--num_fewshot") + 1] == "5" for command in commands)
    assert all("--limit" not in command for command in commands)
    assert all("--apply_chat_template" not in command for command in commands)
    assert "--include_path" in commands[-1]
    assert "--include_path" not in commands[0]


def test_four_gpu_partition_is_complete_and_balanced() -> None:
    entries = [
        {"name": name, "total_requests": requests}
        for name, requests in (
            ("id_validation_full", 3_200),
            ("math500", 4_000),
            ("gsm8k", 2_638),
            ("olympiadbench", 2_696),
            ("minervamath", 2_176),
            ("aime2025", 3_840),
            ("aime2026", 3_840),
        )
    ]
    shards = partition_datasets(entries, 4)
    assert len(shards) == 4
    assert {entry["name"] for shard in shards for entry in shard} == {entry["name"] for entry in entries}
    totals = [sum(entry["total_requests"] for entry in shard) for shard in shards]
    assert max(totals) - min(totals) <= 3_336


def test_clean_id_metrics_are_derived_without_generation(tmp_path: Path) -> None:
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    model = ModelSpec("model", "/models/model", "a" * 64)
    source_path = trace_dir / "model__id_validation_full.jsonl"
    rows = []
    for uid in ("keep", "drop"):
        for sample_index in range(2):
            rows.append(
                {
                    "dataset": "id_validation_full",
                    "uid": uid,
                    "sample_index": sample_index,
                    "acc": uid == "keep",
                    "answer_class": f"answer:{uid}",
                    "answer_class_method": "test",
                    "sequence_entropy": 1.0,
                    "token_entropy_sum": 2.0,
                    "token_entropy_count": 2,
                }
            )
    source_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    clean_path = tmp_path / "clean.parquet"
    import pandas as pd

    pd.DataFrame([{"uid": "keep"}]).to_parquet(clean_path, index=False)
    result = derive_clean_id_result(
        model=model,
        full_entry={"name": "id_validation_full"},
        clean_entry={
            "name": "id_validation_clean",
            "output_path": str(clean_path),
            "unique_questions": 1,
            "samples_per_question": 2,
            "total_requests": 2,
        },
        trace_dir=trace_dir,
    )
    assert result["n_questions"] == 1
    assert result["mean@k"] == 100.0
    assert result["trace_source"] == str(source_path)
    assert not (trace_dir / "model__id_validation_clean.jsonl").exists()


def test_aime_rows_record_the_pinned_train_split() -> None:
    rows = convert_hf_rows("aime2025", [{"problem_idx": 1, "problem": "1+1?", "answer": "2"}])
    assert rows[0]["extra_info"]["split"] == "train"


def test_math_worker_exception_is_recorded_as_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    task = math_scheduler.MathTask(
        task_id="model__shard_00",
        model=ModelSpec("model", "/models/model", "a" * 64),
        datasets=({"name": "math500", "total_requests": 1},),
        total_requests=1,
        metrics_path=tmp_path / "metrics.json",
        trace_dir=tmp_path / "traces",
        log_path=tmp_path / "task.log",
        command=("fake-eval",),
    )

    def fail_run(*args: object, **kwargs: object) -> None:
        raise OSError("worker boom")

    monkeypatch.setattr(math_scheduler.subprocess, "run", fail_run)
    state_path = tmp_path / "state.json"
    with pytest.raises(RuntimeError, match="math evaluation tasks failed"):
        math_scheduler._run_workers([task], gpus=["0"], state_path=state_path, resume=False)
    state = json.loads(state_path.read_text())
    assert state["status"] == "failed"
    assert state["tasks"][task.task_id]["error"] == "OSError: worker boom"


def test_ood_artifact_validation_counts_results_and_logged_samples(tmp_path: Path) -> None:
    output_dir = tmp_path / "gpqa"
    run_dir = output_dir / "gpqa_diamond_cot_n_shot_5shot" / "model"
    run_dir.mkdir(parents=True)
    (output_dir / "ood_eval_manifest.json").write_text(
        json.dumps(
            {
                "config": {"model": "/models/model", "benchmarks": ["gpqa"]},
                "model_identity": {"model_identity_sha256": "a" * 64},
            }
        )
    )
    timestamp = "2026-07-31T00-00-00.000000"
    result_path = run_dir / f"results_{timestamp}.json"
    result_path.write_text(
        json.dumps(
            {
                "results": {"gpqa_diamond_cot_n_shot": {"exact_match,strict-match": 0.25}},
                "n-samples": {"gpqa_diamond_cot_n_shot": {"original": 198, "effective": 198}},
            }
        )
    )
    sample_path = run_dir / f"samples_gpqa_diamond_cot_n_shot_{timestamp}.jsonl"
    sample_path.write_text("{}\n" * 396)
    task = ood_scheduler.OODTask(
        task_id="model__gpqa",
        model=ModelSpec("model", "/models/model", "a" * 64),
        benchmark="gpqa",
        estimated_work=1,
        output_dir=output_dir,
        log_path=tmp_path / "task.log",
        completion_path=output_dir / "complete.json",
        command=("fake-eval",),
    )
    artifacts = ood_scheduler._validate_task_artifacts(task)
    assert artifacts["effective_samples"] == 198
    assert artifacts["logged_sample_rows"] == 396
    sample_path.write_text("{}\n" * 395)
    with pytest.raises(ValueError, match="logged sample count mismatch"):
        ood_scheduler._validate_task_artifacts(task)


def test_ood_worker_exception_is_recorded_as_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    task = ood_scheduler.OODTask(
        task_id="model__gpqa",
        model=ModelSpec("model", "/models/model", "a" * 64),
        benchmark="gpqa",
        estimated_work=1,
        output_dir=tmp_path / "gpqa",
        log_path=tmp_path / "task.log",
        completion_path=tmp_path / "gpqa" / "complete.json",
        command=("fake-eval",),
    )

    def fail_run(*args: object, **kwargs: object) -> None:
        raise OSError("worker boom")

    monkeypatch.setattr(ood_scheduler.subprocess, "run", fail_run)
    state_path = tmp_path / "state.json"
    with pytest.raises(RuntimeError, match="OOD tasks failed"):
        ood_scheduler._run_workers([task], gpus=["0"], state_path=state_path, resume=False)
    state = json.loads(state_path.read_text())
    assert state["status"] == "failed"
    assert state["tasks"][task.task_id]["error"] == "OSError: worker boom"

