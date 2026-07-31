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

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

DATA_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATA_DIR))

import audit_gemma4_cross_engine_topk as audit  # noqa: E402
import gemma4_distill_trace_schema as schema  # noqa: E402


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _semantic(split: str, teacher: dict) -> dict:
    return {
        "schema_version": schema.SCHEMA_VERSION,
        "direction": "e4b_rl100_to_e2b",
        "split": split,
        "source_dataset": f"source-{split}",
        "source_dataset_sha256": ("1" if split == "train" else "2") * 64,
        "prompt_roster_sha256": "3" * 64,
        "source_row_count": 2,
        "unique_question_count": 2,
        "samples_per_question": 5,
        "topk_width": schema.TOPK_WIDTH,
        "global_seed": 42,
        "prompts_per_shard": 2,
        "row_group_rows": 1,
        "total_shards": 1,
        "teacher": teacher,
        "tokenizer": {
            "model": "tokenizer",
            "revision": None,
            "sha256": "4" * 64,
            "vocab_size": 262144,
        },
        "chat_template": {"path": "/template.jinja", "sha256": "5" * 64},
        "sampling": {
            "temperature": 1.0,
            "top_p": 1.0,
            "sampling_top_k": -1,
            "max_prompt_tokens": 4096,
            "max_response_tokens": 8192,
            "max_model_len": 12288,
            "stop": ["<end_of_turn>", "<start_of_turn>"],
            "include_stop_str_in_output": False,
            "skip_special_tokens": False,
            "logprobs": schema.TOPK_WIDTH,
        },
        "engine": {"dtype": "bfloat16", "tensor_parallel_size": 1},
        "generator": {
            "commit": "6" * 40,
            "repository_dirty": False,
            "source_sha256": "7" * 64,
        },
        "environment_versions": {"vllm": "test"},
    }


def _index_record(split: str, row_index: int, response_length: int) -> dict:
    return {
        "trace_id": f"{row_index + (0 if split == 'train' else 100):064x}",
        "split": split,
        "source_uid": f"{split}-{row_index}",
        "sample_index": row_index % 5,
        "prompt_length": 2,
        "response_length": response_length,
        "shard_id": 0,
        "row_within_shard": row_index,
    }


def _payload_record(split: str, row_index: int, response_length: int) -> dict:
    index_record = _index_record(split, row_index, response_length)
    response_ids = [200 + row_index] * response_length
    topk_ids = [list(range(schema.TOPK_WIDTH)) for _ in response_ids]
    topk_logprobs = [[-math.log(schema.TOPK_WIDTH)] * schema.TOPK_WIDTH for _ in response_ids]
    return {
        "schema_version": schema.SCHEMA_VERSION,
        "generation_config_sha256": "8" * 64,
        **index_record,
        "direction": "e4b_rl100_to_e2b",
        "source_dataset": f"source-{split}",
        "source_dataset_sha256": ("1" if split == "train" else "2") * 64,
        "source_uid_original": index_record["source_uid"],
        "question_sha256": "9" * 64,
        "prompt_index": row_index,
        "question_text": f"question-{split}-{row_index}",
        "gold_answer": "1",
        "strict_grade": 1.0,
        "strict_correct": True,
        "strict_prediction": "1",
        "teacher_model": "/teacher",
        "teacher_revision": None,
        "teacher_content_sha256": "a" * 64,
        "tokenizer_model": "/teacher",
        "tokenizer_revision": None,
        "tokenizer_sha256": "4" * 64,
        "tokenizer_vocab_size": 262144,
        "chat_template_path": "/template.jinja",
        "chat_template_sha256": "5" * 64,
        "global_seed": 42,
        "sampling_seed": 1,
        "sampling_parameters_json": "{}",
        "prompt_token_ids": [1, 2],
        "response_token_ids": response_ids,
        "input_ids": [1, 2, *response_ids],
        "response_mask": [0, 0, *([1] * response_length)],
        "teacher_topk_token_ids": topk_ids,
        "teacher_topk_logprobs": topk_logprobs,
        "sampled_token_ids": response_ids,
        "sampled_token_logprobs": [-1.0] * response_length,
        "teacher_topk_rank_order": f"1..{schema.TOPK_WIDTH}",
        "finish_reason": "stop",
        "stop_reason": "<end_of_turn>",
        "matched_stop_string": "<end_of_turn>",
        "reached_max_response_tokens": False,
        "response_text": "response",
        "vllm_response_text": "response",
        "response_text_normalization": schema.RESPONSE_TEXT_NORMALIZATION,
        "generation_timestamp": "2026-07-31T00:00:00+00:00",
        "generator_commit": "6" * 40,
        "generator_source_sha256": "7" * 64,
        "environment_versions_json": "{}",
    }


def _build_contract(tmp_path: Path, monkeypatch) -> tuple[Path, Path, Path, dict]:
    trace_root = tmp_path / "traces"
    model_root = tmp_path / "model"
    model_root.mkdir()
    teacher = {
        "model": "/teacher",
        "revision": None,
        "content_sha256": "a" * 64,
        "content_sha256_kind": "single_model_safetensors_sha256",
        "model_identity_sha256": "b" * 64,
    }
    split_indexes = {}
    for split in audit.SPLITS:
        split_dir = trace_root / split
        split_dir.mkdir(parents=True)
        shard_path = split_dir / f"traces-{split}-000000.parquet"
        records = [
            _payload_record(split, 0, 2),
            _payload_record(split, 1, 7),
        ]
        table = pa.Table.from_pylist(records, schema=schema.trace_arrow_schema())
        pq.write_table(table, shard_path, row_group_size=1)
        semantic = _semantic(split, teacher)
        run_config = {
            "manifest_version": schema.MANIFEST_VERSION,
            "schema_version": schema.SCHEMA_VERSION,
            "generation_config_sha256": schema.hash_json(semantic),
            "semantic_config": semantic,
        }
        run_config_path = split_dir / "run_config.json"
        _write_json(run_config_path, run_config)
        relative_shard = shard_path.relative_to(trace_root).as_posix()
        split_indexes[split] = {
            "complete": True,
            "missing_shard_ids": [],
            "generation_config_sha256": run_config["generation_config_sha256"],
            "run_config_path": run_config_path.relative_to(trace_root).as_posix(),
            "run_config_sha256": schema.sha256_file(run_config_path),
            "parquet_files": [relative_shard],
            "shards": [
                {
                    "shard_id": 0,
                    "path": relative_shard,
                    "sha256": schema.sha256_file(shard_path),
                }
            ],
        }
    index = {
        "manifest_version": schema.MANIFEST_VERSION,
        "schema_version": schema.SCHEMA_VERSION,
        "experiment_sha256": "c" * 64,
        "direction": "e4b_rl100_to_e2b",
        "topk_width": schema.TOPK_WIDTH,
        "teacher": teacher,
        "tokenizer": {"vocab_size": 262144},
        "splits": split_indexes,
    }
    index["dataset_index_sha256"] = schema.hash_json(index)
    index_path = trace_root / "dataset_index.json"
    _write_json(index_path, index)
    monkeypatch.setattr(
        audit,
        "inspect_local_hf_model",
        lambda _path: SimpleNamespace(
            weight_content_sha256="a" * 64,
            weight_content_kind="single_model_safetensors_sha256",
            model_identity_sha256="b" * 64,
        ),
    )
    return trace_root, model_root, index_path, index


def test_module_import_is_cpu_only():
    assert "torch" not in audit.__dict__
    assert "transformers" not in audit.__dict__


def test_dataset_contract_binds_expected_locks_and_model(tmp_path, monkeypatch):
    trace_root, model_root, index_path, index = _build_contract(tmp_path, monkeypatch)
    contract = audit.load_dataset_contract(
        trace_root=trace_root,
        dataset_index=index_path,
        model_root=model_root,
        expected_dataset_index_sha256=index["dataset_index_sha256"],
        expected_teacher_identity_sha256=schema.hash_json(index["teacher"]),
    )
    assert contract["index_sha256"] == index["dataset_index_sha256"]
    assert contract["teacher_identity_sha256"] == schema.hash_json(index["teacher"])
    assert set(contract["semantic_configs"]) == set(audit.SPLITS)


def test_dataset_contract_rejects_expected_lock_mismatch(tmp_path, monkeypatch):
    trace_root, model_root, index_path, index = _build_contract(tmp_path, monkeypatch)
    with pytest.raises(audit.CrossEngineAuditError, match="expected-dataset-index"):
        audit.load_dataset_contract(
            trace_root=trace_root,
            dataset_index=index_path,
            model_root=model_root,
            expected_dataset_index_sha256="d" * 64,
            expected_teacher_identity_sha256=schema.hash_json(index["teacher"]),
        )


def test_dataset_contract_rejects_path_escape(tmp_path, monkeypatch):
    trace_root, model_root, index_path, index = _build_contract(tmp_path, monkeypatch)
    outside = tmp_path / "outside.parquet"
    outside.write_bytes(b"outside")
    index["splits"]["train"]["parquet_files"] = ["../outside.parquet"]
    index["splits"]["train"]["shards"][0]["path"] = "../outside.parquet"
    del index["dataset_index_sha256"]
    index["dataset_index_sha256"] = schema.hash_json(index)
    _write_json(index_path, index)
    with pytest.raises(audit.CrossEngineAuditError, match="escapes the trace root"):
        audit.load_dataset_contract(
            trace_root=trace_root,
            dataset_index=index_path,
            model_root=model_root,
            expected_dataset_index_sha256=index["dataset_index_sha256"],
            expected_teacher_identity_sha256=schema.hash_json(index["teacher"]),
        )


def test_global_stratification_and_row_group_loading(tmp_path, monkeypatch):
    trace_root, model_root, index_path, index = _build_contract(tmp_path, monkeypatch)
    contract = audit.load_dataset_contract(
        trace_root=trace_root,
        dataset_index=index_path,
        model_root=model_root,
        expected_dataset_index_sha256=index["dataset_index_sha256"],
        expected_teacher_identity_sha256=schema.hash_json(index["teacher"]),
    )
    candidates = audit.scan_candidates(contract["shards"])
    selected = audit.stratified_selection(candidates, per_split=2)
    assert [row["response_length"] for row in selected] == [2, 7, 2, 7]
    audit.verify_registered_shards(contract["shards"])
    rows = audit.load_selected_rows(selected)
    assert [row["response_length"] for row in rows] == [2, 7, 2, 7]
    assert all(row["registered_shard_sha256"] for row in rows)


def test_registered_shard_verification_covers_unselected_files(tmp_path, monkeypatch):
    trace_root, model_root, index_path, index = _build_contract(tmp_path, monkeypatch)
    contract = audit.load_dataset_contract(
        trace_root=trace_root,
        dataset_index=index_path,
        model_root=model_root,
        expected_dataset_index_sha256=index["dataset_index_sha256"],
        expected_teacher_identity_sha256=schema.hash_json(index["teacher"]),
    )
    validation_path = Path(contract["shards"]["validation"][0]["path"])
    validation_path.write_bytes(validation_path.read_bytes() + b"tampered")
    with pytest.raises(audit.CrossEngineAuditError, match="registered shard SHA256 mismatch"):
        audit.verify_registered_shards(contract["shards"])


def test_selected_positions_include_exact_endpoints_without_duplicates():
    positions = audit.selected_positions(8192, 64)
    assert len(positions) == len(set(positions)) == 64
    assert positions[0] == 0
    assert positions[-1] == 8191
    assert audit.selected_positions(3, 64) == [0, 1, 2]


def test_compare_topk_position_perfect_and_tie_safe():
    ids = list(range(schema.TOPK_WIDTH))
    logprobs = [-math.log(schema.TOPK_WIDTH)] * schema.TOPK_WIDTH
    perfect = audit.compare_topk_position(
        stored_ids=ids,
        stored_logprobs=logprobs,
        reference_top_ids=ids,
        reference_top_logprobs=logprobs,
        reference_logprobs_on_stored=logprobs,
        stored_sampled_logprob=logprobs[0],
        reference_sampled_logprob=logprobs[0],
        top1_tie_logprob_tolerance=0.02,
    )
    assert perfect["top1_exact"] == 1
    assert perfect["topk_overlap_fraction"] == 1.0
    assert perfect["stored_support_probability_l1"] == pytest.approx(0.0)

    swapped = ids.copy()
    swapped[:2] = swapped[1], swapped[0]
    tied = audit.compare_topk_position(
        stored_ids=ids,
        stored_logprobs=logprobs,
        reference_top_ids=swapped,
        reference_top_logprobs=logprobs,
        reference_logprobs_on_stored=logprobs,
        stored_sampled_logprob=logprobs[0],
        reference_sampled_logprob=logprobs[0],
        top1_tie_logprob_tolerance=0.0,
    )
    assert tied["top1_exact"] == 0
    assert tied["top1_tie_safe"] == 1


def _passing_aggregate() -> dict:
    def metric(mean: float, *, p95: float | None = None, p99: float | None = None):
        return {
            "mean": mean,
            "p95": mean if p95 is None else p95,
            "p99": mean if p99 is None else p99,
        }

    return {
        "top1_tie_safe": metric(0.995),
        "top10_overlap_fraction": metric(0.98),
        "topk_overlap_fraction": metric(0.98),
        "stored_support_weighted_abs_logprob_delta": metric(0.02),
        "stored_support_probability_l1": metric(0.02),
        "sampled_token_abs_logprob_delta": metric(0.02, p95=0.1),
        "stored_only_topk_mass": metric(0.001, p99=0.004),
        "reference_only_topk_mass": metric(0.001, p99=0.004),
    }


def test_threshold_gate_passes_boundaries_and_reports_failure():
    aggregate = _passing_aggregate()
    passed = audit.evaluate_thresholds(
        aggregate,
        native_projection_max_abs=audit.DEFAULT_THRESHOLDS["native_vs_manual_projection_max_abs"],
        thresholds=audit.DEFAULT_THRESHOLDS,
    )
    assert passed["status"] == "pass"
    aggregate["topk_overlap_fraction"]["mean"] = 0.96
    failed = audit.evaluate_thresholds(
        aggregate,
        native_projection_max_abs=0.0,
        thresholds=audit.DEFAULT_THRESHOLDS,
    )
    assert failed["status"] == "fail"
    assert "topk_overlap_fraction_mean" in failed["failure_reasons"][0]


def test_output_path_must_stay_outside_model_and_trace_trees(tmp_path):
    trace_root = tmp_path / "traces"
    model_root = tmp_path / "model"
    trace_root.mkdir()
    model_root.mkdir()
    with pytest.raises(audit.CrossEngineAuditError, match="trace tree"):
        audit.validate_output_path(
            trace_root / "report.json",
            trace_root=trace_root,
            model_root=model_root,
        )
    assert (
        audit.validate_output_path(
            tmp_path / "reports" / "audit.json",
            trace_root=trace_root,
            model_root=model_root,
        )
        == (tmp_path / "reports" / "audit.json").resolve()
    )
