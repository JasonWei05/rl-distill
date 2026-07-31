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
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

DATA_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATA_DIR))

import gemma4_distill_trace_schema as schema  # noqa: E402
import preflight_gemma4_training_topk_overlay as preflight  # noqa: E402
import rescore_gemma4_training_topk as rescorer  # noqa: E402

DIRECTION = "e2b_base_to_e4b"
STUDENT_IDENTITY = "b" * 64
TOKENIZER_SHA256 = "c" * 64
VOCAB_SIZE = 256


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    schema.atomic_write_json(path, value)


def _source_record(
    *,
    split: str,
    trace_id: str,
    generation_config_sha256: str,
    teacher: dict,
) -> dict:
    question_text = f"{split} question"
    topk_ids = list(range(schema.TOPK_WIDTH))
    topk_logprobs = [-10.0] * schema.TOPK_WIDTH
    sampling = {
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
    }
    return {
        "schema_version": schema.SCHEMA_VERSION,
        "generation_config_sha256": generation_config_sha256,
        "trace_id": trace_id,
        "direction": DIRECTION,
        "split": split,
        "source_dataset": f"synthetic-{split}",
        "source_dataset_sha256": ("1" if split == "train" else "2") * 64,
        "source_uid": f"{split}-uid",
        "source_uid_original": f"{split}-uid-original",
        "question_sha256": schema.sha256_text(question_text),
        "prompt_index": 0,
        "sample_index": 0,
        "question_text": question_text,
        "gold_answer": "3",
        "strict_grade": 1.0,
        "strict_correct": True,
        "strict_prediction": "3",
        "teacher_model": teacher["model"],
        "teacher_revision": teacher["revision"],
        "teacher_content_sha256": teacher["content_sha256"],
        "tokenizer_model": "tokenizer",
        "tokenizer_revision": None,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "tokenizer_vocab_size": VOCAB_SIZE,
        "chat_template_path": "/template.jinja",
        "chat_template_sha256": "d" * 64,
        "global_seed": 42,
        "sampling_seed": 17,
        "sampling_parameters_json": json.dumps(sampling, sort_keys=True, separators=(",", ":")),
        "prompt_token_ids": [1, 2],
        "response_token_ids": [3],
        "input_ids": [1, 2, 3],
        "response_mask": [0, 0, 1],
        "teacher_topk_token_ids": [topk_ids],
        "teacher_topk_logprobs": [topk_logprobs],
        "sampled_token_ids": [3],
        "sampled_token_logprobs": [-10.0],
        "teacher_topk_rank_order": f"1..{schema.TOPK_WIDTH}",
        "prompt_length": 2,
        "response_length": 1,
        "finish_reason": "stop",
        "stop_reason": "<end_of_turn>",
        "matched_stop_string": "<end_of_turn>",
        "reached_max_response_tokens": False,
        "response_text": "3",
        "vllm_response_text": "3",
        "response_text_normalization": schema.RESPONSE_TEXT_NORMALIZATION,
        "shard_id": 0,
        "row_within_shard": 0,
        "generation_timestamp": "2026-07-31T00:00:00+00:00",
        "generator_commit": "e" * 40,
        "generator_source_sha256": "f" * 64,
        "environment_versions_json": "{}",
    }


def _build_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source_root = tmp_path / "source"
    overlay_root = tmp_path / "overlay"
    source_root.mkdir()
    overlay_root.mkdir()
    teacher = {
        "model": "teacher",
        "revision": None,
        "content_sha256": "a" * 64,
        "content_sha256_kind": "single_model_safetensors_sha256",
        "model_identity_sha256": "9" * 64,
    }
    target_identity = {
        "kind": "local_hf_safetensors_v1",
        "model_identity_sha256": teacher["model_identity_sha256"],
        "weight_content_sha256": teacher["content_sha256"],
        "weight_content_kind": teacher["content_sha256_kind"],
    }
    experiment_sha256 = "8" * 64
    source_splits = {}
    source_records = {}
    source_total_rows = 0
    source_total_tokens = 0
    for split in ("train", "validation"):
        split_dir = source_root / split
        split_dir.mkdir()
        generation_sha256 = ("6" if split == "train" else "7") * 64
        record = _source_record(
            split=split,
            trace_id=f"trace-{split}",
            generation_config_sha256=generation_sha256,
            teacher=teacher,
        )
        source_records[split] = record
        parquet_path = split_dir / f"traces-{split}-000000.parquet"
        pq.write_table(pa.Table.from_pylist([record], schema=schema.trace_arrow_schema()), parquet_path)
        manifest_path = parquet_path.with_suffix(".manifest.json")
        manifest = {
            "manifest_version": schema.MANIFEST_VERSION,
            "schema_version": schema.SCHEMA_VERSION,
            "split": split,
            "shard_id": 0,
            "row_count": 1,
            "parquet_file": parquet_path.name,
            "parquet_sha256": schema.sha256_file(parquet_path),
            "trace_ids_sha256": schema.hash_json([record["trace_id"]]),
        }
        _write_json(manifest_path, manifest)
        stats = {"row_count": 1, "response_token_count": 1, "empty_response_count": 0}
        source_splits[split] = {
            "generation_config_sha256": generation_sha256,
            "row_count": 1,
            "stats": stats,
            "shards": [
                {
                    "shard_id": 0,
                    "path": parquet_path.relative_to(source_root).as_posix(),
                    "manifest_path": manifest_path.relative_to(source_root).as_posix(),
                    "sha256": schema.sha256_file(parquet_path),
                    "size_bytes": parquet_path.stat().st_size,
                    "rows": 1,
                    "row_groups": pq.ParquetFile(parquet_path).metadata.num_row_groups,
                }
            ],
        }
        source_total_rows += 1
        source_total_tokens += 1
    source_index = {
        "manifest_version": schema.MANIFEST_VERSION,
        "schema_version": schema.SCHEMA_VERSION,
        "experiment_sha256": experiment_sha256,
        "direction": DIRECTION,
        "teacher": teacher,
        "tokenizer": {
            "model": "tokenizer",
            "revision": None,
            "sha256": TOKENIZER_SHA256,
            "vocab_size": VOCAB_SIZE,
        },
        "sampling": {"max_model_len": 12288},
        "total_rows": source_total_rows,
        "total_response_tokens": source_total_tokens,
        "splits": source_splits,
    }
    source_index["dataset_index_sha256"] = schema.hash_json(source_index)
    source_index_path = source_root / "dataset_index.json"
    _write_json(source_index_path, source_index)

    semantic = {
        "schema_version": rescorer.OVERLAY_SCHEMA_VERSION,
        "source_dataset_index_sha256": source_index["dataset_index_sha256"],
        "source_experiment_sha256": experiment_sha256,
        "source_direction": DIRECTION,
        "source_teacher": teacher,
        "target_model_identity": target_identity,
        "loader": preflight.EXPECTED_LOADER,
        "target_engine": preflight.EXPECTED_TARGET_ENGINE,
        "dtype": "bfloat16",
        "attention_implementation": "sdpa",
        "topk_width": rescorer.TOPK_WIDTH,
        "vocab_size": VOCAB_SIZE,
        "lm_head_chunk_tokens": 16,
        "max_sequence_tokens": 12288,
        "final_logit_softcapping": 30.0,
        "causal_alignment": preflight.EXPECTED_ALIGNMENT,
        "normalization": preflight.EXPECTED_NORMALIZATION,
        "storage": preflight.EXPECTED_STORAGE,
        "rescorer_source_sha256": schema.sha256_file(Path(rescorer.__file__)),
        "environment_versions": {"python": "test"},
    }
    run_config = {
        "manifest_version": rescorer.OVERLAY_MANIFEST_VERSION,
        "schema_version": rescorer.OVERLAY_SCHEMA_VERSION,
        "rescore_config_sha256": schema.hash_json(semantic),
        "semantic_config": semantic,
        "runtime": {
            "source_dataset_index": str(source_index_path),
            "model_path": "/models/teacher",
        },
        "created_at": "2026-07-31T00:00:00+00:00",
    }
    _write_json(overlay_root / "rescore_config.json", run_config)
    receipt = {
        "schema_version": 1,
        "passed_at": "2026-07-31T00:00:00+00:00",
        "rescore_config_sha256": run_config["rescore_config_sha256"],
        "source_dataset_index_sha256": source_index["dataset_index_sha256"],
        "target_model_identity_sha256": teacher["model_identity_sha256"],
        "target_engine": preflight.EXPECTED_TARGET_ENGINE,
        "checked_rows": 2,
        "eligibility_response_token_cap": 512,
        "comparison": preflight.EXPECTED_PARITY_COMPARISON,
    }
    receipt["parity_receipt_sha256"] = schema.hash_json(receipt)
    _write_json(overlay_root / rescorer.PARITY_RECEIPT_NAME, receipt)

    overlay_splits = {}
    overlay_records = {}
    for split in ("train", "validation"):
        source_record = source_records[split]
        ids = np.arange(rescorer.TOPK_WIDTH, dtype=np.int32).reshape(1, rescorer.TOPK_WIDTH)
        logprobs = np.full(ids.shape, -10.0, dtype=np.float16)
        sampled = np.asarray([-10.0], dtype=np.float16)
        overlay_record = rescorer.make_overlay_record(
            source_record,
            topk_ids=ids,
            topk_logprobs=logprobs,
            sampled_logprobs=sampled,
            source_index=source_index,
            source_parquet_sha256=source_splits[split]["shards"][0]["sha256"],
            run_config=run_config,
            timestamp="2026-07-31T00:00:00+00:00",
        )
        overlay_records[split] = overlay_record
        split_dir = overlay_root / split
        split_dir.mkdir()
        parquet_path = split_dir / f"targets-{split}-000000.parquet"
        pq.write_table(pa.Table.from_pylist([overlay_record], schema=rescorer.overlay_schema()), parquet_path)
        manifest_path = parquet_path.with_suffix(".manifest.json")
        source_entry = source_splits[split]["shards"][0]
        source_manifest_path = source_root / source_entry["manifest_path"]
        source_trace_ids_sha256 = schema.hash_json([source_record["trace_id"]])
        masses = np.exp(logprobs.astype(np.float32)).sum(axis=1, dtype=np.float64)
        stats = rescorer.ShardStats()
        stats.update(masses, ids, source_record["teacher_topk_token_ids"])
        manifest = {
            "manifest_version": rescorer.OVERLAY_MANIFEST_VERSION,
            "schema_version": rescorer.OVERLAY_SCHEMA_VERSION,
            "rescore_config_sha256": run_config["rescore_config_sha256"],
            "source_dataset_index_sha256": source_index["dataset_index_sha256"],
            "source_experiment_sha256": experiment_sha256,
            "source_parquet_file": source_entry["path"],
            "source_parquet_sha256": source_entry["sha256"],
            "source_manifest_sha256": schema.sha256_file(source_manifest_path),
            "source_trace_ids_sha256": source_trace_ids_sha256,
            "split": split,
            "shard_id": 0,
            "row_count": 1,
            "ordered_trace_ids_sha256": schema.hash_json([overlay_record["trace_id"]]),
            "parquet_file": parquet_path.name,
            "parquet_sha256": schema.sha256_file(parquet_path),
            "parquet_size_bytes": parquet_path.stat().st_size,
            "parquet_row_groups": pq.ParquetFile(parquet_path).metadata.num_row_groups,
            "stats": stats.as_dict(),
            "model_class": "Gemma4ForConditionalGeneration",
            "created_at": "2026-07-31T00:00:00+00:00",
        }
        _write_json(manifest_path, manifest)
        overlay_splits[split] = {
            "row_count": 1,
            "response_token_count": 1,
            "source_generation_config_sha256": source_splits[split]["generation_config_sha256"],
            "shards": [
                {
                    "shard_id": 0,
                    "path": parquet_path.relative_to(overlay_root).as_posix(),
                    "manifest_path": manifest_path.relative_to(overlay_root).as_posix(),
                    "sha256": manifest["parquet_sha256"],
                    "size_bytes": manifest["parquet_size_bytes"],
                    "rows": 1,
                    "response_tokens": 1,
                    "source_parquet_sha256": source_entry["sha256"],
                    "source_manifest_sha256": manifest["source_manifest_sha256"],
                    "source_trace_ids_sha256": source_trace_ids_sha256,
                    "ordered_trace_ids_sha256": manifest["ordered_trace_ids_sha256"],
                }
            ],
        }
    overlay_index = {
        "manifest_version": rescorer.OVERLAY_MANIFEST_VERSION,
        "schema_version": rescorer.OVERLAY_SCHEMA_VERSION,
        "created_at": run_config["created_at"],
        "rescore_config_sha256": run_config["rescore_config_sha256"],
        "source_dataset_index_sha256": source_index["dataset_index_sha256"],
        "source_experiment_sha256": experiment_sha256,
        "direction": DIRECTION,
        "target_model_identity": target_identity,
        "target_engine": preflight.EXPECTED_TARGET_ENGINE,
        "topk_width": rescorer.TOPK_WIDTH,
        "total_rows": 2,
        "total_response_tokens": 2,
        "splits": overlay_splits,
    }
    overlay_index["dataset_index_sha256"] = schema.hash_json(overlay_index)
    overlay_index_path = overlay_root / "dataset_index.json"
    _write_json(overlay_index_path, overlay_index)

    source_result = preflight.source_preflight.PreflightResult(
        train_files=(str((source_root / source_splits["train"]["shards"][0]["path"]).resolve()),),
        validation_files=(str((source_root / source_splits["validation"]["shards"][0]["path"]).resolve()),),
        topk_width=schema.TOPK_WIDTH,
        topk_validation_tolerance=schema.FP16_TOPK_MASS_TOLERANCE,
        dataset_index_sha256=source_index["dataset_index_sha256"],
        experiment_sha256=experiment_sha256,
        direction=DIRECTION,
        teacher_identity_sha256=schema.hash_json(teacher),
        student_identity_sha256=STUDENT_IDENTITY,
        student_tokenizer_sha256=TOKENIZER_SHA256,
    )
    monkeypatch.setattr(preflight.source_preflight, "run_preflight", lambda **_kwargs: source_result)
    return {
        "source_root": source_root,
        "source_index": source_index,
        "source_index_path": source_index_path,
        "overlay_root": overlay_root,
        "overlay_index": overlay_index,
        "overlay_index_path": overlay_index_path,
        "overlay_records": overlay_records,
        "run_config": run_config,
        "source_result": source_result,
        "teacher_identity_sha256": schema.hash_json(teacher),
    }


def _kwargs(fixture):
    return {
        "dataset_index": fixture["overlay_index_path"],
        "source_dataset_index": fixture["source_index_path"],
        "student_model": "/models/student",
        "student_revision": None,
        "expected_direction": DIRECTION,
        "expected_teacher_identity_sha256": fixture["teacher_identity_sha256"],
        "expected_student_identity_sha256": STUDENT_IDENTITY,
    }


def _rehash_index(index: dict, path: Path) -> None:
    index.pop("dataset_index_sha256", None)
    index["dataset_index_sha256"] = schema.hash_json(index)
    _write_json(path, index)


def _rewrite_overlay_record(fixture, split: str) -> None:
    record = fixture["overlay_records"][split]
    entry = fixture["overlay_index"]["splits"][split]["shards"][0]
    parquet_path = fixture["overlay_root"] / entry["path"]
    pq.write_table(pa.Table.from_pylist([record], schema=rescorer.overlay_schema()), parquet_path)
    manifest_path = fixture["overlay_root"] / entry["manifest_path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    parquet_sha256 = schema.sha256_file(parquet_path)
    ordered_sha256 = schema.hash_json([record["trace_id"]])
    manifest["parquet_sha256"] = parquet_sha256
    manifest["parquet_size_bytes"] = parquet_path.stat().st_size
    manifest["ordered_trace_ids_sha256"] = ordered_sha256
    _write_json(manifest_path, manifest)
    entry["sha256"] = parquet_sha256
    entry["size_bytes"] = parquet_path.stat().st_size
    entry["ordered_trace_ids_sha256"] = ordered_sha256
    _rehash_index(fixture["overlay_index"], fixture["overlay_index_path"])


def _publish_rescore_config_change(fixture) -> None:
    config = fixture["run_config"]
    config["rescore_config_sha256"] = schema.hash_json(config["semantic_config"])
    _write_json(fixture["overlay_root"] / "rescore_config.json", config)
    fixture["overlay_index"]["rescore_config_sha256"] = config["rescore_config_sha256"]
    _rehash_index(fixture["overlay_index"], fixture["overlay_index_path"])


def test_overlay_preflight_emits_only_verified_overlay_files(tmp_path, monkeypatch):
    fixture = _build_fixture(tmp_path, monkeypatch)

    result = preflight.run_preflight(**_kwargs(fixture))

    values = dict(line.split("=", 1) for line in result.lines())
    assert json.loads(values["TRAIN_FILES_HYDRA"]) == [
        str((fixture["overlay_root"] / "train/targets-train-000000.parquet").resolve())
    ]
    assert json.loads(values["VAL_FILES_HYDRA"]) == [
        str((fixture["overlay_root"] / "validation/targets-validation-000000.parquet").resolve())
    ]
    assert values["DATASET_INDEX_SHA256"] == fixture["overlay_index"]["dataset_index_sha256"]
    assert values["TEACHER_IDENTITY_SHA256"] == fixture["teacher_identity_sha256"]
    assert values["STUDENT_IDENTITY_SHA256"] == STUDENT_IDENTITY


def test_overlay_preflight_forwards_split_specific_source_contract(tmp_path, monkeypatch):
    fixture = _build_fixture(tmp_path, monkeypatch)
    received = {}

    def fake_source_preflight(**kwargs):
        received.update(kwargs)
        return fixture["source_result"]

    monkeypatch.setattr(preflight.source_preflight, "run_preflight", fake_source_preflight)
    preflight.run_preflight(
        **_kwargs(fixture),
        expected_questions={"train": 9723, "validation": 128},
        expected_samples_per_question={"train": 5, "validation": 1},
    )

    assert received["expected_questions"] == {"train": 9723, "validation": 128}
    assert received["expected_samples_per_question"] == {"train": 5, "validation": 1}


def test_overlay_preflight_cli_accepts_split_specific_source_contract():
    args = preflight.parse_args(
        [
            "--dataset-index",
            "/overlay/dataset_index.json",
            "--source-dataset-index",
            "/source/dataset_index.json",
            "--student-model",
            "/models/student",
            "--expected-direction",
            DIRECTION,
            "--expected-teacher-identity-sha256",
            "a" * 64,
            "--expected-student-identity-sha256",
            "b" * 64,
            "--expected-train-questions",
            "9723",
            "--expected-validation-questions",
            "128",
            "--expected-train-samples-per-question",
            "5",
            "--expected-validation-samples-per-question",
            "1",
        ]
    )

    assert args.expected_train_questions == 9723
    assert args.expected_validation_questions == 128
    assert args.expected_train_samples_per_question == 5
    assert args.expected_validation_samples_per_question == 1


def test_overlay_preflight_rejects_overlay_index_self_hash_mismatch(tmp_path, monkeypatch):
    fixture = _build_fixture(tmp_path, monkeypatch)
    fixture["overlay_index"]["total_rows"] += 1
    _write_json(fixture["overlay_index_path"], fixture["overlay_index"])

    with pytest.raises(preflight.OverlayPreflightError, match="overlay dataset index self-hash mismatch"):
        preflight.run_preflight(**_kwargs(fixture))


def test_overlay_preflight_rejects_source_index_self_hash_mismatch(tmp_path, monkeypatch):
    fixture = _build_fixture(tmp_path, monkeypatch)
    fixture["source_index"]["total_rows"] += 1
    _write_json(fixture["source_index_path"], fixture["source_index"])

    with pytest.raises(preflight.OverlayPreflightError, match="source dataset index self-hash mismatch"):
        preflight.run_preflight(**_kwargs(fixture))


def test_overlay_preflight_rejects_rescore_config_semantic_tampering(tmp_path, monkeypatch):
    fixture = _build_fixture(tmp_path, monkeypatch)
    fixture["run_config"]["semantic_config"]["dtype"] = "float32"
    _write_json(fixture["overlay_root"] / "rescore_config.json", fixture["run_config"])

    with pytest.raises(preflight.OverlayPreflightError, match="rescore semantic hash mismatch"):
        preflight.run_preflight(**_kwargs(fixture))


def test_overlay_preflight_rejects_self_consistent_trace_id_rebinding(tmp_path, monkeypatch):
    fixture = _build_fixture(tmp_path, monkeypatch)
    fixture["overlay_records"]["train"]["trace_id"] = "different-trace"
    _rewrite_overlay_record(fixture, "train")

    with pytest.raises(preflight.OverlayPreflightError, match="copied field trace_id does not match"):
        preflight.run_preflight(**_kwargs(fixture))


def test_overlay_preflight_rejects_self_consistent_invalid_topk_tensor(tmp_path, monkeypatch):
    fixture = _build_fixture(tmp_path, monkeypatch)
    ids = fixture["overlay_records"]["train"]["teacher_topk_token_ids"][0]
    ids[1] = ids[0]
    _rewrite_overlay_record(fixture, "train")

    with pytest.raises(preflight.OverlayPreflightError, match="duplicate token ID"):
        preflight.run_preflight(**_kwargs(fixture))


def test_overlay_preflight_rejects_incomplete_split_set(tmp_path, monkeypatch):
    fixture = _build_fixture(tmp_path, monkeypatch)
    del fixture["overlay_index"]["splits"]["validation"]
    _rehash_index(fixture["overlay_index"], fixture["overlay_index_path"])

    with pytest.raises(preflight.OverlayPreflightError, match="exactly complete train and validation"):
        preflight.run_preflight(**_kwargs(fixture))


def test_overlay_preflight_rejects_target_that_is_not_exact_source_teacher(tmp_path, monkeypatch):
    fixture = _build_fixture(tmp_path, monkeypatch)
    semantic = fixture["run_config"]["semantic_config"]
    semantic["target_model_identity"]["model_identity_sha256"] = "0" * 64
    fixture["overlay_index"]["target_model_identity"] = semantic["target_model_identity"]
    _publish_rescore_config_change(fixture)

    with pytest.raises(preflight.OverlayPreflightError, match="not the exact source teacher"):
        preflight.run_preflight(**_kwargs(fixture))


def test_overlay_preflight_rejects_student_tokenizer_mismatch(tmp_path, monkeypatch):
    fixture = _build_fixture(tmp_path, monkeypatch)
    bad_result = fixture["source_result"].__class__(
        **{
            **fixture["source_result"].__dict__,
            "student_tokenizer_sha256": "0" * 64,
        }
    )
    monkeypatch.setattr(preflight.source_preflight, "run_preflight", lambda **_kwargs: bad_result)

    with pytest.raises(preflight.OverlayPreflightError, match="student tokenizer identity"):
        preflight.run_preflight(**_kwargs(fixture))


def test_overlay_preflight_requires_current_reviewed_rescorer_source(tmp_path, monkeypatch):
    fixture = _build_fixture(tmp_path, monkeypatch)
    fixture["run_config"]["semantic_config"]["rescorer_source_sha256"] = "0" * 64
    _publish_rescore_config_change(fixture)

    with pytest.raises(preflight.OverlayPreflightError, match="current reviewed rescorer source"):
        preflight.run_preflight(**_kwargs(fixture))


def test_overlay_preflight_requires_registered_gemma4_softcap(tmp_path, monkeypatch):
    fixture = _build_fixture(tmp_path, monkeypatch)
    fixture["run_config"]["semantic_config"]["final_logit_softcapping"] = 29.0
    _publish_rescore_config_change(fixture)

    with pytest.raises(preflight.OverlayPreflightError, match="must be exactly 30.0"):
        preflight.run_preflight(**_kwargs(fixture))


def test_overlay_preflight_binds_row_timestamp_to_manifest(tmp_path, monkeypatch):
    fixture = _build_fixture(tmp_path, monkeypatch)
    fixture["overlay_records"]["train"]["rescoring_timestamp"] = "2026-08-01T00:00:00+00:00"
    _rewrite_overlay_record(fixture, "train")

    with pytest.raises(preflight.OverlayPreflightError, match="constant rescoring_timestamp mismatch"):
        preflight.run_preflight(**_kwargs(fixture))


def test_overlay_preflight_requires_canonical_manifest_sibling(tmp_path, monkeypatch):
    fixture = _build_fixture(tmp_path, monkeypatch)
    entry = fixture["overlay_index"]["splits"]["train"]["shards"][0]
    canonical = fixture["overlay_root"] / entry["manifest_path"]
    alternate = canonical.with_name("alternate.manifest.json")
    alternate.write_bytes(canonical.read_bytes())
    entry["manifest_path"] = alternate.relative_to(fixture["overlay_root"]).as_posix()
    _rehash_index(fixture["overlay_index"], fixture["overlay_index_path"])

    with pytest.raises(preflight.OverlayPreflightError, match="canonical parquet sibling"):
        preflight.run_preflight(**_kwargs(fixture))
