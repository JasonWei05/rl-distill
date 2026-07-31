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

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

DATA_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATA_DIR))

import gemma4_distill_trace_schema as schema  # noqa: E402
import preflight_gemma4_topk_distill as preflight  # noqa: E402


@pytest.fixture(autouse=True)
def _use_tiny_registered_question_counts(monkeypatch):
    monkeypatch.setattr(preflight, "EXPECTED_QUESTIONS", {"train": 1, "validation": 1})


class FakeTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0
    unk_token_id = 3
    special_tokens_map = {"bos_token": "<bos>"}
    model_max_length = 12288
    padding_side = "right"
    truncation_side = "right"

    def __init__(self, extra_token: bool = False):
        self.extra_token = extra_token

    def get_vocab(self):
        vocabulary = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
        if self.extra_token:
            vocabulary["extra"] = 4
        return vocabulary


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


def _semantic(
    split: str,
    tokenizer_metadata: dict,
    *,
    question_count: int = 1,
    engine_dtype: str = "bfloat16",
):
    return {
        "schema_version": schema.SCHEMA_VERSION,
        "direction": "e4b_rl100_to_e2b",
        "split": split,
        "source_dataset": f"source-{split}",
        "source_dataset_sha256": ("1" if split == "train" else "2") * 64,
        "prompt_roster_sha256": "3" * 64,
        "source_row_count": question_count,
        "unique_question_count": question_count,
        "samples_per_question": 5,
        "topk_width": schema.TOPK_WIDTH,
        "global_seed": 42,
        "prompts_per_shard": 1,
        "row_group_rows": 1,
        "total_shards": 1,
        "teacher": {
            "model": "teacher",
            "revision": "a" * 40,
            "content_sha256": None,
            "content_sha256_kind": None,
            "model_identity_sha256": schema.hash_json({"model": "teacher", "revision": "a" * 40}),
        },
        "tokenizer": tokenizer_metadata,
        "chat_template": {
            "path": str(DATA_DIR / "gemma3_it_fewshot_math.jinja"),
            "sha256": schema.sha256_file(DATA_DIR / "gemma3_it_fewshot_math.jinja"),
        },
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
        "engine": {"dtype": engine_dtype, "tensor_parallel_size": 1},
        "generator": {"commit": "b" * 40, "repository_dirty": False, "source_sha256": "5" * 64},
        "environment_versions": {"vllm": "test"},
    }


def _build_fixture(
    tmp_path: Path,
    *,
    question_counts: dict[str, int] | None = None,
    question_overlap: bool = False,
    bad_train_sample_coverage: bool = False,
):
    question_counts = question_counts or {"train": 1, "validation": 1}
    tokenizer = FakeTokenizer()
    tokenizer_sha256, vocab_size = schema.tokenizer_fingerprint(tokenizer)
    tokenizer_metadata = {
        "model": "teacher-tokenizer",
        "revision": "c" * 40,
        "sha256": tokenizer_sha256,
        "vocab_size": vocab_size,
    }
    student_model = tmp_path / "student"
    student_model.mkdir()
    _write_json(student_model / "config.json", {"model_type": "gemma4", "hidden_size": 1536})
    _write_json(student_model / "processor_config.json", {"processor_class": "Gemma4Processor"})
    (student_model / "model-00001-of-00002.safetensors").write_bytes(b"student-shard-one")
    (student_model / "model-00002-of-00002.safetensors").write_bytes(b"student-shard-two")
    _write_json(
        student_model / "model.safetensors.index.json",
        {
            "metadata": {"total_size": 34},
            "weight_map": {
                "layer.0.weight": "model-00001-of-00002.safetensors",
                "layer.1.weight": "model-00002-of-00002.safetensors",
            },
        },
    )
    index_path = tmp_path / "dataset_index.json"
    split_indexes = {}
    semantics = {}
    total_rows = 0
    total_response_tokens = 0
    split_question_hashes: dict[str, set[str]] = {}
    for split in ("train", "validation"):
        split_dir = tmp_path / split
        split_dir.mkdir()
        shard_path = split_dir / f"traces-{split}-000000.parquet"
        question_count = question_counts[split]
        source_uids: list[str] = []
        question_hashes: list[str] = []
        question_texts: list[str] = []
        sample_indices: list[int] = []
        for question_index in range(question_count):
            source_uid = f"{split}-question-{question_index}"
            question_prefix = "train" if split == "train" or question_overlap else "validation"
            question_text = f"{question_prefix}-question-text-{question_index}"
            question_sha256 = schema.sha256_text(question_text)
            for sample_index in range(5):
                source_uids.append(source_uid)
                question_hashes.append(question_sha256)
                question_texts.append(question_text)
                sample_indices.append(sample_index)
        if split == "train" and bad_train_sample_coverage:
            assert question_count >= 2
            source_uids[4] = source_uids[5]
            question_hashes[4] = question_hashes[5]
            question_texts[4] = question_texts[5]
            sample_indices[4] = 5
        row_count = question_count * 5
        table = pa.table(
            {
                "source_uid": source_uids,
                "question_sha256": question_hashes,
                "question_text": question_texts,
                "sample_index": sample_indices,
                "response_length": [1] * row_count,
            }
        )
        pq.write_table(table, shard_path, row_group_size=row_count)
        split_question_hashes[split] = set(question_hashes)
        semantic = _semantic(split, tokenizer_metadata, question_count=question_count)
        semantics[split] = semantic
        run_config = {
            "manifest_version": schema.MANIFEST_VERSION,
            "schema_version": schema.SCHEMA_VERSION,
            "generation_config_sha256": schema.hash_json(semantic),
            "semantic_config": semantic,
        }
        run_config_path = split_dir / "run_config.json"
        _write_json(run_config_path, run_config)
        response_tokens = row_count
        total_rows += row_count
        total_response_tokens += response_tokens
        stats = {
            "row_count": row_count,
            "response_token_count": response_tokens,
            "empty_response_count": 0,
        }
        shard = {
            "shard_id": 0,
            "path": str(shard_path.relative_to(tmp_path)),
            "sha256": schema.sha256_file(shard_path),
            "size_bytes": shard_path.stat().st_size,
            "rows": row_count,
            "row_groups": pq.ParquetFile(shard_path).metadata.num_row_groups,
            "stats": stats,
        }
        split_indexes[split] = {
            "source_dataset": semantic["source_dataset"],
            "source_dataset_sha256": semantic["source_dataset_sha256"],
            "generation_config_sha256": run_config["generation_config_sha256"],
            "run_config_path": str(run_config_path.relative_to(tmp_path)),
            "run_config_sha256": schema.sha256_file(run_config_path),
            "question_count": question_count,
            "row_count": row_count,
            "complete": True,
            "missing_shard_ids": [],
            "stats": stats,
            "parquet_files": [str(shard_path.relative_to(tmp_path))],
            "shards": [shard],
        }
    common = preflight._common_generation_config(semantics["train"])
    question_overlap_hashes = sorted(split_question_hashes["train"].intersection(split_question_hashes["validation"]))
    index = {
        "manifest_version": schema.MANIFEST_VERSION,
        "schema_version": schema.SCHEMA_VERSION,
        "created_at": "2026-07-31T00:00:00+00:00",
        "experiment_sha256": schema.hash_json(common),
        "direction": common["direction"],
        "topk_width": schema.TOPK_WIDTH,
        "recommended_training_topk_validation_tolerance": schema.FP16_TOPK_MASS_TOLERANCE,
        "samples_per_question": 5,
        "decode_check_performed": True,
        "teacher": common["teacher"],
        "tokenizer": common["tokenizer"],
        "chat_template": common["chat_template"],
        "sampling": common["sampling"],
        "total_rows": total_rows,
        "total_response_tokens": total_response_tokens,
        "cross_split_question_text_overlap_count": len(question_overlap_hashes),
        "cross_split_question_text_overlap_sha256s": question_overlap_hashes,
        "splits": split_indexes,
    }
    index["dataset_index_sha256"] = schema.hash_json(index)
    _write_json(index_path, index)
    teacher_identity_sha256 = schema.hash_json(index["teacher"])
    student_identity_sha256 = preflight._student_identity_sha256(str(student_model), None)
    return {
        "index": index,
        "index_path": index_path,
        "semantics": semantics,
        "student_model": student_model,
        "student_identity_sha256": student_identity_sha256,
        "teacher_identity_sha256": teacher_identity_sha256,
        "tokenizer": tokenizer,
    }


def _rehash_index(fixture):
    index = fixture["index"]
    index.pop("dataset_index_sha256", None)
    index["dataset_index_sha256"] = schema.hash_json(index)
    _write_json(fixture["index_path"], index)


def _preflight_kwargs(fixture, *, tokenizer=None):
    return {
        "dataset_index": fixture["index_path"],
        "student_model": str(fixture["student_model"]),
        "student_revision": None,
        "expected_direction": fixture["index"]["direction"],
        "expected_teacher_identity_sha256": fixture["teacher_identity_sha256"],
        "expected_student_identity_sha256": fixture["student_identity_sha256"],
        "tokenizer_loader": lambda *_args: tokenizer or fixture["tokenizer"],
    }


def test_preflight_emits_hydra_lists_and_tolerance(tmp_path):
    fixture = _build_fixture(tmp_path)
    result = preflight.run_preflight(**_preflight_kwargs(fixture))
    values = dict(line.split("=", 1) for line in result.lines())
    assert json.loads(values["TRAIN_FILES_HYDRA"]) == [str((tmp_path / "train/traces-train-000000.parquet").resolve())]
    assert json.loads(values["VAL_FILES_HYDRA"]) == [
        str((tmp_path / "validation/traces-validation-000000.parquet").resolve())
    ]
    assert values["TOPK_WIDTH"] == "128"
    assert float(values["TOPK_VALIDATION_TOLERANCE"]) == schema.FP16_TOPK_MASS_TOLERANCE
    assert values["TEACHER_IDENTITY_SHA256"] == fixture["teacher_identity_sha256"]
    assert values["STUDENT_IDENTITY_SHA256"] == fixture["student_identity_sha256"]


def test_preflight_rejects_unexpected_direction(tmp_path):
    fixture = _build_fixture(tmp_path)
    kwargs = _preflight_kwargs(fixture)
    kwargs["expected_direction"] = "e2b_base_to_e4b"
    with pytest.raises(preflight.PreflightError, match="does not match expected direction"):
        preflight.run_preflight(**kwargs)


def test_preflight_rejects_unexpected_teacher_identity(tmp_path):
    fixture = _build_fixture(tmp_path)
    kwargs = _preflight_kwargs(fixture)
    kwargs["expected_teacher_identity_sha256"] = "f" * 64
    with pytest.raises(preflight.PreflightError, match="teacher identity does not match"):
        preflight.run_preflight(**kwargs)


def test_preflight_rejects_unexpected_student_identity(tmp_path):
    fixture = _build_fixture(tmp_path)
    kwargs = _preflight_kwargs(fixture)
    kwargs["expected_student_identity_sha256"] = "f" * 64
    with pytest.raises(preflight.PreflightError, match="student identity does not match"):
        preflight.run_preflight(**kwargs)


def test_local_student_identity_is_semantic_for_json_and_content_bound_for_shards(tmp_path):
    fixture = _build_fixture(tmp_path)
    student_model = fixture["student_model"]
    expected = fixture["student_identity_sha256"]
    config = json.loads((student_model / "config.json").read_text(encoding="utf-8"))
    (student_model / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    weight_index = json.loads((student_model / "model.safetensors.index.json").read_text(encoding="utf-8"))
    (student_model / "model.safetensors.index.json").write_text(
        json.dumps(weight_index, indent=4) + "\n", encoding="utf-8"
    )
    assert preflight._student_identity_sha256(str(student_model), None) == expected
    (student_model / "model-00002-of-00002.safetensors").write_bytes(b"changed-student-shard-two")
    assert preflight._student_identity_sha256(str(student_model), None) != expected


def test_local_student_identity_rejects_unindexed_safetensors(tmp_path):
    fixture = _build_fixture(tmp_path)
    (fixture["student_model"] / "unexpected.safetensors").write_bytes(b"unexpected")
    with pytest.raises(preflight.PreflightError, match="do not exactly match"):
        preflight._student_identity_sha256(str(fixture["student_model"]), None)


@pytest.mark.parametrize(
    "relative_path",
    ["config.json", "model.safetensors.index.json", "model-00001-of-00002.safetensors"],
)
def test_local_student_identity_rejects_escaping_symlinks(tmp_path, relative_path):
    fixture = _build_fixture(tmp_path)
    model_path = fixture["student_model"] / relative_path
    outside_path = tmp_path / "outside" / relative_path
    outside_path.parent.mkdir(parents=True, exist_ok=True)
    outside_path.write_bytes(model_path.read_bytes())
    model_path.unlink()
    model_path.symlink_to(outside_path)
    with pytest.raises(preflight.PreflightError, match="escapes the model directory"):
        preflight._student_identity_sha256(str(fixture["student_model"]), None)


def test_local_student_identity_allows_hf_snapshot_blob_symlinks(tmp_path):
    revision = "e" * 40
    repository_root = tmp_path / "models--example--student"
    snapshot = repository_root / "snapshots" / revision
    blobs = repository_root / "blobs"
    snapshot.mkdir(parents=True)
    blobs.mkdir()
    _write_json(blobs / "config-blob", {"model_type": "gemma4"})
    _write_json(blobs / "processor-blob", {"processor_class": "Gemma4Processor"})
    (blobs / "weights-blob").write_bytes(b"weights")
    (snapshot / "config.json").symlink_to("../../blobs/config-blob")
    (snapshot / "processor_config.json").symlink_to("../../blobs/processor-blob")
    (snapshot / "model.safetensors").symlink_to("../../blobs/weights-blob")
    identity = preflight._student_identity_sha256(str(snapshot), None)
    assert len(identity) == 64


def test_remote_student_identity_requires_immutable_revision(tmp_path):
    remote_model = f"example/nonlocal-{tmp_path.name}"
    revision = "d" * 40
    assert preflight._student_identity_sha256(remote_model, revision) == schema.hash_json(
        {"model": remote_model, "revision": revision}
    )
    with pytest.raises(preflight.PreflightError, match="student_revision"):
        preflight._student_identity_sha256(remote_model, None)
    with pytest.raises(preflight.PreflightError, match="immutable"):
        preflight._student_identity_sha256(remote_model, "main")
    assert preflight._student_identity_sha256(f"  {remote_model}  ", revision) == schema.hash_json(
        {"model": remote_model, "revision": revision}
    )


def test_preflight_rejects_index_self_hash_mismatch(tmp_path):
    fixture = _build_fixture(tmp_path)
    fixture["index"]["total_rows"] += 1
    _write_json(fixture["index_path"], fixture["index"])
    with pytest.raises(preflight.PreflightError, match="self-hash mismatch"):
        preflight.run_preflight(**_preflight_kwargs(fixture))


def test_preflight_rejects_corrupt_shard_with_same_size(tmp_path):
    fixture = _build_fixture(tmp_path)
    shard_path = tmp_path / "train/traces-train-000000.parquet"
    shard_path.write_bytes(b"X" * shard_path.stat().st_size)
    with pytest.raises(preflight.PreflightError, match="SHA256 mismatch"):
        preflight.run_preflight(**_preflight_kwargs(fixture))


def test_preflight_rejects_mixed_generation_configs(tmp_path):
    fixture = _build_fixture(tmp_path)
    validation_semantic = fixture["semantics"]["validation"]
    validation_semantic["engine"]["dtype"] = "float32"
    run_config_path = tmp_path / "validation/run_config.json"
    run_config = {
        "manifest_version": schema.MANIFEST_VERSION,
        "schema_version": schema.SCHEMA_VERSION,
        "generation_config_sha256": schema.hash_json(validation_semantic),
        "semantic_config": validation_semantic,
    }
    _write_json(run_config_path, run_config)
    split = fixture["index"]["splits"]["validation"]
    split["generation_config_sha256"] = run_config["generation_config_sha256"]
    split["run_config_sha256"] = schema.sha256_file(run_config_path)
    _rehash_index(fixture)
    with pytest.raises(preflight.PreflightError, match="mixed generation/teacher"):
        preflight.run_preflight(**_preflight_kwargs(fixture))


def test_preflight_rejects_run_config_file_hash_mismatch(tmp_path):
    fixture = _build_fixture(tmp_path)
    run_config_path = tmp_path / "train/run_config.json"
    run_config_path.write_text(run_config_path.read_text() + "\n", encoding="utf-8")
    with pytest.raises(preflight.PreflightError, match="run config SHA256 mismatch"):
        preflight.run_preflight(**_preflight_kwargs(fixture))


def test_preflight_rejects_student_tokenizer_mismatch(tmp_path):
    fixture = _build_fixture(tmp_path)
    with pytest.raises(preflight.PreflightError, match="student tokenizer does not match"):
        preflight.run_preflight(**_preflight_kwargs(fixture, tokenizer=FakeTokenizer(extra_token=True)))


def test_preflight_rejects_inflated_topk_tolerance(tmp_path):
    fixture = _build_fixture(tmp_path)
    fixture["index"]["recommended_training_topk_validation_tolerance"] = 0.5
    _rehash_index(fixture)
    with pytest.raises(preflight.PreflightError, match="must equal the schema constant"):
        preflight.run_preflight(**_preflight_kwargs(fixture))


def test_preflight_rejects_wrong_generation_sampling_contract(tmp_path):
    fixture = _build_fixture(tmp_path)
    common = preflight._common_generation_config(fixture["semantics"]["train"])
    common["sampling"]["max_response_tokens"] = 128
    with pytest.raises(preflight.PreflightError, match="max_response_tokens must be exactly 8192"):
        preflight._verify_generation_contract(common)


def test_preflight_rejects_dirty_generator_repository(tmp_path):
    fixture = _build_fixture(tmp_path)
    common = preflight._common_generation_config(fixture["semantics"]["train"])
    common["generator"]["repository_dirty"] = True
    with pytest.raises(preflight.PreflightError, match="clean repository"):
        preflight._verify_generation_contract(common)


def test_preflight_pins_registered_question_counts(tmp_path, monkeypatch):
    fixture = _build_fixture(tmp_path)
    monkeypatch.setattr(preflight, "EXPECTED_QUESTIONS", {"train": 2, "validation": 1})
    with pytest.raises(preflight.PreflightError, match="train question_count must be exactly 2"):
        preflight.run_preflight(**_preflight_kwargs(fixture))


def test_preflight_rejects_balanced_but_incomplete_sample_sets(tmp_path, monkeypatch):
    expected_questions = {"train": 2, "validation": 1}
    monkeypatch.setattr(preflight, "EXPECTED_QUESTIONS", expected_questions)
    fixture = _build_fixture(
        tmp_path,
        question_counts=expected_questions,
        bad_train_sample_coverage=True,
    )
    with pytest.raises(preflight.PreflightError, match="has sample indices"):
        preflight.run_preflight(**_preflight_kwargs(fixture))


def test_preflight_rejects_response_token_stats_not_backed_by_rows(tmp_path):
    fixture = _build_fixture(tmp_path)
    train_split = fixture["index"]["splits"]["train"]
    train_split["shards"][0]["stats"]["response_token_count"] += 1
    train_split["stats"]["response_token_count"] += 1
    fixture["index"]["total_response_tokens"] += 1
    _rehash_index(fixture)
    with pytest.raises(preflight.PreflightError, match="actual response-token sum"):
        preflight.run_preflight(**_preflight_kwargs(fixture))


@pytest.mark.parametrize("bad_value", [True, 5.0, "5", None, {}])
def test_preflight_rejects_non_integer_manifest_counters(tmp_path, bad_value):
    fixture = _build_fixture(tmp_path)
    fixture["index"]["splits"]["train"]["stats"]["response_token_count"] = bad_value
    _rehash_index(fixture)
    with pytest.raises(preflight.PreflightError, match="response_token_count must be a JSON integer"):
        preflight.run_preflight(**_preflight_kwargs(fixture))


def test_preflight_question_overlap_is_fail_closed(tmp_path):
    fixture = _build_fixture(tmp_path, question_overlap=True)
    kwargs = _preflight_kwargs(fixture)
    with pytest.raises(preflight.PreflightError, match="question-text overlaps"):
        preflight.run_preflight(**kwargs)
    preflight.run_preflight(**kwargs, allow_question_overlap=True)
