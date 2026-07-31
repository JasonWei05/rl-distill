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

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

MODULE_PATH = Path(__file__).with_name("rescore_gemma4_training_topk.py")
if not MODULE_PATH.is_file():
    MODULE_PATH = Path(__file__).resolve().parents[1] / "rescore_gemma4_training_topk.py"
SPEC = importlib.util.spec_from_file_location("rescorer", MODULE_PATH)
rescorer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rescorer)


class FakeBackbone:
    def __call__(self, *, input_ids, **_kwargs):
        positions = torch.arange(input_ids.shape[1], dtype=torch.float32).view(1, -1, 1)
        return SimpleNamespace(last_hidden_state=positions)


class FakeHead:
    def __init__(self, vocab_size=256):
        self.vocab = (torch.arange(vocab_size, dtype=torch.float32) / 1000).view(1, -1)

    def __call__(self, hidden):
        return hidden * self.vocab


class FakeConfig:
    def get_text_config(self):
        return SimpleNamespace(final_logit_softcapping=30.0)


class FakeModel:
    model = FakeBackbone()
    lm_head = FakeHead()
    config = FakeConfig()

    def __call__(self, *, input_ids, logits_to_keep, **kwargs):
        hidden = self.model(input_ids=input_ids, **kwargs).last_hidden_state
        logits = self.lm_head(hidden[:, logits_to_keep, :])
        logits = torch.tanh(logits / 30.0) * 30.0
        return SimpleNamespace(logits=logits)


class FakeTraceSchema:
    @staticmethod
    def hash_json(value):
        payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def sha256_file(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    @staticmethod
    def atomic_write_json(path, value):
        Path(path).write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _write_valid_source_and_overlay(tmp_path):
    trace_id = "trace-a"
    source_manifest = {"trace_ids_sha256": FakeTraceSchema.hash_json([trace_id])}
    source_manifest_sha256 = "b" * 64
    run_config = {
        "rescore_config_sha256": "c" * 64,
        "created_at": "2026-01-01T00:00:00+00:00",
        "semantic_config": {
            "source_dataset_index_sha256": "3" * 64,
            "source_experiment_sha256": "4" * 64,
            "vocab_size": 256,
            "target_engine": "hf_bf16_sdpa_full_forward",
            "target_model_identity": {"model_identity_sha256": "d" * 64},
            "dtype": "bfloat16",
            "attention_implementation": "sdpa",
            "final_logit_softcapping": 30.0,
        },
    }
    source = {
        "generation_config_sha256": "e" * 64,
        "trace_id": trace_id,
        "direction": "e2b_base_to_e4b",
        "split": "train",
        "source_dataset": "source.parquet",
        "source_dataset_sha256": "f" * 64,
        "source_uid": "uid-a",
        "question_sha256": "1" * 64,
        "prompt_index": 0,
        "sample_index": 0,
        "question_text": "question",
        "gold_answer": "answer",
        "strict_grade": 0.0,
        "strict_correct": False,
        "strict_prediction": "prediction",
        "response_text": "response",
        "vllm_response_text": "response",
        "prompt_token_ids": [2],
        "response_token_ids": [5],
        "input_ids": [2, 5],
        "response_mask": [0, 1],
        "prompt_length": 1,
        "response_length": 1,
        "shard_id": 0,
        "row_within_shard": 0,
        "teacher_model": "teacher",
        "teacher_revision": None,
        "teacher_content_sha256": "2" * 64,
        "sampling_parameters_json": "{}",
        "environment_versions_json": "{}",
    }
    source_path = tmp_path / "source.parquet"
    pq.write_table(pa.Table.from_pylist([source]), source_path)
    source_entry = {
        "sha256": FakeTraceSchema.sha256_file(source_path),
        "shard_id": 0,
        "rows": 1,
        "path": source_path.name,
    }
    source_index = {
        "dataset_index_sha256": run_config["semantic_config"]["source_dataset_index_sha256"],
        "experiment_sha256": run_config["semantic_config"]["source_experiment_sha256"],
    }
    ids = np.arange(128, dtype=np.int32).reshape(1, 128)
    logprobs = np.full((1, 128), -10.0, dtype=np.float16)
    sampled_logprobs = np.array([-10.0], dtype=np.float16)
    timestamp = "2026-01-02T00:00:00+00:00"
    row = rescorer.make_overlay_record(
        source,
        topk_ids=ids,
        topk_logprobs=logprobs,
        sampled_logprobs=sampled_logprobs,
        source_index=source_index,
        source_parquet_sha256=source_entry["sha256"],
        run_config=run_config,
        timestamp=timestamp,
    )
    parquet_path = tmp_path / "targets-train-000000.parquet"
    pq.write_table(pa.Table.from_pylist([row], schema=rescorer.overlay_schema()), parquet_path)
    manifest_path = parquet_path.with_suffix(".manifest.json")
    manifest = {
        "schema_version": rescorer.OVERLAY_SCHEMA_VERSION,
        "rescore_config_sha256": run_config["rescore_config_sha256"],
        "source_parquet_sha256": source_entry["sha256"],
        "source_manifest_sha256": source_manifest_sha256,
        "source_trace_ids_sha256": source_manifest["trace_ids_sha256"],
        "shard_id": 0,
        "row_count": 1,
        "parquet_file": parquet_path.name,
        "parquet_sha256": FakeTraceSchema.sha256_file(parquet_path),
        "parquet_size_bytes": parquet_path.stat().st_size,
        "ordered_trace_ids_sha256": FakeTraceSchema.hash_json([trace_id]),
        "stats": {"response_token_count": 1},
        "created_at": timestamp,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return SimpleNamespace(
        source=source,
        source_path=source_path,
        source_entry=source_entry,
        source_manifest=source_manifest,
        source_manifest_sha256=source_manifest_sha256,
        run_config=run_config,
        source_index=source_index,
        row=row,
        parquet_path=parquet_path,
        manifest_path=manifest_path,
        manifest=manifest,
    )


def test_causal_shift_contract():
    assert rescorer.shifted_prediction_positions([2, 7, 11, 13], [0, 0, 1, 1]) == [1, 2]


def test_causal_shift_rejects_first_token_response():
    try:
        rescorer.shifted_prediction_positions([2, 7], [1, 1])
    except ValueError as error:
        assert "preceding context" in str(error)
    else:
        raise AssertionError("expected ValueError")


def test_teacher_forced_scoring_uses_predecessor_hidden_and_storage_dtypes():
    ids, lp, sampled_lp = rescorer.score_topk(
        model=FakeModel(),
        input_ids=[2, 7, 11, 13],
        response_mask=[0, 0, 1, 1],
        topk_width=128,
        chunk_tokens=1,
        device="cpu",
    )
    assert ids.shape == (2, 128)
    assert lp.shape == ids.shape
    assert ids.dtype == np.int32
    assert lp.dtype == np.float16
    assert sampled_lp.dtype == np.float16
    # Response tokens at positions 2 and 3 use hidden states at 1 and 2.
    # Both are positive, so the highest vocabulary ID is top-1.
    assert ids[:, 0].tolist() == [255, 255]
    masses = rescorer.validate_stored_targets(
        ids,
        lp,
        vocab_size=256,
        response_token_ids=[11, 13],
        sampled_logprobs=sampled_lp,
    )
    assert np.all(masses <= 1.0 + rescorer.FP16_MASS_TOLERANCE)


def test_chunked_path_matches_native_full_forward_reference():
    kwargs = dict(
        model=FakeModel(),
        input_ids=[2, 7, 11, 13],
        response_mask=[0, 0, 1, 1],
        topk_width=128,
        device="cpu",
    )
    chunked = rescorer.score_topk(**kwargs, chunk_tokens=1)
    native = rescorer.score_topk_native_forward(**kwargs)
    for actual, expected in zip(chunked, native, strict=True):
        assert np.array_equal(actual, expected)


def test_overlay_schema_has_fixed_width_and_requested_storage_types():
    schema = rescorer.overlay_schema()
    ids_type = schema.field("teacher_topk_token_ids").type
    lp_type = schema.field("teacher_topk_logprobs").type
    sampled_type = schema.field("teacher_sampled_token_logprobs").type
    assert pa.types.is_list(ids_type)
    assert pa.types.is_fixed_size_list(ids_type.value_type)
    assert ids_type.value_type.list_size == 128
    assert pa.types.is_int32(ids_type.value_type.value_type)
    assert pa.types.is_float16(lp_type.value_type.value_type)
    assert pa.types.is_float16(sampled_type.value_type)


def test_overlap_fraction_uses_tokens_times_topk_denominator():
    old_ids = np.stack([np.arange(128), np.arange(128)])
    new_ids = np.stack([np.arange(128), np.concatenate([np.arange(64), np.arange(128, 192)])]).astype(np.int32)
    stats = rescorer.ShardStats()
    stats.update(np.array([0.9, 0.8]), new_ids, old_ids)
    result = stats.as_dict()
    assert result["training_vs_vllm_top128_overlap_fraction"] == pytest.approx(0.75)


def test_source_manifest_trace_id_binding_is_set_based_and_fail_closed():
    trace_ids = ["trace-b", "trace-a"]
    source_manifest = {"trace_ids_sha256": FakeTraceSchema.hash_json(sorted(trace_ids))}
    assert (
        rescorer.validate_source_trace_id_binding(
            trace_ids,
            source_manifest=source_manifest,
            trace_schema=FakeTraceSchema,
        )
        == source_manifest["trace_ids_sha256"]
    )
    with pytest.raises(ValueError, match="does not match source manifest"):
        rescorer.validate_source_trace_id_binding(
            ["trace-a", "trace-c"],
            source_manifest=source_manifest,
            trace_schema=FakeTraceSchema,
        )
    with pytest.raises(ValueError, match="duplicate trace IDs"):
        rescorer.validate_source_trace_id_binding(
            ["trace-a", "trace-a"],
            source_manifest=source_manifest,
            trace_schema=FakeTraceSchema,
        )


def test_output_root_must_be_disjoint_from_immutable_source(tmp_path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    sibling_output = tmp_path / "overlay"
    rescorer.require_disjoint_source_and_output(source_root, sibling_output)
    with pytest.raises(ValueError, match="disjoint from the immutable source bundle"):
        rescorer.require_disjoint_source_and_output(source_root, source_root)
    with pytest.raises(ValueError, match="disjoint from the immutable source bundle"):
        rescorer.require_disjoint_source_and_output(source_root, source_root / "overlay")
    with pytest.raises(ValueError, match="disjoint from the immutable source bundle"):
        rescorer.require_disjoint_source_and_output(source_root, tmp_path)


def test_resume_validation_rejects_source_manifest_trace_id_mismatch(tmp_path):
    fixture = _write_valid_source_and_overlay(tmp_path)
    rescorer.validate_output_shard(
        fixture.parquet_path,
        fixture.manifest_path,
        source_parquet_path=fixture.source_path,
        source_entry=fixture.source_entry,
        source_manifest=fixture.source_manifest,
        source_manifest_sha256=fixture.source_manifest_sha256,
        run_config=fixture.run_config,
        trace_schema=FakeTraceSchema,
    )
    tampered_source_manifest = {"trace_ids_sha256": FakeTraceSchema.hash_json(["trace-b"])}
    with pytest.raises(ValueError, match="source_trace_ids_sha256 mismatch"):
        rescorer.validate_output_shard(
            fixture.parquet_path,
            fixture.manifest_path,
            source_parquet_path=fixture.source_path,
            source_entry=fixture.source_entry,
            source_manifest=tampered_source_manifest,
            source_manifest_sha256=fixture.source_manifest_sha256,
            run_config=fixture.run_config,
            trace_schema=FakeTraceSchema,
        )


def test_resume_validation_rejects_tampered_copied_source_field(tmp_path):
    fixture = _write_valid_source_and_overlay(tmp_path)
    tampered = dict(fixture.row)
    tampered["question_text"] = "different question"
    pq.write_table(
        pa.Table.from_pylist([tampered], schema=rescorer.overlay_schema()),
        fixture.parquet_path,
    )
    fixture.manifest["parquet_sha256"] = FakeTraceSchema.sha256_file(fixture.parquet_path)
    fixture.manifest_path.write_text(json.dumps(fixture.manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="copied field question_text"):
        rescorer.validate_output_shard(
            fixture.parquet_path,
            fixture.manifest_path,
            source_parquet_path=fixture.source_path,
            source_entry=fixture.source_entry,
            source_manifest=fixture.source_manifest,
            source_manifest_sha256=fixture.source_manifest_sha256,
            run_config=fixture.run_config,
            trace_schema=FakeTraceSchema,
        )


def test_parity_receipt_is_mandatory_and_bound_to_the_run(tmp_path):
    run_config = {
        "rescore_config_sha256": "a" * 64,
        "semantic_config": {
            "target_model_identity": {"model_identity_sha256": "b" * 64},
            "target_engine": "hf_bf16_sdpa_full_forward",
        },
    }
    source_index = {"dataset_index_sha256": "c" * 64}
    with pytest.raises(ValueError, match="cannot read parity receipt"):
        rescorer.validate_parity_receipt(
            tmp_path,
            run_config=run_config,
            source_index=source_index,
            trace_schema=FakeTraceSchema,
        )
    receipt = rescorer.write_parity_receipt(
        tmp_path,
        run_config=run_config,
        source_index=source_index,
        checked_rows=8,
        parity_max_response_tokens=512,
        trace_schema=FakeTraceSchema,
    )
    assert (
        rescorer.validate_parity_receipt(
            tmp_path,
            run_config=run_config,
            source_index=source_index,
            trace_schema=FakeTraceSchema,
        )
        == receipt
    )
    with pytest.raises(ValueError, match="source_dataset_index_sha256 mismatch"):
        rescorer.validate_parity_receipt(
            tmp_path,
            run_config=run_config,
            source_index={"dataset_index_sha256": "d" * 64},
            trace_schema=FakeTraceSchema,
        )


def test_existing_run_config_preserves_original_timestamp(tmp_path):
    first = {
        "rescore_config_sha256": "a" * 64,
        "semantic_config": {"stable": True},
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    second = dict(first, created_at="2026-02-01T00:00:00+00:00")
    persisted = rescorer.ensure_run_config(tmp_path, first, FakeTraceSchema)
    resumed = rescorer.ensure_run_config(tmp_path, second, FakeTraceSchema)
    assert persisted == resumed == first


def test_finalize_is_idempotent(monkeypatch, tmp_path):
    source_root = tmp_path / "source"
    output_root = tmp_path / "overlay"
    output_root.mkdir()
    shards = {}
    splits = {}
    for index, split in enumerate(("train", "validation")):
        relative = Path(split) / "source.parquet"
        source_path = source_root / relative
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_bytes(b"source")
        entry = {
            "shard_id": 0,
            "path": relative.as_posix(),
            "manifest_path": (relative.parent / "source.manifest.json").as_posix(),
            "sha256": str(index + 1) * 64,
            "rows": 1,
        }
        shards[split] = entry
        splits[split] = {
            "row_count": 1,
            "generation_config_sha256": str(index + 3) * 64,
            "shards": [entry],
        }
    source_index = {
        "dataset_index_sha256": "5" * 64,
        "experiment_sha256": "6" * 64,
        "direction": "e2b_base_to_e4b",
        "splits": splits,
    }
    run_config = {
        "created_at": "2026-01-01T00:00:00+00:00",
        "rescore_config_sha256": "7" * 64,
        "semantic_config": {
            "target_model_identity": {"model_identity_sha256": "8" * 64},
            "target_engine": "hf_bf16_sdpa_full_forward",
        },
    }

    def fake_load_source_manifest(_root, entry, _trace_schema):
        return Path(entry["manifest_path"]), {"trace_ids_sha256": "9" * 64}, "a" * 64

    def fake_validate_output_shard(parquet_path, _manifest_path, *, source_entry, **_kwargs):
        split = parquet_path.parent.name
        return {
            "shard_id": source_entry["shard_id"],
            "parquet_sha256": ("b" if split == "train" else "c") * 64,
            "parquet_size_bytes": 100,
            "row_count": 1,
            "stats": {"response_token_count": 1},
            "source_parquet_sha256": source_entry["sha256"],
            "source_manifest_sha256": "a" * 64,
            "source_trace_ids_sha256": "9" * 64,
            "ordered_trace_ids_sha256": "d" * 64,
        }

    monkeypatch.setattr(rescorer, "load_source_manifest", fake_load_source_manifest)
    monkeypatch.setattr(rescorer, "validate_output_shard", fake_validate_output_shard)
    first = rescorer.finalize(
        output_root=output_root,
        source_root=source_root,
        source_index=source_index,
        run_config=run_config,
        trace_schema=FakeTraceSchema,
    )
    second = rescorer.finalize(
        output_root=output_root,
        source_root=source_root,
        source_index=source_index,
        run_config=run_config,
        trace_schema=FakeTraceSchema,
    )
    assert first == second
    assert first["created_at"] == run_config["created_at"]
