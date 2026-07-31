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

import gemma4_distill_trace_schema as schema  # noqa: E402
import gemma4_model_identity as model_identity  # noqa: E402
import generate_gemma4_distill_traces as generate  # noqa: E402
import validate_gemma4_distill_traces as validate  # noqa: E402


class FakeTokenizer:
    chat_template = None
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0
    unk_token_id = 3
    special_tokens_map = {"bos_token": "<bos>"}
    model_max_length = 12288
    padding_side = "right"
    truncation_side = "right"

    def get_vocab(self):
        return {f"token-{token_id}": token_id for token_id in range(2048)}

    def apply_chat_template(self, messages, **_kwargs):
        return messages[-1]["content"]

    def encode(self, _text, **_kwargs):
        return [1, 2, 3]

    def decode(self, token_ids, **_kwargs):
        return "|".join(str(token_id) for token_id in token_ids)


def _position_logprobs(sampled_token_id: int, *, sampled_rank: int = 300):
    entries = {
        1000 + rank: SimpleNamespace(logprob=-5.0 - rank / 100.0, rank=rank) for rank in range(1, schema.TOPK_WIDTH + 1)
    }
    if sampled_token_id not in entries:
        entries[sampled_token_id] = SimpleNamespace(logprob=-9.25, rank=sampled_rank)
    return entries


def _semantic(split: str, *, unique_questions: int, samples_per_question: int = 2):
    value = {
        "schema_version": schema.SCHEMA_VERSION,
        "direction": "e4b_rl100_to_e2b",
        "split": split,
        "source_dataset": f"synthetic-{split}",
        "source_dataset_sha256": ("1" if split == "train" else "2") * 64,
        "prompt_roster_sha256": "3" * 64,
        "source_row_count": unique_questions,
        "unique_question_count": unique_questions,
        "samples_per_question": samples_per_question,
        "topk_width": schema.TOPK_WIDTH,
        "global_seed": 42,
        "prompts_per_shard": unique_questions,
        "row_group_rows": 1,
        "total_shards": 1,
        "teacher": {"model": "teacher", "revision": "a" * 40, "content_sha256": None},
        "tokenizer": {"model": "tokenizer", "revision": "b" * 40, "sha256": "4" * 64, "vocab_size": 2048},
        "chat_template": {"path": "/template.jinja", "sha256": "5" * 64},
        "sampling": {
            "temperature": 1.0,
            "top_p": 1.0,
            "sampling_top_k": -1,
            "max_prompt_tokens": 4096,
            "max_response_tokens": 8192,
            "max_model_len": 12288,
            "stop": list(generate.DEFAULT_STOP_STRINGS),
            "include_stop_str_in_output": False,
            "skip_special_tokens": False,
            "logprobs": schema.TOPK_WIDTH,
        },
        "engine": {
            "tensor_parallel_size": 1,
            "dtype": "bfloat16",
            "gpu_memory_utilization": 0.9,
            "enforce_eager": False,
            "enable_prefix_caching": True,
            "distributed_executor_backend": None,
            "mm_encoder_attn_backend": None,
        },
        "generator": {"commit": "c" * 40, "repository_dirty": False, "source_sha256": "6" * 64},
        "environment_versions": {"python": "test", "pyarrow": pa.__version__, "vllm": "test"},
    }
    return value


def _run_config(split: str, *, unique_questions: int, samples_per_question: int = 2):
    semantic = _semantic(split, unique_questions=unique_questions, samples_per_question=samples_per_question)
    return {
        "manifest_version": schema.MANIFEST_VERSION,
        "schema_version": schema.SCHEMA_VERSION,
        "generation_config_sha256": schema.hash_json(semantic),
        "semantic_config": semantic,
        "input_parquet": "/synthetic.parquet",
        "created_at": "2026-07-30T00:00:00+00:00",
    }


def _output(prompt_ids: list[int], response_ids: list[int]):
    completion = SimpleNamespace(
        token_ids=response_ids,
        logprobs=[_position_logprobs(token_id) for token_id in response_ids],
        finish_reason="stop",
        stop_reason="<end_of_turn>",
        text="vllm text",
    )
    return SimpleNamespace(prompt_token_ids=prompt_ids, outputs=[completion])


def _record(
    *,
    run_config,
    split: str,
    question: str,
    source_uid: str,
    sample_index: int,
    row_within_shard: int,
):
    question_hash = schema.sha256_text(question)
    source = generate.SourcePrompt(
        prompt_index=row_within_shard,
        messages=[{"role": "user", "content": question}],
        question_text=question,
        gold_answer="7",
        source_uid=source_uid,
        source_uid_original=source_uid,
        question_sha256=question_hash,
    )
    prompt_ids = [1, 2, 3]
    request = generate.PreparedRequest(
        source=source,
        sample_index=sample_index,
        sampling_seed=generate.derive_sampling_seed(42, split, source_uid, sample_index),
        prompt_token_ids=prompt_ids,
    )
    return generate.build_trace_record(
        request=request,
        output=_output(prompt_ids, [250]),
        shard_id=0,
        row_within_shard=row_within_shard,
        run_config=run_config,
        tokenizer=FakeTokenizer(),
        strict_grade=1.0,
        strict_prediction="7",
        generation_timestamp="2026-07-30T00:00:00+00:00",
    )


def _write_split(
    split_dir: Path,
    split: str,
    questions: list[tuple[str, str]],
    samples_per_question: int = 2,
    *,
    declared_questions: int | None = None,
    total_shards: int = 1,
):
    split_dir.mkdir(parents=True)
    run_config = _run_config(
        split,
        unique_questions=declared_questions or len(questions),
        samples_per_question=samples_per_question,
    )
    run_config["semantic_config"]["total_shards"] = total_shards
    run_config["generation_config_sha256"] = schema.hash_json(run_config["semantic_config"])
    schema.atomic_write_json(split_dir / "run_config.json", run_config)
    records = []
    row_index = 0
    for source_uid, question in questions:
        for sample_index in range(samples_per_question):
            records.append(
                _record(
                    run_config=run_config,
                    split=split,
                    question=question,
                    source_uid=source_uid,
                    sample_index=sample_index,
                    row_within_shard=row_index,
                )
            )
            row_index += 1
    parquet_path = split_dir / f"traces-{split}-000000.parquet"
    generate._write_validated_shard(
        records,
        parquet_path=parquet_path,
        shard_id=0,
        prompt_start=0,
        prompt_end=len(questions),
        run_config=run_config,
        tokenizer=FakeTokenizer(),
        row_group_rows=1,
    )
    return run_config, parquet_path


def test_generator_import_does_not_import_vllm():
    assert "vllm" not in generate.__dict__


def test_tokenizer_fingerprint_ignores_vllm_cached_wrapper_class():
    class Tokenizer:
        bos_token_id = 1
        eos_token_id = 2
        pad_token_id = 0
        unk_token_id = 3
        special_tokens_map = {"bos_token": "<bos>"}
        model_max_length = 4096
        padding_side = "right"
        truncation_side = "right"

        def get_vocab(self):
            return {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}

    class CachedTokenizer(Tokenizer):
        truncation_side = "left"

    assert schema.tokenizer_fingerprint(Tokenizer()) == schema.tokenizer_fingerprint(CachedTokenizer())


def test_extract_ranked_topk_excludes_extra_sampled_token_and_sorts_by_rank():
    sampled_token_id = 250
    token_ids, logprobs, sampled_logprob = generate.extract_ranked_topk(
        _position_logprobs(sampled_token_id), sampled_token_id
    )
    assert len(token_ids) == schema.TOPK_WIDTH
    assert token_ids[:3] == [1001, 1002, 1003]
    assert sampled_token_id not in token_ids
    assert logprobs == sorted(logprobs, reverse=True)
    assert sampled_logprob == -9.25


def test_extract_ranked_topk_fails_closed_on_missing_rank():
    values = _position_logprobs(250)
    del values[1000 + schema.TOPK_WIDTH]
    with pytest.raises(ValueError, match="exact ranks"):
        generate.extract_ranked_topk(values, 250)


def test_sampled_topk_logprob_comparison_accounts_for_float16_storage():
    run_config = _run_config("train", unique_questions=1, samples_per_question=1)
    source = generate.SourcePrompt(
        prompt_index=0,
        messages=[{"role": "user", "content": "question"}],
        question_text="question",
        gold_answer="7",
        source_uid="train-0",
        source_uid_original="train-0",
        question_sha256=schema.sha256_text("question"),
    )
    request = generate.PreparedRequest(
        source,
        0,
        generate.derive_sampling_seed(42, "train", source.source_uid, 0),
        [1, 2, 3],
    )
    record = generate.build_trace_record(
        request=request,
        output=_output([1, 2, 3], [1001]),
        shard_id=0,
        row_within_shard=0,
        run_config=run_config,
        tokenizer=FakeTokenizer(),
        strict_grade=1.0,
        strict_prediction="7",
        generation_timestamp="2026-07-30T00:00:00+00:00",
    )
    fp16_record = pa.Table.from_pylist([record], schema=schema.trace_arrow_schema()).to_pylist()[0]
    schema.validate_trace_record(
        fp16_record,
        decoder=lambda ids: FakeTokenizer().decode(ids),
        expected_config_sha256=run_config["generation_config_sha256"],
        expected_direction="e4b_rl100_to_e2b",
        expected_split="train",
        expected_shard_id=0,
        expected_row_within_shard=0,
    )


def test_topk_mass_validation_accounts_for_float16_rounding():
    run_config = _run_config("train", unique_questions=1, samples_per_question=1)
    source = generate.SourcePrompt(
        prompt_index=0,
        messages=[{"role": "user", "content": "question"}],
        question_text="question",
        gold_answer="7",
        source_uid="train-0",
        source_uid_original="train-0",
        question_sha256=schema.sha256_text("question"),
    )
    request = generate.PreparedRequest(
        source,
        0,
        generate.derive_sampling_seed(42, "train", source.source_uid, 0),
        [1, 2, 3],
    )
    record = generate.build_trace_record(
        request=request,
        output=_output([1, 2, 3], [1001]),
        shard_id=0,
        row_within_shard=0,
        run_config=run_config,
        tokenizer=FakeTokenizer(),
        strict_grade=1.0,
        strict_prediction="7",
        generation_timestamp="2026-07-30T00:00:00+00:00",
    )
    equal_logprob = math.log(1.0 / 103.0)
    record["teacher_topk_logprobs"] = [[equal_logprob] * 103 + [-100.0] * 25]
    record["sampled_token_logprobs"] = [equal_logprob]
    fp16_record = pa.Table.from_pylist([record], schema=schema.trace_arrow_schema()).to_pylist()[0]
    reconstructed_mass = sum(math.exp(value) for value in fp16_record["teacher_topk_logprobs"][0])
    assert 1.0005 < reconstructed_mass < 1.0 + schema.FP16_TOPK_MASS_TOLERANCE
    schema.validate_trace_record(
        fp16_record,
        decoder=lambda ids: FakeTokenizer().decode(ids),
        expected_config_sha256=run_config["generation_config_sha256"],
        expected_direction="e4b_rl100_to_e2b",
        expected_split="train",
        expected_shard_id=0,
        expected_row_within_shard=0,
    )


def test_source_loader_deduplicates_repeated_validation_rows(tmp_path):
    row = {
        "prompt": [{"role": "user", "content": "question"}],
        "reward_model": {"ground_truth": "7", "style": "rule"},
        "uid": "validation-0",
    }
    source = tmp_path / "source.parquet"
    pq.write_table(pa.Table.from_pylist([row] * 16), source)
    prompts, source_rows = generate.load_unique_source_prompts(source)
    assert source_rows == 16
    assert len(prompts) == 1
    assert prompts[0].source_uid == "validation-0"


def test_source_loader_rejects_same_uid_with_different_prompt_messages(tmp_path):
    rows = [
        {
            "prompt": [{"role": "user", "content": "question"}],
            "reward_model": {"ground_truth": "7"},
            "uid": "validation-0",
        },
        {
            "prompt": [
                {"role": "system", "content": "different context"},
                {"role": "user", "content": "question"},
            ],
            "reward_model": {"ground_truth": "7"},
            "uid": "validation-0",
        },
    ]
    source = tmp_path / "source.parquet"
    pq.write_table(pa.Table.from_pylist(rows), source)
    with pytest.raises(ValueError, match="conflicting rows"):
        generate.load_unique_source_prompts(source)


def _validate_test_record(record, run_config):
    return schema.validate_trace_record(
        record,
        decoder=lambda ids: FakeTokenizer().decode(ids),
        expected_config_sha256=run_config["generation_config_sha256"],
        expected_direction="e4b_rl100_to_e2b",
        expected_split="train",
        expected_shard_id=0,
        expected_row_within_shard=0,
        expected_semantic_config=run_config["semantic_config"],
    )


def test_trace_validator_rejects_corrupt_trace_id():
    run_config = _run_config("train", unique_questions=1, samples_per_question=1)
    record = _record(
        run_config=run_config,
        split="train",
        question="question",
        source_uid="train-0",
        sample_index=0,
        row_within_shard=0,
    )
    record["trace_id"] = "f" * 64
    with pytest.raises(schema.TraceValidationError, match="trace_id does not match"):
        _validate_test_record(record, run_config)


def test_trace_validator_rejects_nonfinite_strict_grade():
    run_config = _run_config("train", unique_questions=1, samples_per_question=1)
    record = _record(
        run_config=run_config,
        split="train",
        question="question",
        source_uid="train-0",
        sample_index=0,
        row_within_shard=0,
    )
    record["strict_grade"] = float("nan")
    record["strict_correct"] = False
    with pytest.raises(schema.TraceValidationError, match="strict_grade must be finite"):
        _validate_test_record(record, run_config)


def test_trace_validator_rejects_corrupt_sampling_seed():
    run_config = _run_config("train", unique_questions=1, samples_per_question=1)
    record = _record(
        run_config=run_config,
        split="train",
        question="question",
        source_uid="train-0",
        sample_index=0,
        row_within_shard=0,
    )
    record["sampling_seed"] += 1
    with pytest.raises(schema.TraceValidationError, match="sampling_seed does not match"):
        _validate_test_record(record, run_config)


def test_trace_validator_rejects_row_provenance_drift():
    run_config = _run_config("train", unique_questions=1, samples_per_question=1)
    record = _record(
        run_config=run_config,
        split="train",
        question="question",
        source_uid="train-0",
        sample_index=0,
        row_within_shard=0,
    )
    record["teacher_model"] = "different-teacher"
    with pytest.raises(schema.TraceValidationError, match="teacher_model does not match"):
        _validate_test_record(record, run_config)


def test_full_generator_cpu_smoke_with_fake_vllm(tmp_path, monkeypatch):
    source = tmp_path / "source.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "prompt": [{"role": "user", "content": "question"}],
                    "reward_model": {"ground_truth": "7", "style": "rule"},
                }
            ]
        ),
        source,
    )
    teacher = tmp_path / "teacher"
    teacher.mkdir()
    (teacher / "config.json").write_text('{"model_type":"gemma4"}\n', encoding="utf-8")
    (teacher / "processor_config.json").write_text('{"processor_class":"Gemma4Processor"}\n', encoding="utf-8")
    (teacher / "model.safetensors").write_bytes(b"fake-local-teacher-weights")
    output_dir = tmp_path / "traces"

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeLLM:
        tokenizer = FakeTokenizer()

        def get_tokenizer(self):
            return self.tokenizer

        def generate(self, requests, sampling_params, **_kwargs):
            assert len(requests) == len(sampling_params) == 5
            assert all(params.kwargs["logprobs"] == schema.TOPK_WIDTH for params in sampling_params)
            return [_output(request["prompt_token_ids"], [250]) for request in requests]

    grader = SimpleNamespace(compute_score=lambda _text, _gold: 1.0, extract_prediction=lambda _text: "7")
    monkeypatch.setattr(generate, "_make_llm", lambda _args: (FakeLLM(), FakeSamplingParams, {}, "fake-vllm"))
    monkeypatch.setattr(generate, "_load_grader", lambda: grader)
    monkeypatch.setattr(generate, "repository_state", lambda _root: ("c" * 40, False))

    args = generate.parse_args(
        [
            "--teacher-model",
            str(teacher),
            "--teacher-content-sha256",
            schema.sha256_file(teacher / "model.safetensors"),
            "--input-parquet",
            str(source),
            "--output-dir",
            str(output_dir),
            "--direction",
            "e4b_rl100_to_e2b",
            "--split",
            "train",
        ]
    )
    generate.run_generation(args)
    parquet_path = output_dir / "traces-train-000000.parquet"
    assert pq.ParquetFile(parquet_path).metadata.num_rows == 5
    run_config = json.loads((output_dir / "run_config.json").read_text())
    assert generate._valid_completed_shard(parquet_path, run_config, FakeTokenizer(), 0)


def test_local_teacher_content_hash_is_verified(tmp_path):
    teacher = tmp_path / "teacher"
    teacher.mkdir()
    (teacher / "config.json").write_text('{"model_type":"gemma4"}\n', encoding="utf-8")
    (teacher / "processor_config.json").write_text("{}\n", encoding="utf-8")
    (teacher / "model.safetensors").write_bytes(b"weights")
    args = generate.parse_args(
        [
            "--teacher-model",
            str(teacher),
            "--teacher-content-sha256",
            "d" * 64,
            "--input-parquet",
            str(tmp_path / "unused.parquet"),
            "--output-dir",
            str(tmp_path / "traces"),
            "--direction",
            "e4b_rl100_to_e2b",
            "--split",
            "train",
        ]
    )
    with pytest.raises(ValueError, match="local teacher weights do not match"):
        generate._resolve_teacher_identity(args)


def test_local_model_identity_binds_weights_config_and_processor(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"gemma4","hidden_size":8}\n', encoding="utf-8")
    (model / "processor_config.json").write_text('{"processor_class":"Gemma4Processor"}\n', encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"weights")

    first = model_identity.inspect_local_hf_model(model)
    assert first.weight_content_sha256 == schema.sha256_file(model / "model.safetensors")
    assert first.weight_content_kind == "single_model_safetensors_sha256"

    (model / "processor_config.json").write_text('{"processor_class":"Changed"}\n', encoding="utf-8")
    second = model_identity.inspect_local_hf_model(model)
    assert second.weight_content_sha256 == first.weight_content_sha256
    assert second.model_identity_sha256 != first.model_identity_sha256


def test_typed_shard_roundtrip_resume_hash_and_dataset_index(tmp_path):
    train_config, train_path = _write_split(tmp_path / "train", "train", [("train-0", "train question")])
    _write_split(tmp_path / "validation", "validation", [("validation-0", "validation question")])

    parquet_schema = pq.ParquetFile(train_path).schema_arrow
    topk_ids_type = parquet_schema.field("teacher_topk_token_ids").type
    topk_logprobs_type = parquet_schema.field("teacher_topk_logprobs").type
    assert topk_ids_type.value_type.list_size == schema.TOPK_WIDTH
    assert topk_ids_type.value_type.value_type == pa.int32()
    assert topk_logprobs_type.value_type.list_size == schema.TOPK_WIDTH
    assert topk_logprobs_type.value_type.value_type == pa.float16()
    assert pq.ParquetFile(train_path).metadata.num_row_groups == 2

    row = pq.read_table(train_path).to_pylist()[0]
    assert row["input_ids"] == row["prompt_token_ids"] + row["response_token_ids"]
    assert row["response_mask"] == [0, 0, 0, 1]
    assert len(row["teacher_topk_token_ids"][0]) == schema.TOPK_WIDTH
    assert 250 not in row["teacher_topk_token_ids"][0]
    assert row["sampled_token_ids"] == [250]
    assert generate._valid_completed_shard(train_path, train_config, FakeTokenizer(), 0)

    output_index = tmp_path / "dataset_index.json"
    index = validate.validate_dataset(
        {"train": tmp_path / "train", "validation": tmp_path / "validation"},
        output_index=output_index,
        decoder=lambda ids: FakeTokenizer().decode(ids),
        expected_questions={"train": 1, "validation": 1},
        expected_samples_per_question=2,
    )
    assert index["total_rows"] == 4
    assert index["splits"]["validation"]["question_count"] == 1
    assert json.loads(output_index.read_text())["dataset_index_sha256"] == index["dataset_index_sha256"]

    with train_path.open("ab") as handle:
        handle.write(b"tamper")
    assert not generate._valid_completed_shard(train_path, train_config, FakeTokenizer(), 0)
    with pytest.raises(schema.TraceValidationError, match="SHA256 mismatch"):
        schema.validate_shard_bundle(
            train_path,
            run_config=train_config,
            decoder=lambda ids: FakeTokenizer().decode(ids),
        )


def test_dataset_index_records_and_can_reject_train_validation_question_overlap(tmp_path):
    _write_split(tmp_path / "train", "train", [("train-0", "same question")])
    _write_split(tmp_path / "validation", "validation", [("validation-0", "same question")])
    index = validate.validate_dataset(
        {"train": tmp_path / "train", "validation": tmp_path / "validation"},
        output_index=tmp_path / "dataset_index.json",
        decoder=lambda ids: FakeTokenizer().decode(ids),
        expected_questions={"train": 1, "validation": 1},
        expected_samples_per_question=2,
    )
    assert index["cross_split_question_text_overlap_count"] == 1
    with pytest.raises(schema.TraceValidationError, match="question text overlaps"):
        validate.validate_dataset(
            {"train": tmp_path / "train", "validation": tmp_path / "validation"},
            output_index=tmp_path / "dataset_index.json",
            decoder=lambda ids: FakeTokenizer().decode(ids),
            expected_questions={"train": 1, "validation": 1},
            expected_samples_per_question=2,
            fail_on_question_overlap=True,
        )


def test_incomplete_index_skips_full_dataset_count_checks(tmp_path):
    _write_split(
        tmp_path / "train",
        "train",
        [("train-0", "question")],
        declared_questions=2,
        total_shards=2,
    )
    index = validate.validate_dataset(
        {"train": tmp_path / "train"},
        output_index=tmp_path / "dataset_index.json",
        decoder=lambda ids: FakeTokenizer().decode(ids),
        expected_questions={"train": 9723},
        expected_samples_per_question=2,
        allow_incomplete=True,
    )
    assert not index["splits"]["train"]["complete"]
    assert index["splits"]["train"]["missing_shard_ids"] == [1]


def test_training_index_rejects_empty_response_by_default(tmp_path):
    split_dir = tmp_path / "train"
    split_dir.mkdir()
    run_config = _run_config("train", unique_questions=1, samples_per_question=1)
    schema.atomic_write_json(split_dir / "run_config.json", run_config)
    source = generate.SourcePrompt(
        prompt_index=0,
        messages=[{"role": "user", "content": "question"}],
        question_text="question",
        gold_answer="7",
        source_uid="train-0",
        source_uid_original="train-0",
        question_sha256=schema.sha256_text("question"),
    )
    request = generate.PreparedRequest(
        source,
        0,
        generate.derive_sampling_seed(42, "train", source.source_uid, 0),
        [1, 2, 3],
    )
    record = generate.build_trace_record(
        request=request,
        output=_output([1, 2, 3], []),
        shard_id=0,
        row_within_shard=0,
        run_config=run_config,
        tokenizer=FakeTokenizer(),
        strict_grade=0.0,
        strict_prediction="",
        generation_timestamp="2026-07-30T00:00:00+00:00",
    )
    shard_path = split_dir / "traces-train-000000.parquet"
    with pytest.raises(schema.TraceValidationError, match="empty response"):
        generate._write_validated_shard(
            [record],
            parquet_path=shard_path,
            shard_id=0,
            prompt_start=0,
            prompt_end=1,
            run_config=run_config,
            tokenizer=FakeTokenizer(),
            row_group_rows=1,
        )
    assert not shard_path.exists()
    assert not schema.parquet_manifest_path(shard_path).exists()
