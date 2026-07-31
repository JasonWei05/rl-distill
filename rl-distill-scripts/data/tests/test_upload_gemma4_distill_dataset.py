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
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

DATA_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATA_DIR))

import gemma4_distill_trace_schema as schema  # noqa: E402
import generate_gemma4_distill_traces as generate  # noqa: E402
import upload_gemma4_distill_dataset as uploader  # noqa: E402
import validate_gemma4_distill_traces as validate  # noqa: E402


class FakeTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0
    unk_token_id = 3
    special_tokens_map = {"bos_token": "<bos>"}
    model_max_length = 12288

    def get_vocab(self):
        return {f"token-{token_id}": token_id for token_id in range(2048)}

    def decode(self, token_ids, **_kwargs):
        return "|".join(str(token_id) for token_id in token_ids)


def _position_logprobs(sampled_token_id: int):
    entries = {
        1000 + rank: SimpleNamespace(logprob=-5.0 - rank / 100.0, rank=rank) for rank in range(1, schema.TOPK_WIDTH + 1)
    }
    entries[sampled_token_id] = SimpleNamespace(logprob=-9.25, rank=300)
    return entries


def _semantic(split: str, tokenizer_sha256: str, vocab_size: int):
    teacher_revision = "a" * 40
    teacher = {
        "model": "example/teacher",
        "revision": teacher_revision,
        "content_sha256": None,
        "content_sha256_kind": None,
        "model_identity_sha256": schema.hash_json({"model": "example/teacher", "revision": teacher_revision}),
    }
    return {
        "schema_version": schema.SCHEMA_VERSION,
        "direction": "e4b_rl100_to_e2b",
        "split": split,
        "source_dataset": f"synthetic-{split}",
        "source_dataset_sha256": ("1" if split == "train" else "2") * 64,
        "prompt_roster_sha256": "3" * 64,
        "source_row_count": 1,
        "unique_question_count": 1,
        "samples_per_question": 5,
        "topk_width": schema.TOPK_WIDTH,
        "global_seed": 42,
        "prompts_per_shard": 1,
        "row_group_rows": 1,
        "total_shards": 1,
        "teacher": teacher,
        "tokenizer": {
            "model": "example/tokenizer",
            "revision": "b" * 40,
            "sha256": tokenizer_sha256,
            "vocab_size": vocab_size,
        },
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


def _run_config(split: str, tokenizer_sha256: str, vocab_size: int):
    semantic = _semantic(split, tokenizer_sha256, vocab_size)
    return {
        "manifest_version": schema.MANIFEST_VERSION,
        "schema_version": schema.SCHEMA_VERSION,
        "generation_config_sha256": schema.hash_json(semantic),
        "semantic_config": semantic,
        "input_parquet": "/synthetic.parquet",
        "created_at": "2026-07-31T00:00:00+00:00",
    }


def _record(run_config, *, split: str, sample_index: int, row_index: int):
    question = f"{split} question"
    source_uid = f"{split}-question-0"
    source = generate.SourcePrompt(
        prompt_index=0,
        messages=[{"role": "user", "content": question}],
        question_text=question,
        gold_answer="7",
        source_uid=source_uid,
        source_uid_original=source_uid,
        question_sha256=schema.sha256_text(question),
    )
    request = generate.PreparedRequest(
        source=source,
        sample_index=sample_index,
        sampling_seed=schema.derive_sampling_seed(42, split, source_uid, sample_index),
        prompt_token_ids=[1, 2, 3],
    )
    completion = SimpleNamespace(
        token_ids=[250],
        logprobs=[_position_logprobs(250)],
        finish_reason="stop",
        stop_reason="<end_of_turn>",
        text="vllm text",
    )
    output = SimpleNamespace(prompt_token_ids=[1, 2, 3], outputs=[completion])
    return generate.build_trace_record(
        request=request,
        output=output,
        shard_id=0,
        row_within_shard=row_index,
        run_config=run_config,
        tokenizer=FakeTokenizer(),
        strict_grade=1.0,
        strict_prediction="7",
        generation_timestamp="2026-07-31T00:00:00+00:00",
    )


def _write_complete_dataset(root: Path) -> Path:
    tokenizer = FakeTokenizer()
    tokenizer_sha256, vocab_size = schema.tokenizer_fingerprint(tokenizer)
    split_dirs = {}
    for split in ("train", "validation"):
        split_dir = root / split
        split_dir.mkdir(parents=True)
        run_config = _run_config(split, tokenizer_sha256, vocab_size)
        schema.atomic_write_json(split_dir / "run_config.json", run_config)
        records = [_record(run_config, split=split, sample_index=index, row_index=index) for index in range(5)]
        generate._write_validated_shard(
            records,
            parquet_path=split_dir / f"traces-{split}-000000.parquet",
            shard_id=0,
            prompt_start=0,
            prompt_end=1,
            run_config=run_config,
            tokenizer=tokenizer,
            row_group_rows=1,
        )
        split_dirs[split] = split_dir
    index_path = root / "dataset_index.json"
    validate.validate_dataset(
        split_dirs,
        output_index=index_path,
        decoder=lambda ids: tokenizer.decode(ids),
        expected_questions={"train": 1, "validation": 1},
        expected_samples_per_question=5,
    )
    return index_path


class FakeApi:
    def __init__(self, *, private=True, error=None, existing_files=()):
        self.private = private
        self.error = error
        self.calls = []
        self.operations = []
        self.files = set(existing_files)

    def create_repo(self, repo_id, **kwargs):
        self.calls.append(("create_repo", repo_id, kwargs))
        if self.error is not None:
            raise self.error
        return SimpleNamespace()

    def repo_info(self, repo_id, **kwargs):
        self.calls.append(("repo_info", repo_id, kwargs))
        return SimpleNamespace(private=self.private)

    def list_repo_files(self, repo_id, **kwargs):
        self.calls.append(("list_repo_files", repo_id, kwargs))
        return sorted(self.files)

    def create_commit(self, **kwargs):
        self.calls.append(("create_commit", kwargs["repo_id"], kwargs))
        self.operations = list(kwargs["operations"])
        self.files.update(operation.path_in_repo for operation in self.operations)
        return SimpleNamespace(oid="d" * 40)


def _validated_fixture(tmp_path: Path):
    index_path = _write_complete_dataset(tmp_path)
    bundle = uploader.validate_upload_bundle(
        index_path,
        _expected_questions={"train": 1, "validation": 1},
    )
    return index_path, bundle


def test_complete_bundle_preserves_index_run_configs_manifests_and_shards(tmp_path):
    index_path, bundle = _validated_fixture(tmp_path)
    assert bundle.index_path == index_path.resolve()
    assert bundle.total_rows == 10
    assert {item.path_in_repo for item in bundle.files} == {
        "dataset_index.json",
        "train/run_config.json",
        "train/traces-train-000000.manifest.json",
        "train/traces-train-000000.parquet",
        "validation/run_config.json",
        "validation/traces-validation-000000.manifest.json",
        "validation/traces-validation-000000.parquet",
    }
    directory_bundle = uploader.validate_upload_bundle(
        tmp_path,
        _expected_questions={"train": 1, "validation": 1},
    )
    assert directory_bundle.dataset_index_sha256 == bundle.dataset_index_sha256


def test_complete_bundle_remains_valid_after_directory_move(tmp_path):
    original_root = tmp_path / "original"
    _write_complete_dataset(original_root)
    moved_root = tmp_path / "downloaded-elsewhere"
    shutil.move(original_root, moved_root)

    index = json.loads((moved_root / "dataset_index.json").read_text(encoding="utf-8"))
    for split in ("train", "validation"):
        assert not Path(index["splits"][split]["run_config_path"]).is_absolute()
        assert all(not Path(path).is_absolute() for path in index["splits"][split]["parquet_files"])
        assert all("absolute_path" not in shard for shard in index["splits"][split]["shards"])
        assert all(not Path(shard["path"]).is_absolute() for shard in index["splits"][split]["shards"])

    bundle = uploader.validate_upload_bundle(
        moved_root,
        _expected_questions={"train": 1, "validation": 1},
    )
    assert bundle.total_rows == 10


def test_production_validation_refuses_smoke_sized_dataset(tmp_path):
    index_path = _write_complete_dataset(tmp_path)
    with pytest.raises(uploader.DatasetUploadError, match="train has 1 questions, expected 9723"):
        uploader.validate_upload_bundle(index_path)


def test_direct_parquet_artifact_is_rejected_before_hf_mutation(tmp_path):
    index_path = _write_complete_dataset(tmp_path)
    parquet_path = tmp_path / "train/traces-train-000000.parquet"
    api = FakeApi()
    with pytest.raises(uploader.DatasetUploadError, match="not a direct shard/manifest smoke artifact"):
        uploader.upload_dataset(
            parquet_path,
            repo_id="example/private-traces",
            revision="main",
            token="secret-token",
            api=api,
        )
    assert api.calls == []
    assert index_path.is_file()


def test_index_missing_validation_split_is_rejected(tmp_path):
    index_path = _write_complete_dataset(tmp_path)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    del index["splits"]["validation"]
    index.pop("dataset_index_sha256")
    index["dataset_index_sha256"] = schema.hash_json(index)
    schema.atomic_write_json(index_path, index)
    with pytest.raises(uploader.DatasetUploadError, match="exactly complete train and validation"):
        uploader.validate_upload_bundle(index_path, _expected_questions={"train": 1, "validation": 1})


def test_full_parquet_schema_is_revalidated_before_upload(tmp_path):
    index_path = _write_complete_dataset(tmp_path)
    parquet_path = tmp_path / "train/traces-train-000000.parquet"
    table = pq.read_table(parquet_path).drop(["response_mask"])
    pq.write_table(table, parquet_path, row_group_size=1)

    manifest_path = schema.parquet_manifest_path(parquet_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["parquet_sha256"] = schema.sha256_file(parquet_path)
    manifest["parquet_size_bytes"] = parquet_path.stat().st_size
    manifest["parquet_row_groups"] = pq.ParquetFile(parquet_path).metadata.num_row_groups
    schema.atomic_write_json(manifest_path, manifest)

    index = json.loads(index_path.read_text(encoding="utf-8"))
    shard = index["splits"]["train"]["shards"][0]
    shard["sha256"] = manifest["parquet_sha256"]
    shard["size_bytes"] = manifest["parquet_size_bytes"]
    shard["row_groups"] = manifest["parquet_row_groups"]
    index.pop("dataset_index_sha256")
    index["dataset_index_sha256"] = schema.hash_json(index)
    schema.atomic_write_json(index_path, index)

    with pytest.raises(uploader.DatasetUploadError, match="unexpected parquet schema"):
        uploader.validate_upload_bundle(index_path, _expected_questions={"train": 1, "validation": 1})


def test_private_atomic_upload_reports_branch_and_immutable_revision(tmp_path):
    _, bundle = _validated_fixture(tmp_path)
    api = FakeApi()
    result = uploader.upload_validated_bundle(
        bundle,
        repo_id="example/private-traces",
        revision="trace-v1",
        token="secret-token",
        api=api,
    )
    assert result.repo_id == "example/private-traces"
    assert result.requested_revision == "trace-v1"
    assert result.commit_oid == "d" * 40
    assert result.file_count == 7
    assert "secret-token" not in "\n".join(result.lines())
    assert api.calls[0][0] == "create_repo"
    assert api.calls[0][2]["private"] is True
    assert api.calls[0][2]["repo_type"] == "dataset"
    create_commit_call = next(call for call in api.calls if call[0] == "create_commit")
    assert create_commit_call[2]["revision"] == "trace-v1"
    assert {operation.path_in_repo for operation in api.operations} == {item.path_in_repo for item in bundle.files}


def test_existing_stale_hub_file_is_refused_before_commit(tmp_path):
    _, bundle = _validated_fixture(tmp_path)
    api = FakeApi(existing_files={"obsolete-shard.parquet"})
    with pytest.raises(uploader.DatasetUploadError, match="stale Hub files"):
        uploader.upload_validated_bundle(
            bundle,
            repo_id="example/private-traces",
            revision="main",
            token="secret-token",
            api=api,
        )
    assert "create_commit" not in [call[0] for call in api.calls]


def test_bundle_changed_after_validation_is_refused_before_hf_mutation(tmp_path):
    _, bundle = _validated_fixture(tmp_path)
    api = FakeApi()
    bundle.index_path.write_text(bundle.index_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(uploader.DatasetUploadError, match="file size changed"):
        uploader.upload_validated_bundle(
            bundle,
            repo_id="example/private-traces",
            revision="main",
            token="secret-token",
            api=api,
        )
    assert api.calls == []


def test_existing_public_repo_is_refused_before_commit(tmp_path):
    _, bundle = _validated_fixture(tmp_path)
    api = FakeApi(private=False)
    with pytest.raises(uploader.DatasetUploadError, match="is not private"):
        uploader.upload_validated_bundle(
            bundle,
            repo_id="example/public-traces",
            revision="main",
            token="secret-token",
            api=api,
        )
    assert [call[0] for call in api.calls] == ["create_repo", "repo_info"]


def test_hf_errors_redact_token(tmp_path):
    _, bundle = _validated_fixture(tmp_path)
    api = FakeApi(error=RuntimeError("request rejected for secret-token"))
    with pytest.raises(uploader.DatasetUploadError) as captured:
        uploader.upload_validated_bundle(
            bundle,
            repo_id="example/private-traces",
            revision="main",
            token="secret-token",
            api=api,
        )
    assert "secret-token" not in str(captured.value)
    assert "[REDACTED]" in str(captured.value)


def test_cli_error_output_redacts_token(monkeypatch, capsys):
    monkeypatch.setenv("GEMMA4_UPLOAD_TEST_TOKEN", "secret-token")

    def fail_upload(*_args, **_kwargs):
        raise RuntimeError("request rejected for secret-token")

    monkeypatch.setattr(uploader, "upload_dataset", fail_upload)
    status = uploader.main(
        [
            "--dataset-path",
            "/unused/dataset_index.json",
            "--repo-id",
            "example/private-traces",
            "--token-env",
            "GEMMA4_UPLOAD_TEST_TOKEN",
        ]
    )
    captured = capsys.readouterr()
    assert status == 2
    assert "secret-token" not in captured.out + captured.err
    assert "[REDACTED]" in captured.err
