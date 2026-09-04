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
"""Training-view builder + preflight on a synthetic two-split trace bundle.

Exercises the study's default (validation rows taken from the teacher's own validation split, every
train question used for training) and the legacy carve-out mode, on a bundle whose ``source/`` holds
both roster parquets like real collections do.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPOSITORY_ROOT / "rl-distill-scripts" / "data"
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

import build_gemma4_distill_training_view as builder  # noqa: E402
import preflight_gemma4_distill_training_view as preflight  # noqa: E402
import preflight_gemma4_topk_distill as source_preflight  # noqa: E402
from gemma4_distill_trace_schema import (  # noqa: E402
    SCHEMA_VERSION,
    TOPK_WIDTH,
    atomic_write_json,
    derive_sampling_seed,
    hash_json,
    sha256_file,
    sha256_text,
    trace_arrow_schema,
)

DIRECTION = sorted(preflight.ALLOWED_DIRECTIONS)[0]
GLOBAL_SEED = 42


def _semantic(split: str, questions: int, samples: int) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "topk_width": TOPK_WIDTH,
        "split": split,
        "direction": DIRECTION,
        "global_seed": GLOBAL_SEED,
        "samples_per_question": samples,
        "unique_question_count": questions,
        "teacher": {"model": "teacher", "content_sha256": "b" * 64},
        "tokenizer": {"model": "tok", "sha256": "c" * 64, "vocab_size": 1000},
        "chat_template": {"path": "t.jinja", "sha256": "d" * 64},
        "sampling": dict(source_preflight.EXPECTED_SAMPLING),
    }


def _row(*, gen_sha: str, split: str, uid: str, sample_index: int, question: str) -> dict:
    response = [5, 6, 7]
    return {
        "schema_version": SCHEMA_VERSION,
        "generation_config_sha256": gen_sha,
        "trace_id": hash_json(
            {
                "generation_config_sha256": gen_sha,
                "source_uid": uid,
                "question_sha256": sha256_text(question),
                "sample_index": sample_index,
            }
        ),
        "direction": DIRECTION,
        "split": split,
        "source_dataset": "ds",
        "source_dataset_sha256": "e" * 64,
        "source_uid": uid,
        "source_uid_original": uid,
        "question_sha256": sha256_text(question),
        "prompt_index": 0,
        "sample_index": sample_index,
        "question_text": question,
        "gold_answer": "1",
        "strict_grade": 1.0,
        "strict_correct": True,
        "strict_prediction": "1",
        "teacher_model": "teacher",
        "teacher_revision": "rev",
        "teacher_content_sha256": "b" * 64,
        "tokenizer_model": "tok",
        "tokenizer_revision": "rev",
        "tokenizer_sha256": "c" * 64,
        "tokenizer_vocab_size": 1000,
        "chat_template_path": "t.jinja",
        "chat_template_sha256": "d" * 64,
        "global_seed": GLOBAL_SEED,
        "sampling_seed": derive_sampling_seed(GLOBAL_SEED, split, uid, sample_index),
        "sampling_parameters_json": "{}",
        "prompt_token_ids": [1, 2],
        "response_token_ids": response,
        "input_ids": [1, 2, *response],
        "response_mask": [0, 0, 1, 1, 1],
        "teacher_topk_token_ids": [list(range(TOPK_WIDTH)) for _ in response],
        "teacher_topk_logprobs": [[-1.0] * TOPK_WIDTH for _ in response],
        "sampled_token_ids": response,
        "sampled_token_logprobs": [-1.0] * len(response),
        "teacher_topk_rank_order": "descending",
        "prompt_length": 2,
        "response_length": len(response),
        "finish_reason": "stop",
        "stop_reason": None,
        "matched_stop_string": "<end_of_turn>",
        "reached_max_response_tokens": False,
        "response_text": "1",
        "vllm_response_text": "1",
        "response_text_normalization": "none",
        "shard_id": 0,
        "row_within_shard": 0,
        "generation_timestamp": "2026-09-04T00:00:00Z",
        "generator_commit": "test",
        "generator_source_sha256": "0" * 64,
        "environment_versions_json": "{}",
    }


def _write_split(root: Path, split: str, uids: list[str], samples: int, rows_per_shard: int) -> dict:
    gen_sha = hash_json(_semantic(split, len(uids), samples))
    rows = [
        _row(gen_sha=gen_sha, split=split, uid=uid, sample_index=index, question=f"Question {uid}?")
        for uid in uids
        for index in range(samples)
    ]
    split_dir = root / split
    split_dir.mkdir(parents=True)
    shards = []
    for shard_id, start in enumerate(range(0, len(rows), rows_per_shard)):
        table = pa.Table.from_pylist(rows[start : start + rows_per_shard], schema=trace_arrow_schema())
        path = split_dir / f"traces-{split}-{shard_id:06d}.parquet"
        pq.write_table(table, path, row_group_size=2)
        shards.append(
            {
                "shard_id": shard_id,
                "path": f"{split}/{path.name}",
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "rows": table.num_rows,
                "row_groups": pq.ParquetFile(path).metadata.num_row_groups,
            }
        )
    run_config = {"semantic_config": _semantic(split, len(uids), samples), "generation_config_sha256": gen_sha}
    atomic_write_json(split_dir / "run_config.json", run_config)
    return {
        "question_count": len(uids),
        "row_count": len(rows),
        "run_config_path": f"{split}/run_config.json",
        "parquet_files": [entry["path"] for entry in shards],
        "shards": shards,
    }


@pytest.fixture
def bundle(tmp_path: Path) -> Path:
    root = tmp_path / "bundle"
    train_uids = [f"train-{i}" for i in range(6)]
    validation_uids = [f"val-{i}" for i in range(4)]
    (root / "source").mkdir(parents=True)
    # Real bundles keep BOTH RL data-prep rosters under source/.
    pq.write_table(pa.table({"uid": train_uids}), root / "source" / "deepscaler_gemma4_26b_hard_train.parquet")
    pq.write_table(
        pa.table({"uid": validation_uids}), root / "source" / "deepscaler_gemma4_26b_hard_val300_x16.parquet"
    )
    splits = {
        "train": _write_split(root, "train", train_uids, samples=2, rows_per_shard=4),
        "validation": _write_split(root, "validation", validation_uids, samples=1, rows_per_shard=4),
    }
    index = {"schema_version": "test-bundle", "splits": splits}
    index["dataset_index_sha256"] = hash_json(index)
    atomic_write_json(root / "dataset_index.json", index)
    atomic_write_json(
        root / "COMPLETE.json", {"dataset_index_sha256": index["dataset_index_sha256"], "trace_spec": "26b-hard"}
    )
    return root


def _build(bundle: Path, out: Path, **overrides) -> dict:
    kwargs = dict(
        source_root=bundle,
        output_root=out,
        source_s3_uri="hf://not-used",
        output_s3_uri=None,
        seed=GLOBAL_SEED,
        train_questions=6,
        validation_questions=3,
        train_samples_per_question=2,
        validation_sample_index=0,
        expected_source_questions=6,
        expected_source_samples_per_question=2,
        rows_per_shard=5,
        row_group_rows=2,
        validation_source="validation",
    )
    kwargs.update(overrides)
    return builder.build_training_view(**kwargs)


def _run_preflight(view_root: Path, monkeypatch, **expected) -> source_preflight.PreflightResult:
    index = json.loads((view_root / "dataset_index.json").read_text())
    monkeypatch.setattr(source_preflight, "_student_identity_sha256", lambda *_: "f" * 64)
    monkeypatch.setattr(source_preflight, "_verify_student_tokenizer", lambda **_: "c" * 64)
    kwargs = dict(
        dataset_index=str(view_root / "dataset_index.json"),
        student_model="/nonexistent/student",
        expected_direction=DIRECTION,
        expected_teacher_identity_sha256=hash_json(index["teacher"]),
        expected_student_identity_sha256="f" * 64,
        local_files_only=True,
        expected_train_questions=6,
        expected_validation_questions=3,
        expected_train_samples_per_question=2,
        expected_validation_samples_per_question=1,
    )
    kwargs.update(expected)
    return preflight.run_preflight(**kwargs)


def test_validation_split_source_trains_on_every_question(bundle: Path, tmp_path: Path, monkeypatch):
    view = _build(bundle, tmp_path / "view")
    selection = json.loads((tmp_path / "view" / "selection.json").read_text())
    assert selection["validation_source_split"] == "validation"
    assert selection["train_question_count"] == 6 and selection["unused_question_count"] == 0
    assert sorted(selection["train_source_uids"]) == [f"train-{i}" for i in range(6)]
    assert len(selection["validation_source_uids"]) == 3
    assert all(uid.startswith("val-") for uid in selection["validation_source_uids"])
    assert selection["validation_source_question_count"] == 4 and selection["validation_unused_question_count"] == 1
    assert view["splits"]["train"]["row_count"] == 12 and view["splits"]["validation"]["row_count"] == 3
    assert view["source_trace"]["validation_split"] == "validation"
    assert (tmp_path / "view" / "source_validation_run_config.json").is_file()
    # the validation rows keep their immutable provenance: split=validation, validation generation identity
    rows = pq.read_table(tmp_path / "view" / view["splits"]["validation"]["parquet_files"][0]).to_pydict()
    assert set(rows["split"]) == {"validation"}
    assert set(rows["generation_config_sha256"]) == {view["source_trace"]["validation_generation_config_sha256"]}

    result = _run_preflight(tmp_path / "view", monkeypatch)
    assert len(result.train_files) == 3 and len(result.validation_files) == 1  # 12 rows / 5 per shard; 3 rows
    assert result.dataset_index_sha256 == view["dataset_index_sha256"]


def test_validation_split_source_is_deterministic(bundle: Path, tmp_path: Path):
    a = _build(bundle, tmp_path / "a")
    b = _build(bundle, tmp_path / "b")
    sel_a = json.loads((tmp_path / "a" / "selection.json").read_text())
    sel_b = json.loads((tmp_path / "b" / "selection.json").read_text())
    assert sel_a["validation_source_uids"] == sel_b["validation_source_uids"]
    assert a["splits"]["train"]["shards"][0]["sha256"] == b["splits"]["train"]["shards"][0]["sha256"]


def test_train_carveout_mode_still_works_with_two_rosters(bundle: Path, tmp_path: Path, monkeypatch):
    view = _build(bundle, tmp_path / "view", validation_source="train", train_questions=4, validation_questions=2)
    selection = json.loads((tmp_path / "view" / "selection.json").read_text())
    assert selection["validation_source_split"] == "train"
    assert (
        selection["train_question_count"] + selection["validation_question_count"] + selection["unused_question_count"]
        == 6
    )
    assert all(uid.startswith("train-") for uid in selection["validation_source_uids"])
    assert view["source_trace"]["validation_split"] == "train"
    assert "validation_generation_config_sha256" not in view["source_trace"]
    _run_preflight(tmp_path / "view", monkeypatch, expected_train_questions=4, expected_validation_questions=2)


def test_validation_source_rejects_more_questions_than_the_split_holds(bundle: Path, tmp_path: Path):
    with pytest.raises(builder.TrainingViewError, match="exceed the validation roster"):
        _build(bundle, tmp_path / "view", validation_questions=5)


def test_preflight_rejects_tampered_validation_provenance(bundle: Path, tmp_path: Path, monkeypatch):
    view = _build(bundle, tmp_path / "view")
    index_path = tmp_path / "view" / "dataset_index.json"
    index = json.loads(index_path.read_text())
    # Claim the validation rows came from the train split: the split provenance no longer matches.
    index["source_trace"]["validation_split"] = "train"
    del index["dataset_index_sha256"]
    index["dataset_index_sha256"] = hash_json(index)
    index_path.write_text(json.dumps(index))
    (tmp_path / "view" / "COMPLETE.json").write_text(
        json.dumps(
            {
                **json.loads((tmp_path / "view" / "COMPLETE.json").read_text()),
                "dataset_index_sha256": index["dataset_index_sha256"],
            }
        )
    )
    with pytest.raises(preflight.TrainingViewPreflightError, match="validation_split disagrees"):
        _run_preflight(tmp_path / "view", monkeypatch)
    assert view["source_trace"]["validation_split"] == "validation"
