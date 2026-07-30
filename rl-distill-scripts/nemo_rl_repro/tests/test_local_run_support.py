# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal.datastore import DataStore

LOCAL_DIR = Path(__file__).resolve().parents[1] / "local"


def load_local_module(name: str):
    sys.path.insert(0, str(LOCAL_DIR))
    spec = importlib.util.spec_from_file_location(name, LOCAL_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def make_finalized_checkpoint(root: Path, step: int) -> Path:
    step_dir = root / f"step_{step}"
    consolidated = step_dir / "policy/weights/model/consolidated"
    tokenizer = step_dir / "policy/tokenizer"
    consolidated.mkdir(parents=True)
    tokenizer.mkdir(parents=True)
    (step_dir / "training_info.json").write_text('{"total_steps": 20}\n')
    (step_dir / "config.yaml").write_text("checkpointing: {}\n")
    (consolidated / "config.json").write_text("{}\n")
    (consolidated / "model.safetensors").write_bytes(b"test")
    (tokenizer / "tokenizer_config.json").write_text("{}\n")
    return step_dir


def test_uploader_requires_and_selects_only_consolidated_model(tmp_path):
    uploader = load_local_module("hf_checkpoint_uploader")
    step_dir = make_finalized_checkpoint(tmp_path, 20)
    shard_dir = step_dir / "policy/weights/model"
    (shard_dir / "shard-00001.safetensors").write_bytes(b"large shard")

    expected = shard_dir / "consolidated"
    assert uploader.checkpoint_model_root(step_dir) == expected
    assert uploader.checkpoint_is_ready(step_dir)
    assert uploader.discover_steps(tmp_path, 20) == [(20, step_dir)]
    assert uploader.discover_steps(tmp_path, 25) == []


def test_uploader_rejects_incomplete_checkpoint(tmp_path):
    uploader = load_local_module("hf_checkpoint_uploader")
    step_dir = make_finalized_checkpoint(tmp_path, 20)
    (step_dir / "policy/weights/model/consolidated/config.json").unlink()

    assert uploader.checkpoint_model_root(step_dir) is None
    assert not uploader.checkpoint_is_ready(step_dir)


def test_uploader_stages_fp32_checkpoint_as_bfloat16(tmp_path):
    uploader = load_local_module("hf_checkpoint_uploader")
    step_dir = make_finalized_checkpoint(tmp_path, 20)
    model_root = step_dir / "policy/weights/model/consolidated"
    source = {
        "model.weight": torch.tensor([[1.25, -2.5], [3.0, 4.5]], dtype=torch.float32),
        "model.bias": torch.tensor([0.5, -0.25], dtype=torch.float32),
    }
    save_file(source, model_root / "model.safetensors")
    (model_root / "config.json").write_text(
        json.dumps({"dtype": "float32", "text_config": {"dtype": "float32"}}) + "\n"
    )

    staging = uploader.prepare_upload_staging(step_dir)
    staged_model = staging / "policy/weights/model/consolidated/model.safetensors"
    with safe_open(staged_model, framework="pt", device="cpu") as staged:
        assert set(staged.keys()) == set(source)
        assert {str(staged.get_slice(key).get_dtype()) for key in staged.keys()} == {"BF16"}
        assert torch.equal(staged.get_tensor("model.weight"), source["model.weight"].bfloat16())

    staged_config = json.loads((staging / "policy/weights/model/consolidated/config.json").read_text())
    assert staged_config["dtype"] == "bfloat16"
    assert staged_config["text_config"]["dtype"] == "bfloat16"
    manifest = json.loads((staging / "upload_manifest.json").read_text())
    assert manifest["source_dtypes"] == ["float32"]
    assert manifest["uploaded_dtype"] == "bfloat16"
    assert manifest["tensor_count"] == 2


def test_sparse_checkpoint_delta_round_trip(tmp_path):
    codec = load_local_module("hf_checkpoint_delta")
    base = tmp_path / "base.safetensors"
    target = tmp_path / "target.safetensors"
    output = tmp_path / "reconstructed.safetensors"
    base_bytes = bytearray(range(256)) * 64
    target_bytes = bytearray(base_bytes)
    for index, value in ((2, 19), (100, 201), (4096, 7), (12000, 99)):
        target_bytes[index : index + 2] = value.to_bytes(2, "little")
    base.write_bytes(base_bytes)
    target.write_bytes(target_bytes)

    delta_dir = tmp_path / "delta"
    manifest = codec.create_delta(base, target, delta_dir, chunk_bytes=1024, zstd_level=3)
    codec.reconstruct_delta(base, delta_dir, output, manifest)

    assert output.read_bytes() == target.read_bytes()
    assert manifest["codec"] == codec.CODEC
    assert manifest["changed_word_count"] == 4
    assert manifest["delta_bytes"] < len(target_bytes)


def test_sparse_checkpoint_delta_zigzag_varint_edges():
    codec = load_local_module("hf_checkpoint_delta")
    signed = np.array(
        [
            np.iinfo(np.int16).min,
            -8193,
            -8192,
            -65,
            -64,
            -2,
            -1,
            0,
            1,
            2,
            63,
            64,
            8191,
            8192,
            np.iinfo(np.int16).max,
        ],
        dtype=np.int16,
    )

    encoded = codec._encode_zigzag_varints(signed)
    decoded = codec._decode_zigzag_varints(encoded, len(signed)).view(np.int16)

    assert np.array_equal(decoded, signed)
    with pytest.raises(ValueError, match="expected number"):
        codec._decode_zigzag_varints(encoded[:-1], len(signed))


def test_base_canonicalizer_selects_target_layout(tmp_path):
    canonicalizer = load_local_module("hf_base_canonicalizer")
    source = tmp_path / "source.safetensors"
    target_index = tmp_path / "model.safetensors.index.json"
    output = tmp_path / "canonical.safetensors"
    save_file(
        {
            "model.a": torch.tensor([1.0, 2.0], dtype=torch.bfloat16),
            "model.b": torch.tensor([3.0], dtype=torch.bfloat16),
            "model.extra": torch.tensor([4.0], dtype=torch.bfloat16),
        },
        source,
        metadata={"format": "pt"},
    )
    target_index.write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {
                    "model.a": "model.safetensors",
                    "model.b": "model.safetensors",
                },
            }
        )
    )

    metadata = canonicalizer.canonicalize_base(source, target_index, output)

    with safe_open(output, framework="pt", device="cpu") as canonical:
        assert list(canonical.keys()) == ["model.a", "model.b"]
        assert torch.equal(canonical.get_tensor("model.a"), torch.tensor([1.0, 2.0]))
    assert metadata["tensor_count"] == 2
    assert metadata["canonical_sha256"] == canonicalizer.sha256_file(output)


def test_uploader_prepares_public_base_anchor_delta(tmp_path):
    uploader = load_local_module("hf_checkpoint_uploader")
    step_dir = make_finalized_checkpoint(tmp_path, 20)
    model_root = step_dir / "policy/weights/model/consolidated"
    target_tensors = {
        "model.weight": torch.tensor([[1.125, 2.0], [3.0, 4.0]], dtype=torch.float32),
        "model.bias": torch.tensor([0.5, -0.25], dtype=torch.float32),
    }
    save_file(target_tensors, model_root / "model.safetensors")
    (model_root / "config.json").write_text(json.dumps({"dtype": "float32"}) + "\n")
    canonical_base = tmp_path / "canonical-base.safetensors"
    save_file(
        {
            "model.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16),
            "model.bias": torch.tensor([0.5, -0.25], dtype=torch.bfloat16),
        },
        canonical_base,
        metadata={"format": "pt"},
    )

    staging = uploader.prepare_anchor_delta_staging(
        step_dir,
        20,
        canonical_base_path=canonical_base,
    )
    model_relative = Path("policy/weights/model/consolidated/model.safetensors")
    target_model = step_dir / ".hf_upload" / model_relative
    reconstructed = tmp_path / "anchor-reconstructed.safetensors"
    manifest = json.loads((staging / "delta/delta_manifest.json").read_text())
    codec = load_local_module("hf_checkpoint_delta")
    codec.reconstruct_delta(canonical_base, staging / "delta", reconstructed, manifest)

    assert not (staging / model_relative).exists()
    assert (staging / "delta/canonicalize_base.py").is_file()
    assert reconstructed.read_bytes() == target_model.read_bytes()
    assert manifest["target_step"] == 20
    assert manifest["base_model_id"] == uploader.BASE_MODEL_ID


def test_uploader_prepares_chained_delta_after_anchor(tmp_path):
    uploader = load_local_module("hf_checkpoint_uploader")
    for step, offset in ((20, 0.0), (40, 0.125)):
        step_dir = make_finalized_checkpoint(tmp_path, step)
        model_root = step_dir / "policy/weights/model/consolidated"
        save_file(
            {"model.weight": torch.tensor([[1.0 + offset, 2.0], [3.0, 4.0]])},
            model_root / "model.safetensors",
        )
        (model_root / "config.json").write_text(json.dumps({"dtype": "float32"}) + "\n")

    delta_staging = uploader.prepare_delta_staging(tmp_path / "step_40", 40, 20)
    delta_dir = delta_staging / "delta"
    manifest = json.loads((delta_dir / "delta_manifest.json").read_text())
    base_model = tmp_path / "step_20/.hf_upload/policy/weights/model/consolidated/model.safetensors"
    target_model = tmp_path / "step_40/.hf_upload/policy/weights/model/consolidated/model.safetensors"
    reconstructed = tmp_path / "reconstructed.safetensors"
    codec = load_local_module("hf_checkpoint_delta")
    codec.reconstruct_delta(base_model, delta_dir, reconstructed, manifest)

    assert reconstructed.read_bytes() == target_model.read_bytes()
    assert manifest["base_step"] == 20
    assert manifest["target_step"] == 40


def test_supervisor_checkpoint_and_metric_gates(tmp_path):
    supervisor = load_local_module("full_run_supervisor")
    make_finalized_checkpoint(tmp_path, 20)
    make_finalized_checkpoint(tmp_path, 40)
    (tmp_path / "step_60").mkdir()

    assert supervisor.latest_checkpoint_step(tmp_path) == 40
    assert (
        supervisor.health_is_bad(
            {
                "probs_ratio": 1.001,
                "probs_ratio_clamped": 0.999,
                "grad_norm": 2.5,
            }
        )
        is None
    )
    assert "gate" in supervisor.health_is_bad({"probs_ratio": 0.8})
    assert "non-finite" in supervisor.health_is_bad({"grad_norm": float("nan")})


def test_supervisor_reads_live_local_wandb_history(tmp_path):
    supervisor = load_local_module("full_run_supervisor")
    run_id = "abcd1234"
    run_dir = tmp_path / "exp_001/wandb/wandb" / f"run-20260730_000000-{run_id}"
    run_dir.mkdir(parents=True)
    wandb_path = run_dir / f"run-{run_id}.wandb"

    datastore = DataStore()
    datastore.open_for_write(str(wandb_path))
    for values in (
        {"_step": 0, "validation/accuracy": 0.12},
        {
            "_step": 1,
            "train/probs_ratio": 1.0001,
            "train/probs_ratio_clamped": 1.0002,
            "train/grad_norm": 2.5,
        },
    ):
        record = wandb_internal_pb2.Record()
        for key, value in values.items():
            item = record.history.item.add()
            item.key = key
            item.value_json = json.dumps(value)
        datastore.write(record)
    datastore.close()

    health = supervisor.get_local_wandb_health(logs_dir=tmp_path, run_id=run_id)
    assert health == {
        "state": "running",
        "source": "local_wandb_stream",
        "step": 1,
        "probs_ratio": 1.0001,
        "probs_ratio_clamped": 1.0002,
        "grad_norm": 2.5,
        "validation_accuracy": 0.12,
    }
