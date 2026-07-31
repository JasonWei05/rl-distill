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

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

LOCAL_DIR = Path(__file__).resolve().parents[1] / "local"
CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


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


def locked_manifests(module):
    lock = module.load_locked_chain(CONFIG_DIR / "e4b_step100_delta_chain.lock.json")
    manifests = {}
    for item in lock["steps"]:
        step = int(item["step"])
        manifest = copy.deepcopy(item)
        manifest["target_step"] = step
        manifest["mask_bytes"] = int(item["delta_bytes"]) - 1
        manifest["value_bytes"] = 1
        manifest["word_count"] = int(item["target_bytes"]) // 2
        if step == int(lock["steps"][0]["step"]):
            manifest.update(
                {
                    "base_canonical_sha256": lock["base"]["canonical_sha256"],
                    "base_model_id": lock["base"]["model_id"],
                    "base_revision": lock["base"]["revision"],
                }
            )
        manifests[step] = manifest
    return lock, manifests


def test_locked_e4b_step100_chain_is_self_consistent():
    module = load_local_module("materialize_hf_checkpoint_chain")
    lock, manifests = locked_manifests(module)

    module.validate_chain(lock, manifests)

    assert lock["target_step"] == 100
    assert [item["step"] for item in lock["steps"]] == [20, 40, 60, 80, 100]
    assert lock["steps"][-1]["target_sha256"] == "d565a3ff371906ca31a5e355472d70366b6956c0e82a914de4ea8a7c0085630c"


def test_chain_validator_rejects_broken_hash_link():
    module = load_local_module("materialize_hf_checkpoint_chain")
    lock, manifests = locked_manifests(module)
    manifests[60]["base_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="differs from the lock"):
        module.validate_chain(lock, manifests)


def test_chain_validator_rejects_inconsistent_payload_sizes():
    module = load_local_module("materialize_hf_checkpoint_chain")
    lock, manifests = locked_manifests(module)
    manifests[80]["value_bytes"] = 2

    with pytest.raises(ValueError, match="delta byte counts are inconsistent"):
        module.validate_chain(lock, manifests)


def test_materialize_two_step_chain(tmp_path, monkeypatch):
    module = load_local_module("materialize_hf_checkpoint_chain")
    codec = load_local_module("hf_checkpoint_delta")
    canonicalizer = load_local_module("hf_base_canonicalizer")
    snapshot = tmp_path / "snapshot"
    anchor_model_root = snapshot / "checkpoints/step_20/policy/weights/model/consolidated"
    tokenizer_root = snapshot / "checkpoints/step_20/policy/tokenizer"
    anchor_model_root.mkdir(parents=True)
    tokenizer_root.mkdir(parents=True)
    (anchor_model_root / "config.json").write_text('{"model_type": "test"}\n')
    (anchor_model_root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": {"model.weight": "model.safetensors"}}) + "\n"
    )
    (tokenizer_root / "tokenizer_config.json").write_text("{}\n")
    processor_config = tmp_path / "processor_config.json"
    processor_config.write_text('{"processor_class": "TestProcessor"}\n')
    target_root = snapshot / "checkpoints/step_40"
    target_root.mkdir()
    (target_root / "config.yaml").write_text("checkpointing: {}\n")
    (target_root / "training_info.json").write_text('{"current_step": 40}\n')

    source_base = tmp_path / "source_base.safetensors"
    canonical_base = tmp_path / "canonical_base.safetensors"
    target_20 = tmp_path / "target_20.safetensors"
    target_40 = tmp_path / "target_40.safetensors"
    save_file(
        {
            "model.extra": torch.tensor([9.0], dtype=torch.bfloat16),
            "model.weight": torch.tensor([1.0, 2.0], dtype=torch.bfloat16),
        },
        source_base,
        metadata={"format": "pt"},
    )
    base_metadata = canonicalizer.canonicalize_base(
        source_base,
        anchor_model_root / "model.safetensors.index.json",
        canonical_base,
    )
    save_file(
        {"model.weight": torch.tensor([1.25, 2.0], dtype=torch.bfloat16)},
        target_20,
        metadata={"format": "pt"},
    )
    save_file(
        {"model.weight": torch.tensor([1.25, 2.5], dtype=torch.bfloat16)},
        target_40,
        metadata={"format": "pt"},
    )

    manifest_20 = codec.create_delta(
        canonical_base,
        target_20,
        snapshot / "checkpoints/step_20/delta",
        chunk_bytes=32,
        zstd_level=1,
    )
    manifest_20.update(
        {
            "base_canonical_sha256": base_metadata["canonical_sha256"],
            "base_model_id": "test/base",
            "base_revision": "base-revision",
            "target_step": 20,
        }
    )
    manifest_40 = codec.create_delta(
        target_20,
        target_40,
        snapshot / "checkpoints/step_40/delta",
        chunk_bytes=32,
        zstd_level=1,
    )
    manifest_40.update({"base_step": 20, "target_step": 40})
    manifests = {20: manifest_20, 40: manifest_40}
    lock = {
        "base": {
            "canonical_sha256": base_metadata["canonical_sha256"],
            "metadata_files": {"processor_config.json": canonicalizer.sha256_file(processor_config)},
            "model_id": "test/base",
            "revision": "base-revision",
        },
        "repo_id": "test/checkpoints",
        "repo_revision": "checkpoint-revision",
        "schema_version": 1,
        "steps": [
            {key: manifest_20[key] for key in ("base_sha256", "codec", "delta_bytes", "target_bytes", "target_sha256")}
            | {"step": 20},
            {key: manifest_40[key] for key in ("base_sha256", "codec", "delta_bytes", "target_bytes", "target_sha256")}
            | {"base_step": 20, "step": 40},
        ],
        "target_step": 40,
    }
    module.validate_chain(lock, manifests)

    def fake_hf_hub_download(**kwargs):
        if kwargs["filename"] == "model.safetensors":
            return str(source_base)
        if kwargs["filename"] == "processor_config.json":
            return str(processor_config)
        raise AssertionError(kwargs)

    monkeypatch.setattr(module, "hf_hub_download", fake_hf_hub_download)

    output_dir = tmp_path / "output"
    work_dir = tmp_path / "work"
    module.materialize(
        lock,
        manifests,
        snapshot=snapshot,
        output_dir=output_dir,
        work_dir=work_dir,
    )

    assert (output_dir / "model.safetensors").read_bytes() == target_40.read_bytes()
    assert (output_dir / "config.json").is_file()
    assert (output_dir / "processor_config.json").read_bytes() == processor_config.read_bytes()
    assert (output_dir / "training_info.json").is_file()
    assert not work_dir.exists()
    provenance = json.loads((output_dir / "materialization_manifest.json").read_text())
    assert provenance["materialized_step"] == 40
    assert provenance["materialized_sha256"] == manifest_40["target_sha256"]
    assert provenance["chain_target_sha256"] == manifest_40["target_sha256"]
    assert provenance["base_metadata_files"] == lock["base"]["metadata_files"]
    assert provenance["steps"][-1] == {"step": 40, "target_sha256": manifest_40["target_sha256"]}


def test_expand_gemma4_shared_kv_aliases(tmp_path):
    module = load_local_module("materialize_hf_checkpoint_chain")
    source = tmp_path / "source.safetensors"
    output = tmp_path / "output.safetensors"
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "text_config": {
                    "num_hidden_layers": 4,
                    "num_kv_shared_layers": 2,
                    "layer_types": ["sliding_attention", "full_attention"] * 2,
                }
            }
        )
    )
    tensors = {}
    for layer in range(4):
        tensors[f"model.language_model.layers.{layer}.self_attn.q_norm.weight"] = torch.tensor(
            [layer], dtype=torch.bfloat16
        )
    for layer in range(2):
        for offset, suffix in enumerate(("k_norm.weight", "k_proj.weight", "v_proj.weight")):
            tensors[f"model.language_model.layers.{layer}.self_attn.{suffix}"] = torch.tensor(
                [10 * layer + offset], dtype=torch.bfloat16
            )
    save_file(tensors, source, metadata={"format": "pt"})

    aliases = module.expand_gemma4_shared_kv_aliases(source, config, output)

    assert len(aliases) == 6
    with module.safe_open(output, framework="pt", device="cpu") as expanded:
        assert len(list(expanded.keys())) == len(tensors) + 6
        for layer, source_layer in ((2, 0), (3, 1)):
            for suffix in ("k_norm.weight", "k_proj.weight", "v_proj.weight"):
                target = expanded.get_tensor(f"model.language_model.layers.{layer}.self_attn.{suffix}")
                expected = expanded.get_tensor(f"model.language_model.layers.{source_layer}.self_attn.{suffix}")
                assert torch.equal(target, expected)
