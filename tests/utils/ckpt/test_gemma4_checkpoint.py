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

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from verl.utils.checkpoint.gemma4_checkpoint import (
    ensure_gemma4_processor_metadata,
    expand_gemma4_shared_kv_state_dict,
    verify_gemma4_shared_kv_checkpoint,
    verify_gemma4_shared_kv_tensor_names,
)


def _gemma4_config(model_source=""):
    return SimpleNamespace(
        model_type="gemma4",
        name_or_path=str(model_source),
        _commit_hash=None,
        text_config=SimpleNamespace(
            model_type="gemma4_text",
            num_hidden_layers=4,
            num_kv_shared_layers=2,
            layer_types=["sliding_attention", "full_attention"] * 2,
        ),
    )


def _sparse_state_dict():
    state_dict = {}
    prefix = "model.language_model.layers"
    for layer in range(4):
        state_dict[f"{prefix}.{layer}.self_attn.q_proj.weight"] = torch.tensor([layer], dtype=torch.bfloat16)
    for layer in range(2):
        for offset, suffix in enumerate(("k_norm.weight", "k_proj.weight", "v_proj.weight")):
            state_dict[f"{prefix}.{layer}.self_attn.{suffix}"] = torch.tensor(
                [10 * layer + offset], dtype=torch.bfloat16
            )
    return state_dict


def test_expand_gemma4_shared_kv_state_dict_clones_every_serving_alias():
    original = _sparse_state_dict()

    expanded, aliases = expand_gemma4_shared_kv_state_dict(original, _gemma4_config())

    assert len(aliases) == 6
    assert len(expanded) == len(original) + 6
    for alias in aliases:
        assert alias.target not in original
        torch.testing.assert_close(expanded[alias.target], expanded[alias.source])
        assert expanded[alias.target].data_ptr() != expanded[alias.source].data_ptr()
    verify_gemma4_shared_kv_tensor_names(_gemma4_config(), set(expanded))


def test_verify_gemma4_shared_kv_tensor_names_rejects_sparse_artifact():
    with pytest.raises(ValueError, match="missing 6 shared-KV serving tensors"):
        verify_gemma4_shared_kv_tensor_names(_gemma4_config(), set(_sparse_state_dict()))


def test_verify_gemma4_shared_kv_checkpoint_reads_safetensors(tmp_path):
    expanded, aliases = expand_gemma4_shared_kv_state_dict(_sparse_state_dict(), _gemma4_config())
    save_file(expanded, tmp_path / "model.safetensors")

    verified = verify_gemma4_shared_kv_checkpoint(tmp_path, _gemma4_config())

    assert verified == aliases


def test_ensure_gemma4_processor_metadata_copies_pinned_local_file(tmp_path):
    model_source = tmp_path / "model-source"
    checkpoint = tmp_path / "checkpoint"
    model_source.mkdir()
    processor_config = {"processor_class": "Gemma4Processor", "image_seq_length": 280}
    (model_source / "processor_config.json").write_text(json.dumps(processor_config) + "\n", encoding="utf-8")

    destination = ensure_gemma4_processor_metadata(_gemma4_config(model_source), checkpoint)

    assert destination == checkpoint / "processor_config.json"
    assert json.loads(destination.read_text(encoding="utf-8")) == processor_config


def test_ensure_gemma4_processor_metadata_fails_closed_when_missing(tmp_path):
    model_source = tmp_path / "model-source"
    model_source.mkdir()

    with pytest.raises(FileNotFoundError, match="missing processor_config.json"):
        ensure_gemma4_processor_metadata(_gemma4_config(model_source), tmp_path / "checkpoint")
