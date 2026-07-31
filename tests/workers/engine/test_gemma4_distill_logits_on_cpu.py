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

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import SFTTensorCollator


def test_gemma4_skip_lm_head_returns_only_selected_hidden_states():
    # Importing the FSDP implementation installs the guarded Gemma 4 patch.
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration

    import verl.workers.engine.fsdp.transformer_impl  # noqa: F401

    class DummyBackbone(torch.nn.Module):
        def __init__(self, hidden_states):
            super().__init__()
            self.hidden_states = hidden_states

        def forward(self, *args, **kwargs):
            del args, kwargs
            return SimpleNamespace(
                last_hidden_state=self.hidden_states,
                past_key_values="cache",
                hidden_states=(self.hidden_states,),
                attentions=None,
                image_hidden_states=None,
                audio_hidden_states=None,
                shared_kv_states={3: "shared"},
            )

    model = object.__new__(Gemma4ForConditionalGeneration)
    torch.nn.Module.__init__(model)
    hidden_states = torch.arange(30, dtype=torch.float32).reshape(1, 5, 6)
    model.model = DummyBackbone(hidden_states)

    selected = torch.tensor([0, 3], dtype=torch.long)
    output = model(
        input_ids=torch.ones((1, 5), dtype=torch.long),
        logits_to_keep=selected,
        skip_lm_head=True,
        return_dict=True,
    )

    torch.testing.assert_close(output.logits, hidden_states[:, selected, :])
    assert output.loss is None
    assert output.past_key_values == "cache"
    assert output.shared_kv_states == {3: "shared"}


def test_distill_lm_head_uses_explicit_gemma4_softcap():
    scripts_dir = Path(__file__).parents[3] / "rl-distill-scripts"
    sys.path.insert(0, str(scripts_dir))
    from full_vocab_kl_loss import FullVocabKLLoss

    loss = FullVocabKLLoss(teacher_model_path="unused")
    lm_head = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        lm_head.weight.copy_(torch.eye(2))

    hidden = torch.tensor([[100.0, -100.0]])
    config_with_removed_softcap = SimpleNamespace(final_logit_softcapping=None)
    logits = loss._apply_lm_head(
        lm_head,
        hidden,
        config_with_removed_softcap,
        logit_softcap_override=30.0,
    )
    expected = torch.tanh(hidden / 30.0) * 30.0
    torch.testing.assert_close(logits, expected)


def test_precomputed_topk_partial_kl_matches_reference_and_backpropagates():
    scripts_dir = Path(__file__).parents[3] / "rl-distill-scripts"
    sys.path.insert(0, str(scripts_dir))
    from full_vocab_kl_loss import FullVocabKLLoss

    top_k = 2
    vocab_size = 7
    teacher_full_logits = torch.tensor(
        [
            [0.1, 1.2, -0.3, 0.0, 0.5, -0.7, 0.2],
            [0.7, -0.1, 0.3, 1.5, -0.5, 0.2, 0.0],
            [1.0, 0.2, -0.4, 0.6, 0.1, -0.2, 0.4],
            [-0.3, 0.8, 1.1, 0.2, -0.6, 0.4, 0.0],
        ]
    )
    teacher_full_logprobs = teacher_full_logits.log_softmax(dim=-1)
    teacher_topk_logprobs, teacher_topk_ids = teacher_full_logprobs.topk(top_k, dim=-1)

    samples = [
        {
            "input_ids": torch.tensor([1, 2, 3, 4]),
            "position_ids": torch.arange(4),
            "loss_mask": torch.tensor([0, 0, 1, 1]),
            "teacher_topk_token_ids": teacher_topk_ids[:2].to(torch.int32).flatten(),
            "teacher_topk_logprobs": teacher_topk_logprobs[:2].to(torch.float16).flatten(),
        },
        {
            "input_ids": torch.tensor([1, 5, 6]),
            "position_ids": torch.arange(3),
            "loss_mask": torch.tensor([0, 1, 1]),
            "teacher_topk_token_ids": teacher_topk_ids[2:].to(torch.int32).flatten(),
            "teacher_topk_logprobs": teacher_topk_logprobs[2:].to(torch.float16).flatten(),
        },
    ]
    collated = SFTTensorCollator("no_padding")(samples)
    data = tu.get_tensordict(
        collated,
        non_tensor_dict={"sp_size": 1, "pad_token_id": 0, "pad_mode": "no_padding"},
    )

    student_logits = torch.randn(1, 7, vocab_size, requires_grad=True)
    loss_impl = FullVocabKLLoss(
        precomputed_topk=True,
        top_k=top_k,
        chunk_size=1,
    )
    output = loss_impl(student_logits=student_logits, data=data)

    active_indices = torch.tensor([1, 2, 4, 5])
    active_student_logprobs = student_logits[0, active_indices].log_softmax(dim=-1)
    stored_teacher_logprobs = teacher_topk_logprobs.to(torch.float16).float()
    expected_student_topk_logprobs = active_student_logprobs.gather(1, teacher_topk_ids)
    expected_kl = (stored_teacher_logprobs.exp() * (stored_teacher_logprobs - expected_student_topk_logprobs)).sum(
        dim=-1
    )
    expected_teacher_mass = stored_teacher_logprobs.exp().sum(dim=-1)
    expected_student_mass = expected_student_topk_logprobs.exp().sum(dim=-1)

    torch.testing.assert_close(output["full_vocab_kl"][0, active_indices], expected_kl)
    torch.testing.assert_close(output["teacher_topk_mass"][0, active_indices], expected_teacher_mass)
    torch.testing.assert_close(output["student_topk_mass"][0, active_indices], expected_student_mass)
    output["full_vocab_kl"].sum().backward()
    assert student_logits.grad is not None
    assert torch.isfinite(student_logits.grad).all()
    assert student_logits.grad.abs().sum() > 0


def test_precomputed_topk_hidden_projection_checkpoints_full_vocab_activations():
    scripts_dir = Path(__file__).parents[3] / "rl-distill-scripts"
    sys.path.insert(0, str(scripts_dir))
    from full_vocab_kl_loss import FullVocabKLLoss

    top_k = 2
    vocab_size = 257
    samples = [
        {
            "input_ids": torch.tensor([1, 2, 3, 4]),
            "position_ids": torch.arange(4),
            "loss_mask": torch.tensor([0, 0, 1, 1]),
            "teacher_topk_token_ids": torch.tensor([[1, 2], [3, 4]], dtype=torch.int32).flatten(),
            "teacher_topk_logprobs": torch.tensor([[-0.1, -1.0], [-0.2, -1.2]], dtype=torch.float16).flatten(),
        }
    ]
    collated = SFTTensorCollator("no_padding")(samples)
    data = tu.get_tensordict(
        collated,
        non_tensor_dict={"sp_size": 1, "pad_token_id": 0, "pad_mode": "no_padding"},
    )
    active_indices = torch.tensor([1, 2])
    hidden = torch.randn(1, 2, 8, requires_grad=True)
    lm_head = torch.nn.Linear(8, vocab_size, bias=False)
    loss_impl = FullVocabKLLoss(
        precomputed_topk=True,
        top_k=top_k,
        chunk_size=1,
        checkpoint_student_chunks=True,
    )

    saved_shapes = []

    def pack(tensor):
        saved_shapes.append(tuple(tensor.shape))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        output = loss_impl(
            student_hidden=hidden,
            student_active_flat_idx=active_indices,
            student_lm_head=lm_head,
            student_config=SimpleNamespace(final_logit_softcapping=None),
            data=data,
        )

    assert not any(shape and shape[-1] == vocab_size for shape in saved_shapes)
    output["full_vocab_kl"].sum().backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    assert lm_head.weight.grad is not None and torch.isfinite(lm_head.weight.grad).all()


def test_precomputed_topk_dataset_flattens_response_targets_for_no_padding(tmp_path):
    scripts_dir = Path(__file__).parents[3] / "rl-distill-scripts"
    sys.path.insert(0, str(scripts_dir))
    from full_vocab_distill_dataset import FullVocabDistillDataset

    parquet = tmp_path / "traces.parquet"
    frame = pd.DataFrame(
        {
            "input_ids": [[1, 2, 3, 4], [1, 5, 6]],
            "response_mask": [[0, 0, 1, 1], [0, 1, 1]],
            "teacher_topk_token_ids": [
                [[1, 2], [3, 4]],
                [[2, 3], [4, 5]],
            ],
            "teacher_topk_logprobs": [
                [[-0.5, -1.5], [-0.6, -1.6]],
                [[-0.7, -1.7], [-0.8, -1.8]],
            ],
        }
    )
    frame.to_parquet(parquet, index=False)
    config = OmegaConf.create(
        {
            "pad_mode": "no_padding",
            "max_length": 16,
            "truncation": "error",
            "shuffle": False,
            "use_precomputed_topk": True,
            "teacher_topk_width": 2,
        }
    )
    tokenizer = SimpleNamespace(vocab_size=7, pad_token_id=0)
    dataset = FullVocabDistillDataset(str(parquet), tokenizer, config)

    first = dataset[0]
    assert first["teacher_topk_token_ids"].shape == (4,)
    assert first["teacher_topk_token_ids"].dtype == torch.int32
    assert first["teacher_topk_logprobs"].shape == (4,)
    assert first["teacher_topk_logprobs"].dtype == torch.float16

    collated = SFTTensorCollator("no_padding")([dataset[0], dataset[1]])
    data = tu.get_tensordict(collated)
    chunks = tu.chunk_tensordict(data, chunks=2)
    assert [chunk["teacher_topk_token_ids"].values().numel() for chunk in chunks] == [4, 4]


def test_precomputed_topk_left_truncation_rejects_trace_without_retained_prompt(tmp_path):
    scripts_dir = Path(__file__).parents[3] / "rl-distill-scripts"
    sys.path.insert(0, str(scripts_dir))
    from full_vocab_distill_dataset import FullVocabDistillDataset

    parquet = tmp_path / "traces.parquet"
    pd.DataFrame(
        {
            "input_ids": [[10, 11, 12, 13, 14]],
            "response_mask": [[0, 1, 1, 1, 1]],
            "teacher_topk_token_ids": [[[1, 2], [3, 4], [5, 6], [1, 3]]],
            "teacher_topk_logprobs": [[[-0.5, -1.5], [-0.6, -1.6], [-0.7, -1.7], [-0.8, -1.8]]],
        }
    ).to_parquet(parquet, index=False)
    config = OmegaConf.create(
        {
            "pad_mode": "no_padding",
            "max_length": 3,
            "truncation": "left",
            "shuffle": False,
            "use_precomputed_topk": True,
            "teacher_topk_width": 2,
        }
    )
    dataset = FullVocabDistillDataset(
        str(parquet),
        SimpleNamespace(vocab_size=16, pad_token_id=0),
        config,
    )

    with pytest.raises(ValueError, match="retain both prompt and response"):
        dataset[0]
