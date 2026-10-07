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

import pytest
import torch


def _tiny_model(kind):
    if kind == "moe":
        from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

        config = Qwen3MoeConfig(
            vocab_size=64, hidden_size=32, intermediate_size=64, moe_intermediate_size=16, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8, num_experts=4, num_experts_per_tok=2,
            tie_word_embeddings=False,
        )
        return Qwen3MoeForCausalLM(config)
    from transformers import Qwen3Config, Qwen3ForCausalLM

    config = Qwen3Config(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, tie_word_embeddings=True,
    )
    return Qwen3ForCausalLM(config)


@pytest.mark.parametrize("kind", ["moe", "dense"])
def test_qwen3_skip_lm_head_returns_selected_hidden_states(kind):
    import verl.workers.engine.fsdp.transformer_impl  # noqa: F401  (installs the patch)

    torch.manual_seed(0)
    model = _tiny_model(kind).eval()
    assert type(model)._verl_skip_lm_head_patched
    input_ids = torch.randint(0, 64, (2, 7))
    token_idx = torch.tensor([0, 3, 6, 2, 5])
    batch_idx = torch.tensor([0, 0, 0, 1, 1])
    with torch.no_grad():
        full = model(input_ids=input_ids, use_cache=False, return_dict=True).logits
        out = model(
            input_ids=input_ids, use_cache=False, return_dict=True, skip_lm_head=True,
            logits_to_keep=token_idx, logits_to_keep_batch_indices=batch_idx,
        )
    assert out.logits.shape == (1, 5, 32)  # hidden states, not vocab logits
    torch.testing.assert_close(model.lm_head(out.logits)[0], full[batch_idx, token_idx], rtol=0, atol=1e-6)
    if kind == "moe":
        assert out.aux_loss is None
