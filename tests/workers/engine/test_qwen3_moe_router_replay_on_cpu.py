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
"""Qwen3-MoE R3 router replay on a tiny random model (needs transformers 5: fused experts + Qwen3MoeTopKRouter)."""

from types import SimpleNamespace

import pytest
import torch

modeling = pytest.importorskip("transformers.models.qwen3_moe.modeling_qwen3_moe")
if not hasattr(modeling, "Qwen3MoeTopKRouter"):
    pytest.skip("Qwen3-MoE R3 replay targets the transformers 5 router", allow_module_level=True)

from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM  # noqa: E402

from verl.models.transformers.gemma4_router_replay import validate_routed_experts  # noqa: E402
from verl.models.transformers.qwen3_moe_router_replay import (  # noqa: E402
    patch_qwen3_moe_router_replay,
    qwen3_moe_router_replay_installed,
)

LAYERS, EXPERTS, TOP_K = 3, 8, 2


def _model(norm_topk_prob=True):
    torch.manual_seed(0)
    config = Qwen3MoeConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, moe_intermediate_size=16, num_hidden_layers=LAYERS,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16, num_experts=EXPERTS, num_experts_per_tok=TOP_K,
        norm_topk_prob=norm_topk_prob, max_position_embeddings=64, attn_implementation="eager",
    )
    model = Qwen3MoeForCausalLM(config).float().eval()
    with torch.no_grad():  # the init leaves the router at zeros, which would make every route a tie
        for layer in model.model.layers:
            layer.mlp.gate.weight.normal_(std=0.5)
    return model


def _native_routes(model, input_ids):
    """Run the unpatched routing and record each layer's top-k as a [batch, seq, layers, top_k] tensor."""
    routes, hooks = {}, []
    for idx, layer in enumerate(model.model.layers):
        hooks.append(layer.mlp.gate.register_forward_hook(lambda m, a, out, idx=idx: routes.__setitem__(idx, out[2])))
    logits = model(input_ids=input_ids).logits
    for hook in hooks:
        hook.remove()
    batch, seq = input_ids.shape
    return logits, torch.stack([routes[i].reshape(batch, seq, TOP_K) for i in range(LAYERS)], dim=2)


@pytest.mark.parametrize("norm_topk_prob", [True, False])
def test_replaying_native_routes_reproduces_native_forward_exactly(norm_topk_prob):
    assert patch_qwen3_moe_router_replay() and qwen3_moe_router_replay_installed()
    model = _model(norm_topk_prob)
    input_ids = torch.randint(0, 64, (2, 7))
    native_logits, routes = _native_routes(model, input_ids)
    replay_logits = model(input_ids=input_ids, routed_experts=routes.to(torch.int32)).logits
    torch.testing.assert_close(replay_logits, native_logits, rtol=0, atol=0)


def test_replayed_routes_drive_the_experts_and_backpropagate_into_the_router():
    patch_qwen3_moe_router_replay()
    model = _model()
    input_ids = torch.randint(0, 64, (2, 5))
    native_logits, routes = _native_routes(model, input_ids)
    forced = (routes + 3) % EXPERTS  # different experts at every token and layer
    used = []
    hook = model.model.layers[1].mlp.experts.register_forward_hook(lambda m, args, out: used.append(args[1]))
    model.train()
    replay_logits = model(input_ids=input_ids, routed_experts=forced).logits
    hook.remove()
    torch.testing.assert_close(used[0], forced[:, :, 1, :].reshape(-1, TOP_K).long())
    assert not torch.allclose(replay_logits, native_logits)
    replay_logits.square().mean().backward()
    for layer in model.model.layers:
        grad = layer.mlp.gate.weight.grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    # the replay state is scoped to the forward: a plain call afterwards routes natively again
    model.eval()
    torch.testing.assert_close(model(input_ids=input_ids).logits, native_logits)


def test_validate_routed_experts_uses_qwen_config_names():
    config = _model().config
    routes = torch.randint(1, EXPERTS, (1, 4, LAYERS, TOP_K))
    validate_routed_experts(routes, config, expected_batch_size=1, expected_sequence_length=4)
    with pytest.raises(ValueError, match="top_k"):
        validate_routed_experts(routes[..., :1], config)
    with pytest.raises(ValueError, match="IDs must be"):
        validate_routed_experts(torch.full((1, 4, LAYERS, TOP_K), EXPERTS), config)


def test_fsdp_engine_accepts_qwen3_moe_r3_routes():
    from verl.utils import tensordict_utils as tu
    from verl.utils.dataset.dataset_utils import DatasetPadMode
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead

    offsets = torch.tensor([0, 3, 4], dtype=torch.int64)
    routes = torch.nested.nested_tensor_from_jagged(torch.randint(1, EXPERTS, (4, LAYERS, TOP_K)), offsets)
    micro_batch = tu.get_tensordict(
        {
            "input_ids": torch.nested.nested_tensor_from_jagged(torch.tensor([1, 2, 3, 4]), offsets),
            "position_ids": torch.nested.nested_tensor_from_jagged(torch.tensor([0, 1, 2, 0]), offsets),
            "temperature": torch.ones(2),
            "routed_experts": routes,
        },
        non_tensor_dict={
            "use_remove_padding": False, "use_fused_kernels": False, "pad_mode": DatasetPadMode.NO_PADDING,
            "pad_token_id": 0, "enable_routing_replay": True,
        },
    )
    engine = object.__new__(FSDPEngineWithLMHead)
    engine.module = SimpleNamespace(config=_model().config)
    engine.engine_config = SimpleNamespace(router_replay=SimpleNamespace(mode="R3"))
    engine.use_ulysses_sp = False
    model_inputs, _ = engine.prepare_model_inputs(micro_batch)
    assert model_inputs["routed_experts"].shape == (2, 3, LAYERS, TOP_K)
