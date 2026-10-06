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

"""Qwen3-MoE rollout-router replay (R3) for the Hugging Face/FSDP2 path.

Same contract as ``gemma4_router_replay``: the trainer receives the expert IDs vLLM selected for every token and layer
(``routed_experts``, ``[batch, sequence, layers, top_k]``), and each sparse MoE block uses those IDs instead of its own
top-k. The weights are the *current* trainer router's softmax probabilities gathered at the replayed IDs and, when
``norm_topk_prob`` is set (Qwen3-30B-A3B), renormalized over them -- exactly what the native router computes for its own
top-k, so gradients still flow into the router. With the rollout's IDs equal to the trainer's own top-k the output is
unchanged.
"""

from __future__ import annotations

from functools import wraps

import torch

from verl.models.transformers.gemma4_router_replay import _INTEGER_DTYPES

_ROUTED_EXPERTS_ATTR = "_verl_routed_experts"
_MISSING = object()


def replay_qwen3_moe_router(
    router_logits: torch.Tensor, routed_experts: torch.Tensor, *, top_k: int, norm_topk_prob: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (weights, indices) for the rollout-selected experts, matching Qwen3MoeTopKRouter's arithmetic."""
    if router_logits.ndim != 2:
        raise ValueError(f"router_logits must have shape [tokens, experts], got {tuple(router_logits.shape)}")
    if routed_experts.dtype not in _INTEGER_DTYPES:
        raise TypeError(f"routed_experts must use an integer dtype, got {routed_experts.dtype}")
    top_k_index = routed_experts.reshape(-1, routed_experts.shape[-1])
    if top_k_index.shape != (router_logits.shape[0], top_k):
        raise ValueError(
            "layer routed_experts must align one route with every router token: "
            f"got {tuple(top_k_index.shape)}, expected {(router_logits.shape[0], top_k)}"
        )
    top_k_index = top_k_index.to(device=router_logits.device, dtype=torch.long)
    router_probs = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
    top_k_weights = router_probs.gather(dim=-1, index=top_k_index)
    if norm_topk_prob:
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
    return top_k_weights.to(router_logits.dtype), top_k_index


def qwen3_moe_router_replay_installed() -> bool:
    """True once the patch is active (it needs the transformers 5 Qwen3MoeTopKRouter; 4.x has a plain nn.Linear gate)."""
    try:
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeTopKRouter
    except (ImportError, ModuleNotFoundError):
        return False
    return bool(getattr(Qwen3MoeTopKRouter, "_verl_router_replay_patched", False))


def patch_qwen3_moe_router_replay() -> bool:
    """Install an idempotent router-replay patch on HF Qwen3-MoE (transformers 5 fused-expert layout)."""
    try:
        from transformers.models.qwen3_moe.modeling_qwen3_moe import (
            Qwen3MoeDecoderLayer,
            Qwen3MoeSparseMoeBlock,
            Qwen3MoeTopKRouter,
        )
    except (ImportError, ModuleNotFoundError):
        return False
    if getattr(Qwen3MoeTopKRouter, "_verl_router_replay_patched", False):
        return True

    original_router_forward = Qwen3MoeTopKRouter.forward

    @wraps(original_router_forward)
    def router_forward_with_replay(self, hidden_states: torch.Tensor):
        routed_experts = getattr(self, _ROUTED_EXPERTS_ATTR, None)
        if routed_experts is None:
            return original_router_forward(self, hidden_states)
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = torch.nn.functional.linear(hidden_states, self.weight)
        top_k_weights, top_k_index = replay_qwen3_moe_router(
            router_logits, routed_experts, top_k=self.top_k, norm_topk_prob=self.norm_topk_prob
        )
        return router_logits, top_k_weights, top_k_index

    original_decoder_forward = Qwen3MoeDecoderLayer.forward

    @wraps(original_decoder_forward)
    def decoder_forward_with_replay(self, *args, **kwargs):
        routed_experts = kwargs.pop("routed_experts", None)
        if routed_experts is None or not isinstance(self.mlp, Qwen3MoeSparseMoeBlock):
            return original_decoder_forward(self, *args, **kwargs)
        if routed_experts.ndim != 4:
            raise ValueError(
                f"routed_experts must have shape [batch, sequence, layers, top_k], got {tuple(routed_experts.shape)}"
            )
        # vLLM records one slot per decoder layer (indexed by the layer number in the module prefix).
        layer_idx = self.self_attn.layer_idx
        router = self.mlp.gate
        previous = getattr(router, _ROUTED_EXPERTS_ATTR, _MISSING)
        setattr(router, _ROUTED_EXPERTS_ATTR, routed_experts[:, :, layer_idx, :])
        try:
            return original_decoder_forward(self, *args, **kwargs)
        finally:
            if previous is _MISSING:
                delattr(router, _ROUTED_EXPERTS_ATTR)
            else:
                setattr(router, _ROUTED_EXPERTS_ATTR, previous)

    Qwen3MoeTopKRouter.forward = router_forward_with_replay
    Qwen3MoeTopKRouter._verl_router_replay_patched = True
    Qwen3MoeTopKRouter._verl_router_replay_original_forward = original_router_forward
    Qwen3MoeDecoderLayer.forward = decoder_forward_with_replay
    Qwen3MoeDecoderLayer._verl_router_replay_original_forward = original_decoder_forward
    return True
