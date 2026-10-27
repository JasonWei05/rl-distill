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


class _RouterStats:
    """Per-forward MoE router statistics (monitoring only; nothing here enters the loss).

    ``start(token_mask)`` arms collection for one model forward, every patched router call then records one layer, and
    ``stop()`` returns layer-averaged scalars. Padding tokens are excluded via ``token_mask`` ([batch, seq], 1 = real
    token; None = every token is real). Collection is off outside start/stop, so activation-checkpoint recomputation
    in backward is never double counted.
    """

    def __init__(self):
        self.active = False
        self.token_mask = None
        self.layers: list[dict[str, torch.Tensor]] = []

    def start(self, token_mask: torch.Tensor | None) -> None:
        self.active, self.layers = True, []
        self.token_mask = None if token_mask is None else token_mask.reshape(-1).bool()

    def stop(self) -> dict[str, float]:
        self.active, layers, self.layers, self.token_mask = False, self.layers, [], None
        if not layers:
            return {}
        stacked = {key: torch.stack([layer[key] for layer in layers]) for key in layers[0]}
        out = {f"moe/{key}": float(value.mean()) for key, value in stacked.items()}
        out["moe/max_expert_load_worst_layer"] = float(stacked["max_expert_load"].max())
        return out

    @torch.no_grad()
    def record(self, router_logits: torch.Tensor, top_k_index: torch.Tensor, replayed: bool) -> None:
        logits = router_logits.float()
        if self.token_mask is not None and self.token_mask.numel() == logits.shape[0]:
            keep = self.token_mask.to(logits.device)
            logits, top_k_index = logits[keep], top_k_index[keep]
        if logits.shape[0] == 0:
            return
        num_experts, top_k = logits.shape[-1], top_k_index.shape[-1]
        probs = torch.softmax(logits, dim=-1)
        # fraction of top-k assignments per expert (x E: 1.0 = uniform)
        load = torch.bincount(top_k_index.reshape(-1), minlength=num_experts).float() / top_k_index.numel()
        stats = {
            # HF Qwen3-MoE load_balancing_loss_func (Switch aux loss); = top_k when perfectly balanced
            "aux_loss": num_experts * (load * top_k * probs.mean(0)).sum(),
            "z_loss": torch.logsumexp(logits, dim=-1).square().mean(),  # ST-MoE router z-loss
            "router_entropy": -(probs * torch.log(probs.clamp_min(1e-30))).sum(-1).mean(),
            "max_expert_load": load.max() * num_experts,
            "min_expert_load": load.min() * num_experts,
            "dead_expert_frac": (load == 0).float().mean(),
        }
        if replayed:  # share of the rollout's (vLLM's) chosen experts that the trainer's own router would also pick
            own = probs.topk(top_k, dim=-1).indices
            stats["r3_route_agreement"] = (top_k_index.unsqueeze(-1) == own.unsqueeze(-2)).any(-1).float().mean()
        self.layers.append(stats)


ROUTER_STATS = _RouterStats()


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
            router_logits, top_k_weights, top_k_index = original_router_forward(self, hidden_states)
        else:
            hidden_states = hidden_states.reshape(-1, self.hidden_dim)
            router_logits = torch.nn.functional.linear(hidden_states, self.weight)
            top_k_weights, top_k_index = replay_qwen3_moe_router(
                router_logits, routed_experts, top_k=self.top_k, norm_topk_prob=self.norm_topk_prob
            )
        if ROUTER_STATS.active:
            ROUTER_STATS.record(router_logits, top_k_index, replayed=routed_experts is not None)
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
