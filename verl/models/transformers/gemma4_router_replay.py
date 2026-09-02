# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# The replay selection is adapted from Prime-RL:
# https://github.com/PrimeIntellect-ai/prime-rl/blob/d8f3d01016f64e746584cef9b9bd75e0d5ca8fe9/src/prime_rl/trainer/models/layers/moe.py
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# The adapted source is licensed under the BSD-style license identified in
# the Prime-RL source file above.
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

"""Gemma 4 rollout-router replay for the Hugging Face/FSDP2 path.

The replay selection below is adapted from Prime-RL's BSD-attributed MoE router:
https://github.com/PrimeIntellect-ai/prime-rl/blob/d8f3d01016f64e746584cef9b9bd75e0d5ca8fe9/src/prime_rl/trainer/models/layers/moe.py

Prime-RL computes the current trainer router probabilities, gathers those
probabilities at the rollout-selected expert IDs, and renormalizes them.  This
file applies the same R3 behavior to Hugging Face Gemma 4 while retaining
Gemma 4's learned per-expert scale.  The surrounding integration is original
to verl and intentionally avoids copying Prime-RL's trainer or model stack.
"""

from __future__ import annotations

from functools import wraps

import torch

_INTEGER_DTYPES = frozenset(
    dtype
    for dtype in (
        torch.uint8,
        getattr(torch, "uint16", None),
        getattr(torch, "uint32", None),
        getattr(torch, "uint64", None),
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    )
    if dtype is not None
)
_ROUTED_EXPERTS_ATTR = "_verl_routed_experts"
_MISSING = object()


def _text_config(config):
    return getattr(config, "text_config", config)


def validate_routed_experts(
    routed_experts: torch.Tensor,
    config,
    *,
    expected_batch_size: int | None = None,
    expected_sequence_length: int | None = None,
) -> torch.Tensor:
    """Validate the dense ``[batch, sequence, layers, top_k]`` R3 tensor."""
    if not isinstance(routed_experts, torch.Tensor):
        raise TypeError(f"routed_experts must be a torch.Tensor, got {type(routed_experts)!r}")
    if routed_experts.is_nested:
        raise ValueError("routed_experts must be converted from nested to dense/packed form before model forward")
    if routed_experts.ndim != 4:
        raise ValueError(
            f"routed_experts must have shape [batch, sequence, layers, top_k], got {tuple(routed_experts.shape)}"
        )
    if routed_experts.dtype not in _INTEGER_DTYPES:
        raise TypeError(f"routed_experts must use an integer dtype, got {routed_experts.dtype}")

    text_config = _text_config(config)
    expected_layers = int(text_config.num_hidden_layers)
    expected_top_k = int(text_config.top_k_experts)
    if routed_experts.shape[2] != expected_layers:
        raise ValueError(f"routed_experts has {routed_experts.shape[2]} layers, expected {expected_layers} for Gemma 4")
    if routed_experts.shape[3] != expected_top_k:
        raise ValueError(f"routed_experts top_k is {routed_experts.shape[3]}, expected {expected_top_k} for Gemma 4")
    if expected_batch_size is not None and routed_experts.shape[0] != expected_batch_size:
        raise ValueError(f"routed_experts batch size is {routed_experts.shape[0]}, expected {expected_batch_size}")
    if expected_sequence_length is not None and routed_experts.shape[1] != expected_sequence_length:
        raise ValueError(
            f"routed_experts sequence length is {routed_experts.shape[1]}, expected {expected_sequence_length}"
        )

    if routed_experts.numel():
        minimum, maximum = torch.aminmax(routed_experts)
        minimum = int(minimum.item())
        maximum = int(maximum.item())
        num_experts = int(text_config.num_experts)
        if minimum < 0 or maximum >= num_experts:
            raise ValueError(f"routed_experts IDs must be in [0, {num_experts}), got min={minimum}, max={maximum}")
        if expected_top_k > 1 and not routed_experts.any().item():
            raise ValueError(
                "routed_experts from the rollout are entirely zero; rollout-side routing capture is not working"
            )
    return routed_experts


def routed_experts_to_model_input(
    routed_experts: torch.Tensor,
    *,
    use_remove_padding: bool,
    batch_size: int | None = None,
    max_sequence_length: int | None = None,
) -> torch.Tensor:
    """Convert verl's jagged routed-expert tensor to the Hugging Face input shape."""
    if not isinstance(routed_experts, torch.Tensor):
        raise TypeError(f"routed_experts must be a torch.Tensor, got {type(routed_experts)!r}")

    if use_remove_padding:
        if routed_experts.is_nested:
            values = routed_experts.values()
            if values.ndim != 3:
                raise ValueError(
                    f"nested routed_experts values must have shape [tokens, layers, top_k], got {tuple(values.shape)}"
                )
            return values.unsqueeze(0)
        if routed_experts.ndim != 4:
            raise ValueError(
                "packed routed_experts must have shape [batch, sequence, layers, top_k], "
                f"got {tuple(routed_experts.shape)}"
            )
        return routed_experts

    if routed_experts.is_nested:
        if batch_size is None or max_sequence_length is None:
            raise ValueError("batch_size and max_sequence_length are required to pad nested routed_experts")
        values = routed_experts.values()
        if values.ndim != 3:
            raise ValueError(
                f"nested routed_experts values must have shape [tokens, layers, top_k], got {tuple(values.shape)}"
            )
        return torch.nested.to_padded_tensor(
            routed_experts,
            padding=0,
            output_size=(batch_size, max_sequence_length, values.shape[-2], values.shape[-1]),
        )
    if routed_experts.ndim != 4:
        raise ValueError(
            f"padded routed_experts must have shape [batch, sequence, layers, top_k], got {tuple(routed_experts.shape)}"
        )
    return routed_experts


def replay_router_weights(
    router_probabilities: torch.Tensor,
    routed_experts: torch.Tensor,
    per_expert_scale: torch.Tensor,
    *,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather current router scores at rollout-selected IDs and renormalize."""
    if router_probabilities.ndim != 2:
        raise ValueError(
            f"router_probabilities must have shape [tokens, experts], got {tuple(router_probabilities.shape)}"
        )
    if routed_experts.ndim not in (2, 3):
        raise ValueError(
            f"layer routed_experts must have shape [tokens, top_k] or [batch, sequence, top_k], "
            f"got {tuple(routed_experts.shape)}"
        )
    if routed_experts.dtype not in _INTEGER_DTYPES:
        raise TypeError(f"routed_experts must use an integer dtype, got {routed_experts.dtype}")
    if routed_experts.device != router_probabilities.device:
        raise ValueError(
            "routed_experts and router_probabilities must be on the same device, got "
            f"{routed_experts.device} and {router_probabilities.device}"
        )

    top_k_index = routed_experts.reshape(-1, routed_experts.shape[-1])
    if top_k_index.shape != (router_probabilities.shape[0], top_k):
        raise ValueError(
            "layer routed_experts must align one route with every router token: "
            f"got {tuple(top_k_index.shape)}, expected {(router_probabilities.shape[0], top_k)}"
        )
    if per_expert_scale.ndim != 1 or per_expert_scale.shape[0] != router_probabilities.shape[1]:
        raise ValueError(
            "per_expert_scale must have one value per expert: "
            f"got {tuple(per_expert_scale.shape)}, expected {(router_probabilities.shape[1],)}"
        )

    top_k_index = top_k_index.to(torch.long)
    top_k_weights = router_probabilities.gather(dim=-1, index=top_k_index)
    # Prime-RL uses a small denominator epsilon so a severely off-policy route
    # whose selected fp32 probabilities all underflow cannot create NaNs.
    top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-20)
    top_k_weights = top_k_weights * per_expert_scale[top_k_index]
    return top_k_weights, top_k_index


def patch_gemma4_router_replay() -> bool:
    """Install an idempotent, narrow router-replay patch on HF Gemma 4."""
    try:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer, Gemma4TextRouter
    except (ImportError, ModuleNotFoundError):
        return False

    if getattr(Gemma4TextRouter, "_verl_router_replay_patched", False):
        return True

    original_router_forward = Gemma4TextRouter.forward

    @wraps(original_router_forward)
    def router_forward_with_replay(self, hidden_states: torch.Tensor):
        router_probabilities, top_k_weights, top_k_index = original_router_forward(self, hidden_states)
        routed_experts = getattr(self, _ROUTED_EXPERTS_ATTR, None)
        if routed_experts is None:
            return router_probabilities, top_k_weights, top_k_index

        top_k_weights, top_k_index = replay_router_weights(
            router_probabilities,
            routed_experts,
            self.per_expert_scale,
            top_k=int(self.config.top_k_experts),
        )
        return router_probabilities, top_k_weights, top_k_index

    original_decoder_forward = Gemma4TextDecoderLayer.forward

    @wraps(original_decoder_forward)
    def decoder_forward_with_replay(self, *args, **kwargs):
        routed_experts = kwargs.pop("routed_experts", None)
        if routed_experts is None or not self.enable_moe_block:
            return original_decoder_forward(self, *args, **kwargs)
        if routed_experts.ndim != 4:
            raise ValueError(
                f"routed_experts must have shape [batch, sequence, layers, top_k], got {tuple(routed_experts.shape)}"
            )
        expected_layers = int(self.config.num_hidden_layers)
        if routed_experts.shape[2] != expected_layers:
            raise ValueError(
                f"routed_experts has {routed_experts.shape[2]} layers, expected {expected_layers} for Gemma 4"
            )

        layer_routed_experts = routed_experts[:, :, self.layer_idx, :]
        previous = getattr(self.router, _ROUTED_EXPERTS_ATTR, _MISSING)
        setattr(self.router, _ROUTED_EXPERTS_ATTR, layer_routed_experts)
        try:
            return original_decoder_forward(self, *args, **kwargs)
        finally:
            if previous is _MISSING:
                delattr(self.router, _ROUTED_EXPERTS_ATTR)
            else:
                setattr(self.router, _ROUTED_EXPERTS_ATTR, previous)

    Gemma4TextRouter.forward = router_forward_with_replay
    Gemma4TextRouter._verl_router_replay_patched = True
    Gemma4TextRouter._verl_router_replay_original_forward = original_router_forward
    Gemma4TextDecoderLayer.forward = decoder_forward_with_replay
    Gemma4TextDecoderLayer._verl_router_replay_original_forward = original_decoder_forward
    return True
