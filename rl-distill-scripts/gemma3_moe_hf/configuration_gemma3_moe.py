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

"""Configuration for Gemma 3 text models upcycled with top-1 MoE MLPs."""

import math

from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig


class Gemma3MoeConfig(Gemma3TextConfig):
    model_type = "gemma3_moe"

    def __init__(
        self,
        gemma3_moe_num_experts: int | None = None,
        num_experts_per_tok: int = 1,
        router_pre_softmax: bool = False,
        router_score_function: str = "softmax",
        router_aux_loss_coef: float = 1e-3,
        router_dtype: str | None = None,
        gemma3_moe_canonical_dense_init: bool = False,
        **kwargs,
    ):
        # Earlier exports stored the expert count under the generic MoE keys
        # `num_experts`/`num_local_experts`. vLLM's Transformers backend treats
        # any config exposing those keys as a fused-MoE model and swaps the
        # experts for a FusedMoE module, which cannot represent the per-expert
        # post-MLP RMSNorm of this architecture. Fold them into a
        # Gemma3-specific key. The production vLLM path is the registered
        # native Gemma3-MoE plugin; this also keeps the reference Transformers
        # backend structurally correct.
        legacy_num_experts = kwargs.pop("num_experts", None)
        legacy_num_local_experts = kwargs.pop("num_local_experts", None)
        if (
            legacy_num_experts is not None
            and legacy_num_local_experts is not None
            and int(legacy_num_experts) != int(legacy_num_local_experts)
        ):
            raise ValueError(
                f"Legacy num_experts and num_local_experts disagree: {legacy_num_experts} != {legacy_num_local_experts}"
            )
        legacy_expert_count = legacy_num_experts
        if legacy_expert_count is None:
            legacy_expert_count = legacy_num_local_experts
        super().__init__(**kwargs)
        if gemma3_moe_num_experts is None:
            gemma3_moe_num_experts = 2 if legacy_expert_count is None else legacy_expert_count
        self.gemma3_moe_num_experts = int(gemma3_moe_num_experts)
        self.num_experts_per_tok = int(num_experts_per_tok)
        self.router_pre_softmax = bool(router_pre_softmax)
        self.router_score_function = router_score_function
        self.router_aux_loss_coef = float(router_aux_loss_coef)
        self.router_dtype = router_dtype
        if self.gemma3_moe_num_experts <= 0:
            raise ValueError("gemma3_moe_num_experts must be positive")
        if self.num_experts_per_tok != 1:
            raise ValueError("Gemma3MoeConfig supports top-1 routing only")
        if self.router_pre_softmax:
            raise ValueError(
                "Gemma3 dense upcycling requires router_pre_softmax=False so "
                "the selected top-1 expert has combine weight 1"
            )
        if self.router_score_function != "softmax":
            raise ValueError("Gemma3MoeConfig supports the softmax router score function only")
        if not math.isfinite(self.router_aux_loss_coef) or self.router_aux_loss_coef <= 0:
            raise ValueError("router_aux_loss_coef must be positive and finite")
        if self.router_dtype not in (None, "fp32", "fp64"):
            raise ValueError("router_dtype must be null, 'fp32', or 'fp64'")
        # Explicit correctness-only mode.  The runtime still evaluates and
        # records router decisions, but executes the identical dense expert
        # over the full token batch so bf16 route partitioning cannot perturb
        # an otherwise exact upcycle.
        self.gemma3_moe_canonical_dense_init = bool(gemma3_moe_canonical_dense_init)

        # Transformers 5 standardized Gemma 3 RoPE config under
        # `rope_parameters`, while older Gemma 3 model code still reads these
        # aliases. Keep both names populated so the remote-code model works
        # across the Transformers versions used by training and vLLM.
        rope_parameters = getattr(self, "rope_parameters", None) or {}
        full_rope = rope_parameters.get("full_attention") or {}
        sliding_rope = rope_parameters.get("sliding_attention") or {}
        self.rope_theta = full_rope.get("rope_theta", getattr(self, "rope_theta", 1_000_000.0))
        self.rope_local_base_freq = sliding_rope.get("rope_theta", getattr(self, "rope_local_base_freq", 10_000.0))

        # vLLM's Transformers backend reads this from the config object when
        # replacing nn.Linear with tensor-parallel linear layers.
        self.base_model_tp_plan = {
            r"layers\.[0-9]+\.self_attn\.q_proj": "colwise_rep",
            r"layers\.[0-9]+\.self_attn\.k_proj": "colwise_rep",
            r"layers\.[0-9]+\.self_attn\.v_proj": "colwise_rep",
            r"layers\.[0-9]+\.self_attn\.o_proj": "rowwise_rep",
            r"layers\.[0-9]+\.mlp\.experts\.[0-9]+\.gate_proj": "colwise",
            r"layers\.[0-9]+\.mlp\.experts\.[0-9]+\.up_proj": "colwise",
            r"layers\.[0-9]+\.mlp\.experts\.[0-9]+\.down_proj": "rowwise",
            r"layers\.[0-9]+\.mlp\.router": "replicate",
        }
