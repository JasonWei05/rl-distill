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

from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig


class Gemma3MoeConfig(Gemma3TextConfig):
    model_type = "gemma3_moe"

    def __init__(
        self,
        num_experts: int = 2,
        num_experts_per_tok: int = 1,
        router_pre_softmax: bool = False,
        router_score_function: str = "softmax",
        router_aux_loss_coef: float = 1e-3,
        router_dtype: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.num_local_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.router_pre_softmax = router_pre_softmax
        self.router_score_function = router_score_function
        self.router_aux_loss_coef = router_aux_loss_coef
        self.router_dtype = router_dtype

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
