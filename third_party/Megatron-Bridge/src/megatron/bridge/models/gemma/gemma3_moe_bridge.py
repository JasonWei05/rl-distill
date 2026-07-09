# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import math

import torch
from megatron.core.models.gpt.gpt_model import GPTModel

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    QKVMapping,
)
from megatron.bridge.models.conversion.transformers_compat import (
    rope_local_base_freq_from_hf,
    rope_scaling_factor_from_hf,
    rope_theta_from_hf,
)
from megatron.bridge.models.gemma.gemma3_provider import (
    Gemma3MoEModelProvider,
    gemma3_moe_layer_spec,
)
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM


# Register Gemma3-specific module types for AutoMapping
AutoMapping.register_module_type("Gemma3TEDotProductAttention", "replicated")
AutoMapping.register_module_type("TERowParallelLinearLayerNorm", "row")
AutoMapping.register_module_type("TERowParallelLinearTorchRMSNorm", "row")


def _num_experts_from_hf_config(hf_config) -> int:
    """Read the expert count, tolerating the legacy generic MoE key names."""
    for key in ("gemma3_moe_num_experts", "num_experts", "num_local_experts"):
        value = getattr(hf_config, key, None)
        if value:
            return int(value)
    raise ValueError("Gemma3 MoE config does not declare an expert count")


@MegatronModelBridge.register_bridge(
    source="Gemma3MoeForCausalLM",
    target=GPTModel,
    provider=Gemma3MoEModelProvider,
    model_type="gemma3_moe",
)
class Gemma3MoeBridge(MegatronModelBridge):
    """
    Megatron Bridge for dense-to-MoE upcycled Gemma3 text models.

    The HF side is the remote-code ``Gemma3MoeForCausalLM``
    (rl-distill gemma3_moe_hf export): each dense MLP is replaced by a
    bias-free linear router plus full-size Gemma MLP experts that keep the
    dense model's post-MLP RMSNorm per expert. Routing is top-1 with
    post-top-k softmax (combine weight exactly 1.0).
    """

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> Gemma3MoEModelProvider:
        """Convert HuggingFace config to Gemma3MoEModelProvider."""
        provider = super().provider_bridge(hf_pretrained)
        hf_config = hf_pretrained.config

        # The MoE export is text-only, so unlike the dense Gemma3 bridge the
        # top-level config already carries the precision info.
        params_dtype = self.dtype_from_hf(hf_config, default=torch.float32)
        provider.fp16 = params_dtype == torch.float16
        provider.bf16 = params_dtype == torch.bfloat16
        provider.params_dtype = params_dtype
        provider.autocast_dtype = params_dtype

        # Gemma3-specific features not in CONFIG_MAPPING
        provider.window_size = hf_config.sliding_window
        provider.rotary_base = (
            rope_local_base_freq_from_hf(hf_config),
            rope_theta_from_hf(hf_config),
        )
        provider.softmax_scale = 1.0 / math.sqrt(hf_config.query_pre_attn_scalar)
        provider.rope_scaling_factor = rope_scaling_factor_from_hf(hf_config)

        # MoE topology. Top-1 routing with post-top-k softmax matches the
        # upcycled checkpoints; the router trains through the aux loss.
        provider.transformer_layer_spec = gemma3_moe_layer_spec
        provider.num_moe_experts = _num_experts_from_hf_config(hf_config)
        provider.moe_router_topk = int(getattr(hf_config, "num_experts_per_tok", 1))
        provider.moe_router_pre_softmax = bool(getattr(hf_config, "router_pre_softmax", False))
        provider.moe_router_score_function = str(getattr(hf_config, "router_score_function", "softmax"))
        provider.moe_router_load_balancing_type = "aux_loss"
        provider.moe_aux_loss_coeff = float(getattr(hf_config, "router_aux_loss_coef", 1e-3))
        provider.moe_ffn_hidden_size = hf_config.intermediate_size
        provider.moe_grouped_gemm = False
        provider.moe_token_dispatcher_type = "alltoall"
        provider.moe_permute_fusion = True

        return provider

    @classmethod
    def megatron_to_hf_config(cls, provider: Gemma3MoEModelProvider) -> dict:
        """Convert Gemma3 MoE provider config back to HuggingFace config."""
        hf_config = super().megatron_to_hf_config(provider)

        if isinstance(provider.rotary_base, tuple):
            rope_local_base_freq, rope_theta = provider.rotary_base
            hf_config["rope_local_base_freq"] = rope_local_base_freq
            hf_config["rope_theta"] = rope_theta

        hf_config["sliding_window"] = provider.window_size

        if provider.softmax_scale:
            query_pre_attn_scalar = 1.0 / (provider.softmax_scale**2)
            rounded = round(query_pre_attn_scalar)
            if math.isclose(query_pre_attn_scalar, rounded, rel_tol=0.0, abs_tol=1e-9):
                query_pre_attn_scalar = rounded
            hf_config["query_pre_attn_scalar"] = query_pre_attn_scalar

        if getattr(provider, "rope_scaling_factor", 1.0) != 1.0:
            hf_config["rope_scaling"] = {
                "factor": provider.rope_scaling_factor,
                "type": "linear",
            }

        hf_config["gemma3_moe_num_experts"] = provider.num_moe_experts
        hf_config["num_experts_per_tok"] = provider.moe_router_topk
        hf_config["router_pre_softmax"] = provider.moe_router_pre_softmax
        hf_config["router_score_function"] = provider.moe_router_score_function
        hf_config["router_aux_loss_coef"] = provider.moe_aux_loss_coeff

        return hf_config

    def mapping_registry(self) -> MegatronMappingRegistry:
        mapping = {
            # word embedding
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            # attention
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.self_attention.q_layernorm.weight": "model.layers.*.self_attn.q_norm.weight",
            "decoder.layers.*.self_attention.k_layernorm.weight": "model.layers.*.self_attn.k_norm.weight",
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            "decoder.layers.*.self_attention.linear_proj.post_layernorm.weight": (
                "model.layers.*.post_attention_layernorm.weight"
            ),
            # moe mlp: standalone pre-MoE norm, router, and per-expert weights
            "decoder.layers.*.pre_mlp_layernorm.weight": "model.layers.*.pre_feedforward_layernorm.weight",
            "decoder.layers.*.mlp.router.weight": "model.layers.*.mlp.router.weight",
            "decoder.layers.*.mlp.experts.local_experts.*.linear_fc2.weight": (
                "model.layers.*.mlp.experts.*.down_proj.weight"
            ),
            "decoder.layers.*.mlp.experts.local_experts.*.linear_fc2.post_layernorm.weight": (
                "model.layers.*.mlp.experts.*.post_layernorm.weight"
            ),
            # final norm
            "decoder.final_layernorm.weight": "model.norm.weight",
        }
        mapping_list = []
        # Convert each dictionary entry to AutoMapping(megatron_param, hf_param)
        for megatron_param, hf_param in mapping.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        mapping_list.extend(
            [
                # QKV: Combine separate Q, K, V matrices into single QKV matrix
                QKVMapping(
                    megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                    q="model.layers.*.self_attn.q_proj.weight",
                    k="model.layers.*.self_attn.k_proj.weight",
                    v="model.layers.*.self_attn.v_proj.weight",
                ),
                # Gated MLP per expert: combine gate and up projections into FC1
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.experts.local_experts.*.linear_fc1.weight",
                    gate="model.layers.*.mlp.experts.*.gate_proj.weight",
                    up="model.layers.*.mlp.experts.*.up_proj.weight",
                ),
            ]
        )

        return MegatronMappingRegistry(*mapping_list)
