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

"""Native vLLM execution for the Gemma 3 dense-to-top-1-MoE checkpoint.

The checkpoint's remote Transformers model is useful as a portability
reference, but vLLM's generic Transformers backend takes a different execution
path from native Gemma 3 (notably for attention/cache handling).  This module
keeps vLLM's native Gemma 3 attention, KV cache, tensor-parallel linear layers,
and residual fusion, while replacing just the MLP with our per-expert
post-RMSNorm top-1 MoE block.
"""

from __future__ import annotations

from collections.abc import Iterable
from itertools import islice

import torch
from torch import nn
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.gemma3 import Gemma3Attention, Gemma3MLP
from vllm.model_executor.models.interfaces import SupportsLoRA, SupportsPP
from vllm.model_executor.models.utils import (
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors


def _capture_routes(layer_idx: int, selected_experts: torch.Tensor) -> None:
    """Feed vLLM's optional routed-expert capture without requiring FusedMoE."""
    try:
        from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
            RoutedExpertsCapturer,
        )

        capturer = RoutedExpertsCapturer.get_instance()
        if capturer is not None and capturer._device_buffer is not None:
            capturer.capture(layer_idx, selected_experts.unsqueeze(-1))
    except ImportError:
        # The model is sometimes inspected outside a live vLLM worker.
        return


class Gemma3MoeExpert(Gemma3MLP):
    """One full Gemma3 MLP plus the upcycled per-expert output RMSNorm."""

    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        hidden_activation: str,
        rms_norm_eps: float,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_activation=hidden_activation,
            quant_config=quant_config,
            prefix=prefix,
        )
        self.post_layernorm = GemmaRMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.post_layernorm(super().forward(hidden_states))


class Gemma3MoeMLP(nn.Module):
    """Top-1 routed Gemma3 experts with an exact dense-init fast path."""

    def __init__(
        self,
        config,
        *,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.num_experts = int(config.gemma3_moe_num_experts)
        self.canonical_dense_init = bool(getattr(config, "gemma3_moe_canonical_dense_init", False))
        # Routers are replicated in the checkpoint topology.  Keep this a
        # plain parameterized linear layer: it is tiny, works at every TP
        # degree, and preserves the checkpoint's exact name.
        self.router = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                Gemma3MoeExpert(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.intermediate_size,
                    hidden_activation=config.hidden_activation,
                    rms_norm_eps=config.rms_norm_eps,
                    quant_config=quant_config,
                    prefix=f"{prefix}.experts.{expert_idx}",
                )
                for expert_idx in range(self.num_experts)
            ]
        )

    def forward(self, hidden_states: torch.Tensor, *, layer_idx: int) -> torch.Tensor:
        original_shape = hidden_states.shape
        flat_states = hidden_states.reshape(-1, original_shape[-1])
        selected_experts = self.router(flat_states).argmax(dim=-1)
        _capture_routes(layer_idx, selected_experts)

        if self.canonical_dense_init:
            # Each expert is an exact dense copy at upcycle time.  Evaluating
            # expert 0 on the full batch retains the same native GEMM shape as
            # dense Gemma3 and therefore removes route-partition drift.
            return self.experts[0](flat_states).reshape(original_shape)

        output = torch.empty_like(flat_states)
        for expert_idx, expert in enumerate(self.experts):
            token_mask = selected_experts == expert_idx
            if torch.any(token_mask):
                output[token_mask] = expert(flat_states[token_mask])
        return output.reshape(original_shape)


class Gemma3MoeDecoderLayer(nn.Module):
    """Native Gemma3 decoder layer with a per-expert post-norm MoE MLP."""

    def __init__(
        self,
        config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.self_attn = Gemma3Attention(
            config=config,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            cache_config=cache_config,
            quant_config=quant_config,
            attn_logits_soft_cap=None,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = Gemma3MoeMLP(config, quant_config=quant_config, prefix=f"{prefix}.mlp")
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # ``post_feedforward_layernorm`` intentionally lives inside each
        # expert, matching the HF upcycle checkpoint.
        self.layer_idx = int(prefix.rsplit(".", 1)[-1])

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, residual = self.pre_feedforward_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states, layer_idx=self.layer_idx)
        return hidden_states, residual


class Gemma3MoeModel(nn.Module):
    """Text backbone matching vLLM's native Gemma3 model interface."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.embed_tokens",
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Gemma3MoeDecoderLayer(
                config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.register_buffer(
            "normalizer",
            torch.tensor(config.hidden_size**0.5),
            persistent=False,
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids) * self.normalizer

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            hidden_states = inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(positions, hidden_states, residual, **kwargs)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states, "residual": residual})
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Gemma3MoeForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
    """vLLM-native causal LM wrapper for ``Gemma3MoeForCausalLM``."""

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.model = Gemma3MoeModel(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if config.tie_word_embeddings:
            self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        self.logits_processor = LogitsProcessor(
            config.vocab_size,
            soft_cap=config.final_logit_softcapping,
        )
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load native packed projections from the unpacked HF checkpoint.

        ``AutoWeightsLoader`` deliberately does not infer that a pair of
        arbitrary checkpoint tensors belongs in a packed projection.  Native
        Gemma 3 has a model-specific loader for that reason.  The MoE expert
        introduces another such packed pair (``gate_proj`` and ``up_proj``),
        so retain the native mapping here rather than silently loading only
        half of each expert MLP.
        """
        stacked_params_mapping = [
            # (packed parameter name, checkpoint tensor fragment, shard id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for checkpoint_name, loaded_weight in weights:
            # The tied head points at ``model.embed_tokens`` and must not be
            # loaded a second time (the checkpoint may nevertheless contain
            # the alias).
            if self.config.tie_word_embeddings and checkpoint_name == "lm_head.weight":
                continue

            name = checkpoint_name
            if (
                self.quant_config is not None
                and self.quant_config.get_name() == "gguf"
                and name.endswith("norm.weight")
            ):
                # See the corresponding native Gemma 3 loader.
                loaded_weight -= 1

            if self.quant_config is not None and (scale_name := self.quant_config.get_cache_scale(name)):
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight[0])
                loaded_params.add(scale_name)
                continue

            for param_name, shard_name, shard_id in stacked_params_mapping:
                if shard_name not in name:
                    continue
                name = name.replace(shard_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    break
                if is_pp_missing_parameter(name, self):
                    break
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None or is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        return loaded_params
