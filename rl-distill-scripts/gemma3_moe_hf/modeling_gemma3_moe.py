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

"""Gemma 3 text model with top-1 MoE feed-forward layers.

This remote-code module mirrors Transformers' Gemma3 text stack and replaces
each dense MLP with a router plus duplicated Gemma MLP experts. The routing
semantics intentionally match the Megatron configuration used for the upcycled
checkpoints: top-1, softmax score function, and ``router_pre_softmax=False``.
In that setting the selected expert output is not multiplied by a softmax
probability because the softmax is taken over a single selected logit.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3Attention,
    Gemma3PreTrainedModel,
    Gemma3RMSNorm,
    Gemma3RotaryEmbedding,
    Gemma3TextScaledWordEmbedding,
)

from .configuration_gemma3_moe import Gemma3MoeConfig


class Gemma3MoeExpert(nn.Module):
    def __init__(self, config: Gemma3MoeConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.post_layernorm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.act_fn = ACT2FN[config.hidden_activation]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        hidden_states = self.down_proj(hidden_states)
        return self.post_layernorm(hidden_states)


class Gemma3MoeMLP(nn.Module):
    def __init__(self, config: Gemma3MoeConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.router_pre_softmax = config.router_pre_softmax
        self.router_score_function = config.router_score_function
        self.router_dtype = config.router_dtype
        self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList([Gemma3MoeExpert(config) for _ in range(config.num_experts)])

        if self.num_experts_per_tok != 1:
            raise ValueError("Gemma3MoeMLP currently supports top-1 routing only.")
        if self.router_score_function != "softmax":
            raise ValueError("Gemma3MoeMLP currently supports the softmax router score function only.")

    def _router_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.router_dtype == "fp32":
            hidden_states = hidden_states.float()
            weight = self.router.weight.float()
            return torch.nn.functional.linear(hidden_states, weight)
        if self.router_dtype == "fp64":
            hidden_states = hidden_states.double()
            weight = self.router.weight.double()
            return torch.nn.functional.linear(hidden_states, weight)
        return self.router(hidden_states)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape
        flat_states = hidden_states.reshape(-1, original_shape[-1])
        router_logits = self._router_logits(flat_states)

        if self.router_pre_softmax:
            router_scores = torch.softmax(router_logits, dim=-1, dtype=torch.float32)
            route_weights, selected_experts = torch.topk(router_scores, k=1, dim=-1)
            route_weights = route_weights.to(flat_states.dtype).squeeze(-1)
        else:
            selected_experts = torch.argmax(router_logits, dim=-1, keepdim=True)
            route_weights = torch.ones(flat_states.shape[0], dtype=flat_states.dtype, device=flat_states.device)

        selected_experts = selected_experts.squeeze(-1)
        output = torch.empty_like(flat_states)
        for expert_idx, expert in enumerate(self.experts):
            token_mask = selected_experts == expert_idx
            if not torch.any(token_mask):
                continue
            expert_output = expert(flat_states[token_mask])
            output[token_mask] = expert_output * route_weights[token_mask].unsqueeze(-1)

        return output.reshape(original_shape)


class Gemma3MoeDecoderLayer(nn.Module):
    def __init__(self, config: Gemma3MoeConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.attention_type = config.layer_types[layer_idx]
        self.self_attn = Gemma3Attention(config=config, layer_idx=layer_idx)
        self.mlp = Gemma3MoeMLP(config)
        self.input_layernorm = Gemma3RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma3RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma3RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings_global: torch.Tensor,
        position_embeddings_local: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, ...]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        position_embeddings = position_embeddings_local if self.self_attn.is_sliding else position_embeddings_global
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        return outputs


class Gemma3MoePreTrainedModel(Gemma3PreTrainedModel):
    config_class = Gemma3MoeConfig
    _no_split_modules = ["Gemma3MoeDecoderLayer"]


class Gemma3MoeModel(Gemma3MoePreTrainedModel):
    config_class = Gemma3MoeConfig
    config: Gemma3MoeConfig

    def __init__(self, config: Gemma3MoeConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = Gemma3TextScaledWordEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            embed_scale=self.config.hidden_size**0.5,
        )
        self.layers = nn.ModuleList(
            [Gemma3MoeDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Gemma3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast | tuple:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None and not self.training:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            try:
                causal_mask_mapping = {
                    "full_attention": create_causal_mask(**mask_kwargs),
                    "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
                }
            except TypeError:
                mask_kwargs["input_embeds"] = mask_kwargs.pop("inputs_embeds")
                mask_kwargs["cache_position"] = cache_position
                causal_mask_mapping = {
                    "full_attention": create_causal_mask(**mask_kwargs),
                    "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
                }

        hidden_states = inputs_embeds
        position_embeddings_by_type = {
            layer_type: self.rotary_emb(hidden_states, position_ids, layer_type)
            for layer_type in set(self.config.layer_types)
        }

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                position_embeddings_global=position_embeddings_by_type[decoder_layer.attention_type],
                position_embeddings_local=position_embeddings_by_type[decoder_layer.attention_type],
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                v for v in [hidden_states, past_key_values, all_hidden_states, all_self_attns] if v is not None
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class Gemma3MoeForCausalLM(Gemma3MoePreTrainedModel, GenerationMixin):
    config_class = Gemma3MoeConfig
    _tied_weights_keys = ["lm_head.weight"]
    config: Gemma3MoeConfig
    base_model_prefix = "model"

    def __init__(self, config: Gemma3MoeConfig):
        super().__init__(config)
        self.model = Gemma3MoeModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast | tuple:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        if not return_dict:
            output = (logits, outputs.past_key_values, outputs.hidden_states, outputs.attentions)
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
