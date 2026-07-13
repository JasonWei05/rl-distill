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

import copy
import math
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Optional, Tuple, Union

import torch
from megatron.core.activations import fast_gelu
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.common.embeddings.language_model_embedding import (
    LanguageModelEmbedding,
)
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.gpt import GPTModel as MCoreGPTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer import (
    ModuleSpec,
    TransformerConfig,
    TransformerLayer,
    TransformerLayerSubmodules,
)
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnBackend, AttnMaskType
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.moe.experts import SequentialMLP
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from torch import Tensor

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.utils.import_utils import safe_import_from


TENorm, _ = safe_import_from("megatron.core.extensions.transformer_engine", "TENorm")
TELayerNormColumnParallelLinear, _ = safe_import_from(
    "megatron.core.extensions.transformer_engine", "TELayerNormColumnParallelLinear"
)
TEColumnParallelLinear, _ = safe_import_from("megatron.core.extensions.transformer_engine", "TEColumnParallelLinear")
TERowParallelLinear, _ = safe_import_from("megatron.core.extensions.transformer_engine", "TERowParallelLinear")
TEDotProductAttention, _ = safe_import_from("megatron.core.extensions.transformer_engine", "TEDotProductAttention")


@dataclass
class Gemma3ModelProvider(GPTModelProvider):
    """Configuration and provider for Megatron Core Gemma3 models."""

    seq_length: int = 131_072

    # embedding
    position_embedding_type: str = "rope"
    rotary_base: tuple = (10_000, 1_000_000)  # (local, global)
    share_embeddings_and_output_weights: bool = True

    # norm
    normalization: str = "RMSNorm"
    layernorm_zero_centered_gamma: bool = True  # x * (1 + w)
    layernorm_epsilon: float = 1e-6

    # attention
    qk_layernorm: bool = True
    window_size: tuple = 512  # local
    interleaved_attn_pattern: tuple = (5, 1)  # (local, global)
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    rope_scaling_factor: float = 1.0
    # Disable cuDNN attention since TE 1.8 does not support head dim > 128
    attention_backend: AttnBackend = AttnBackend.flash
    softmax_scale: float = 1.0 / math.sqrt(256)

    # mlp
    gated_linear_unit: bool = True
    add_bias_linear: bool = False
    activation_func: Callable = fast_gelu  # identical to openai_gelu

    # Do not change
    is_vision_language: bool = False
    flash_decode: bool = False
    transformer_layer_spec: Union[ModuleSpec, Callable[["Gemma3ModelProvider"], ModuleSpec]] = field(
        default_factory=lambda: gemma3_layer_spec
    )
    scatter_embedding_sequence_parallel: bool = True

    # Data type settings to match HF models
    bf16: bool = True
    fp16: bool = False
    params_dtype: torch.dtype = torch.bfloat16
    autocast_dtype: torch.dtype = torch.bfloat16

    def provide(self, pre_process=None, post_process=None, vp_stage=None) -> "MCoreGPTModel":
        """Configure and instantiate a Megatron Core Gemma3 model.

        Replaces the model's embedding and rope with customized Gemma3 ones.

        Args:
            pre_process: Whether to include pre-processing in the model
            post_process: Whether to include post-processing in the model
            vp_stage: Virtual pipeline stage

        Returns:
            MCoreGPTModel: Configured Megatron Core GPT model instance
        """
        rotary_base_local, rotary_base_global = self.rotary_base
        # Trick megatron's RotaryEmbedding to initialize the model successfully
        self.rotary_base = rotary_base_local
        model = super().provide(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
        self.rotary_base = (rotary_base_local, rotary_base_global)
        # Replace model's embedding and rope with customized ones
        if hasattr(model, "embedding"):
            model.embedding = Gemma3LanguageModelEmbedding(
                config=self,
                vocab_size=self.vocab_size,
                max_sequence_length=self.seq_length,
                position_embedding_type=self.position_embedding_type,
                scatter_to_sequence_parallel=self.scatter_embedding_sequence_parallel,
            )
        model.rotary_pos_emb = Gemma3RotaryEmbedding(
            kv_channels=self.kv_channels,
            rotary_percent=1.0,
            rotary_interleaved=self.rotary_interleaved,
            seq_len_interpolation_factor=self.seq_len_interpolation_factor,
            rotary_base=rotary_base_global,
            rope_scaling=False,
            rope_scaling_factor=self.rope_scaling_factor,
            use_cpu_initialization=self.use_cpu_initialization,
            rotary_base_local=rotary_base_local,
        )
        if hasattr(model, "embedding") or hasattr(model, "output_layer"):
            model.setup_embeddings_and_output_layer()
        return model


def gemma3_layer_spec(config) -> ModuleSpec:
    """Gemma3 custom layer spec."""
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=Gemma3SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TELayerNormColumnParallelLinear,
                    core_attention=Gemma3TEDotProductAttention,  # mixed gloabl/local attn
                    q_layernorm=TENorm if config.qk_layernorm else None,
                    k_layernorm=TENorm if config.qk_layernorm else None,
                    linear_proj=TERowParallelLinearLayerNorm,  # post attn RMSNorm
                ),
            ),
            self_attn_bda=get_bias_dropout_add,  # residual link
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=TELayerNormColumnParallelLinear,
                    linear_fc2=TERowParallelLinearLayerNorm,  # post mlp RMSNorm
                ),
            ),
            mlp_bda=get_bias_dropout_add,  # residual link
        ),
    )


class Gemma3SelfAttention(SelfAttention):
    """Gemma3 self attention.

    Uses local rope embedding for local layers,
    global rope embedding for global layers.
    """

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        key_value_states: Optional[Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        rotary_pos_cos_sin: Optional[Tuple[Tensor, Tensor]] = None,
        attention_bias: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[int] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Switch to either local or global rope embedding before forward"""
        assert isinstance(rotary_pos_emb, torch.Tensor) and rotary_pos_emb.ndim >= 1 and rotary_pos_emb.size(0) == 2
        assert rotary_pos_cos is None and rotary_pos_sin is None

        if _is_local_attn_layer(self.layer_number, self.config.interleaved_attn_pattern):
            final_rotary_pos_emb = rotary_pos_emb[0]
        else:
            final_rotary_pos_emb = rotary_pos_emb[1]
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            key_value_states=key_value_states,
            inference_context=inference_context,
            rotary_pos_emb=final_rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            inference_params=inference_params,
        )


class Gemma3TEDotProductAttention(TEDotProductAttention):
    """Gemma3 core attention.

    Switches between global and local sliding window attention
    based on the layer_number and pre-defined layer pattern.
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: Optional[float] = None,
        **kwargs,
    ):
        # Overwrite config.window_size based on layer_number
        config = copy.deepcopy(config)
        if _is_local_attn_layer(layer_number, config.interleaved_attn_pattern):
            # local attention, (q, k)
            config.window_size = (config.window_size - 1, 0)
        else:
            # global attention
            config.window_size = None

        # The VL model calculates mask manually
        if config.is_vision_language:
            attn_mask_type = AttnMaskType.arbitrary

        super().__init__(
            config=config,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type=attention_type,
            attention_dropout=attention_dropout,
            **kwargs,
        )


class Gemma3LanguageModelEmbedding(LanguageModelEmbedding):
    """Gemma3 language token embedding.

    Adds a normalization to the embedding.
    """

    def forward(self, input_ids: Tensor, position_ids: Tensor, tokentype_ids: int = None) -> Tensor:
        """Calculate embedding and normalize"""
        embeddings = super().forward(input_ids, position_ids, tokentype_ids)
        embeddings = embeddings * (self.config.hidden_size**0.5)
        return embeddings


class Gemma3RotaryEmbedding(RotaryEmbedding):
    """Gemma3 position rope embedding.

    Calculates rope embeddings for both local and global attention layers.
    """

    def __init__(
        self,
        rope_scaling: bool = False,
        rope_scaling_factor: float = 8.0,
        rotary_base: int = 1_000_000,
        rotary_base_local: int = 10_000,
        **kwargs,
    ):
        # The rope scaling in RotaryEmbedding is not linear scaling,
        # so this flag must be off. Will calculate linear scaling below.
        assert rope_scaling is False

        # Get inv_freq for global attention layers
        super().__init__(
            rope_scaling=rope_scaling,
            rotary_base=rotary_base,
            **kwargs,
        )
        self.inv_freq /= rope_scaling_factor

        # Setup Rotary Embedding for local attentions
        self.rope_local = RotaryEmbedding(
            rope_scaling=rope_scaling,
            rotary_base=rotary_base_local,
            **kwargs,
        )

    def forward(
        self,
        max_seq_len: int,
        offset: int = 0,
        packed_seq: bool = False,
        cp_group: torch.distributed.ProcessGroup | None = None,
    ) -> Tensor:
        """Get global and local rope embedding.

        Note: Caching is bypassed when cp_group is provided since ProcessGroup is unhashable.
        """
        # ProcessGroup is unhashable, so bypass caching when cp_group is provided
        if cp_group is not None:
            rope_global = super().forward(max_seq_len, offset, packed_seq, cp_group)
            rope_local = self.rope_local.forward(max_seq_len, offset, packed_seq, cp_group)
            return torch.stack([rope_local, rope_global], dim=0)
        return self._forward_cached(max_seq_len, offset, packed_seq)

    @lru_cache(maxsize=32)
    def _forward_cached(
        self,
        max_seq_len: int,
        offset: int = 0,
        packed_seq: bool = False,
    ) -> Tensor:
        """Cached forward for hashable parameters only."""
        rope_global = super().forward(max_seq_len, offset, packed_seq, None)
        rope_local = self.rope_local.forward(max_seq_len, offset, packed_seq, None)
        return torch.stack([rope_local, rope_global], dim=0)


def _is_local_attn_layer(
    layer_number: int,
    layer_pattern: Tuple[int, int],
) -> bool:
    pattern_size = sum(layer_pattern)
    return layer_number % pattern_size != 0


class TERowParallelLinearLayerNorm(TERowParallelLinear):
    """Modified From TERowParallelLinear with an additional Post-LN."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: TransformerConfig,
        **kwargs,
    ):
        super().__init__(
            input_size,
            output_size,
            config=config,
            **kwargs,
        )
        self.post_layernorm = TENorm(config, output_size)

    def forward(self, x):
        """Forward with additional Post LN on output"""
        output, bias = super().forward(x)
        return self.post_layernorm(output), bias


class Gemma3RMSNorm(torch.nn.Module):
    """Gemma3 zero-centered RMSNorm: ``rmsnorm(x) * (1 + weight)``.

    Torch implementation matching HF's ``Gemma3RMSNorm``. Used instead of
    TENorm inside SequentialMLP experts, where TE's norm hits an
    "Output rsigma is not allocated" runtime error, and to keep the MoE
    expert output norm bit-identical to the HF reference implementation.
    """

    def __init__(self, dim: int, eps: float = 1e-6, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.zeros(dim, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        output = x.float()
        output = output * torch.rsqrt(output.pow(2).mean(-1, keepdim=True) + self.eps)
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


class TERowParallelLinearTorchRMSNorm(TERowParallelLinear):
    """TERowParallelLinear followed by a torch Gemma3 RMSNorm on the output.

    The per-expert post-MLP norm for upcycled Gemma3 MoE experts.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: TransformerConfig,
        **kwargs,
    ):
        super().__init__(
            input_size,
            output_size,
            config=config,
            **kwargs,
        )
        self.post_layernorm = Gemma3RMSNorm(output_size, eps=config.layernorm_epsilon, dtype=config.params_dtype)

    def forward(self, x):
        """Forward with additional post RMSNorm on output"""
        output, bias = super().forward(x)
        return self.post_layernorm(output), bias


class Gemma3MoETransformerLayer(TransformerLayer):
    """Gemma3 MoE layer with an opt-in dense-equivalent initialization path.

    Normal MoE execution must use a standalone pre-MLP norm and sequential
    experts: the router needs normalized activations before dispatch and TE
    cannot place its fused post-MLP RMSNorm inside ``SequentialMLP``.  That is
    mathematically equivalent to the dense block, but it is not numerically
    identical in bf16.  In particular, it changes both the fused norm/GEMM
    graph and the GEMM batch geometry.

    ``gemma3_moe_canonical_dense_init`` is intentionally a correctness-only
    initialization mode.  It computes the same dense MLP graph as the dense
    Gemma3 provider from the raw residual stream, while still running the MoE
    router so R2/R3 replay and the router auxiliary loss are exercised.  The
    dense MLP's pre/post norms alias the canonical expert state; its TP-shaped
    FC tensors are derived from that expert with autograd links.  The helper
    itself has no registered state, so it adds no checkpoint or optimizer
    parameters.

    This mode must be disabled before normal sparse-MoE training: the first
    EP-local expert is the canonical dense copy and the branch deliberately
    bypasses dispatch.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gemma3_moe_canonical_dense_init = bool(getattr(self.config, "gemma3_moe_canonical_dense_init", False))
        if not self.gemma3_moe_canonical_dense_init:
            return

        if not isinstance(self.mlp, MoELayer):
            raise TypeError("Gemma3 canonical dense-init path requires an MoELayer")
        if not isinstance(self.mlp.experts, SequentialMLP):
            raise TypeError("Gemma3 canonical dense-init path requires SequentialMLP experts")
        if not self.mlp.experts.local_experts:
            raise RuntimeError("Gemma3 canonical dense-init path requires a local expert")

        # Build the exact dense MCore MLP kernel graph off-module.  Using
        # ``object.__setattr__`` keeps this helper out of state_dict()/the
        # optimizer; the existing MoE parameters remain authoritative.
        canonical_mlp = MLP(
            self.config,
            MLPSubmodules(
                linear_fc1=TELayerNormColumnParallelLinear,
                linear_fc2=TERowParallelLinearLayerNorm,
            ),
            tp_group=self.pg_collection.tp,
        )
        expert = self.mlp.experts.local_experts[0]
        canonical_mlp.linear_fc1.layer_norm_weight = self.pre_mlp_layernorm.weight
        canonical_mlp.linear_fc2.post_layernorm.weight = expert.linear_fc2.post_layernorm.weight
        object.__setattr__(self, "_canonical_dense_mlp", canonical_mlp)
        # Bridge loads modify the expert tensors in-place after construction.
        # Bind TP tensors lazily (or whenever their source versions change) so
        # no stale autograd tensor survives the first forward after loading.
        self._canonical_dense_mlp_source_versions = None

    def _canonical_dense_source_versions(self):
        expert = self.mlp.experts.local_experts[0]
        return (expert.linear_fc1.weight._version, expert.linear_fc2.weight._version)

    def refresh_canonical_dense_mlp_weights(self) -> None:
        """Bind dense TP tensors to differentiable pieces of one local expert.

        With EP>1 and ETP=1, SequentialMLP experts are full-width replicas
        while dense Gemma MLPs shard FC1 on dim 0 and FC2 on dim 1 across TP.
        FC1's fused layout needs special handling: its dense shard is
        ``[gate_shard; up_shard]``, not a contiguous chunk of the full
        ``[gate; up]`` expert tensor.  The derived tensors retain autograd
        links to the authoritative expert parameters and give the unregistered
        helper exactly the dense kernel shapes.
        """
        if not self.gemma3_moe_canonical_dense_init:
            return

        canonical_mlp = self._canonical_dense_mlp
        expert = self.mlp.experts.local_experts[0]
        tp_size = canonical_mlp.linear_fc1.tp_size
        tp_rank = canonical_mlp.linear_fc1.tp_rank
        # This method may be reached under inference_mode during a rollout.
        # Explicitly create normal autograd tensors so a subsequent training
        # forward still propagates gradients to the expert tensors.
        with torch.inference_mode(False):
            gate_weight, up_weight = expert.linear_fc1.weight.chunk(2, dim=0)
            fc1_weight = torch.cat(
                (
                    gate_weight.chunk(tp_size, dim=0)[tp_rank],
                    up_weight.chunk(tp_size, dim=0)[tp_rank],
                ),
                dim=0,
            )
            # TE's row-parallel path may cache a transposed representation.
            # Give it an independent contiguous tensor rather than a strided
            # view into a full expert; ``clone`` still backpropagates to the
            # expert parameter through CloneBackward.
            fc2_weight = expert.linear_fc2.weight.chunk(tp_size, dim=1)[tp_rank].clone()
        if fc1_weight.shape != canonical_mlp.linear_fc1.weight.shape:
            raise RuntimeError(
                "Gemma3 canonical dense-init FC1 TP slice has the wrong shape: "
                f"{tuple(fc1_weight.shape)} != "
                f"{tuple(canonical_mlp.linear_fc1.weight.shape)}"
            )
        if fc2_weight.shape != canonical_mlp.linear_fc2.weight.shape:
            raise RuntimeError(
                "Gemma3 canonical dense-init FC2 TP slice has the wrong shape: "
                f"{tuple(fc2_weight.shape)} != "
                f"{tuple(canonical_mlp.linear_fc2.weight.shape)}"
            )

        # These helpers are deliberately unregistered.  A normal ``Module``
        # assignment only accepts ``Parameter`` objects and would detach the
        # derived tensor; bypass registration so TE reads it directly.
        # Clearing the original helper allocation avoids a
        # second, unused full MLP in GPU memory.
        canonical_mlp.linear_fc1._parameters["weight"] = None
        object.__setattr__(canonical_mlp.linear_fc1, "weight", fc1_weight)
        canonical_mlp.linear_fc2._parameters["weight"] = None
        object.__setattr__(canonical_mlp.linear_fc2, "weight", fc2_weight)
        self._canonical_dense_mlp_source_versions = self._canonical_dense_source_versions()

    def _forward_mlp(self, hidden_states, inference_context=None, padding_mask=None):
        if not self.gemma3_moe_canonical_dense_init:
            return super()._forward_mlp(hidden_states, inference_context, padding_mask)

        # Differentiable TP slices carry an autograd graph. Rebuild them for
        # every grad-enabled forward so gradient accumulation never reuses a
        # graph that an earlier microbatch already freed. In inference, cache
        # the slices until bridge loading or an optimizer step changes the
        # authoritative expert tensors.
        if torch.is_grad_enabled() or (
            self._canonical_dense_mlp_source_versions != self._canonical_dense_source_versions()
        ):
            self.refresh_canonical_dense_mlp_weights()

        # ``hidden_states`` is the raw residual stream expected by the dense
        # fused LayerNormLinear.  Compute normalized activations separately for
        # the router only; this exactly preserves route replay and aux-loss
        # accounting without changing the canonical MLP output.
        residual = hidden_states
        router_input = self._forward_pre_mlp_layernorm(hidden_states)
        router_padding_mask = padding_mask.transpose(0, 1).bool() if padding_mask is not None else None
        router_probs, _ = self.mlp.route(router_input, router_padding_mask)

        canonical_mlp = self._canonical_dense_mlp
        # The helper is deliberately not a registered child module, so mirror
        # the enclosing layer's training state explicitly.
        canonical_mlp.train(self.training)
        mlp_output, mlp_bias = canonical_mlp(hidden_states)

        if self.training and torch.is_grad_enabled():
            # Top-1 post-softmax routing has no policy-gradient path, but the
            # router's aux loss is attached to ``router_probs``.  A zero-valued
            # connection retains that auxiliary gradient without perturbing
            # the forward activation.
            mlp_output = mlp_output + router_probs.sum().to(mlp_output.dtype) * 0.0

        return self._forward_post_mlp((mlp_output, mlp_bias), residual)


def gemma3_moe_layer_spec(config) -> ModuleSpec:
    """Gemma3 MoE layer spec for dense-to-MoE upcycled models.

    Keeps Gemma3 attention identical to :func:`gemma3_layer_spec` and replaces
    the dense MLP with a Megatron-Core MoE layer:

    - standalone pre-MoE RMSNorm (the dense spec fuses it into ``linear_fc1``,
      which is not possible ahead of a router),
    - Megatron-Core ``TopKRouter`` (top-1, post-top-k softmax => combine
      weight exactly 1.0),
    - ``SequentialMLP`` experts, each a full Gemma3 MLP with its own post-MLP
      RMSNorm. ``SequentialMLP`` is required because grouped GEMM cannot
      represent the per-expert output norm.
    """
    return ModuleSpec(
        module=Gemma3MoETransformerLayer,
        submodules=TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=Gemma3SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TELayerNormColumnParallelLinear,
                    core_attention=Gemma3TEDotProductAttention,  # mixed global/local attn
                    q_layernorm=TENorm if config.qk_layernorm else None,
                    k_layernorm=TENorm if config.qk_layernorm else None,
                    linear_proj=TERowParallelLinearLayerNorm,  # post attn RMSNorm
                ),
            ),
            self_attn_bda=get_bias_dropout_add,  # residual link
            pre_mlp_layernorm=TENorm,
            mlp=ModuleSpec(
                module=MoELayer,
                submodules=MoESubmodules(
                    experts=ModuleSpec(
                        module=SequentialMLP,
                        submodules=MLPSubmodules(
                            linear_fc1=TEColumnParallelLinear,
                            linear_fc2=TERowParallelLinearTorchRMSNorm,  # per-expert post mlp RMSNorm
                        ),
                    ),
                    shared_experts=None,
                ),
                metainfo={"fuse_pre_mlp_layernorm": False},
            ),
            mlp_bda=get_bias_dropout_add,  # residual link
        ),
    )


@dataclass
class Gemma3ModelProvider4B(Gemma3ModelProvider):
    """Dense Gemma3 4B (text stack)."""

    num_layers: int = 34
    hidden_size: int = 2560
    ffn_hidden_size: int = 10240
    num_attention_heads: int = 8
    num_query_groups: int = 4
    kv_channels: int = 256
    window_size: int = 1024
    rope_scaling_factor: float = 8.0
    vocab_size: int = 262_208


@dataclass
class Gemma3MoEModelProvider(Gemma3ModelProvider):
    """Gemma3 dense-to-MoE upcycled model provider.

    Every dense MLP is replaced by a top-1 MoE layer whose experts are
    full-size copies of the dense MLP (including the per-expert post-MLP
    RMSNorm). ``moe_router_pre_softmax=False`` with top-1 makes the combine
    weight exactly 1.0, so a freshly upcycled model is mathematically
    dense-equivalent. Normal bf16 sparse dispatch is not bit-exact because
    it changes norm/GEMM execution geometry; use
    ``gemma3_moe_canonical_dense_init`` for the strict initialization gate.
    The router then trains through the aux load-balancing loss.
    """

    transformer_layer_spec: Union[ModuleSpec, Callable[["Gemma3ModelProvider"], ModuleSpec]] = field(
        default_factory=lambda: gemma3_moe_layer_spec
    )
    num_moe_experts: int = 2
    moe_router_topk: int = 1
    moe_router_pre_softmax: bool = False
    moe_router_score_function: str = "softmax"
    moe_router_load_balancing_type: str = "aux_loss"
    moe_aux_loss_coeff: float = 1e-3
    moe_grouped_gemm: bool = False  # SequentialMLP: experts carry a post-MLP norm
    moe_token_dispatcher_type: str = "alltoall"
    moe_permute_fusion: bool = True
    # Exact dense-equivalent activation path for one-time correctness gates.
    # Do not enable this for normal sparse-MoE training.
    gemma3_moe_canonical_dense_init: bool = False


@dataclass
class Gemma3MoEModelProvider4B(Gemma3MoEModelProvider):
    """Gemma3 4B upcycled to 2 experts, top-1 routing."""

    num_layers: int = 34
    hidden_size: int = 2560
    ffn_hidden_size: int = 10240
    moe_ffn_hidden_size: int = 10240
    num_attention_heads: int = 8
    num_query_groups: int = 4
    kv_channels: int = 256
    window_size: int = 1024
    rope_scaling_factor: float = 8.0
    vocab_size: int = 262_208
    num_moe_experts: int = 2


@dataclass
class Gemma3MoEModelProvider4B4E(Gemma3MoEModelProvider4B):
    """Gemma3 4B upcycled to 4 experts, top-1 routing."""

    num_moe_experts: int = 4


def _relax_top1_post_topk_softmax_validation() -> None:
    """Allow ``moe_router_topk=1`` with post-top-k softmax in TransformerConfig.

    Megatron-Core rejects ``moe_router_topk=1`` + ``softmax`` +
    ``moe_router_pre_softmax=False`` because softmax over the single selected
    logit is constant 1.0 and gives the router no main-loss gradient. For
    upcycled Gemma3 MoE that constant 1.0 is exactly what preserves the dense
    model's logits, and the router still receives gradients through the aux
    load-balancing loss, so the combination is permitted when aux-loss
    balancing is active.
    """
    original_post_init = TransformerConfig.__post_init__
    if getattr(original_post_init, "_gemma3_moe_top1_relaxed", False):
        return

    def patched_post_init(self):
        relax = (
            getattr(self, "num_moe_experts", None)
            and getattr(self, "moe_router_topk", None) == 1
            and getattr(self, "moe_router_score_function", None) == "softmax"
            and not getattr(self, "moe_router_pre_softmax", True)
            and getattr(self, "moe_router_load_balancing_type", None) == "aux_loss"
            and (getattr(self, "moe_aux_loss_coeff", 0) or 0) > 0
        )
        if not relax:
            return original_post_init(self)
        warnings.warn(
            "Allowing moe_router_topk=1 with post-top-k softmax: the combine weight is "
            "exactly 1.0 and the router trains only through the aux load-balancing loss.",
            stacklevel=2,
        )
        self.moe_router_pre_softmax = True
        try:
            return original_post_init(self)
        finally:
            self.moe_router_pre_softmax = False

    patched_post_init._gemma3_moe_top1_relaxed = True
    TransformerConfig.__post_init__ = patched_post_init


_relax_top1_post_topk_softmax_validation()
