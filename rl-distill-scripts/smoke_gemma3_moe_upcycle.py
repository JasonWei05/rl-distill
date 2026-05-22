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

"""Smoke-test Gemma3 dense-to-MoE upcycling on a tiny model."""

import os
import socket

import torch
import torch.distributed as dist
from megatron.bridge.models.gemma.gemma3_provider import Gemma3ModelProvider, gemma3_moe_layer_spec
from megatron.core import parallel_state
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.moe import upcycling_utils


def _free_port() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return str(sock.getsockname()[1])


def _init_dist() -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", _free_port())
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    torch.cuda.set_device(0)
    dist.init_process_group(backend="nccl", rank=0, world_size=1)
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=1,
    )
    model_parallel_cuda_manual_seed(1234)


def _tiny_provider(*, moe: bool) -> Gemma3ModelProvider:
    provider = Gemma3ModelProvider(
        is_vision_language=False,
        num_layers=2,
        hidden_size=128,
        num_attention_heads=4,
        num_query_groups=2,
        kv_channels=32,
        ffn_hidden_size=256,
        seq_length=128,
        vocab_size=512,
        window_size=16,
        softmax_scale=1.0 / (32**0.5),
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=1,
        use_cpu_initialization=False,
        gradient_accumulation_fusion=False,
    )
    if moe:
        provider.num_moe_experts = 2
        provider.moe_router_topk = 1
        provider.moe_ffn_hidden_size = provider.ffn_hidden_size
        provider.moe_grouped_gemm = False
        provider.moe_router_load_balancing_type = "aux_loss"
        provider.moe_aux_loss_coeff = 1e-3
        provider.moe_router_pre_softmax = True
        provider.moe_token_dispatcher_type = "alltoall"
        provider.transformer_layer_spec = gemma3_moe_layer_spec
    provider.finalize()
    provider._pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    return provider


def _assert_same(
    dense_state: dict[str, torch.Tensor],
    moe_state: dict[str, torch.Tensor],
    left: str,
    right: str,
) -> None:
    if left not in dense_state:
        raise AssertionError(f"missing {left}")
    if right not in moe_state:
        raise AssertionError(f"missing {right}")
    if not torch.equal(dense_state[left].detach().cpu(), moe_state[right].detach().cpu()):
        raise AssertionError(f"{left} != {right}")


def main() -> None:
    _init_dist()
    try:
        dense_model = _tiny_provider(moe=False).provide().cuda().bfloat16()
        moe_model = _tiny_provider(moe=True).provide().cuda().bfloat16()

        dense_state = dense_model.state_dict()
        converted = upcycling_utils.upcycle_state_dict([moe_model], [dense_model])["model"]
        moe_model.load_state_dict(converted, strict=True)

        for layer_idx in range(2):
            dense_fc1 = f"decoder.layers.{layer_idx}.mlp.linear_fc1.weight"
            dense_fc2 = f"decoder.layers.{layer_idx}.mlp.linear_fc2.weight"
            dense_postnorm = f"decoder.layers.{layer_idx}.mlp.linear_fc2.post_layernorm.weight"
            for expert_idx in range(2):
                expert_prefix = f"decoder.layers.{layer_idx}.mlp.experts.local_experts.{expert_idx}"
                _assert_same(dense_state, converted, dense_fc1, f"{expert_prefix}.linear_fc1.weight")
                _assert_same(dense_state, converted, dense_fc2, f"{expert_prefix}.linear_fc2.weight")
                _assert_same(
                    dense_state,
                    converted,
                    dense_postnorm,
                    f"{expert_prefix}.linear_fc2.post_layernorm.weight",
                )

        print("Gemma3 tiny MoE upcycling smoke test passed")
    finally:
        if parallel_state.model_parallel_is_initialized():
            parallel_state.destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
