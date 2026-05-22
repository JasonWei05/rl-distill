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

"""Compare dense Gemma3 4B logits against 2E and 4E upcycled MoE variants."""

import gc
import os
import socket

import torch
import torch.distributed as dist
from megatron.bridge.models.gemma import (
    Gemma3ModelProvider4B,
    Gemma3MoEModelProvider4B,
    Gemma3MoEModelProvider4B4E,
)
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


def _provider(provider_cls):
    provider = provider_cls(
        seq_length=16,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=1,
        use_cpu_initialization=False,
        gradient_accumulation_fusion=False,
        moe_permute_fusion=False,
    )
    provider.finalize()
    provider._pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    return provider


def _inputs(vocab_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids = torch.tensor([[1, 42, 320, 2048, 17, 4096, 23, 9]], dtype=torch.long, device="cuda")
    input_ids %= vocab_size
    position_ids = torch.arange(input_ids.size(1), dtype=torch.long, device="cuda").unsqueeze(0)
    attention_mask = torch.ones(
        (input_ids.size(0), 1, input_ids.size(1), input_ids.size(1)),
        dtype=torch.bool,
        device="cuda",
    )
    return input_ids, position_ids, attention_mask


@torch.no_grad()
def _logits(model, input_ids, position_ids, attention_mask):
    model.eval()
    return model(
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
    ).detach()


def _compare(name: str, dense_model, dense_logits, input_ids, position_ids, attention_mask) -> None:
    moe_model = _provider(Gemma3MoEModelProvider4B if name == "2E" else Gemma3MoEModelProvider4B4E).provide().cuda()
    moe_model.load_state_dict(upcycling_utils.upcycle_state_dict([moe_model], [dense_model])["model"], strict=True)
    moe_logits = _logits(moe_model, input_ids, position_ids, attention_mask)

    diff = (moe_logits - dense_logits).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    same = torch.equal(moe_logits, dense_logits)
    close = torch.allclose(moe_logits, dense_logits, rtol=0.0, atol=0.0)
    print(f"{name}: equal={same} allclose_atol0={close} max_abs={max_abs:.8g} mean_abs={mean_abs:.8g}")
    if not same:
        raise AssertionError(f"{name} logits differ from dense Gemma3 4B")

    del moe_logits, moe_model
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    _init_dist()
    try:
        dense_provider = _provider(Gemma3ModelProvider4B)
        dense_model = dense_provider.provide().cuda()
        input_ids, position_ids, attention_mask = _inputs(dense_provider.vocab_size)
        dense_logits = _logits(dense_model, input_ids, position_ids, attention_mask)

        _compare("2E", dense_model, dense_logits, input_ids, position_ids, attention_mask)
        _compare("4E", dense_model, dense_logits, input_ids, position_ids, attention_mask)
        print("Dense Gemma3 4B, 2E MoE, and 4E MoE logits are exactly identical")
    finally:
        if parallel_state.model_parallel_is_initialized():
            parallel_state.destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
