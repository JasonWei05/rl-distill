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

"""Compare a Gemma3 MoE HF checkpoint against its Megatron-Bridge load.

Loads the same converted MoE checkpoint twice — through the HF remote-code
model and through AutoBridge.load_hf_weights into the Megatron MoE model —
and compares next-token logits. This validates the fork's HF<->Megatron
parameter mapping with real SFT weights, including expert-parallel sharding.

Single rank:
    CUDA_VISIBLE_DEVICES=4 python logit_parity_gemma3_moe_hf_vs_megatron.py <snapshot>

Expert parallel (EP=2 across two GPUs):
    CUDA_VISIBLE_DEVICES=4,6 torchrun --nproc-per-node 2 \
        logit_parity_gemma3_moe_hf_vs_megatron.py <snapshot> --ep 2
"""

import argparse
import os
import socket

import torch
import torch.distributed as dist


def _free_port() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return str(sock.getsockname()[1])


def _init_dist(ep_size: int) -> int:
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", _free_port())
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=ep_size,
        expert_tensor_parallel_size=1,
    )
    model_parallel_cuda_manual_seed(1234)
    return local_rank


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", help="Path to the converted MoE HF checkpoint")
    parser.add_argument("--ep", type=int, default=1, help="expert_model_parallel_size")
    parser.add_argument("--prompt", default="The capital of France is the city of")
    args = parser.parse_args()

    local_rank = _init_dist(args.ep)
    rank = dist.get_rank()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.snapshot)
    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.cuda()
    seq_len = input_ids.size(1)
    position_ids = torch.arange(seq_len, dtype=torch.long, device="cuda").unsqueeze(0)
    attention_mask = torch.ones((1, 1, seq_len, seq_len), dtype=torch.bool, device="cuda")

    from megatron.bridge import AutoBridge
    from megatron.core.process_groups_config import ProcessGroupCollection

    bridge = AutoBridge.from_hf_pretrained(args.snapshot, trust_remote_code=True)
    provider = bridge.to_megatron_provider(load_weights=False)
    provider.params_dtype = torch.bfloat16
    provider.bf16 = True
    provider.fp16 = False
    provider.tensor_model_parallel_size = 1
    provider.pipeline_model_parallel_size = 1
    provider.expert_model_parallel_size = args.ep
    provider.expert_tensor_parallel_size = 1
    provider.variable_seq_lengths = True
    provider.seq_length = max(seq_len, 16)
    provider.gradient_accumulation_fusion = False
    provider.moe_permute_fusion = False
    provider.finalize()
    provider._pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    if rank == 0:
        print(
            f"provider: {type(provider).__name__} experts={provider.num_moe_experts} "
            f"topk={provider.moe_router_topk} pre_softmax={provider.moe_router_pre_softmax} "
            f"ep={provider.expert_model_parallel_size}"
        )

    megatron_model = provider.provide().cuda()
    bridge.load_hf_weights([megatron_model], args.snapshot)
    megatron_model.eval()
    with torch.no_grad():
        megatron_logits = megatron_model(
            input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask
        ).detach()
    if megatron_logits.shape[0] == seq_len:  # [s, b, v] -> [b, s, v]
        megatron_logits = megatron_logits.transpose(0, 1)
    megatron_logits = megatron_logits.float()

    if rank == 0:
        from transformers import AutoModelForCausalLM

        hf_model = (
            AutoModelForCausalLM.from_pretrained(args.snapshot, trust_remote_code=True, dtype=torch.bfloat16)
            .cuda()
            .eval()
        )
        with torch.no_grad():
            hf_logits = hf_model(input_ids=input_ids).logits.detach().float()
        del hf_model
        torch.cuda.empty_cache()

        vocab = hf_logits.shape[-1]
        megatron_logits = megatron_logits[..., :vocab]
        diff = (megatron_logits - hf_logits).abs()
        hf_top1 = hf_logits.argmax(-1)
        mg_top1 = megatron_logits.argmax(-1)
        agree = (hf_top1 == mg_top1).float().mean().item()
        print(f"max_abs_diff={diff.max().item():.6g} mean_abs_diff={diff.mean().item():.6g}")
        print(f"top1_agreement={agree:.4f}")
        print("hf   next tokens:", tokenizer.decode(hf_top1[0][-8:]))
        print("mcore next tokens:", tokenizer.decode(mg_top1[0][-8:]))
        assert agree >= 0.99, "top-1 token disagreement between HF and Megatron"
        assert diff.mean().item() < 0.05, "mean logit divergence too large"
        print("LOGIT_PARITY_OK")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
