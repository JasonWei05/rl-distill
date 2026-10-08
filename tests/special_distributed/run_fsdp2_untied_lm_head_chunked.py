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
"""Chunked LM-head calls outside the forward with an untied head under FSDP2 (2 ranks, gloo/CPU).

The hidden-state distillation loss calls ``lm_head`` once per vocab chunk, and the chunk count differs per rank. With
``lm_head`` wrapped as its own FSDP unit every call reshards/all-gathers and the ranks deadlock; with
VERL_FSDP2_WRAP_EMBED_LM_HEAD=0 it stays in the root unit and both ranks finish with identical reduced gradients.
    VERL_FSDP2_WRAP_EMBED_LM_HEAD=0 torchrun --standalone --nproc_per_node=2 tests/special_distributed/run_fsdp2_untied_lm_head_chunked.py
"""
import os, sys, torch, torch.distributed as dist
from torch.distributed.fsdp import fully_shard
from torch.distributed.device_mesh import init_device_mesh
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
sys.path.insert(0, os.getcwd())
from verl.utils.fsdp_utils import _select_fsdp2_wrap_targets
import verl.workers.engine.fsdp.transformer_impl  # noqa: F401  (skip_lm_head patch)

dist.init_process_group("gloo"); rank = dist.get_rank()
torch.manual_seed(0)
cfg = Qwen3MoeConfig(vocab_size=64, hidden_size=32, intermediate_size=64, moe_intermediate_size=16, num_hidden_layers=2,
                     num_attention_heads=4, num_key_value_heads=2, head_dim=8, num_experts=4, num_experts_per_tok=2, tie_word_embeddings=False)
model = Qwen3MoeForCausalLM(cfg)
mesh = init_device_mesh("cpu", (2,))
targets = _select_fsdp2_wrap_targets(model, ["Qwen3MoeDecoderLayer"])
for m in targets: fully_shard(m, mesh=mesh)
fully_shard(model, mesh=mesh)
if rank == 0: print("separately wrapped:", sorted({type(m).__name__ for m in targets}), flush=True)
ids = torch.randint(0, 64, (1, 8 + 8 * rank))            # rank 1 has more tokens -> more chunks
L = ids.shape[1]
hidden = model(input_ids=ids, use_cache=False, return_dict=True, skip_lm_head=True, logits_to_keep=torch.arange(L), logits_to_keep_batch_indices=torch.zeros(L, dtype=torch.long)).logits[0]
head = model.lm_head
if hasattr(head, "unshard"): head.unshard()
loss = sum(head(hidden[i:i + 4]).float().logsumexp(-1).sum() for i in range(0, hidden.shape[0], 4))  # chunk=4
if hasattr(head, "reshard"): head.reshard()
loss.backward()
g = model.lm_head.weight.grad
g = g.full_tensor() if hasattr(g, "full_tensor") else g
print(f"rank{rank} OK chunks={(hidden.shape[0]+3)//4} lm_head_grad_norm={g.norm():.4f}", flush=True)
dist.destroy_process_group()
