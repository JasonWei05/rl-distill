"""Local single-step gemma-4 update-phase memory harness.

Runs verl's REAL TrainingWorker (FSDP2 engine, prod config from
run_gemma4_pt_deepscaler_4of4strict_rl.sh) on a worst-case mini-batch of four
max-length samples (1577 prompt + 20480 response = 22057 tokens each), through
both passes that have OOMed on ScaleTrain:
  1. infer_batch with calculate_entropy=True  (== _compute_old_log_prob)
  2. train_batch with calculate_entropy=False (== update_actor, entropy_coeff=0)
Records CUDA memory history, dumps a snapshot on OOM, and checks the peak
against the prod budget: 79.19 GiB - ~23.5 GiB resident vLLM = 55.7 GiB.

Usage: CUDA_VISIBLE_DEVICES=<free gpu> VERL_FSDP2_LOCAL_LOAD=1 python local_update_memtest.py
"""

import os
import sys
from functools import partial

os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29617")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

sys.path.insert(0, "/mnt/efs/jasonwei/rl-distill")

from verl import DataProto  # noqa: E402
from verl.utils import tensordict_utils as tu  # noqa: E402
from verl.workers.config import (  # noqa: E402
    ActorConfig,
    FSDPEngineConfig,
    FSDPOptimizerConfig,
    HFModelConfig,
)
from verl.workers.engine_workers import TrainingWorker, TrainingWorkerConfig  # noqa: E402
from verl.workers.utils.losses import ppo_loss  # noqa: E402
from verl.workers.utils.padding import left_right_2_no_padding  # noqa: E402

SNAP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memtest_snapshot.pickle")
BUDGET_GIB = 55.7  # prod: 79.19 total - ~23.5 vLLM resident
PROMPT_LEN, RESP_LEN, BSZ = 1577, 20480, 4

dist.init_process_group(backend="nccl")
torch.cuda.memory._record_memory_history(max_entries=200000)


def gib(x):
    return x / 2**30


def build_batch(vocab):
    torch.manual_seed(0)
    seqlen = PROMPT_LEN + RESP_LEN
    input_ids = torch.randint(3, min(vocab, 200000), (BSZ, seqlen))
    attention_mask = torch.ones(BSZ, seqlen, dtype=torch.int64)
    position_ids = torch.arange(seqlen).unsqueeze(0).expand(BSZ, -1).contiguous()
    responses = input_ids[:, PROMPT_LEN:]
    data = DataProto.from_single_dict(
        {
            "input_ids": input_ids,
            "prompts": input_ids[:, :PROMPT_LEN],
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": responses,
            "response_mask": torch.ones_like(responses),
            "old_log_probs": torch.rand(BSZ, RESP_LEN) * -1.0,
            "advantages": torch.randn(BSZ, RESP_LEN),
            "ref_log_prob": torch.rand(BSZ, RESP_LEN) * -1.0,
        },
        meta_info={"temperature": 1.0, "global_token_num": [seqlen] * BSZ},
    )
    return data


def phase(name, fn):
    torch.cuda.reset_peak_memory_stats()
    try:
        out = fn()
    except torch.OutOfMemoryError:
        torch.cuda.memory._dump_snapshot(SNAP)
        print(f"[FAIL] {name}: OOM — snapshot dumped to {SNAP}", flush=True)
        raise
    peak = gib(torch.cuda.max_memory_allocated())
    verdict = "OK " if peak < BUDGET_GIB else "OVER"
    print(f"[{verdict}] {name}: peak={peak:.2f} GiB (budget {BUDGET_GIB})", flush=True)
    return out


def main():
    model_config = HFModelConfig(
        path="google/gemma-4-E2B",
        use_remove_padding=False,
        override_config={"attn_implementation": "sdpa"},
        enable_gradient_checkpointing=True,
    )
    engine_config = FSDPEngineConfig(
        forward_only=False,
        strategy="fsdp2",
        fsdp_size=1,
        # bf16 load halves the load-time spike so the harness fits next to local GPU
        # tenants; prod loads fp32 but the update-phase compute dtype (bf16 autocast)
        # and offloaded fp32 optimizer are unaffected.
        model_dtype="bfloat16",
        use_remove_padding=False,
        use_dynamic_bsz=False,
        param_offload=True,
        optimizer_offload=True,
        grad_offload=True,
        entropy_from_logits_with_chunking=True,
        entropy_checkpointing=True,
        wrap_policy={"transformer_layer_cls_to_wrap": ["Gemma4TextDecoderLayer"]},
    )
    config = TrainingWorkerConfig(
        model_type="language_model",
        model_config=model_config,
        engine_config=engine_config,
        optimizer_config=FSDPOptimizerConfig(),
        checkpoint_config=None,
    )
    worker = TrainingWorker(config)
    worker.reset()
    print("[init] model loaded", flush=True)

    actor_config = ActorConfig(strategy="fsdp2", rollout_n=1, ppo_micro_batch_size_per_gpu=1)
    worker.set_loss_fn(partial(ppo_loss, config=actor_config))

    hf_cfg = model_config.hf_config
    vocab = hf_cfg.text_config.vocab_size if hasattr(hf_cfg, "text_config") else hf_cfg.vocab_size
    data = build_batch(vocab)

    # ---- pass 1: old-logprob (forward-only, entropy on) — mirrors ray_trainer.py:1239
    td = left_right_2_no_padding(data.to_tensordict())
    tu.assign_non_tensor(
        td,
        calculate_entropy=True,
        compute_loss=False,
        use_dynamic_bsz=False,
        micro_batch_size_per_gpu=1,
        use_remove_padding=False,
        max_response_length=RESP_LEN,
    )
    phase("old_logprob (fwd-only, entropy)", lambda: worker.infer_batch(td))

    # ---- pass 2: update (grad, entropy off) — mirrors ray_trainer.py:1273-1295
    td2 = left_right_2_no_padding(data.to_tensordict())
    tu.assign_non_tensor(
        td2,
        calculate_entropy=False,
        compute_loss=True,
        global_batch_size=BSZ,
        mini_batch_size=BSZ,
        epochs=1,
        seed=1,
        dataloader_kwargs={"shuffle": False},
        use_dynamic_bsz=False,
        micro_batch_size_per_gpu=1,
        use_remove_padding=False,
        max_response_length=RESP_LEN,
    )
    metrics = phase("update (fwd+bwd, 4 micros)", lambda: worker.train_batch(td2))
    m = tu.get(metrics, "metrics")
    print(f"[metrics] pg_loss present: {'pg_loss' in str(m)[:2000]}", flush=True)

    # ---- pass 3: second update — Adam states now materialized (the step>=2 regime)
    td3 = td2.clone()
    phase("update #2 (post-Adam)", lambda: worker.train_batch(td3))

    print("LOCAL_UPDATE_MEMTEST_PASSED", flush=True)


if __name__ == "__main__":
    main()
