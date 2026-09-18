"""E4B update-phase micro-batch sweep: memory + speed of one optimizer step on ONE GPU.

Derived from gemma4_local_update_memtest.py (real verl TrainingWorker, FSDP2 engine, prod-style
config) but for the Gemma 4 E4B RL recipe:
  * realistic batch: BATCH_SIZE sequences with a 12-shot prompt of PROMPT_LEN tokens and response
    lengths drawn from a clipped log-normal matching the medium-band RL runs (mean ~340, max ~1.3k),
    or WORST_CASE=1 -> every response at RESP_LEN (the 8k cap) to check the long-tail batch;
  * both production passes: old-logprob (fwd, entropy on) and update (fwd+bwd, entropy off), the
    update timed on its second call (Adam states materialised);
  * one `SWEEP_RESULT {json}` line: baseline memory after load, peak per phase, wall time per phase,
    micro-batch count actually produced by verl's prepare_micro_batches for this layout.

Runs under torchrun with fsdp_size=WORLD_SIZE (3-4 local GPUs; one GPU cannot hold the unsharded update). Params bf16,
optimizer+grads offloaded to CPU as in the memtest harness, so:
  * activation/logits peaks are per-GPU quantities and transfer directly to the 4-GPU pod;
  * the fixed footprint differs from prod (prod holds fp32 sharded params + Adam + grads ~30 GiB during
    the update); estimate prod peak = 30 GiB + (peak - baseline) reported here;
  * wall time excludes FSDP all-gather traffic (fewer micro-batches -> fewer all-gathers, so the real
    speed-up from larger micro-batches is >= the one measured here) and the CPU Adam step is excluded
    from the timed region.

Usage: CUDA_VISIBLE_DEVICES=<gpu> MICRO_BATCH_SIZE_PER_GPU=8 MAX_PADDED_TOKENS_PER_MICROBATCH=16384 \
         /tmp/.venv-gemma4/bin/python rl-distill-scripts/gemma4_e4b_microbatch_sweep.py
"""

import json
import os
import sys
import time

os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29631")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

sys.path.insert(0, "/mnt/efs/jasonwei/rl-distill")

from verl import DataProto  # noqa: E402
from verl.utils import tensordict_utils as tu  # noqa: E402
from verl.workers.config import ActorConfig, FSDPEngineConfig, FSDPOptimizerConfig, HFModelConfig  # noqa: E402
from verl.workers.engine.utils import prepare_micro_batches  # noqa: E402
from verl.workers.engine_workers import TrainingWorker, TrainingWorkerConfig  # noqa: E402
from verl.workers.utils.losses import ppo_loss  # noqa: E402
from verl.workers.utils.padding import left_right_2_no_padding  # noqa: E402
from functools import partial  # noqa: E402

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "/mnt/efs/jasonwei/.cache/huggingface/hub/models--google--gemma-4-E4B/snapshots/411aa17b749aa952df1359d2dcea73917a544d9a",
)
PROMPT_LEN = int(os.environ.get("PROMPT_LEN", "1577"))
RESP_LEN = int(os.environ.get("RESP_LEN", "8192"))  # cap (= max_response_length in the RL recipe)
BSZ = int(os.environ.get("BATCH_SIZE", "128"))  # sequences per optimizer step per GPU (64 prompts x 16 / 4 GPUs / 2 minis)
MICRO_BSZ = int(os.environ.get("MICRO_BATCH_SIZE_PER_GPU", "4"))
MAX_PADDED_TOKENS = int(os.environ.get("MAX_PADDED_TOKENS_PER_MICROBATCH", "8192"))
WORST_CASE = os.environ.get("WORST_CASE", "0") == "1"
RESP_MEAN = float(os.environ.get("RESP_MEAN", "340"))
RESP_MAX = int(os.environ.get("RESP_MAX", "1300"))
SEED = int(os.environ.get("SEED", "0"))
PROD_LAYOUT = os.environ.get("PROD_LAYOUT", "1") == "1"
TAG = os.environ.get("TAG", f"mbs{MICRO_BSZ}_cap{MAX_PADDED_TOKENS}{'_worst' if WORST_CASE else ''}")

local_rank = int(os.environ.get("LOCAL_RANK", "0"))
torch.cuda.set_device(local_rank)
dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))


def gib(x):
    return x / 2**30


def response_lengths():
    g = torch.Generator().manual_seed(SEED)
    if WORST_CASE:
        return torch.full((BSZ,), RESP_LEN, dtype=torch.long)
    # log-normal with the requested mean, sigma chosen so the max over ~128 draws lands near RESP_MAX
    sigma = 0.55
    mu = torch.log(torch.tensor(RESP_MEAN)) - sigma**2 / 2
    lens = torch.exp(mu + sigma * torch.randn(BSZ, generator=g)).round().long()
    return lens.clamp(64, min(RESP_MAX, RESP_LEN))


def build_batch(vocab):
    torch.manual_seed(SEED)
    lens = response_lengths()
    total = PROMPT_LEN + RESP_LEN
    input_ids = torch.randint(3, min(vocab, 200000), (BSZ, total))
    attention_mask = torch.zeros(BSZ, total, dtype=torch.int64)
    response_mask = torch.zeros(BSZ, RESP_LEN, dtype=torch.int64)
    for i, n in enumerate(lens.tolist()):
        attention_mask[i, : PROMPT_LEN + n] = 1
        response_mask[i, :n] = 1
    position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)
    responses = input_ids[:, PROMPT_LEN:]
    data = DataProto.from_single_dict(
        {
            "input_ids": input_ids,
            "prompts": input_ids[:, :PROMPT_LEN],
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": responses,
            "response_mask": response_mask,
            "old_log_probs": torch.rand(BSZ, RESP_LEN) * -1.0,
            "advantages": torch.randn(BSZ, RESP_LEN),
            "ref_log_prob": torch.rand(BSZ, RESP_LEN) * -1.0,
        },
        meta_info={"temperature": 1.0, "global_token_num": (PROMPT_LEN + lens).tolist()},
    )
    return data, lens


def timed(name, fn, out):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    try:
        res = fn()
    except (torch.OutOfMemoryError, RuntimeError) as e:  # cuDNN INTERNAL_ERROR near the memory limit is an OOM in disguise
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        kind = "oom" if isinstance(e, torch.OutOfMemoryError) else "runtime_error"
        out[name] = {"oom": True, "kind": kind, "peak_gib": round(gib(torch.cuda.max_memory_allocated()), 2),
                     "reserved_gib": round(gib(torch.cuda.memory_reserved()), 2), "error": str(e)[:200]}
        print(f"[OOM ] {name}: {out[name]}", flush=True)
        return None
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    out[name] = {"oom": False, "peak_gib": round(gib(torch.cuda.max_memory_allocated()), 2), "seconds": round(dt, 2)}
    print(f"[OK  ] {name}: peak={out[name]['peak_gib']} GiB time={dt:.1f}s", flush=True)
    return res


def main():
    result = {"tag": TAG, "prod_layout": PROD_LAYOUT, "mbs": MICRO_BSZ, "cap": MAX_PADDED_TOKENS, "worst_case": WORST_CASE, "bsz": BSZ, "prompt_len": PROMPT_LEN}
    model_config = HFModelConfig(
        path=MODEL_PATH,
        use_remove_padding=False,
        override_config={"attn_implementation": "sdpa"},
        enable_gradient_checkpointing=True,
    )
    engine_config = FSDPEngineConfig(
        forward_only=False,
        strategy="fsdp2",
        fsdp_size=int(os.environ.get("WORLD_SIZE", "1")),
        # PROD_LAYOUT=1 (default): fp32 sharded master weights + Adam + grads resident on GPU during the update, exactly
        # what the RL pod holds inside update_actor (OFFLOAD=True only moves them off between phases). PROD_LAYOUT=0 =
        # the memtest harness layout (bf16 params, CPU Adam) — smaller footprint but the CPU Adam step confounds timing.
        model_dtype="float32" if PROD_LAYOUT else "bfloat16",
        use_remove_padding=False,
        use_dynamic_bsz=False,
        param_offload=not PROD_LAYOUT,
        optimizer_offload=not PROD_LAYOUT,
        grad_offload=not PROD_LAYOUT,
        offload_policy=False,
        entropy_from_logits_with_chunking=True,
        entropy_checkpointing=True,
        max_padded_tokens_per_microbatch=MAX_PADDED_TOKENS,
        infer_max_padded_tokens_per_microbatch=MAX_PADDED_TOKENS,
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
    torch.cuda.synchronize()
    result["baseline_gib_after_load"] = round(gib(torch.cuda.memory_allocated()), 2)
    print(f"[init] model loaded, baseline={result['baseline_gib_after_load']} GiB", flush=True)
    bc = torch.backends.cuda
    result["sdp_backends"] = {"flash": bc.flash_sdp_enabled(), "mem_efficient": bc.mem_efficient_sdp_enabled(), "math": bc.math_sdp_enabled(), "cudnn": bc.cudnn_sdp_enabled()}
    print(f"[sdp] {result['sdp_backends']}", flush=True)

    actor_config = ActorConfig(strategy="fsdp2", rollout_n=1, ppo_micro_batch_size_per_gpu=MICRO_BSZ)
    worker.set_loss_fn(partial(ppo_loss, config=actor_config))

    hf_cfg = model_config.hf_config
    vocab = hf_cfg.text_config.vocab_size if hasattr(hf_cfg, "text_config") else hf_cfg.vocab_size
    data, lens = build_batch(vocab)
    result["resp_len_mean"] = round(float(lens.float().mean()), 1)
    result["resp_len_max"] = int(lens.max())
    result["tokens_total"] = int((PROMPT_LEN + lens).sum())

    common = dict(use_dynamic_bsz=False, micro_batch_size_per_gpu=MICRO_BSZ, use_remove_padding=False, max_response_length=RESP_LEN,
                  max_padded_tokens_per_microbatch=MAX_PADDED_TOKENS)

    td = left_right_2_no_padding(data.to_tensordict())
    tu.assign_non_tensor(td, calculate_entropy=True, compute_loss=False, **common)
    micro, _ = prepare_micro_batches(td)
    result["num_micro_batches"] = len(micro)
    def padded_tokens(m):
        ids = m["input_ids"]
        n = ids.offsets().diff() if getattr(ids, "is_nested", False) else m["attention_mask"].sum(-1)
        return int(n.max().item() * n.numel())
    try:
        result["padded_tokens_per_micro_max"] = max(padded_tokens(m) for m in micro)
    except Exception as e:  # diagnostic only
        result["padded_tokens_per_micro_max"] = f"n/a ({type(e).__name__})"
    print(f"[layout] {len(micro)} micro-batches for {BSZ} seqs (mbs={MICRO_BSZ}, cap={MAX_PADDED_TOKENS}); resp mean={result['resp_len_mean']} max={result['resp_len_max']}", flush=True)

    phases = {}
    timed("old_logprob", lambda: worker.infer_batch(td), phases)
    if phases["old_logprob"]["oom"]:
        result["phases"] = phases
        result["world_size"] = dist.get_world_size()
        if dist.get_rank() == 0:
            print("SWEEP_RESULT " + json.dumps(result), flush=True)
        return

    def make_update_td():
        td2 = left_right_2_no_padding(data.to_tensordict())
        tu.assign_non_tensor(td2, calculate_entropy=False, compute_loss=True, global_batch_size=BSZ, mini_batch_size=BSZ, epochs=1, seed=1,
                             dataloader_kwargs={"shuffle": False}, **common)
        return td2

    timed("update_1", lambda: worker.train_batch(make_update_td()), phases)
    if not phases["update_1"]["oom"]:
        timed("update_2", lambda: worker.train_batch(make_update_td()), phases)
    if not phases["update_1"]["oom"] and not phases["update_2"]["oom"]:
        timed("update_3", lambda: worker.train_batch(make_update_td()), phases)
    result["phases"] = phases
    result["world_size"] = dist.get_world_size()
    for ph in phases.values():  # report the max peak across ranks from rank 0 only
        t = torch.tensor([ph.get("peak_gib", 0.0)], device="cuda")
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        ph["peak_gib"] = round(t.item(), 2)
    if dist.get_rank() == 0:
        print("SWEEP_RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
