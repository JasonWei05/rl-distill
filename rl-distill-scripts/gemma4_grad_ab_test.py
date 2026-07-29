"""Differential grad-norm experiment: v20 fork code path vs reconstructed stock path.

Runs verl's REAL TrainingWorker (FSDP2, gemma-4-E2B, sdpa, non-rmpad NO_PADDING path —
the exact prod configuration of run_gemma4_pt_*_rl.sh) on ONE deterministic synthetic
batch and reports:
  - forward parity: infer_batch log_probs (saved to disk, compared across arms offline)
  - grad_norm from a single identical train_batch step (old_log_probs = own infer
    log_probs, so PPO ratio ~= 1: the exact first-mini-batch regime of a prod step)

Arms (selected via ARM env var, each run in its own process):
  A: current working-tree code, untouched (fork "v20" path: fused/chunked softcap +
     _ChunkedLogprobsFromLogits with inplace backward + padded-direct labels +
     unit-temperature div skip).
  B: stock reconstruction via monkeypatch BEFORE any forward:
     1) final_logit_softcapping=30.0 restored into the model config (modeling code
        reads it per-forward) and engine._logit_softcap=None so prepare_model_outputs
        applies no fused/in-place softcap;
     2) verl.workers.engine.fsdp.transformer_impl.logprobs_from_logits replaced by
        stock fp32 log_softmax + gather (== logprobs_from_logits_v2 semantics,
        out-of-place, full autograd).
     (padded-direct labels + temp-div-skip stay active — isolated by arm C)
  C: arm B + prepare_model_outputs restored to the HEAD (stock) non-rmpad structure:
     in-place temperature div + narrow/unbind/cat logits packing.

Usage:
  ARM=A SEED=0 CUDA_VISIBLE_DEVICES=2 VERL_FSDP2_LOCAL_LOAD=1 \
    python rl-distill-scripts/gemma4_grad_ab_test.py
Outputs: <OUT_DIR>/result_{ARM}_{SEED}.json and logprobs_{ARM}_{SEED}.pt
"""

import json
import os
import sys
import types
from functools import partial

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
from verl.workers.config import (  # noqa: E402
    ActorConfig,
    FSDPEngineConfig,
    FSDPOptimizerConfig,
    HFModelConfig,
)
from verl.workers.engine_workers import TrainingWorker, TrainingWorkerConfig  # noqa: E402
from verl.workers.utils.losses import ppo_loss  # noqa: E402
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding  # noqa: E402

ARM = os.environ.get("ARM", "A")  # A | B | C; reassigned per-spec in MULTI mode
SEED = int(os.environ.get("SEED", "0"))
MULTI = os.environ.get("MULTI", "")  # e.g. "A:0,A:7,B:0,B:7,C:0" — run all in one process
OUT_DIR = os.environ["OUT_DIR"]
PROMPT_LEN, RESP_LEN, BSZ = 200, 800, 4
# ragged real lengths (left-padded prompts, right-padded responses) so the
# padded-labels-vs-narrow/cat difference is actually exercised
PLENS = [200, 187, 174, 161]
RLENS = [800, 763, 726, 689]

if os.environ.get("GPU_PICK") == "1":
    # pick the freest GPU (never index 3) at run time, before any CUDA init
    import subprocess
    import time

    for _ in range(360):
        rows = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"], text=True
        ).strip().splitlines()
        cand = sorted(
            ((int(f), i) for i, f in (r.split(", ") for r in rows) if i != "3"),
            reverse=True,
        )
        if cand and cand[0][0] >= 32000:
            os.environ["CUDA_VISIBLE_DEVICES"] = cand[0][1]
            print(f"[abtest] picked GPU {cand[0][1]} ({cand[0][0]} MiB free)", flush=True)
            break
        time.sleep(10)
    else:
        raise RuntimeError("no GPU with >=32000 MiB free")

dist.init_process_group(backend="nccl")


def log(msg):
    print(f"[abtest {ARM} seed={SEED}] {msg}", flush=True)


# ---------------------------------------------------------------- batch construction
def build_dataproto(vocab, old_log_probs=None):
    """Deterministic ragged batch in standard verl left-right padded layout."""
    g = torch.Generator().manual_seed(SEED)
    seqlen = PROMPT_LEN + RESP_LEN
    input_ids = torch.randint(3, min(vocab, 200000), (BSZ, seqlen), generator=g)
    attention_mask = torch.zeros(BSZ, seqlen, dtype=torch.int64)
    response_mask = torch.zeros(BSZ, RESP_LEN, dtype=torch.int64)
    for i, (pl, rl) in enumerate(zip(PLENS, RLENS)):
        attention_mask[i, PROMPT_LEN - pl : PROMPT_LEN + rl] = 1
        response_mask[i, :rl] = 1
    position_ids = torch.clip(torch.cumsum(attention_mask, dim=-1) - 1, min=0)
    responses = input_ids[:, PROMPT_LEN:]
    advantages = torch.randn(BSZ, RESP_LEN, generator=g) * response_mask
    if old_log_probs is None:
        old_log_probs = torch.rand(BSZ, RESP_LEN, generator=g) * -1.0
    data = DataProto.from_single_dict(
        {
            "input_ids": input_ids,
            "prompts": input_ids[:, :PROMPT_LEN],
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": responses,
            "response_mask": response_mask,
            "old_log_probs": old_log_probs,
            "advantages": advantages,
            "ref_log_prob": old_log_probs.clone(),
        },
        meta_info={"temperature": 1.0, "global_token_num": [pl + rl for pl, rl in zip(PLENS, RLENS)]},
    )
    return data


# ---------------------------------------------------------------- stock monkeypatches
def stock_logprobs_from_logits(logits, labels, inplace_backward=True, softcap=None, **kw):
    """Stock semantics (== logprobs_from_logits_v2): fp32 log_softmax + gather,
    out-of-place, full autograd. Ignores the fork's fused-softcap/inplace knobs."""
    assert softcap is None, "stock arm must never receive a fused softcap"
    lp = torch.log_softmax(logits.float(), dim=-1)
    return torch.gather(lp, -1, labels.unsqueeze(-1)).squeeze(-1)


def stock_prepare_model_outputs(self, output, output_args, micro_batch, logits_processor_func):
    """Byte-faithful reconstruction of the HEAD (stock) non-rmpad NO_PADDING branch:
    in-place temperature div, narrow/unbind/cat logits packing, plain logprobs call."""
    import verl.utils.torch_functional as verl_F  # noqa: F401
    import verl.workers.engine.fsdp.transformer_impl as timpl

    calculate_entropy = tu.get_non_tensor_data(data=micro_batch, key="calculate_entropy", default=False)
    use_remove_padding = tu.get_non_tensor_data(data=micro_batch, key="use_remove_padding", default=True)
    use_fused_kernels = tu.get_non_tensor_data(data=micro_batch, key="use_fused_kernels", default=False)
    assert not use_remove_padding and not use_fused_kernels

    model_output = {}
    input_ids = micro_batch["input_ids"]

    logits = output.logits  # (bsz, seqlen, vocab)
    temperature = output_args["temperature"].unsqueeze(-1).unsqueeze(-1)
    logits.div_(temperature.clamp(min=1e-8).to(logits.dtype))

    entropy = None
    if calculate_entropy:
        if not self.engine_config.entropy_checkpointing:
            entropy = verl_F.entropy_from_logits(logits)
        else:
            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

    cu_seqlens = input_ids.offsets()
    seq_lengths = cu_seqlens.diff()
    starts = torch.zeros_like(seq_lengths, dtype=torch.int64)
    logits_j = torch.nested.narrow(logits, 1, starts, seq_lengths, layout=torch.jagged)
    logits_rmpad = torch.cat([t for t in logits_j.unbind()])
    input_ids_rmpad_rolled = output_args["input_ids_rmpad_rolled"]
    log_probs = timpl.logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)
    log_probs = torch.nested.nested_tensor_from_jagged(log_probs, cu_seqlens)
    if calculate_entropy:
        entropy_j = torch.nested.narrow(entropy, 1, starts, seq_lengths, layout=torch.jagged)
        entropy = torch.nested.nested_tensor_from_jagged(torch.cat([t for t in entropy_j.unbind()]), cu_seqlens)
        model_output["entropy"] = entropy
    model_output["log_probs"] = log_probs
    return model_output


def apply_stock_patches(worker):
    import verl.workers.engine.fsdp.transformer_impl as timpl

    eng = worker.engine
    module = eng.module  # FSDP2-wrapped; config object is shared with the inner model
    cfg = getattr(module, "config", None)
    text_cfg = getattr(cfg, "text_config", cfg)
    stash = getattr(eng, "_logit_softcap", None)
    assert stash is not None, f"expected engine._logit_softcap stash, got {stash}"
    text_cfg.final_logit_softcapping = float(stash)  # modeling code reads this per-forward
    eng._logit_softcap = None  # prepare_model_outputs now skips fused/in-place softcap
    timpl.logprobs_from_logits = stock_logprobs_from_logits
    log(f"stock patch: in-model softcap restored to {text_cfg.final_logit_softcapping}, stock logprobs installed")
    if ARM == "C":
        eng.prepare_model_outputs = types.MethodType(stock_prepare_model_outputs, eng)
        log("stock patch: HEAD prepare_model_outputs (narrow/unbind/cat + in-place temp div) installed")


# ---------------------------------------------------------------- serialization
def ser(x):
    if hasattr(x, "values") and hasattr(x, "aggregation") and not isinstance(x, dict):
        return {"metric_values": ser(x.values), "agg": str(x.aggregation)}
    if isinstance(x, torch.Tensor):
        return x.item() if x.numel() == 1 else x.tolist()
    if isinstance(x, (list, tuple)):
        return [ser(v) for v in x]
    if isinstance(x, dict):
        return {k: ser(v) for k, v in x.items()}
    if isinstance(x, (int, float, str, bool, type(None))):
        return x
    return str(x)


def main():
    model_config = HFModelConfig(
        path="google/gemma-4-E2B",
        use_remove_padding=False,
        override_config={"attn_implementation": "sdpa"},
        enable_gradient_checkpointing=True,
    )
    # optional: REDUCE_DTYPE=bf16 shaves the ~20 GiB fp32 grad-reduce spike so the run
    # fits beside GPU co-tenants. Applied identically to every arm, so the A/B comparison
    # is unaffected; absolute grad_norm shifts only by bf16 reduction rounding.
    mp_kwargs = {}
    if os.environ.get("REDUCE_DTYPE"):
        mp_kwargs["mixed_precision"] = {
            "param_dtype": "bf16",
            "reduce_dtype": os.environ["REDUCE_DTYPE"],
            "buffer_dtype": "fp32",
        }
    engine_config = FSDPEngineConfig(
        forward_only=False,
        strategy="fsdp2",
        fsdp_size=1,
        model_dtype="bfloat16",
        **mp_kwargs,
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
    log("model loaded")

    if ARM in ("B", "C"):
        apply_stock_patches(worker)
    else:
        assert ARM == "A"
        log(f"arm A: v20 working-tree code, engine._logit_softcap={worker.engine._logit_softcap}")

    actor_config = ActorConfig(strategy="fsdp2", rollout_n=1, ppo_micro_batch_size_per_gpu=1)
    worker.set_loss_fn(partial(ppo_loss, config=actor_config))

    hf_cfg = model_config.hf_config
    vocab = hf_cfg.text_config.vocab_size if hasattr(hf_cfg, "text_config") else hf_cfg.vocab_size

    # ---- pass 1: forward-only infer (old-logprob pass, entropy off) — parity artifact
    data = build_dataproto(vocab)
    td = left_right_2_no_padding(data.to_tensordict())
    tu.assign_non_tensor(
        td,
        calculate_entropy=False,
        compute_loss=False,
        use_dynamic_bsz=False,
        micro_batch_size_per_gpu=1,
        use_remove_padding=False,
        max_response_length=RESP_LEN,
    )
    out = worker.infer_batch(td)
    lp_nested = out["log_probs"]
    flat_lp = lp_nested.values().float().cpu()
    torch.save(flat_lp, os.path.join(OUT_DIR, f"logprobs_{ARM}_{SEED}.pt"))
    log(f"infer done: flat log_probs shape={tuple(flat_lp.shape)} mean={flat_lp.mean():.6f}")

    # slice response log_probs into padded (BSZ, RESP_LEN) — becomes old_log_probs, ratio ~= 1
    ref_td = left_right_2_no_padding(build_dataproto(vocab).to_tensordict())
    old_lp = no_padding_2_padding(lp_nested, ref_td).float()
    resp_mask = torch.zeros(BSZ, RESP_LEN)
    for i, rl in enumerate(RLENS):
        resp_mask[i, :rl] = 1
    old_lp = old_lp * resp_mask
    log(f"old_log_probs from infer: masked mean={(old_lp.sum() / resp_mask.sum()):.6f}")

    # ---- pass 2: one identical train step
    data2 = build_dataproto(vocab, old_log_probs=old_lp)
    td2 = left_right_2_no_padding(data2.to_tensordict())
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
    metrics = worker.train_batch(td2)
    m = tu.get(metrics, "metrics")
    m_ser = ser(dict(m))
    grad_norm = m_ser.get("grad_norm")
    log(f"train done: grad_norm={grad_norm} loss={m_ser.get('loss')}")
    log(f"full metrics: {json.dumps(m_ser)}")

    with open(os.path.join(OUT_DIR, f"result_{ARM}_{SEED}.json"), "w") as f:
        json.dump(
            {
                "arm": ARM,
                "seed": SEED,
                "grad_norm": grad_norm,
                "loss": m_ser.get("loss"),
                "metrics": m_ser,
                "infer_logprobs_mean": flat_lp.mean().item(),
                "old_logprobs_masked_mean": (old_lp.sum() / resp_mask.sum()).item(),
            },
            f,
            indent=2,
        )
    print(f"ABTEST_ARM_{ARM}_SEED_{SEED}_DONE", flush=True)

    # release this worker's GPU state so the next spec in MULTI mode starts clean
    del worker
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def run_all():
    """Run several arm:seed specs in one process (imports cost ~15 min on a loaded box).
    Arms MUST be ordered A before B/C: the stock logprobs monkeypatch is module-level
    and is never unwound. Each spec builds a fresh TrainingWorker (fresh weights)."""
    global ARM, SEED
    specs = [s.split(":") for s in MULTI.split(",")]
    arms = [a for a, _ in specs]
    assert arms == sorted(arms), f"arms must be ordered A..B..C, got {arms}"
    import gc
    import time

    for arm, seed in specs:
        if os.path.exists(os.path.join(OUT_DIR, f"result_{arm}_{seed}.json")):
            print(f"[abtest] skip {arm}:{seed} (result exists)", flush=True)
            continue
        ARM, SEED = arm, int(seed)
        # co-tenants on this box balloon by tens of GiB without warning; CUDA pins us to
        # the picked GPU, so on OOM just wait for it to drain and retry the whole spec.
        for attempt in range(30):
            log(f"=== MULTI spec start attempt={attempt} (alloc={torch.cuda.memory_allocated() / 2**30:.2f} GiB) ===")
            try:
                main()
                break
            except torch.OutOfMemoryError as e:
                gc.collect()
                torch.cuda.empty_cache()
                log(f"OOM ({str(e)[:160]}) — waiting 60s for co-tenant to drain, then retrying spec")
                time.sleep(60)
        else:
            log("giving up on spec after 30 OOM retries")
    print("ABTEST_MULTI_ALL_DONE", flush=True)


if __name__ == "__main__":
    if MULTI:
        run_all()
    else:
        main()
