#!/usr/bin/env python3
"""Calibrate the analytic FLOPs formula against torch.utils.flop_counter for the maze models.

For S (TailRL's released checkpoint) and L (the 13.3M config, random init) it measures one
training pass (forward + backward) and one generation pass (prefill + 60 decoded tokens) with
FlopCounterMode, and prints the measured / analytic ratio per phase. The analytic numbers come
from the fork's FlopsCounter through FlopsLedger, i.e. exactly what the trainer logs.

    PYTHONPATH=~/TailRL/experiments/text_maze ~/TailRL/.venv/bin/python \
        rl-distill-scripts/maze/calibrate_flops.py --ckpt ~/.cache/tailrl_maze_sft_ckpts/ckpt-3650
"""

import argparse
import ast
import os

import pyarrow.parquet as pq
import torch
from torch.utils.flop_counter import FlopCounterMode
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Qwen2Config

from verl.utils.flops_ledger import FlopsLedger

HOME = os.path.expanduser("~")
L_CONFIG = dict(hidden_size=384, num_hidden_layers=6, num_attention_heads=6, num_key_value_heads=3, intermediate_size=1536)
XL_CONFIG = dict(hidden_size=512, num_hidden_layers=8, num_attention_heads=8, num_key_value_heads=4, intermediate_size=2048)


def field(v):
    return ast.literal_eval(v) if isinstance(v, str) else v


def prompts_from_parquet(path, n):
    rows = pq.read_table(path).to_pylist()[:n]
    out = []
    for r in rows:
        p = field(r["prompt"])
        out.append(p[0]["content"] if isinstance(p, list) else p)
    return out


def measure(model, ledger, prompt_ids, attn, resp_len, device):
    """Returns {phase: (measured_gflop_per_seq, analytic_gflop_per_seq)} for one batch."""
    bsz, plen = prompt_ids.shape
    model.to(device)
    results = {}

    # generation: prefill + resp_len decoded tokens, EOS suppressed so every sequence is full length
    model.eval()
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        out = model.generate(
            input_ids=prompt_ids, attention_mask=attn, max_new_tokens=resp_len, min_new_tokens=resp_len,
            do_sample=False, pad_token_id=0,
        )
    gen_measured = fc.get_total_flops()
    seqlens = [plen + resp_len] * bsz
    gen_analytic = ledger.forward_tflops(seqlens) * 1e12
    results["gen (prefill+decode)"] = (gen_measured / bsz / 1e9, gen_analytic / bsz / 1e9)

    # forward-only pass over the full sequences (old_log_prob / teacher scoring)
    full_ids = out[:, : plen + resp_len]
    full_attn = torch.ones_like(full_ids)
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(input_ids=full_ids, attention_mask=full_attn)
    results["forward (log-prob)"] = (fc.get_total_flops() / bsz / 1e9, gen_analytic / bsz / 1e9)

    # training pass: forward + backward over the full sequences
    model.train()
    with FlopCounterMode(display=False) as fc:
        loss = model(input_ids=full_ids, attention_mask=full_attn, labels=full_ids).loss
        loss.backward()
    model.zero_grad(set_to_none=True)
    train_analytic = ledger.train_tflops(seqlens) * 1e12
    results["update (fwd+bwd)"] = (fc.get_total_flops() / bsz / 1e9, train_analytic / bsz / 1e9)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(HOME, ".cache/tailrl_maze_sft_ckpts/ckpt-3650"))
    ap.add_argument("--parquet", default=os.path.join(HOME, "TailRL/experiments/text_maze/data/eval1000/test.parquet"))
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--resp-len", type=int, default=60)
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager"])
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    tok = AutoTokenizer.from_pretrained(args.ckpt)
    tok.padding_side = "left"
    prompts = prompts_from_parquet(args.parquet, args.batch)
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    prompt_ids, attn = enc.input_ids.to(device), enc.attention_mask.to(device)

    s_config = AutoConfig.from_pretrained(args.ckpt, attn_implementation=args.attn)
    models = {"S": AutoModelForCausalLM.from_pretrained(args.ckpt, config=s_config, torch_dtype=dtype)}
    base = s_config.to_dict()
    base.pop("layer_types", None)  # per-layer list sized for S; Qwen2Config regenerates it for the new depth
    for name, overrides in (("L", L_CONFIG), ("XL", XL_CONFIG)):
        cfg = Qwen2Config(**{**base, **overrides})
        cfg._attn_implementation = args.attn
        models[name] = AutoModelForCausalLM.from_config(cfg, torch_dtype=dtype)

    print(f"attn={args.attn} dtype={args.dtype} batch={args.batch} prompt_len={prompt_ids.shape[1]} resp_len={args.resp_len}")
    print(f"{'model':>5} {'params(M)':>9}  {'phase':<22} {'measured GF/seq':>16} {'analytic GF/seq':>16} {'ratio':>7}")
    for name, model in models.items():
        ledger = FlopsLedger(model.config)
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        res = measure(model, ledger, prompt_ids, attn, args.resp_len, device)
        for phase, (meas, ana) in res.items():
            print(f"{name:>5} {n_params:9.2f}  {phase:<22} {meas:16.3f} {ana:16.3f} {meas / ana:7.3f}")
        model.cpu()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
