#!/usr/bin/env python3
"""Generate off-policy distillation traces from a maze teacher: K samples per prompt with the teacher's
full 32-way log-probabilities at every response position.

Output parquet matches rl-distill's FullVocabDistillDataset schema (full_vocab_distill_dataset.py):
  input_ids                [prompt + response] token ids (no padding)
  response_mask            1 on response positions (the generated tokens, incl. DONE/eos), 0 on the prompt
  teacher_topk_token_ids   [response_len, V] ids sorted by descending teacher probability (V = 32 = full vocab)
  teacher_topk_logprobs    [response_len, V] the teacher's full-softmax log-probs (temperature 1), same order
  plus prompt_idx, goal_reached, is_shortest, response_len for bookkeeping.
With V = full vocabulary the tail bucket of the top-k forward KL is identically zero, so the loss is the exact
forward KL. Teacher FLOPs (prefill + decode, forward-only formula) are written to <out>.flops.json.

    CUDA_VISIBLE_DEVICES=2 ~/TailRL/.venv/bin/python rl-distill-scripts/maze/gen_teacher_traces.py \
        --teacher <L_RL>/global_step_N/actor/huggingface --prompts ~/tailrl_maze_data/main_parquet/train.parquet \
        --n-prompts 3000 --k 16 --out ~/tailrl_maze_runs/traces/L0rl_to_S0/train.parquet
"""

import argparse
import ast
import importlib.util
import json
import os
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

HOME = os.path.expanduser("~")


def field(v):
    return ast.literal_eval(v) if isinstance(v, str) else v


def load_judge(tailrl_root):
    path = os.path.join(tailrl_root, "experiments/text_maze/verl/utils/reward_score/maze.py")
    spec = importlib.util.spec_from_file_location("tailrl_maze_judge", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def forward_tflops(config, seqlens):
    """Forward-only FLOPs with verl's FlopsCounter formula (dense 2N + attention), in TFLOPs."""
    h, L, V = config.hidden_size, config.num_hidden_layers, config.vocab_size
    heads, kv = config.num_attention_heads, config.num_key_value_heads
    hd = getattr(config, "head_dim", None) or h // heads
    n_dense = L * (h * config.intermediate_size * 3 + h * (heads * hd + 2 * kv * hd + heads * hd)) + V * h * 2
    tokens = sum(seqlens)
    return (2 * n_dense * tokens + 4 * sum(s * s for s in seqlens) * hd * heads * L) / 1e12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--prompts", required=True, help="RL parquet with prompt / reward_model columns")
    ap.add_argument("--n-prompts", type=int, default=3000)
    ap.add_argument("--offset", type=int, default=0, help="skip this many rows first (to take a disjoint val slice)")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--batch-prompts", type=int, default=64)
    ap.add_argument("--max-new-tokens", type=int, default=180)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tailrl-root", default=os.path.join(HOME, "TailRL"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    device = "cuda"

    tok = AutoTokenizer.from_pretrained(args.teacher)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = "<pad>"
    pad_id = tok.convert_tokens_to_ids(tok.pad_token)
    config = AutoConfig.from_pretrained(args.teacher)
    model = AutoModelForCausalLM.from_pretrained(args.teacher, torch_dtype=torch.bfloat16).to(device).eval()
    V = model.get_output_embeddings().weight.shape[0]
    judge = load_judge(args.tailrl_root)

    rows = pq.read_table(args.prompts).to_pylist()[args.offset : args.offset + args.n_prompts]
    prompts, gts = [], []
    for r in rows:
        p = field(r["prompt"])
        prompts.append(p[0]["content"] if isinstance(p, list) else p)
        gts.append(field(r["reward_model"])["ground_truth"])

    out_rows = {"prompt_idx": [], "input_ids": [], "response_mask": [], "teacher_topk_token_ids": [], "teacher_topk_logprobs": [],
                "goal_reached": [], "is_shortest": [], "response_len": []}
    seqlens, t0 = [], time.time()
    for i0 in range(0, len(prompts), args.batch_prompts):
        enc = tok(prompts[i0 : i0 + args.batch_prompts], return_tensors="pt", padding=True, add_special_tokens=False)
        ids = enc.input_ids.to(device).repeat_interleave(args.k, 0)
        am = enc.attention_mask.to(device).repeat_interleave(args.k, 0)
        plen = ids.shape[1]
        with torch.no_grad():
            gen = model.generate(input_ids=ids, attention_mask=am, max_new_tokens=args.max_new_tokens, do_sample=True,
                                 temperature=1.0, top_p=1.0, pad_token_id=pad_id)
            full_am = (gen != pad_id).long()
            full_am[:, :plen] = am  # prompt padding is left-side; responses are right-padded with pad_id after eos
            logits = model(input_ids=gen, attention_mask=full_am).logits.float()
            logp = torch.log_softmax(logits, dim=-1)  # [B, T, V]; position t predicts token t+1
        texts = tok.batch_decode(gen[:, plen:], skip_special_tokens=False)
        for j in range(gen.shape[0]):
            resp = gen[j, plen:]
            rlen = int((resp != pad_id).sum().item())
            if rlen == 0:
                continue
            prompt_ids = ids[j][am[j].bool()].tolist()
            resp_ids = resp[:rlen].tolist()
            # teacher distribution for response token r (r = 0..rlen-1) sits at position plen-1+r of the full sequence
            lp = logp[j, plen - 1 : plen - 1 + rlen, :]  # [rlen, V]
            sorted_lp, sorted_ids = torch.sort(lp, dim=-1, descending=True)
            res = judge.judge_maze(solution_str=texts[j], ground_truth=gts[i0 + j // args.k])
            out_rows["prompt_idx"].append(args.offset + i0 + j // args.k)
            out_rows["input_ids"].append(prompt_ids + resp_ids)
            out_rows["response_mask"].append([0] * len(prompt_ids) + [1] * rlen)
            out_rows["teacher_topk_token_ids"].append(sorted_ids.cpu().numpy().astype(np.int32).tolist())
            out_rows["teacher_topk_logprobs"].append(sorted_lp.cpu().numpy().astype(np.float32).tolist())
            out_rows["goal_reached"].append(float(res.get("goal_reached", 0.0)))
            out_rows["is_shortest"].append(float(res.get("is_shortest", 0.0)))
            out_rows["response_len"].append(rlen)
            seqlens.append(len(prompt_ids) + rlen)
        print(f"{i0 + args.batch_prompts}/{len(prompts)} prompts, {len(seqlens)} traces, goal {np.mean(out_rows['goal_reached']):.3f} "
              f"short {np.mean(out_rows['is_shortest']):.3f} ({time.time() - t0:.0f}s)", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    table = pa.table({
        "prompt_idx": pa.array(out_rows["prompt_idx"], pa.int32()),
        "input_ids": pa.array(out_rows["input_ids"], pa.list_(pa.int32())),
        "response_mask": pa.array(out_rows["response_mask"], pa.list_(pa.int8())),
        "teacher_topk_token_ids": pa.array(out_rows["teacher_topk_token_ids"], pa.list_(pa.list_(pa.int32()))),
        "teacher_topk_logprobs": pa.array(out_rows["teacher_topk_logprobs"], pa.list_(pa.list_(pa.float32()))),
        "goal_reached": pa.array(out_rows["goal_reached"], pa.float32()),
        "is_shortest": pa.array(out_rows["is_shortest"], pa.float32()),
        "response_len": pa.array(out_rows["response_len"], pa.int32()),
    })
    pq.write_table(table, args.out, row_group_size=2048)
    tf = forward_tflops(config, seqlens)
    meta = {"teacher": args.teacher, "prompts": args.prompts, "n_prompts": len(prompts), "k": args.k, "traces": len(seqlens),
            "vocab": V, "teacher_params_nonembed": None, "teacher_forward_tflops": tf, "total_tokens": int(sum(seqlens)),
            "goal_reached_mean": float(np.mean(out_rows["goal_reached"])), "is_shortest_mean": float(np.mean(out_rows["is_shortest"]))}
    with open(args.out + ".flops.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {args.out}: {len(seqlens)} traces, teacher forward {tf:.3f} TFLOPs, goal {meta['goal_reached_mean']:.3f} short {meta['is_shortest_mean']:.3f}")


if __name__ == "__main__":
    main()
