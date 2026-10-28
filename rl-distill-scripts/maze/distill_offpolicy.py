#!/usr/bin/env python3
"""Off-policy forward-KL distillation for the maze study, mirroring rl-distill's recipe 2 at maze scale.

Loss (rl-distill-scripts/docs/RECIPES.md 2.2, full_vocab_kl_loss.py): per response position t,
    l_t = sum_v p_v (log p_v - log q_v) over the stored teacher set S_t (+ tail bucket; zero here since S_t = V),
with the response mask shifted so the student logits at t-1 predict response token t; batch loss = global token
mean over response positions. Temperature 1, no clamp.
Optimizer and schedule shape follow recipe 2.3 (AdamW, weight decay 0.1, betas (0.9, 0.98), grad clip 1.0,
linear warmup then linear decay to MIN_LR_RATIO), with maze-scale values.
Logs train loss, validation forward KL on held-out teacher traces, and analytic FLOPs (student update 6N;
the teacher's trace-generation FLOPs come from <traces>.flops.json), to <out>/train_metrics.jsonl.

    CUDA_VISIBLE_DEVICES=2 ~/TailRL/.venv/bin/python rl-distill-scripts/maze/distill_offpolicy.py \
        --student ~/tailrl_maze_runs/matched/primary_S0 --train ~/tailrl_maze_runs/traces/<spec>/train.parquet \
        --val ~/tailrl_maze_runs/traces/<spec>/val.parquet --out ~/tailrl_maze_runs/distill/<spec> --steps 1000
"""

import argparse
import json
import math
import os
import time

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler


class TraceDataset(Dataset):
    def __init__(self, path):
        t = pq.read_table(path, columns=["input_ids", "response_mask", "teacher_topk_token_ids", "teacher_topk_logprobs"])
        self.ids = t.column("input_ids").to_pylist()
        self.mask = t.column("response_mask").to_pylist()
        self.tids = t.column("teacher_topk_token_ids").to_pylist()
        self.tlp = t.column("teacher_topk_logprobs").to_pylist()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return self.ids[i], self.mask[i], self.tids[i], self.tlp[i]


def collate(batch, pad_id):
    B = len(batch)
    T = max(len(b[0]) for b in batch)
    R = max(len(b[2]) for b in batch)
    V = len(batch[0][2][0])
    ids = torch.full((B, T), pad_id, dtype=torch.long)
    am = torch.zeros((B, T), dtype=torch.long)
    rmask = torch.zeros((B, T), dtype=torch.bool)
    tids = torch.zeros((B, R, V), dtype=torch.long)
    tlp = torch.full((B, R, V), -1e4, dtype=torch.float32)
    for b, (x, m, ti, tl) in enumerate(batch):
        ids[b, : len(x)] = torch.tensor(x)
        am[b, : len(x)] = 1
        rmask[b, : len(x)] = torch.tensor(m, dtype=torch.bool)
        tids[b, : len(ti)] = torch.tensor(ti)
        tlp[b, : len(tl)] = torch.tensor(tl)
    return ids, am, rmask, tids, tlp


def forward_kl(model, ids, am, rmask, tids, tlp):
    """Sum over response positions of sum_v p_v (log p_v - log q_v), and the number of positions."""
    logits = model(input_ids=ids, attention_mask=am).logits.float()  # [B, T, V]
    logq = torch.log_softmax(logits[:, :-1, :], dim=-1)  # position t-1 predicts token t
    tgt_mask = rmask[:, 1:]  # response tokens as prediction targets
    B = ids.shape[0]
    total, n = logits.new_zeros(()), 0
    for b in range(B):
        pos = tgt_mask[b].nonzero(as_tuple=True)[0]  # positions t (in shifted coordinates) of response tokens
        r = pos.numel()
        if r == 0:
            continue
        q = logq[b, pos, :]  # [r, V]
        ti, tl = tids[b, :r], tlp[b, :r]  # teacher ids / log-probs for response tokens 0..r-1
        p = tl.exp()
        qg = q.gather(-1, ti)
        total = total + (p * (tl - qg)).sum()
        n += r
    return total, n


def train_tflops(config, seqlens):
    h, L, V = config.hidden_size, config.num_hidden_layers, config.vocab_size
    heads, kv = config.num_attention_heads, config.num_key_value_heads
    hd = getattr(config, "head_dim", None) or h // heads
    n_dense = L * (h * config.intermediate_size * 3 + h * (heads * hd + 2 * kv * hd + heads * hd)) + V * h * 2
    return (6 * n_dense * sum(seqlens) + 12 * sum(s * s for s in seqlens) * hd * heads * L) / 1e12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--micro-batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--warmup-steps", type=int, default=100)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--val-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"], help="master weights; forward runs under bf16 autocast")
    ap.add_argument("--tflops-budget", type=float, default=None,
                    help="stop (and save) once teacher trace TFLOPs + cumulative student TFLOPs reach this budget, even before --steps")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = "cuda"

    tok = AutoTokenizer.from_pretrained(args.student)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    model = AutoModelForCausalLM.from_pretrained(args.student, torch_dtype=torch.float32 if args.dtype == "fp32" else torch.bfloat16).to(device)
    train_ds, val_ds = TraceDataset(args.train), TraceDataset(args.val)
    g = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=g, collate_fn=lambda b: collate(b, pad_id), drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.micro_batch_size, shuffle=False, collate_fn=lambda b: collate(b, pad_id))

    decay, no_decay = [], []
    for n_, p in model.named_parameters():
        (no_decay if p.ndim < 2 or "norm" in n_ else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.98))
    # linear warmup, then linear decay to min_lr_ratio * lr at --steps
    def lr_lambda(step):
        if step < args.warmup_steps:
            return (step + 1) / max(1, args.warmup_steps)
        frac = min(1.0, (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps))
        return 1.0 - (1.0 - args.min_lr_ratio) * frac
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    teacher_flops = json.load(open(args.train + ".flops.json")).get("teacher_forward_tflops", 0.0) if os.path.exists(args.train + ".flops.json") else 0.0

    @torch.no_grad()
    def validate():
        model.eval()
        tot, n = 0.0, 0
        for ids, am, rmask, tids, tlp in val_loader:
            ids, am, rmask, tids, tlp = (x.to(device) for x in (ids, am, rmask, tids, tlp))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                s, k = forward_kl(model, ids, am, rmask, tids, tlp)
            tot += s.item(); n += k
        model.train()
        return tot / max(n, 1)

    metrics_path = os.path.join(args.out, "train_metrics.jsonl")
    cum_tflops, step, t0 = 0.0, 0, time.time()
    val_kl = validate()
    with open(metrics_path, "a") as f:
        f.write(json.dumps({"step": 0, "val_forward_kl": val_kl, "cumulative_student_tflops": 0.0, "teacher_trace_tflops": teacher_flops}) + "\n")
    print(f"step 0 val forward KL {val_kl:.5f}", flush=True)
    model.train()
    done = False
    while not done:
        for ids, am, rmask, tids, tlp in train_loader:
            ids, am, rmask, tids, tlp = (x.to(device) for x in (ids, am, rmask, tids, tlp))
            n_tokens = int(rmask[:, 1:].sum().item())
            opt.zero_grad(set_to_none=True)
            loss_sum = 0.0
            for i in range(0, ids.shape[0], args.micro_batch_size):
                sl = slice(i, i + args.micro_batch_size)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    s, _ = forward_kl(model, ids[sl], am[sl], rmask[sl], tids[sl], tlp[sl])
                (s / n_tokens).backward()  # global token mean over the whole batch
                loss_sum += s.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            cum_tflops += train_tflops(model.config, am.sum(-1).tolist())
            rec = {"step": step, "train_forward_kl": loss_sum / n_tokens, "lr": sched.get_last_lr()[0], "cumulative_student_tflops": cum_tflops,
                   "teacher_trace_tflops": teacher_flops, "elapsed_s": round(time.time() - t0, 1)}
            if step % args.val_every == 0:
                rec["val_forward_kl"] = validate()
                print(f"step {step} train KL {rec['train_forward_kl']:.5f} val KL {rec['val_forward_kl']:.5f} lr {rec['lr']:.2e} ({rec['elapsed_s']}s)", flush=True)
            over_budget = args.tflops_budget is not None and teacher_flops + cum_tflops >= args.tflops_budget
            if over_budget:
                rec["budget_stop"] = True
                print(f"FLOPs budget reached at step {step}: teacher {teacher_flops:.3f} + student {cum_tflops:.3f} >= {args.tflops_budget:.3f} TFLOPs", flush=True)
            with open(metrics_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            if step % args.save_every == 0 or step == args.steps or over_budget:
                path = os.path.join(args.out, f"ckpt-{step}")
                model.save_pretrained(path, safe_serialization=True); tok.save_pretrained(path)
            if step >= args.steps or over_budget:
                done = True
                break
    print(f"done: {step} steps, student {cum_tflops:.3f} TFLOPs + teacher traces {teacher_flops:.3f} TFLOPs")


if __name__ == "__main__":
    main()
