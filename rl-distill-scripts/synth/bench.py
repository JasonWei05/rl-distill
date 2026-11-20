"""Seconds per fastrl training step (median over steps 4..--steps, W&B off), split into generation, reward and update,
one validation, and the peak GPU memory, for each number of prompts per step. Run several at once on one GPU to
measure packing.

    python bench.py --base <model dir> --data <dataset dir> --batches 16 64 256 1024 [fastrl args]
"""

import argparse
import os
import statistics
import tempfile
import time

import torch
import torch.distributed as dist
import wandb

import fastrl

os.environ["WANDB_MODE"] = "disabled"

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--batches", type=int, nargs="+", default=[16, 64, 256, 1024])
ap.add_argument("--steps", type=int, default=20)
args, extra = ap.parse_known_args()  # the rest go to fastrl (e.g. --minibatches 4 --n 16 --micro 32)
dist.init_process_group("nccl", store=dist.HashStore(), rank=0, world_size=1)
for batch in args.batches:
    tr = fastrl.Trainer(fastrl.parser().parse_args([
        "--base", args.base, "--data", args.data, "--out", tempfile.mkdtemp(), "--batch", str(batch), "--seed", "3", "--steps", str(args.steps), "--wandb-project", "bench", "--wandb-id", f"bench-b{batch}", *extra]))
    tr.validate()  # warm-up: compiles and captures the decode graphs
    t = time.perf_counter()
    tr.validate()
    val_s = time.perf_counter() - t
    rows = []
    for step in range(1, args.steps + 1):
        tr.step = step
        t = time.perf_counter()
        m = tr.train_step()
        rows.append((time.perf_counter() - t, m["timing_s/gen"], m["timing_s/reward"], m["timing_s/update_actor"]))
    step_s, gen, reward, update = (statistics.median(r[i] for r in rows[3:]) for i in range(4))
    print(f"BENCH {os.path.basename(args.base.rstrip('/'))} b{batch}: step {step_s:.3f} s (gen {gen:.3f}, reward {reward:.3f}, "
          f"update {update:.3f}), validation {val_s:.2f} s, peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB", flush=True)
    wandb.finish()
    del tr
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
