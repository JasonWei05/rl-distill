"""Seconds per training step of fastrl.Trainer (median over steps 4..--steps; W&B off), split into generation, reward and
update, plus one validation and the peak GPU memory, for each prompts-per-step:micro-batch pair.

    WANDB_MODE=disabled python bench.py --base <HF dir> --data <dir with main_parquet/ and eval500/> --batches 16:512 64:512
"""

import argparse
import os
import statistics
import tempfile
import time

import torch

import fastrl

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--batches", nargs="+", default=["16:512", "64:512", "256:1024", "1024:1024"])
ap.add_argument("--steps", type=int, default=20)
args = ap.parse_args()
for spec in args.batches:
    batch, micro = map(int, spec.split(":"))
    a = argparse.Namespace(base=args.base, train=f"{args.data}/main_parquet/train.parquet", val=f"{args.data}/eval500/test.parquet",
                           out=tempfile.mkdtemp(), batch=batch, seed=3, n=16, micro=micro, lr=1e-5, warmup=20, tis_cap=2.0,
                           steps=args.steps, test_freq=args.steps + 1, patience=args.steps, val_n=16, max_response=300,
                           wandb_project="bench", wandb_id=f"bench-b{batch}")
    tr = fastrl.Trainer(a)
    tr.validate()
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
    print(f"BENCH fastrl {os.path.basename(args.base.rstrip('/'))} b{batch}: step {step_s:.3f} s (gen {gen:.3f}, reward {reward:.3f}, "
          f"update {update:.3f}), validation {val_s:.2f} s, peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB", flush=True)
    del tr
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
