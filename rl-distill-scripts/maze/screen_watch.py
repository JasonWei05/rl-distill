#!/usr/bin/env python3
"""Stream-screen pretraining checkpoints as they land (text-maze study, step 3).

Polls run directories for ckpt-<step>/ folders, screens the ones it is responsible for on the fixed
dev set with screen_sft_ckpts.screen_one, and appends results to <run>/screen_<tag>.json. Work is
split across workers by step index so several GPUs share the load evenly:
    worker w of n takes checkpoints with (step // stride) % n == w.
Stops when every run has a DONE or FAILED marker and nothing is left to screen.

    CUDA_VISIBLE_DEVICES=2 ~/TailRL/.venv/bin/python rl-distill-scripts/maze/screen_watch.py \
        --run-dirs ~/tailrl_maze_runs/pretrain/* --stride 50 --k 16 --worker 0 --n-workers 6 --tag stride50_k16

Phase 2 (every checkpoint within +-100 steps of the crossing at k=64) and phase 3 (finalists at
k=1024) are the same command with --stride 10 --min-step/--max-step and --k 64/1024, or
screen_sft_ckpts.py --paths for a hand-picked list.
"""

import argparse
import glob
import json
import os
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import screen_sft_ckpts as ss  # noqa: E402

import torch  # noqa: E402


def ckpt_steps(run_dir):
    out = []
    for d in os.listdir(run_dir):
        if d.startswith("ckpt-") and d[5:].isdigit() and os.path.isfile(os.path.join(run_dir, d, "model.safetensors")):
            out.append(int(d[5:]))
    return sorted(out)


def complete(path, settle_s=5.0):
    """A checkpoint is complete once its files have not changed for settle_s seconds."""
    newest = max(os.path.getmtime(os.path.join(path, f)) for f in os.listdir(path))
    return time.time() - newest > settle_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dirs", nargs="+", required=True)
    ap.add_argument("--stride", type=int, default=50)
    ap.add_argument("--min-step", type=int, default=0)
    ap.add_argument("--max-step", type=int, default=10**9)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--n-workers", type=int, default=1)
    ap.add_argument("--tag", default=None, help="results file suffix (default: stride<stride>_k<k>)")
    ap.add_argument("--poll", type=float, default=30.0)
    ap.add_argument("--once", action="store_true", help="one pass, no polling")
    ap.add_argument("--tailrl-root", default=ss.DEFAULT_TAILRL)
    ap.add_argument("--parquet", default=None)
    ap.add_argument("--n-prompts", type=int, default=1000)
    ap.add_argument("--batch-prompts", type=int, default=128)
    ap.add_argument("--max-new-tokens", type=int, default=180)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    tag = args.tag or f"stride{args.stride}_k{args.k}"

    parquet = args.parquet or os.path.join(args.tailrl_root, "experiments/text_maze/data/eval1000/test.parquet")
    judge = ss.load_judge(args.tailrl_root)
    prompts, gts = ss.load_prompts(parquet, args.n_prompts)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sargs = SimpleNamespace(k=args.k, batch_prompts=args.batch_prompts, max_new_tokens=args.max_new_tokens)

    def mine(step):
        return (args.min_step <= step <= args.max_step and step % args.stride == 0
                and (step // args.stride) % args.n_workers == args.worker)

    while True:
        did = 0
        all_finished = True
        for run in args.run_dirs:
            run = run.rstrip("/")
            if not os.path.isdir(run):
                continue
            finished = os.path.exists(os.path.join(run, "DONE")) or os.path.exists(os.path.join(run, "FAILED"))
            all_finished &= finished
            # one results file per worker (concurrent writers to a shared file lose each other's entries);
            # "already screened" is the union over every worker's file for this tag
            out_path = os.path.join(run, f"screen_{tag}.w{args.worker}.json" if args.n_workers > 1 else f"screen_{tag}.json")
            results = json.load(open(out_path)) if os.path.exists(out_path) else {}
            done_keys = set()
            for f in glob.glob(os.path.join(run, f"screen_{tag}*.json")):
                try:
                    done_keys.update(json.load(open(f)).keys())
                except (json.JSONDecodeError, OSError):
                    pass
            for step in ckpt_steps(run):
                key = f"ckpt-{step}"
                if key in done_keys or not mine(step):
                    continue
                path = os.path.join(run, key)
                if not complete(path):
                    all_finished = False
                    continue
                torch.manual_seed(args.seed)
                r = ss.screen_one(path, prompts, gts, judge, sargs, device)
                results[key] = r
                g, s = r["goal_reached"], r["is_shortest"]
                print(f"{os.path.basename(run):>16} {key:>10} goal p@1 {g['pass@1']:.4f} short p@1 {s['pass@1']:.4f} "
                      f"goal mix16 {g['mixed16']:.3f} len {r['mean_response_tokens']:.1f} ({r['seconds']}s)", flush=True)
                tmp = out_path + f".tmp{os.getpid()}"
                with open(tmp, "w") as f:
                    json.dump(results, f, indent=2)
                os.replace(tmp, out_path)
                did += 1
        if args.once or (all_finished and did == 0):
            break
        if did == 0:
            time.sleep(args.poll)
    print("screen_watch done")


if __name__ == "__main__":
    main()
