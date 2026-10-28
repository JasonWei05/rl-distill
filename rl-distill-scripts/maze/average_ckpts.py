#!/usr/bin/env python3
"""Average consecutive pretraining checkpoints (SWA-style) to smooth step-to-step policy jitter.

At lr 5e-4 in pure bf16 the maze policies fluctuate by several points of goal_reached between
checkpoints 10 steps apart, so a single checkpoint "closest to the target" is mostly noise. This
averages the weights of the checkpoints within [center - half_window, center + half_window] (fp32
accumulation, saved back in the checkpoints' dtype) and writes a new checkpoint directory with the
tokenizer and configs copied from the center checkpoint.

    ~/TailRL/.venv/bin/python rl-distill-scripts/maze/average_ckpts.py --run ~/tailrl_maze_runs/pretrain20k/S_ds0_is0 \
        --center 7000 --half-window 50 [--stride 20] [--out <run>/avg-7000-w50]

Screen the result with screen_sft_ckpts.py --paths <out>.
"""

import argparse
import os
import shutil

import torch
from safetensors.torch import load_file, save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--center", type=int, required=True)
    ap.add_argument("--half-window", type=int, default=50)
    ap.add_argument("--stride", type=int, default=None, help="checkpoint spacing (default: inferred from the run)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    steps = sorted(int(d[5:]) for d in os.listdir(args.run) if d.startswith("ckpt-") and d[5:].isdigit())
    stride = args.stride or min(b - a for a, b in zip(steps, steps[1:]))
    members = [s for s in steps if args.center - args.half_window <= s <= args.center + args.half_window and (s - args.center) % stride == 0]
    if args.center not in steps:
        raise SystemExit(f"center ckpt-{args.center} not found in {args.run}")
    out = args.out or os.path.join(args.run, f"avg-{args.center}-w{args.half_window}")
    os.makedirs(out, exist_ok=True)

    acc, dtype = None, None
    for s in members:
        sd = load_file(os.path.join(args.run, f"ckpt-{s}", "model.safetensors"))
        if acc is None:
            acc = {k: v.to(torch.float32).clone() for k, v in sd.items()}
            dtype = next(iter(sd.values())).dtype
        else:
            for k, v in sd.items():
                acc[k] += v.to(torch.float32)
    avg = {k: (v / len(members)).to(dtype).contiguous() for k, v in acc.items()}
    save_file(avg, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
    center_dir = os.path.join(args.run, f"ckpt-{args.center}")
    for f in os.listdir(center_dir):
        if f != "model.safetensors":
            shutil.copy(os.path.join(center_dir, f), os.path.join(out, f))
    with open(os.path.join(out, "AVERAGED_FROM.txt"), "w") as f:
        f.write(f"run={args.run}\ncenter={args.center}\nhalf_window={args.half_window}\nstride={stride}\nmembers={members}\n")
    print(f"averaged {len(members)} checkpoints ({members[0]}..{members[-1]}, stride {stride}) -> {out}")


if __name__ == "__main__":
    main()
