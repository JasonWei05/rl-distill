#!/usr/bin/env python3
"""Find where each pretraining run crosses the starting-capability target, and emit phase-2 commands.

Reads <run>/screen_<tag>.json (written by screen_watch.py / screen_sft_ckpts.py), reports per run the
last screened step below the target and the first at or above it on the chosen metric, and prints the
screen_watch.py commands that screen every checkpoint in the bracket (+- margin) densely.

    ~/TailRL/.venv/bin/python rl-distill-scripts/maze/find_crossing.py --runs ~/tailrl_maze_runs/pretrain/*_ds0_* \
        --tag stride50_k16 --metric goal_reached --target 0.10 --margin 100 --emit --k 64 --gpus 2 3 4 5 6 7
"""

import argparse
import json
import os


def load(run, tag):
    """Merge screen_<tag>.json and the per-worker screen_<tag>.w<N>.json files."""
    import glob
    d = {}
    for path in sorted(glob.glob(os.path.join(run, f"screen_{tag}*.json"))):
        try:
            d.update(json.load(open(path)))
        except (json.JSONDecodeError, OSError):
            pass
    return {int(k.split("-")[1]): v for k, v in d.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--tag", default="stride50_k16")
    ap.add_argument("--metric", default="goal_reached", choices=["goal_reached", "is_shortest"])
    ap.add_argument("--target", type=float, default=0.10)
    ap.add_argument("--margin", type=int, default=100, help="steps on each side of the crossing to screen densely")
    ap.add_argument("--emit", action="store_true", help="print phase-2 screen_watch.py commands")
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--gpus", type=int, nargs="*", default=[2, 3, 4, 5, 6, 7])
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    watch = os.path.join(here, "screen_watch.py")
    py = os.path.expanduser("~/TailRL/.venv/bin/python")
    print(f"{'run':>16} {'screened':>8} {'below':>7} {'p@1':>7} {'at/above':>9} {'p@1':>7}  bracket")
    cmds = []
    for i, run in enumerate(sorted(args.runs)):
        run = run.rstrip("/")
        res = load(run, args.tag)
        steps = sorted(res)
        below = above = None
        for s in steps:
            p = res[s][args.metric]["pass@1"]
            if p >= args.target:
                above = s
                break
            below = s
        name = os.path.basename(run)
        if above is None:
            top = f"{res[steps[-1]][args.metric]['pass@1']:.4f} @ {steps[-1]}" if steps else "-"
            print(f"{name:>16} {len(steps):>8}   not crossed yet (max {top})")
            continue
        pb = res[below][args.metric]["pass@1"] if below is not None else float("nan")
        lo, hi = max(0, (below if below is not None else above) - args.margin), above + args.margin
        print(f"{name:>16} {len(steps):>8} {below if below is not None else '-':>7} {pb:7.4f} {above:>9} {res[above][args.metric]['pass@1']:7.4f}  [{lo}, {hi}]")
        gpu = args.gpus[i % len(args.gpus)] if args.gpus else 0
        cmds.append(
            f"CUDA_VISIBLE_DEVICES={gpu} {py} {watch} --run-dirs {run} --stride {args.stride} --min-step {lo} --max-step {hi} "
            f"--k {args.k} --once --tag bracket_k{args.k} > {run}/screen_bracket_k{args.k}.log 2>&1 &"
        )
    if args.emit and cmds:
        print("\n# phase 2: dense screening of each bracket (one process per run, round-robin over GPUs)")
        for c in cmds:
            print(c)


if __name__ == "__main__":
    main()
