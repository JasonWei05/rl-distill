#!/usr/bin/env python3
"""Per-run candidate selection for the matched starting capability (text-maze study, step 3).

1. Read the run's stride-50 screening (screen_<tag>.json) and find the goal_reached crossing bracket.
2. Build SWA-style averaged candidates (average_ckpts.py) centered on every saved checkpoint in the
   bracket that lies on the --center-step grid, each averaging the checkpoints within --half-window.
3. Screen every candidate at --k samples per maze on the dev set and print the table.
4. Rank candidates by |goal_reached pass@1 - target|; re-screen the --n-final closest at --final-k.
Writes <run>/select_<metric>.json with every candidate's metrics and per-maze counts.

    ~/TailRL/.venv/bin/python rl-distill-scripts/maze/select_matched_init.py --run ~/tailrl_maze_runs/pretrain20k/S_ds0_is0 \
        --center-step 20 --half-window 60 --k 64 --final-k 1024 --n-final 2
"""

import argparse
import json
import os
import subprocess
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import screen_sft_ckpts as ss  # noqa: E402

import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def crossing(run, tag, metric, target):
    import glob
    d = {}
    for p in sorted(glob.glob(os.path.join(run, f"screen_{tag}*.json"))):  # shared file + per-worker files
        try:
            d.update(json.load(open(p)))
        except (json.JSONDecodeError, OSError):
            pass
    r = {int(k.split("-")[1]): v for k, v in d.items()}
    below = None
    for s in sorted(r):
        if r[s][metric]["pass@1"] >= target:
            return below, s
        below = s
    return below, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--tag", default="stride50_k16")
    ap.add_argument("--metric", default="goal_reached")
    ap.add_argument("--target", type=float, default=0.10)
    ap.add_argument("--margin", type=int, default=100, help="steps above the raw crossing to include")
    ap.add_argument("--margin-below", type=int, default=250, help="steps below the raw crossing to include (averaged candidates are stronger than raw ones, so the target sits earlier)")
    ap.add_argument("--bracket", type=int, nargs=2, metavar=("LO", "HI"), default=None,
                    help="explicit candidate step range instead of the screening crossing (for low targets, where the k=16 screening is too noisy to bracket)")
    ap.add_argument("--out-name", default=None, help="results file name under the run (default select_<metric>.json)")
    ap.add_argument("--center-step", type=int, default=20)
    ap.add_argument("--half-window", type=int, default=60)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--final-k", type=int, default=1024)
    ap.add_argument("--n-final", type=int, default=2)
    ap.add_argument("--tailrl-root", default=ss.DEFAULT_TAILRL)
    ap.add_argument("--n-prompts", type=int, default=1000)
    ap.add_argument("--batch-prompts", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run = args.run.rstrip("/")

    if args.bracket:
        below = above = None
        lo, hi = args.bracket
    else:
        below, above = crossing(run, args.tag, args.metric, args.target)
        if above is None:
            raise SystemExit(f"{os.path.basename(run)}: no crossing of {args.target} on {args.metric} in screen_{args.tag}.json")
        lo, hi = max(0, (below if below is not None else above) - args.margin_below), above + args.margin
    steps = sorted(int(d[5:]) for d in os.listdir(run) if d.startswith("ckpt-") and d[5:].isdigit())
    stride = min(b - a for a, b in zip(steps, steps[1:]))
    centers = [s for s in steps if lo <= s <= hi and s % args.center_step == 0
               and s - args.half_window >= steps[0] and s + args.half_window <= steps[-1]]
    print(f"{os.path.basename(run)}: crossing between {below} and {above}; bracket [{lo}, {hi}]; {len(centers)} centers, "
          f"window +-{args.half_window} over stride {stride}")

    cands = []
    for c in centers:
        out = os.path.join(run, f"avg-{c}-w{args.half_window}")
        if not os.path.isfile(os.path.join(out, "model.safetensors")):
            subprocess.run([sys.executable, os.path.join(HERE, "average_ckpts.py"), "--run", run, "--center", str(c),
                            "--half-window", str(args.half_window), "--stride", str(stride), "--out", out], check=True,
                           stdout=subprocess.DEVNULL)
        cands.append((c, out))

    judge = ss.load_judge(args.tailrl_root)
    parquet = os.path.join(args.tailrl_root, "experiments/text_maze/data/eval1000/test.parquet")
    prompts, gts = ss.load_prompts(parquet, args.n_prompts)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_path = os.path.join(run, args.out_name or f"select_{args.metric}.json")
    results = json.load(open(out_path)) if os.path.exists(out_path) else {}

    def screen(path, k, key):
        if key in results and results[key]["k"] == k:
            return results[key]
        torch.manual_seed(args.seed)
        r = ss.screen_one(path, prompts, gts, judge, SimpleNamespace(k=k, batch_prompts=args.batch_prompts, max_new_tokens=180), device)
        results[key] = r
        tmp = out_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(results, f)
        os.replace(tmp, out_path)
        return r

    print(f"{'center':>7} {'goal p@1':>9} {'short p@1':>9} {'goal p@16':>9} {'goal p@64':>9} {'goal mix16':>10} {'short mix16':>11} {'len':>6}")
    table = []
    for c, path in cands:
        r = screen(path, args.k, f"avg-{c}-w{args.half_window}@k{args.k}")
        g, s = r["goal_reached"], r["is_shortest"]
        table.append((abs(g["pass@1"] - args.target), c, path))
        print(f"{c:>7} {g['pass@1']:9.4f} {s['pass@1']:9.4f} {g['pass@16']:9.4f} {g['pass@64']:9.4f} {g['mixed16']:10.3f} {s['mixed16']:11.3f} {r['mean_response_tokens']:6.1f}", flush=True)

    table.sort()
    print(f"\nclosest to target {args.target} on {args.metric}: " + ", ".join(f"center {c} (|d|={d:.4f})" for d, c, _ in table[: args.n_final]))
    for d, c, path in table[: args.n_final]:
        r = screen(path, args.final_k, f"avg-{c}-w{args.half_window}@k{args.final_k}")
        g, s = r["goal_reached"], r["is_shortest"]
        pk = lambda m: " ".join(f"p@{k} {m[f'pass@{k}']:.4f}" for k in (1, 16, 64, 256, 1024) if f"pass@{k}" in m)
        print(f"final k={args.final_k} center {c}: goal {pk(g)} | short {pk(s)} | goal mix16 {g['mixed16']:.3f} ({r['seconds']}s)", flush=True)
    results["_selection"] = {"metric": args.metric, "target": args.target, "bracket": [lo, hi], "ranked_centers": [c for _, c, _ in table]}
    with open(out_path, "w") as f:
        json.dump(results, f)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
