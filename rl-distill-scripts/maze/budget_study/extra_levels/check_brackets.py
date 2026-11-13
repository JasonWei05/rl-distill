#!/usr/bin/env python3
"""Before Step 2: for each run and target, the bracket select_matched_init.py will use (first stride-50 k=16 screening
point >= target, [below - 250, above + 100]) and whether every SWA member (center +- 60, stride 20) is still on disk."""
import glob
import json
import os
import sys

R = os.path.expanduser("~/tailrl_maze_runs/pretrain20k_rerun")


def screening(run):
    d = {}
    for p in sorted(glob.glob(os.path.join(R, run, "screen_stride50_k16*.json"))):
        d.update(json.load(open(p)))
    return {int(k.split("-")[1]): v["goal_reached"]["pass@1"] for k, v in d.items()}


def main():
    runs = [a for a in sys.argv[1:] if not a[0].isdigit()] or ["S_ds0_is4", "L_ds0_is0_rep2"]
    targets = [float(t) for t in ([a for a in sys.argv[1:] if a[0].isdigit()] or ["0.003", "0.03", "0.05"])]
    for run in runs:
        scr = screening(run)
        have = {int(d[5:]) for d in os.listdir(os.path.join(R, run)) if d.startswith("ckpt-") and d[5:].isdigit()}
        print(f"{run}: {len(scr)} screened grid points (steps {min(scr, default=None)}..{max(scr, default=None)}), {len(have)} ckpts on disk")
        for t in targets:
            below = above = None
            for s in sorted(scr):
                if scr[s] >= t:
                    above = s
                    break
                below = s
            if above is None:
                print(f"  target {t}: NO CROSSING in the screening")
                continue
            lo, hi = max(0, (below if below is not None else above) - 250), above + 100
            centers = [c for c in range(lo - lo % 20, hi + 1, 20) if c >= lo]
            missing = sorted({m for c in centers for m in range(c - 60, c + 61, 20)} - have)
            around = " ".join(f"{s}:{scr[s]:.4f}" for s in sorted(scr) if below is not None and below - 300 <= s <= above + 300)
            print(f"  target {t}: crossing between {below} and {above}; bracket [{lo}, {hi}], {len(centers)} centers; "
                  f"missing SWA members: {missing if missing else 'none'}\n    screening near it: {around}")


if __name__ == "__main__":
    main()
