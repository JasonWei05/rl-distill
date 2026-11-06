#!/usr/bin/env python3
"""Keep the 20k-step pretraining re-run within /work2's free space while it trains (federer, 2026-11-03).

Deletes a raw ckpt-<step>/ only when it is settled (untouched for --settle seconds) and NOT needed later:
  kept = every checkpoint inside the size's windows (the original trajectories' 0.3% / 3% / 30% crossings, S near
         8,500-8,900 / 9,500-9,800 / 16,000-18,000 and L near 2,900-3,100 / 3,500-3,800 / 5,000-6,000, +- ~800 steps,
         which covers the selection brackets [cross-350, cross+100] plus the +-60 SWA windows), the last step, and
         screening-grid checkpoints (step % 100 == 0: screen_watch --stride 50 over SAVE_STEPS=20) until they are screened.
  If a re-run crossing lands outside its window, select_matched_init.py's bracket is incomplete: check before Step 2.
The full trajectory is cheap to regenerate (~40 min on 2 GPUs), so pruned checkpoints are not uploaded.
    python prune_pretrain_rerun.py --root ~/tailrl_maze_runs/pretrain20k_rerun [--loop 60] [--dry-run]
"""
import argparse
import glob
import json
import os
import shutil
import time

WINDOWS = {  # run name -> inclusive [lo, hi] step windows kept whole (others: screening grid until screened)
    # 08:10: the L re-run crossed ~1,000 steps earlier than the original (no loss spike near step 2,000), so its 0.3%
    # bracket fell below the first window and was pruned; L 0.3% comes from a second L re-run (rep2) instead.
    "S_ds0_is4": [(8740, 10460), (11190, 11960)],  # 08:20: the 0.3% [8800, 10400] (wide, center-step 80), 3% and 5% brackets +- 60
    # 08:15: 30% dropped, 5% added (user); all new L levels come from rep2 (one trajectory), rep's 3% bracket kept as a fallback
    "L_ds0_is0_rep": [],                  # 08:45: rep2 reproduced it bit for bit (identical k=16 screening), not needed
    "L_ds0_is0_rep2": [(1490, 2860)],     # 08:45: the 0.3% / 3% / 5% brackets [1550, 2000] / [2150, 2600] / [2350, 2800] +- 60
}


def screened_keys(rdir):
    keys = set()
    for f in glob.glob(os.path.join(rdir, "screen_stride50_k16*.json")):
        try:
            keys.update(json.load(open(f)).keys())
        except (json.JSONDecodeError, OSError):
            pass
    return keys


def keep(run, step, last, screened):
    if step == last:
        return True
    wins = WINDOWS.get(run)
    if wins is None:
        return True  # unknown run: never prune
    if any(lo <= step <= hi for lo, hi in wins):
        return True
    return step % 100 == 0 and f"ckpt-{step}" not in screened


def one_pass(root, settle, dry):
    freed = 0
    for run in sorted(os.listdir(root)):
        rdir = os.path.join(root, run)
        if not os.path.isdir(rdir):
            continue
        steps = sorted(int(d[5:]) for d in os.listdir(rdir) if d.startswith("ckpt-") and d[5:].isdigit())
        if not steps:
            continue
        finished = os.path.exists(os.path.join(rdir, "DONE"))
        last = steps[-1] if finished else -1  # while training, the newest checkpoint is never pruned anyway (not settled)
        screened = screened_keys(rdir)
        for s in steps:
            if keep(run, s, last, screened):
                continue
            path = os.path.join(rdir, f"ckpt-{s}")
            try:
                newest = max(os.path.getmtime(os.path.join(path, f)) for f in os.listdir(path))
            except (ValueError, FileNotFoundError):
                continue
            if time.time() - newest < settle or not os.path.isfile(os.path.join(path, "model.safetensors")):
                continue
            size = sum(os.path.getsize(os.path.join(path, f)) for f in os.listdir(path))
            if not dry:
                shutil.rmtree(path, ignore_errors=True)
            freed += size
    return freed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expanduser("~/tailrl_maze_runs/pretrain20k_rerun"))
    ap.add_argument("--settle", type=float, default=120.0)
    ap.add_argument("--loop", type=float, default=0.0, help="seconds between passes; 0 = one pass")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    while True:
        freed = one_pass(args.root, args.settle, args.dry_run)
        runs = [d for d in os.listdir(args.root) if os.path.isdir(os.path.join(args.root, d))]
        done = all(os.path.exists(os.path.join(args.root, r, "DONE")) or os.path.exists(os.path.join(args.root, r, "FAILED")) for r in runs)
        print(f"{time.strftime('%H:%M:%S')} {'would free' if args.dry_run else 'freed'} {freed / 1e9:.2f} GB; all runs finished: {done}", flush=True)
        if not args.loop or (done and runs):
            if args.loop and done:  # one last pass after the final checkpoints settled
                time.sleep(args.settle + 5)
                one_pass(args.root, args.settle, args.dry_run)
            break
        time.sleep(args.loop)


if __name__ == "__main__":
    main()
