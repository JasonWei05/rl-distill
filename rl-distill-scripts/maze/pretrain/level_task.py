#!/usr/bin/env python3
"""One (pretraining run, pass@1 level) task of the shortest-path maze study (maze_pt_pipeline.py runs these in parallel).

select_matched_init.py (SWA-averaged candidates around the run's screened crossing of --target, screened at --select-k,
the 4 closest re-screened at --search-k) -> pick_tune_endpoints.py (an adjacent pair that brackets the target) ->
tune_pass1.py (interpolate to the target at --search-k, correct at --final-k, confirm with a fresh sampling seed at
--final-k) -> <run>/tune-<level>/COMPLETE.
The raw k-sample screen crossing is noisy and averaged candidates differ from raw checkpoints, so when no candidate pair
brackets the target the selection window is widened on both sides (--margin and --margin-below x2, x4); already screened
candidates are reused.

    python3 level_task.py --run <run dir> --level p1 --target 0.01 --tag stride25_k16 --tailrl-root /tmp/maze/TailRL
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--level", required=True)
    ap.add_argument("--target", type=float, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--tailrl-root", required=True)
    ap.add_argument("--metric", default="is_shortest")
    ap.add_argument("--margin", type=int, default=50)
    ap.add_argument("--margin-below", type=int, default=200)
    ap.add_argument("--center-step", type=int, default=10)
    ap.add_argument("--half-window", type=int, default=15)
    ap.add_argument("--select-k", type=int, default=64)
    ap.add_argument("--search-k", type=int, default=256)
    ap.add_argument("--final-k", type=int, default=2048)
    ap.add_argument("--fine-mode", default="fine", choices=["fine", "confirm"])
    args = ap.parse_args()
    py, sel = sys.executable, os.path.join(args.run, f"select_{args.level}_set.json")

    ends = None
    for f in (1, 2, 4):
        mb, ma = f * args.margin_below, f * args.margin
        subprocess.run([py, os.path.join(HERE, "select_matched_init.py"), "--tailrl-root", args.tailrl_root, "--run", args.run,
                        "--tag", args.tag, "--metric", args.metric, "--target", str(args.target), "--margin", str(ma),
                        "--margin-below", str(mb), "--center-step", str(args.center_step), "--half-window", str(args.half_window),
                        "--k", str(args.select_k), "--final-k", str(args.search_k), "--n-final", "4",
                        "--out-name", os.path.basename(sel)], check=True)
        p = subprocess.run([py, os.path.join(HERE, "pick_tune_endpoints.py"), "--select", sel, "--target", str(args.target),
                            "--final-k", str(args.search_k), "--metric", args.metric], capture_output=True, text=True)
        print(p.stdout.strip() or p.stderr.strip()[-300:], flush=True)
        if p.returncode == 0:
            ends = json.loads(p.stdout)
            break
        print(f"LEVEL_TASK widen: margins below/above {mb}/{ma} -> {2 * mb}/{2 * ma}", flush=True)
    if ends is None:
        raise SystemExit(f"LEVEL_TASK no bracketing pair for {args.level} in {args.run}")

    for k in ("a", "b"):   # an endpoint can come from an earlier, wider selection pass whose candidate was not rebuilt here
        path = os.path.join(args.run, ends[k])
        if not os.path.isfile(os.path.join(path, "model.safetensors")):
            _, center, hw = ends[k].split("-")
            subprocess.run([py, os.path.join(HERE, "average_ckpts.py"), "--run", args.run, "--center", center,
                            "--half-window", hw.lstrip("w"), "--out", path], check=True, stdout=subprocess.DEVNULL)
    out = os.path.join(args.run, f"tune-{args.level}")
    cmd = [py, os.path.join(HERE, "tune_pass1.py"), "--tailrl-root", args.tailrl_root, "--a", os.path.join(args.run, ends["a"]),
           "--b", os.path.join(args.run, ends["b"]), "--target", str(args.target), "--out", out, "--k", str(args.final_k),
           "--search-k", str(args.search_k),
           "--metric", args.metric, "--fine-mode", args.fine_mode]
    for k in ("pa", "pb"):
        if k in ends:
            cmd += [f"--{k}", str(ends[k])]
    subprocess.run(cmd, check=True)
    open(os.path.join(out, "COMPLETE"), "w").close()
    print(f"LEVEL_TASK_DONE {os.path.basename(args.run)} {args.level}", flush=True)


if __name__ == "__main__":
    main()
