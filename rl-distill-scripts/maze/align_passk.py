#!/usr/bin/env python3
"""Tune one pretraining run to a whole shortest-path pass@k curve (pass@1 exactly, pass@1024 matched), sampling-free.

pass@1 alone leaves the pass@k shape free: models of different sizes at the same pass@1 spread their success over
different numbers of mazes. Two knobs give a 2-D family of bases:
  * the training point: SWA-averaged checkpoints at --center-step spacing (as exact_tune.py) and weight interpolation
    between two of them;
  * a logit scale beta, applied by multiplying the final RMSNorm weight (model.norm.weight) by beta. This is exactly sampling
    the unscaled model at temperature 1/beta, and leaves the tied input embeddings untouched.
For a candidate, one forward pass caches the logits at every position of every shortest path (+ the DONE after it;
exact_pass.py), so the exact per-maze shortest-path probability -- hence pass@k for every k -- is available for any beta
without another forward pass. For each candidate, beta is solved so pass@1 hits --target exactly; the candidate is then
scored by its pass@1024 at that beta, and the training point (center walk, then interpolation) is searched until pass@1024
matches the reference (--ref-passk JSON {k: pass@k} or --ref-ckpt, scored exactly) within --tol-abs. The chosen model
(interpolated weights, norm scaled by beta) is written to <work>/<run>/align-<level>/best with align.json (both curves).

    ~/TailRL/.venv/bin/python rl-distill-scripts/maze/align_passk.py --run-s3 s3://.../L/L_is0_... --level p1 --target 0.01 \
        --ref-ckpt <small base dir> --work /tmp/align --parquet <dev test.parquet>
"""
import argparse
import glob
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time

import numpy as np
import torch
from safetensors.torch import load_file, save_file

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import exact_pass as ep  # noqa: E402
import exact_tune as et  # noqa: E402
import screen_sft_ckpts as ss  # noqa: E402

KS = ep.KS


@torch.no_grad()
def logit_cache(ckpt, prompts, gts, max_paths, device="cuda", batch_tokens=262144):
    """Logits Z [N, V] and targets Y [N] at every scored position (path moves + DONE), the path id of each position and
    the maze id of each path, for exact per-maze shortest-path probabilities at any logit scale."""
    sc = ep.Scorer(ckpt, device, batch_tokens)
    Zs, Ys, pos_path, path_maze, pid = [], [], [], [], 0
    for m, (prompt, gt) in enumerate(zip(prompts, gts)):
        g = ep.grid_of(gt)
        pids = sc.tok.encode(prompt, add_special_tokens=False)
        paths = ep.shortest_paths(g, max_paths) or sc.best_first(pids, g, max_paths)
        seqs = [pids + [sc.ids[w] for w in p] + [sc.ids["DONE"]] for p in paths]
        bs = max(1, batch_tokens // max(len(s) for s in seqs))
        for b in range(0, len(seqs), bs):
            chunk = seqs[b:b + bs]
            L = max(len(s) for s in chunk)
            x = torch.zeros(len(chunk), L, dtype=torch.long)
            att = torch.zeros(len(chunk), L, dtype=torch.long)
            for r, s in enumerate(chunk):
                x[r, :len(s)] = torch.tensor(s)
                att[r, :len(s)] = 1
            logits = sc.model(input_ids=x.to(device), attention_mask=att.to(device)).logits.float()
            for r, s in enumerate(chunk):
                n0, n1 = len(pids), len(s)
                Zs.append(logits[r, n0 - 1:n1 - 1])
                Ys.append(torch.tensor(s[n0:n1], device=device))
                pos_path.append(torch.full((n1 - n0,), pid, device=device, dtype=torch.long))
                path_maze.append(m)
                pid += 1
    return {"Z": torch.cat(Zs), "Y": torch.cat(Ys), "pos_path": torch.cat(pos_path),
            "path_maze": torch.tensor(path_maze, device=device), "n_paths": pid, "n_mazes": len(prompts)}


def per_maze_p(cache, beta):
    lp = torch.log_softmax(beta * cache["Z"], -1).gather(1, cache["Y"][:, None]).squeeze(1)
    path_lp = torch.zeros(cache["n_paths"], device=lp.device).index_add_(0, cache["pos_path"], lp)
    p = torch.zeros(cache["n_mazes"], device=lp.device).index_add_(0, cache["path_maze"], path_lp.exp())
    return p.clamp(max=1.0)


def curve(p):
    p = p.double()
    return {k: float((1 - (1 - p) ** k).mean()) for k in KS}


def solve_beta(cache, target, lo=0.3, hi=3.0):
    """beta with pass@1 == target (pass@1 is monotone in beta near a policy's mode-seeking regime; bisection in log beta)."""
    f = lambda b: float(per_maze_p(cache, b).mean()) - target
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0:   # pass@1 not bracketed on [lo, hi]: take the closer end
        return lo if abs(flo) < abs(fhi) else hi
    for _ in range(40):
        mid = math.sqrt(lo * hi)
        fm = f(mid)
        if (fm > 0) == (fhi > 0):
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return math.sqrt(lo * hi)


def write_scaled(src, beta, out):
    if os.path.exists(out):
        shutil.rmtree(out)
    shutil.copytree(src, out)
    sd = load_file(os.path.join(out, "model.safetensors"))
    key = "model.norm.weight"
    sd[key] = (sd[key].float() * beta).to(sd[key].dtype)
    save_file(sd, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
    with open(os.path.join(out, "SCALED_FROM.txt"), "w") as f:
        f.write(f"{src}\nmodel.norm.weight x {beta:.6f} (= sampling {os.path.basename(src)} at temperature {1 / beta:.4f})\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-s3", required=True)
    ap.add_argument("--level", required=True)
    ap.add_argument("--target", type=float, required=True)
    ap.add_argument("--ref-ckpt", default=None, help="reference base (scored exactly) whose pass@k curve to match")
    ap.add_argument("--ref-passk", default=None, help="reference curve as JSON {k: pass@k}")
    ap.add_argument("--match-k", type=int, default=1024)
    ap.add_argument("--tol-abs", type=float, default=0.005, help="absolute tolerance on pass@match_k")
    ap.add_argument("--work", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--n-prompts", type=int, default=1000)
    ap.add_argument("--center-step", type=int, default=25)
    ap.add_argument("--half-window", type=int, default=15)
    ap.add_argument("--max-centers", type=int, default=24)
    ap.add_argument("--scan-below", type=int, default=500, help="scan centers this many steps before the screened crossing")
    ap.add_argument("--scan-above", type=int, default=250)
    ap.add_argument("--scan-step", type=int, default=50)
    ap.add_argument("--max-iters", type=int, default=8)
    ap.add_argument("--max-paths", type=int, default=2500)
    ap.add_argument("--upload", action="store_true", help="write <run-s3>/align-<level>/{best,align.json,COMPLETE}")
    args = ap.parse_args()
    t0 = time.time()
    prompts, gts = ss.load_prompts(args.parquet, args.n_prompts)
    if args.ref_ckpt:
        ref = curve(per_maze_p(logit_cache(args.ref_ckpt, prompts, gts, args.max_paths), 1.0))
    else:
        ref = {int(k): v for k, v in json.loads(args.ref_passk).items()}
    q = ref[args.match_k]
    print(f"reference pass@k: " + " ".join(f"@{k} {v:.4f}" for k, v in ref.items()), flush=True)

    run = os.path.join(args.work, os.path.basename(args.run_s3.rstrip("/")))
    out = os.path.join(run, f"align-{args.level}")
    os.makedirs(out, exist_ok=True)
    et.aws("s3", "sync", "--only-show-errors", args.run_s3, run, "--exclude", "*", "--include", "screen_*.json")
    sc = {}
    for f in glob.glob(os.path.join(run, "screen_*.json")):
        try:
            sc.update(json.load(open(f)))
        except (json.JSONDecodeError, OSError):
            pass
    pts = sorted((int(k.split("-")[1]), v["is_shortest"]["pass@1"]) for k, v in sc.items() if k.startswith("ckpt-"))
    smooth = [(s, statistics.median([x for _, x in pts[max(0, i - 2):i + 3]])) for i, (s, _) in enumerate(pts)]
    cross = next((s for s, x in smooth if x >= args.target), pts[-1][0])
    hw, cs = args.half_window, args.center_step
    log = {"target": args.target, "ref": ref, "match_k": args.match_k, "evals": {}}

    def score(path, key):
        cache = logit_cache(path, prompts, gts, args.max_paths)
        beta = solve_beta(cache, args.target)
        c = curve(per_maze_p(cache, beta))
        log["evals"][key] = {"beta": beta, "passk": c}
        print(f"{key:>26} beta {beta:.4f}: pass@1 {c[1]:.6f} pass@{args.match_k} {c[args.match_k]:.4f} (ref {q:.4f}); "
              f"max |pass@k - ref| {max(abs(c[k] - ref[k]) for k in KS):.4f}", flush=True)
        return beta, c

    cache_c = {}

    def center(c):
        if c not in cache_c:
            inc = []
            for s in range(c - hw, c + hw + 1, 5):
                inc += ["--include", f"ckpt-{s}/*"]
            et.aws("s3", "sync", "--only-show-errors", args.run_s3, run, "--exclude", "*", *inc)
            path = os.path.join(run, f"avg-{c}-w{hw}")
            if not os.path.isfile(os.path.join(path, "model.safetensors")):
                et.sh(sys.executable, os.path.join(HERE, "average_ckpts.py"), "--run", run, "--center", str(c), "--half-window",
                      str(hw), "--stride", "5", "--out", path)
            beta, cv = score(path, f"avg-{c}-w{hw}")
            cache_c[c] = (cv[args.match_k], beta, path)
        return cache_c[c]

    # objective: max over k of |pass@k - ref| at the beta that pins pass@1. Walk centers from the crossing in the improving
    # direction until it stops improving, then golden-section on the interpolation weight towards the better neighbour.
    def h(cv):
        return max(abs(cv[k] - ref[k]) for k in KS)

    def try_center(cc):
        try:
            center(cc)
            return h(log["evals"][f"avg-{cc}-w{hw}"]["passk"])
        except (SystemExit, subprocess.CalledProcessError) as e:   # checkpoints outside the mirrored range
            print(f"center {cc} unavailable: {e}", flush=True)
            return None
    # coarse scan of the training point (pass@k shape at matched pass@1 moves slowly with the step, and in either direction),
    # then the 25-step neighbours of the best scanned center
    c0 = max(hw + 5, int(round(cross / cs)) * cs)
    ss_ = max(cs, int(round(args.scan_step / cs)) * cs)
    hs = {}
    for cc in range(c0 - (args.scan_below // ss_) * ss_, c0 + args.scan_above + 1, ss_):
        if cc - hw >= 5:
            v = try_center(cc)
            if v is not None:
                hs[cc] = v
    c = min(hs, key=hs.get)
    for d in (cs, -cs):
        cc = c + d
        if cc not in hs and cc - hw >= 5:
            v = try_center(cc)
            if v is not None:
                hs[cc] = v
    c = min(hs, key=hs.get)
    nb = [k for k in (c - cs, c + cs) if k in hs]
    cands = [(hs[c], 0.0, cache_c[c][1], cache_c[c][2])]
    if nb:
        n = min(nb, key=hs.get)
        a, b = cache_c[c][2], cache_c[n][2]

        def at(alpha):
            path = os.path.join(out, f"interp-a{alpha:.5f}")
            if not os.path.isfile(os.path.join(path, "model.safetensors")):
                et.sh(sys.executable, os.path.join(HERE, "interpolate_ckpts.py"), "--a", a, "--b", b, "--alpha", f"{alpha:.5f}",
                      "--out", path)
            beta, cv = score(path, f"interp-a{alpha:.5f}")
            cands.append((h(cv), alpha, beta, path))
            return h(cv)
        gr = (math.sqrt(5) - 1) / 2
        lo, hi = 0.0, 1.0
        x1, x2 = hi - gr * (hi - lo), lo + gr * (hi - lo)
        f1, f2 = at(x1), at(x2)
        for _ in range(max(0, args.max_iters - 2)):
            if min(f1, f2) <= args.tol_abs:
                break
            if f1 < f2:
                hi, x2, f2 = x2, x1, f1
                x1 = hi - gr * (hi - lo)
                f1 = at(x1)
            else:
                lo, x1, f1 = x1, x2, f2
                x2 = lo + gr * (hi - lo)
                f2 = at(x2)
    g, alpha, beta, path = min(cands, key=lambda x: x[0])
    best = os.path.join(out, "best")
    write_scaled(path, beta, best)
    final = curve(per_maze_p(logit_cache(best, prompts, gts, args.max_paths), 1.0))   # re-score the written model at beta 1
    log.update(best={"path": path, "alpha": alpha, "beta": beta, "passk": final,
                     "pass1_confirm": final[1], "confirm_rel_vs_target": (final[1] - args.target) / args.target,
                     "max_abs_diff_vs_ref": max(abs(final[k] - ref[k]) for k in KS), "method": "exact-align"},
               seconds=round(time.time() - t0, 1))
    with open(os.path.join(out, "align.json"), "w") as f:
        json.dump(log, f)
    print(f"ALIGNED {os.path.basename(run)} {args.level}: {os.path.basename(path)} beta {beta:.4f} -> pass@1 {final[1]:.6f} "
          f"({100 * (final[1] - args.target) / args.target:+.2f}%), max |pass@k - ref| {log['best']['max_abs_diff_vs_ref']:.4f} "
          f"over k=1..1024 in {log['seconds']}s", flush=True)
    print("   ours: " + " ".join(f"{final[k]:.3f}" for k in (1, 4, 16, 64, 256, 1024)), flush=True)
    print("   ref : " + " ".join(f"{ref[k]:.3f}" for k in (1, 4, 16, 64, 256, 1024)), flush=True)
    if args.upload:
        dst = f"{args.run_s3.rstrip('/')}/align-{args.level}"
        et.aws("s3", "sync", "--only-show-errors", best, f"{dst}/best")
        et.aws("s3", "cp", "--only-show-errors", os.path.join(out, "align.json"), f"{dst}/align.json")
        et.aws("s3", "cp", "-", f"{dst}/COMPLETE", input=b"")
        print(f"UPLOADED {dst}", flush=True)


if __name__ == "__main__":
    main()
