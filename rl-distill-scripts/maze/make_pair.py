#!/usr/bin/env python3
"""Freeze a matched S/L pair at a given starting level from two selection files (select_matched_init.py --out-name).

Takes the L candidate closest to --target (at the final k), then the S candidate closest to that L value. If the gap is
above --tol, interpolates weights between the two adjacent S candidates that bracket the L value (interpolate_ckpts.py),
measures the interpolated model at the same k, and refines once more (secant); if S cannot bracket, the roles swap.
Writes the matched/<pair>_{S0,L0} symlinks, the <pair>_{S0,L0}_base/ckpt-0 dirs train.sh takes, and the manifest entry.

    CUDA_VISIBLE_DEVICES=6 ~/TailRL/.venv/bin/python rl-distill-scripts/maze/make_pair.py --pair p1 --target 0.01 --tol 0.0005 \
        --s-run ~/tailrl_maze_runs/pretrain20k/S_ds0_is4 --l-run ~/tailrl_maze_runs/pretrain20k/L_ds0_is0_rep \
        --select-name select_goal_reached_1pct.json --k 1024
"""

import argparse
import datetime
import json
import os
import re
import subprocess
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import screen_sft_ckpts as ss  # noqa: E402

import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def center_of(name):
    m = re.match(r"avg-(\d+)-w\d+$", name)
    if m:
        return float(m.group(1))
    m = re.match(r"interp-(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)-a([0-9.]+)$", name)
    if m:
        a, b, al = float(m.group(1)), float(m.group(2)), float(m.group(3))
        return a + al * (b - a)
    return None


class Side:
    def __init__(self, label, run, select_name, k, metric, screen):
        self.label, self.run, self.k, self.metric, self.screen = label, run, k, metric, screen
        self.path = os.path.join(run, select_name)
        self.results = json.load(open(self.path))
        self.cands = {}  # name -> {k: metrics}
        for key, r in self.results.items():
            if key.startswith("_"):
                continue
            name, kk = key.rsplit("@k", 1)
            self.cands.setdefault(name, {})[int(kk)] = r

    def p(self, name, k=None):
        ks = self.cands[name]
        k = k if k in ks else max(ks)
        return ks[k][self.metric]["pass@1"], k

    def measure(self, name):
        """pass@1 at the final k, screening the candidate if it has not been measured at that k."""
        if self.k not in self.cands.get(name, {}):
            print(f"[{self.label}] screening {name} at k={self.k}", flush=True)
            r = self.screen(os.path.join(self.run, name), self.k)
            self.cands.setdefault(name, {})[self.k] = r
            self.results[f"{name}@k{self.k}"] = r
            tmp = self.path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self.results, f)
            os.replace(tmp, self.path)
        return self.cands[name][self.k][self.metric]["pass@1"]

    def finals(self):
        return {n: ks[self.k][self.metric]["pass@1"] for n, ks in self.cands.items() if self.k in ks}

    def bracket(self, v):
        """Adjacent candidates (by trajectory position) whose estimated pass@1 bracket v."""
        est = [(center_of(n), n, self.p(n)[0]) for n in self.cands if center_of(n) is not None]
        est.sort()
        for (ca, a, pa), (cb, b, pb) in zip(est, est[1:]):
            lo, hi = min(pa, pb), max(pa, pb)
            if lo <= v <= hi and pa != pb:
                return (a, b) if pa <= pb else (b, a)
        return None

    def interpolate(self, a, b, alpha):
        ca, cb = center_of(a), center_of(b)
        name = f"interp-{ca:g}-{cb:g}-a{alpha:.3f}"
        out = os.path.join(self.run, name)
        if not os.path.isfile(os.path.join(out, "model.safetensors")):
            subprocess.run([sys.executable, os.path.join(HERE, "interpolate_ckpts.py"), "--a", os.path.join(self.run, a),
                            "--b", os.path.join(self.run, b), "--alpha", f"{alpha:.6f}", "--out", out], check=True, stdout=subprocess.DEVNULL)
        return name


def match(fixed, moving, tol, max_iters):
    """Return (moving candidate name, its pass@1) closest to fixed's value, interpolating if the finals are not within tol."""
    v = fixed.value
    finals = moving.finals()
    best = min(finals, key=lambda n: abs(finals[n] - v))
    tried = [(abs(finals[best] - v), best, finals[best])]
    if tried[0][0] <= tol:
        return best, finals[best], tried
    br = moving.bracket(v)
    if br is None:
        print(f"[{moving.label}] no candidates bracket {v:.5f}; nearest final is {best} ({finals[best]:.5f})")
        return best, finals[best], tried
    a, b = br
    pa, pb = moving.measure(a), moving.measure(b)
    print(f"[{moving.label}] bracket {a} ({pa:.5f}) .. {b} ({pb:.5f}) around {v:.5f}", flush=True)
    for _ in range(max_iters):
        if pb == pa:
            break
        alpha = min(1.0, max(0.0, (v - pa) / (pb - pa)))
        name = moving.interpolate(a, b, alpha)
        p = moving.measure(name)
        tried.append((abs(p - v), name, p))
        print(f"[{moving.label}] {name}: pass@1 {p:.5f} (target {v:.5f}, |d| {abs(p - v):.5f})", flush=True)
        if abs(p - v) <= tol:
            break
        if p < v:
            a, pa = name, p
        else:
            b, pb = name, p
    tried.sort()
    return tried[0][1], tried[0][2], tried


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True)
    ap.add_argument("--target", type=float, required=True)
    ap.add_argument("--tol", type=float, required=True, help="absolute pass@1 tolerance (0.0005 = 0.05 pt)")
    ap.add_argument("--s-run", required=True)
    ap.add_argument("--l-run", required=True)
    ap.add_argument("--select-name", required=True, help="selection file name under each run")
    ap.add_argument("--k", type=int, required=True, help="the final k the candidates were measured at")
    ap.add_argument("--metric", default="goal_reached")
    ap.add_argument("--max-iters", type=int, default=2)
    ap.add_argument("--matched-dir", default=os.path.expanduser("~/tailrl_maze_runs/matched"))
    ap.add_argument("--tailrl-root", default=ss.DEFAULT_TAILRL)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    judge = ss.load_judge(args.tailrl_root)
    parquet = os.path.join(args.tailrl_root, "experiments/text_maze/data/eval1000/test.parquet")
    prompts, gts = ss.load_prompts(parquet, 1000)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def screen(path, k):
        torch.manual_seed(args.seed)
        return ss.screen_one(path, prompts, gts, judge, SimpleNamespace(k=k, batch_prompts=128, max_new_tokens=180), device)

    S = Side("S", args.s_run.rstrip("/"), args.select_name, args.k, args.metric, screen)
    L = Side("L", args.l_run.rstrip("/"), args.select_name, args.k, args.metric, screen)
    lf = L.finals()
    if not lf or not S.finals():
        raise SystemExit("no finalists at the requested k on one side; run the selection first")
    l_name = min(lf, key=lambda n: abs(lf[n] - args.target))
    L.value = lf[l_name]
    print(f"L anchor {l_name}: pass@1 {L.value:.5f} (target {args.target})")
    s_name, s_val, tried = match(L, S, args.tol, args.max_iters)
    if abs(s_val - L.value) > args.tol:  # S could not reach L's value: fix S instead and move L
        print("S side could not match within tol; interpolating on the L side instead", flush=True)
        S.value = s_val
        l_name, l_val, tried_l = match(S, L, args.tol, args.max_iters)
        L.value = l_val
    d = abs(s_val - L.value)
    status = "MATCHED" if d <= args.tol else "BEST EFFORT (outside tol)"
    s_m, l_m = S.cands[s_name][args.k], L.cands[l_name][args.k]
    pk = lambda m: {k: m[args.metric][k] for k in m[args.metric] if k.startswith("pass@") or k == "mixed16"}
    print(f"\n{status}: {args.pair}  S {s_name} pass@1 {s_val:.5f}  L {l_name} pass@1 {L.value:.5f}  |d| {d:.5f} (tol {args.tol})")
    print(f"  S goal {pk(s_m)}\n  L goal {pk(l_m)}")
    print(f"  shortest pass@1: S {s_m['is_shortest']['pass@1']:.5f}  L {l_m['is_shortest']['pass@1']:.5f}")

    md = args.matched_dir
    os.makedirs(md, exist_ok=True)
    for side, run, name in (("S0", S.run, s_name), ("L0", L.run, l_name)):
        link = os.path.join(md, f"{args.pair}_{side}")
        if os.path.islink(link):
            os.remove(link)
        os.symlink(os.path.join(run, name), link)
        base = os.path.join(md, f"{args.pair}_{side}_base")
        os.makedirs(base, exist_ok=True)
        ck = os.path.join(base, "ckpt-0")
        if os.path.islink(ck):
            os.remove(ck)
        os.symlink(f"../{args.pair}_{side}", ck)
    mpath = os.path.join(md, "manifest.json")
    manifest = json.load(open(mpath)) if os.path.exists(mpath) else {"pairs": {}}
    manifest.setdefault("pairs", {})[args.pair] = {
        "target": args.target, "tol": args.tol, "k": args.k, "status": status, "d_pass1": d,
        "created": datetime.date.today().isoformat(),
        "S0": {"run": os.path.basename(S.run), "candidate": s_name, "path": os.path.join(S.run, s_name), f"metrics_k{args.k}": s_m},
        "L0": {"run": os.path.basename(L.run), "candidate": l_name, "path": os.path.join(L.run, l_name), f"metrics_k{args.k}": l_m},
    }
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"wrote {md}/{args.pair}_{{S0,L0}} (+_base) and manifest entry; {status}")
    if d > args.tol:
        sys.exit(2)


if __name__ == "__main__":
    main()
