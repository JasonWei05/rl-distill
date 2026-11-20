"""Tune a finished SFT run to the config's pass@1 levels (tune.levels): one RL base per level.

Pass@1 is the mean judge score of tune.k samples (temperature 1, fp16) on each of the 1000 dev prompts. For each target:
  1. candidates are checkpoint averages: the mean of the checkpoints within +-tune.window steps of a center step (any
     checkpoint step; near the ends the window is cut);
  2. walk the centers from the screens' first crossing of the target, sft.screen_every steps at a time, until two adjacent
     centers bracket it, then bisect the bracket down to sft.save_every steps;
  3. interpolate the weights of that pair, (1 - a) lo + a hi, with secant steps on log pass@1 (Illinois), until within
     tune.tol (relative) of the target or tune.max_iters interpolations;
  4. confirm the closest candidate with tune.confirm_k fresh samples per prompt.
A target above every candidate (the run never gets there) is recorded as unreachable. The bases are model directories
tune/<level>/, every evaluation is in tune.json; TUNED marks the end.

    python tune.py --run /tmp/synth/sft/maze33/2XS_s0 --data /tmp/synth/data/maze33
"""

import argparse
import glob
import json
import math
import os

import torch
from safetensors.torch import load_file

from data import Dataset, same_build
from files import read_json, write_json
from model import Decoder, Qwen2, evaluate, half, save_model


class Tuner:
    def __init__(self, run, data):
        self.run, self.data = run, Dataset(data)
        self.meta = read_json(os.path.join(run, "run.json"))
        assert same_build(self.meta["config"], self.data.cfg), "the run was trained on another dataset config"
        self.cfg, self.t, self.task = self.data.cfg, self.data.cfg["tune"], self.data.task
        self.model = Qwen2(self.meta["hf"])
        self.decoder = Decoder(self.model, **self.cfg["decode"])
        self.dev = self.data.prompts("dev").cuda()
        self.parsed = self.task.parse(self.dev)
        self.steps = sorted(int(os.path.basename(p).split(".")[0]) for p in glob.glob(os.path.join(run, "ckpt", "*.safetensors")))
        with open(os.path.join(run, "screen.jsonl")) as f:
            self.screens = [json.loads(line) for line in f]
        self.path = os.path.join(run, "tune.json")
        self.log = read_json(self.path) if os.path.exists(self.path) else {}
        self.cache, self.seed = {}, 1000

    def pass1(self, w, k):
        self.seed += 1
        torch.manual_seed(self.seed)
        return evaluate(self.task, self.decoder, half(w), self.dev, self.parsed, k)["score"]

    def average(self, center):
        """The mean of the checkpoints within +-window steps of center."""
        steps = [s for s in self.steps if abs(s - center) <= self.t["window"]]
        ws = [load_file(os.path.join(self.run, "ckpt", f"{s}.safetensors"), device=torch.cuda.current_device()) for s in steps]
        return {k: sum(w[k] for w in ws) / len(ws) for k in ws[0]}

    def center_score(self, center, evals):
        if center not in self.cache:
            self.cache[center] = self.pass1(self.average(center), self.t["k"])
            evals.append({"center": center, "pass1": self.cache[center]})
        return self.cache[center]

    def bracket(self, target, evals):
        """Adjacent centers (lo, hi), save_every apart, with pass1(lo) < target <= pass1(hi); None if no center reaches it."""
        stride, coarse = self.cfg["sft"]["save_every"], self.cfg["sft"]["screen_every"]
        first, last = self.steps[0], self.steps[-1]
        crossed = [r["step"] for r in self.screens if r["score"] >= target]
        c = crossed[0] if crossed else last
        if self.center_score(c, evals) >= target:
            while c > first and self.center_score(max(c - coarse, first), evals) >= target:
                c = max(c - coarse, first)
            lo, hi = max(c - coarse, first), c
        else:
            while c < last and self.center_score(min(c + coarse, last), evals) < target:
                c = min(c + coarse, last)
            lo, hi = c, min(c + coarse, last)
        if self.center_score(hi, evals) < target:
            return None
        assert self.center_score(lo, evals) < target, f"pass@1 >= {target} already at the first candidate (center {lo})"
        while hi - lo > stride:
            mid = lo + (hi - lo) // 2 // stride * stride
            if self.center_score(mid, evals) >= target:
                hi = mid
            else:
                lo = mid
        return lo, hi

    def tune(self, level, target):
        evals = []
        pair = self.bracket(target, evals)
        if pair is None:
            return {"target": target, "reachable": False, "evals": evals}
        lo, hi = pair
        A, B = self.average(lo), self.average(hi)
        (a0, f0), (a1, f1) = (0.0, self.cache[lo]), (1.0, self.cache[hi])
        best = min(((0.0, f0), (1.0, f1)), key=lambda x: abs(x[1] / target - 1))
        side = 0
        for _ in range(self.t["max_iters"]):
            if abs(best[1] / target - 1) <= self.t["tol"]:
                break
            g = lambda f: math.log(max(f, 1e-6))
            a = a0 + (g(target) - g(f0)) * (a1 - a0) / (g(f1) - g(f0))
            f = self.pass1({k: (1 - a) * A[k] + a * B[k] for k in A}, self.t["k"])
            evals.append({"lo": lo, "hi": hi, "alpha": a, "pass1": f})
            best = min((best, (a, f)), key=lambda x: abs(x[1] / target - 1))
            if f < target:  # Illinois: halve the stale end's weight when the same end is kept twice
                (a0, f0), f1 = (a, f), (f1 * target) ** 0.5 if side == -1 else f1
                side = -1
            else:
                (a1, f1), f0 = (a, f), (f0 * target) ** 0.5 if side == 1 else f0
                side = 1
        a = best[0]
        w = {k: (1 - a) * A[k] + a * B[k] for k in A}
        confirm = self.pass1(w, self.t["confirm_k"])
        save_model(os.path.join(self.run, "tune", level), w, self.meta["hf"],
                   {"config": self.cfg, "size": self.meta["size"], "seed": self.meta["seed"], "level": level,
                    "lo": lo, "hi": hi, "alpha": a, "pass1": confirm})
        return {"target": target, "reachable": True, "lo": lo, "hi": hi, "alpha": a, "search_pass1": best[1],
                "pass1": confirm, "evals": evals}

    def run_all(self):
        for level, target in self.t["levels"].items():
            if level in self.log:
                continue
            self.log[level] = self.tune(level, target)
            write_json(self.path, self.log)
            r = self.log[level]
            print(f"TUNED {self.run} {level}: " + (f"pass1 {r['pass1']:.4f} (centers {r['lo']}-{r['hi']}, alpha {r['alpha']:.3f})"
                                                  if r["reachable"] else "unreachable"), flush=True)
        open(os.path.join(self.run, "TUNED"), "w").close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="SFT run dir (sft.py --out)")
    ap.add_argument("--data", required=True, help="its dataset dir")
    a = ap.parse_args()
    Tuner(a.run, a.data).run_all()
