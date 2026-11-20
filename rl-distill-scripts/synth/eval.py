"""pass@k of a model on a dataset split: n samples per prompt (fp16, temperature 1, the dataset's decode settings), the
unbiased estimator 1 - C(n - c, k) / C(n, k) of each prompt's c correct samples, averaged over the split and over each
of its groups (meta.json; chess test: the benchmark's bins B1-B5), with the judge's other outcomes as sample means.

    python eval.py --model <HF model dir> --data /tmp/synth/data/chess --split test --n 64 --out <dir>/EVAL.json
    python eval.py --run <sft run dir> --step 384 ...     (a run's checkpoint: run.json's config, ckpt/<step>.safetensors)
"""

import argparse
import math
import os
import time

import torch
from safetensors.torch import load_file

from data import Dataset
from files import read_json, write_json
from model import Decoder, Qwen2, half, judged, load_model


def pass_at(c, n, k):
    """The unbiased pass@k of a prompt with c correct samples out of n (Chen et al. 2021)."""
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def summary(out, n, ks, rows):
    """pass@k and outcome means over the prompts in `rows` (a slice)."""
    c = out["score"].view(-1, n)[rows].sum(1).int().tolist()
    return {"prompts": len(c), **{f"pass@{k}": sum(pass_at(x, n, k) for x in c) / len(c) for k in ks},
            **{k: v.view(-1, n)[rows].mean().item() for k, v in out.items() if k != "score"}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="HF model directory")
    ap.add_argument("--run", help="SFT run directory (with --step)")
    ap.add_argument("--step", type=int)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--k", type=int, nargs="+", default=[1, 4, 8, 16])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="the result JSON")
    a = ap.parse_args()
    assert bool(a.model) != bool(a.run) and (a.run is None) == (a.step is None), "--model, or --run and --step"
    assert max(a.k) <= a.n, (a.k, a.n)
    if a.model:
        hf, _, w = load_model(a.model)
    else:
        hf = read_json(os.path.join(a.run, "run.json"))["hf"]
        w = load_file(os.path.join(a.run, "ckpt", f"{a.step}.safetensors"), device=torch.cuda.current_device())
    data = Dataset(a.data)
    prompts = data.prompts(a.split).cuda()
    torch.manual_seed(a.seed)
    t0 = time.time()
    out = judged(data.task, Decoder(Qwen2(hf), **data.cfg["decode"]), half(w), prompts, data.task.parse(prompts), a.n)
    groups = data.meta["splits"][a.split].get("groups", {})
    res = {"model": a.model or f"{a.run}@{a.step}", "split": a.split, "n": a.n, "seed": a.seed, "seconds": round(time.time() - t0),
           "all": summary(out, a.n, a.k, slice(None)),
           "groups": {g: summary(out, a.n, a.k, slice(i, i + m)) for g, (i, m) in groups.items()},
           "correct": out["score"].view(-1, a.n).sum(1).int().tolist()}
    write_json(a.out, res)
    print(res["all"], flush=True)


if __name__ == "__main__":
    main()
