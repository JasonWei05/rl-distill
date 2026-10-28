#!/usr/bin/env python3
"""Screen maze SFT checkpoints for the synthetic RL-vs-distillation study.

For every checkpoint: sample K responses per held-out maze at T = 1.0, score each with TailRL's
maze judge, and report goal_reached and is_shortest pass@1, unbiased pass@k, the fraction of mazes
a 16-rollout group would find "mixed" (the RL signal), and the mean response length. Standalone:
no Ray, one GPU, the TailRL venv.

    ~/TailRL/.venv/bin/python rl-distill-scripts/maze/screen_sft_ckpts.py \
        --steps 4000 4200 4400 --k 64 --n-prompts 1000 --out ~/tailrl_maze_runs/screen_sft_ckpts.json

Checkpoints are <ckpt-dir>/ckpt-<step> (TailRL's layout) or explicit --paths. The judge is loaded
from the TailRL checkout by path, like TailRL's own smoke test does.
"""

import argparse
import ast
import importlib.util
import json
import os
import time

import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HOME = os.path.expanduser("~")
DEFAULT_TAILRL = os.path.join(HOME, "TailRL")
DEFAULT_CKPT_DIR = os.environ.get("HF_CACHE_DIR", os.path.join(HOME, ".cache/tailrl_maze_sft_ckpts"))


def load_judge(tailrl_root):
    path = os.path.join(tailrl_root, "experiments/text_maze/verl/utils/reward_score/maze.py")
    spec = importlib.util.spec_from_file_location("tailrl_maze_judge", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def field(v):
    return ast.literal_eval(v) if isinstance(v, str) else v


def load_prompts(parquet, n):
    rows = pq.read_table(parquet).to_pylist()[:n]
    prompts, gts = [], []
    for r in rows:
        p = field(r["prompt"])
        prompts.append(p[0]["content"] if isinstance(p, list) else p)
        gts.append(field(r["reward_model"])["ground_truth"])
    return prompts, gts


def pass_at_k(n, c, k):
    """Unbiased pass@k for one prompt with n samples and c successes (Chen et al. 2021)."""
    if n - c < k:
        return 1.0
    return 1.0 - float(np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def mixed_group_fraction(p_hat, group=16):
    """P(a group of `group` rollouts has both a success and a failure), averaged over mazes."""
    p = np.asarray(p_hat, dtype=np.float64)
    return float(np.mean(1.0 - p**group - (1.0 - p) ** group))


def summarize(counts, k_samples, ks):
    p_hat = counts / k_samples
    out = {"pass@1": float(p_hat.mean()), "mixed16": mixed_group_fraction(p_hat)}
    for k in ks:
        if k <= k_samples:
            out[f"pass@{k}"] = float(np.mean([pass_at_k(k_samples, int(c), k) for c in counts]))
    out["frac_solved_ge1"] = float(np.mean(counts > 0))
    return out


def screen_one(ckpt, prompts, gts, judge, args, device):
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(ckpt)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = "<pad>"
    pad_id = tok.convert_tokens_to_ids(tok.pad_token)
    model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.bfloat16).to(device).eval()
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)

    n = len(prompts)
    goal = np.zeros(n)
    short = np.zeros(n)
    lengths = []
    bp = max(1, min(args.batch_prompts, 8192 // args.k))  # keep <= 8192 sequences per generate call
    for i0 in range(0, n, bp):
        ids = enc.input_ids[i0 : i0 + bp].to(device).repeat_interleave(args.k, 0)
        am = enc.attention_mask[i0 : i0 + bp].to(device).repeat_interleave(args.k, 0)
        with torch.no_grad():
            out = model.generate(
                input_ids=ids,
                attention_mask=am,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=1.0,
                top_p=1.0,
                pad_token_id=pad_id,
            )
        gen = out[:, ids.shape[1] :]
        lengths.extend((gen != pad_id).sum(-1).tolist())
        for j, text in enumerate(tok.batch_decode(gen, skip_special_tokens=False)):
            m = i0 + j // args.k
            r = judge.judge_maze(solution_str=text, ground_truth=gts[m])
            goal[m] += 1.0 if r.get("goal_reached", 0.0) > 0 else 0.0
            short[m] += 1.0 if r.get("is_shortest", 0.0) > 0 else 0.0

    ks = [1, 4, 16, 64, 256, 1024]
    res = {
        "ckpt": ckpt,
        "n_prompts": n,
        "k": args.k,
        "goal_reached": summarize(goal, args.k, ks),
        "is_shortest": summarize(short, args.k, ks),
        "mean_response_tokens": float(np.mean(lengths)),
        "seconds": round(time.time() - t0, 1),
        # per-maze success counts (out of k) for histogram comparisons between candidates
        "per_maze": {"goal_reached": goal.astype(int).tolist(), "is_shortest": short.astype(int).tolist()},
    }
    del model
    torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tailrl-root", default=DEFAULT_TAILRL)
    ap.add_argument("--ckpt-dir", default=DEFAULT_CKPT_DIR)
    ap.add_argument("--steps", type=int, nargs="*", default=[])
    ap.add_argument("--paths", nargs="*", default=[], help="explicit checkpoint directories")
    ap.add_argument("--parquet", default=None, help="default: TailRL's committed data/eval1000/test.parquet")
    ap.add_argument("--n-prompts", type=int, default=1000)
    ap.add_argument("--k", type=int, default=64, help="samples per maze")
    ap.add_argument("--batch-prompts", type=int, default=128, help="prompts per generate call (x k sequences)")
    ap.add_argument("--max-new-tokens", type=int, default=180)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HOME, "tailrl_maze_runs/screen_sft_ckpts.json"))
    args = ap.parse_args()

    parquet = args.parquet or os.path.join(args.tailrl_root, "experiments/text_maze/data/eval1000/test.parquet")
    ckpts = [os.path.join(args.ckpt_dir, f"ckpt-{s}") for s in args.steps] + list(args.paths)
    if not ckpts:
        ap.error("give --steps and/or --paths")
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    judge = load_judge(args.tailrl_root)
    prompts, gts = load_prompts(parquet, args.n_prompts)

    results = {}
    if os.path.exists(args.out):
        with open(args.out) as f:
            results = json.load(f)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    header = f"{'ckpt':>10} {'goal p@1':>9} {'short p@1':>9} {'goal p@16':>9} {'short p@16':>10} {'goal mix16':>10} {'short mix16':>11} {'len':>6} {'sec':>6}"
    print(header)
    def label(ck):
        # RL exports all end in actor/huggingface: key those by their global_step dir instead of the shared basename
        parts = os.path.normpath(ck).split(os.sep)
        return next((p for p in reversed(parts) if p.startswith("global_step_")), os.path.basename(ck))

    for ck in ckpts:
        r = screen_one(ck, prompts, gts, judge, args, device)
        g, s = r["goal_reached"], r["is_shortest"]
        print(
            f"{label(ck):>16} {g['pass@1']:9.4f} {s['pass@1']:9.4f} {g.get('pass@16', 0):9.4f} "
            f"{s.get('pass@16', 0):10.4f} {g['mixed16']:10.4f} {s['mixed16']:11.4f} {r['mean_response_tokens']:6.1f} {r['seconds']:6.1f}",
            flush=True,
        )
        results[label(ck)] = r
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
