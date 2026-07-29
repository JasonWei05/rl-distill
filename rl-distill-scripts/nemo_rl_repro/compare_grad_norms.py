#!/usr/bin/env python3
"""Compare NeMo-RL vs verl grad norms for the gemma-4 E2B DeepScaleR 8k run.

Both frameworks log the SAME quantity: pre-clip global L2 norm over all trainable
params, computed per optimizer step after gradient accumulation (verl:
fsdp_workers/dp_actor; NeMo-RL: nemo_automodel scale_grads_and_clip_grad_norm ->
torch-style clip that returns the pre-clip total norm). Two optimizer steps per
rollout step in both (1024 responses / 512 global batch); verl logs the mean of the
two, NeMo-RL the last — irrelevant at the magnitude-class level being tested.

verl reference (recovery run recbw9dcxso, 285 steps) established the bimodal law:
  steps with zero at-cap (truncated) responses -> grad_norm 1.3-6.8
  steps with >=1 truncated response            -> grad_norm 83-3108 (83/83 steps)
Mechanism: GRPO ddof=1 std-normalization pins advantages to +/-3.75 on 15-vs-1
reward splits; the overlong penalty (-1) converts silent all-wrong groups
(std=0 -> adv 0 -> no gradient) into dominant gradient sources.

Prediction if verl's implementation is correct: NeMo-RL shows the same two regimes,
correlated with train/truncation_rate > 0.

Usage:
  python compare_grad_norms.py                       # auto-find newest nemorl run
  python compare_grad_norms.py --nemorl-run <id>     # explicit run id
"""

import argparse
import os
import re
import sys

import wandb

ENTITY = "rl-distill"
PROJECT = "DAPO"
NEMORL_NAME_RE = re.compile(r"^nemorl-dapo-gemma4-e2b-pt-DeepScaleR-4of4strict-seed42-8k")


def find_newest_nemorl_run(api):
    runs = api.runs(f"{ENTITY}/{PROJECT}", order="-created_at")
    for r in runs:
        if NEMORL_NAME_RE.match(r.name or ""):
            return r
    return None


def pull_history(run, keys_re):
    """Return {step: {key: value}} for history keys matching keys_re."""
    rows = {}
    for row in run.scan_history():
        step = row.get("step")
        if step is None:
            step = row.get("_step")
        if step is None:
            continue
        picked = {k: v for k, v in row.items() if keys_re.search(k) and v is not None}
        if picked:
            rows.setdefault(int(step), {}).update(picked)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nemorl-run", default=None, help="wandb run id (default: newest by name)")
    args = ap.parse_args()

    api = wandb.Api()

    if args.nemorl_run:
        nemorl = api.run(f"{ENTITY}/{PROJECT}/{args.nemorl_run}")
    else:
        nemorl = find_newest_nemorl_run(api)
        if nemorl is None:
            sys.exit("no nemorl run found in wandb yet")
    print(f"NeMo-RL run: {nemorl.name} ({nemorl.id}) state={nemorl.state}")

    metric_re = re.compile(r"grad_norm|truncation_rate|mean_gen_tokens_per_sample|loss$|reward")
    nem = pull_history(nemorl, metric_re)
    if not nem:
        sys.exit("nemorl run has no matching history rows yet")

    def fmt(v, w):
        return f"{v:>{w}.4g}" if isinstance(v, (int, float)) else " " * (w - 1) + "-"

    def pick(row, substr):
        # deterministic key choice: prefer the *_mean variant, else first sorted match
        keys = sorted(k for k in row if substr in k)
        for k in keys:
            if k.endswith("_mean") or k.endswith(substr):
                return row[k]
        return row[keys[0]] if keys else None

    print(f"\n{'step':>4}  {'grad_norm':>12}  {'trunc_rate':>10}  {'gen_len':>8}  {'reward':>8}")
    for step in sorted(nem):
        row = nem[step]
        gn = pick(row, "grad_norm")
        tr = pick(row, "truncation_rate")
        gl = pick(row, "mean_gen_tokens")
        rw = pick(row, "reward")
        print(f"{step:>4}  {fmt(gn, 12)}  {fmt(tr, 10)}  {fmt(gl, 8)}  {fmt(rw, 8)}")

    # Verdict vs the verl bimodal law
    pairs = []
    for step, row in sorted(nem.items()):
        gn = pick(row, "grad_norm")
        tr = pick(row, "truncation_rate")
        if isinstance(gn, (int, float)) and isinstance(tr, (int, float)):
            pairs.append((step, gn, tr))
    if pairs:
        clean = [gn for _, gn, tr in pairs if tr == 0]
        trunc = [gn for _, gn, tr in pairs if tr > 0]
        print("\n--- vs verl reference (recbw9dcxso): clean steps 1.3-6.8, truncated steps 83-3108 ---")
        if clean:
            print(f"NeMo-RL clean steps     (n={len(clean)}): grad_norm {min(clean):.3g} .. {max(clean):.3g}")
        if trunc:
            print(f"NeMo-RL truncated steps (n={len(trunc)}): grad_norm {min(trunc):.3g} .. {max(trunc):.3g}")
        if clean and trunc and min(trunc) > max(clean):
            print("VERDICT: bimodality REPRODUCED in NeMo-RL — behavior is framework-independent")
        elif trunc and clean:
            print("VERDICT: regimes overlap — differs from verl; investigate")
        else:
            print("VERDICT: only one regime observed so far — need more steps")


if __name__ == "__main__":
    main()
