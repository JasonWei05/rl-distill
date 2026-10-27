#!/usr/bin/env python3
"""Backfill one training step's metrics into a W&B run from the trainer's pod log.

verl prints every step as ``step:N - key:value - key:value ...`` (the same dict it hands to wandb.log). When a job is
cancelled right after step N, the W&B uploader can die before that step reaches the server; this re-logs it by resuming
the run (resume="must") at the same step. No-op if the run already has step N.

    python wandb_backfill_step_from_pod_log.py --run-id q3-opd-q4bm-to-q1p7b-pgk1-v1-s42 --step 120 --pod-log <log> [<log> ...]
"""

from __future__ import annotations

import argparse
import re

STEP_RE = re.compile(r" step:(\d+) - (.*)$")
ITEM_RE = re.compile(r"([^\s:][^:]*?):(np\.float64\(([^)]*)\)|[-+0-9.eE]+|nan|inf|-inf)(?: - |$)")


def parse_step(lines: list[str], step: int) -> dict[str, float]:
    for line in lines:
        m = STEP_RE.search(line.rstrip("\n"))
        if m and int(m.group(1)) == step:
            out = {}
            for item in ITEM_RE.finditer(m.group(2)):
                key, raw = item.group(1).strip(), item.group(3) if item.group(3) is not None else item.group(2)
                try:
                    out[key] = float(raw)
                except ValueError:
                    continue
            return out
    raise SystemExit(f"step {step} not found in the pod logs")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--pod-log", nargs="+", required=True)
    parser.add_argument("--entity", default="rl-distill")
    parser.add_argument("--project", default="DAPO")
    args = parser.parse_args()
    import wandb

    run_path = f"{args.entity}/{args.project}/{args.run_id}"
    history = wandb.Api(timeout=60).run(run_path).history(keys=["actor/distillation/k1"], samples=100000, pandas=False)
    logged = {row["_step"] for row in history}
    if args.step in logged:
        print(f"WANDB_STEP_PRESENT {run_path} step={args.step}")
        return 0
    lines = []
    for path in args.pod_log:
        with open(path, errors="replace") as handle:
            lines.extend(handle)
    metrics = parse_step(lines, args.step)
    run = wandb.init(entity=args.entity, project=args.project, id=args.run_id, resume="must")
    run.log(metrics, step=args.step)
    run.finish()
    print(f"WANDB_STEP_BACKFILLED {run_path} step={args.step} metrics={len(metrics)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
