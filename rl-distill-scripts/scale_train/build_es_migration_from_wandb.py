#!/usr/bin/env python3
"""Build EARLY_STOPPING_LEGACY_STATE_B64 for resuming a run that trained with early stopping OFF.

Replays the run's logged validation metric (W&B, steps that are multiples of --every, incl. the initial step-0 validation)
through dapo.validation_early_stopping with the policy the resumed trainer will use, and prints the identity-bound
migration payload (protocol legacy_early_stopping_migration_v1) for the restored --checkpoint-step. Use --every equal to the
permanent-checkpoint cadence so the all-time best is always a step with an HF export.
  python build_es_migration_from_wandb.py --run-id q3-opd-...-s42 --model Qwen/Qwen3-1.7B-Base --difficulty medium \
      --checkpoint-step 250 --patience 10 --min-delta 0.005 --every 10
"""
import argparse, base64, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dapo.validation_early_stopping import ValidationEarlyStopping  # noqa: E402

METRIC = "val-core/math/acc/mean@16"

ap = argparse.ArgumentParser()
ap.add_argument("--run-id", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--difficulty", required=True)
ap.add_argument("--checkpoint-step", type=int, required=True)
ap.add_argument("--patience", type=int, required=True)
ap.add_argument("--min-delta", type=float, default=0.0)
ap.add_argument("--every", type=int, default=10)
a = ap.parse_args()

import wandb  # noqa: E402

h = wandb.Api(timeout=120).run(f"rl-distill/DAPO/{a.run_id}").history(keys=[METRIC], samples=10000, pandas=True)
vals = sorted((int(s), float(v)) for s, v in zip(h["_step"], h[METRIC]) if s % a.every == 0 and s <= a.checkpoint_step)
if not vals or vals[-1][0] != a.checkpoint_step:
    sys.exit(f"FATAL: no validation logged at checkpoint step {a.checkpoint_step} (last {vals[-1:] })")
es = ValidationEarlyStopping(metric=METRIC, patience=a.patience, mode="max", min_delta=a.min_delta,
                             include_initial_validation=True, miss_tolerance=0.0)
for step, score in vals:
    es.observe({METRIC: score}, step=step)
payload = {"protocol": "legacy_early_stopping_migration_v1", "checkpoint_step": a.checkpoint_step, "model": a.model,
           "difficulty": a.difficulty, "early_stopping_state": es.state_dict()}
print(f"observations={len(vals)} best={es.best_score:.4f}@{es.best_step} misses={es.non_improving_rounds} "
      f"triggered={es.last_observed_triggered}", file=sys.stderr)
print(base64.b64encode(json.dumps(payload, separators=(",", ":")).encode()).decode())
