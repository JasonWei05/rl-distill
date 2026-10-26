#!/usr/bin/env bash
# Re-publish best_hf of a finished Qwen on-policy distillation run at its best-VALIDATION saved step <= MAX_STEP (W&B val
# mean@16; earliest on ties; step 0 = the init is not eligible) via republish_best_hf.py, the selection stop_qwen3_opd_at_step.sh
# uses. Needed for runs that finished on their own: with early stopping off the run-file publishes the final step as best_hf
# (best_score null), and min_delta early stopping can keep a "best" that lags the max. The old completion marker is removed
# first and a fresh verified one written last; the run-file's receipt is kept as run_outcome.runfile.json.
#   SPEC=q30bm-to-q4b bash publish_qwen3_opd_best_hf.sh            # MAX_STEP (default: no cap), RUN_TAG (default pgk1-v1)
#   DRY_RUN=1 SPEC=... bash publish_qwen3_opd_best_hf.sh            # print the selection only
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
SPEC="${SPEC:?SPEC required}"; RUN_TAG="${RUN_TAG:-pgk1-v1}"
S3_BASE="${S3_BASE:-s3://scale-ml/genai/rl-distill/qwen3-opd}"
sup=".scale_train_supervisors/q3-opd-${SPEC}-${RUN_TAG}"
args=(--artifact-s3 "${S3_BASE}/${SPEC}-${RUN_TAG}" --ckpt-s3 "${S3_BASE}-full-checkpoints/${SPEC}-${RUN_TAG}"
      --wandb-run "rl-distill/DAPO/q3-opd-${SPEC}-${RUN_TAG}-s42")
[ -n "${MAX_STEP:-}" ] && args+=(--max-step "${MAX_STEP}")
[ -n "${DRY_RUN:-}" ] && args+=(--dry-run)
mkdir -p "${sup}"
(set -a; source .env; set +a; export AWS_PROFILE=ml-worker AWS_REGION=us-west-2
 .venv/bin/python rl-distill-scripts/scale_train/republish_best_hf.py "${args[@]}") 2>&1 \
  | while read -r l; do echo "$(date -u +%FT%TZ) ${SPEC} ${l}" | tee -a "${sup}/stop-at-step.log"; done
exit "${PIPESTATUS[0]}"
