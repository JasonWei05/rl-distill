#!/usr/bin/env bash
# Stop a running Qwen on-policy distillation run (launched with more steps) at STOP_STEP: wait until the pod log has the
# step-STOP_STEP validation and S3 has the complete permanent global_step_STOP_STEP checkpoint, then stop the supervisor +
# janitor, cancel the ScaleTrain job (by ID), backfill step STOP_STEP into W&B if its upload was cut off, and publish best_hf =
# the HF export of the best-validation saved step <= STOP_STEP (W&B val mean@16; earliest on ties; step 0 = the init is not
# eligible) plus a run_outcome.json via republish_best_hf.py (stale marker removed first, verified marker written last).
#   SPEC=q30bm-to-q1p7b STOP_STEP=120 bash stop_qwen3_opd_at_step.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
SPEC="${SPEC:?SPEC required}"; STOP_STEP="${STOP_STEP:-120}"; RUN_TAG="${RUN_TAG:-pgk1-v1}"
S3_BASE="${S3_BASE:-s3://scale-ml/genai/rl-distill/qwen3-opd}"
CKPT="${S3_BASE}-full-checkpoints/${SPEC}-${RUN_TAG}"; ART="${S3_BASE}/${SPEC}-${RUN_TAG}"
name=q3-opd-${SPEC}; sup=".scale_train_supervisors/${name}-${RUN_TAG}"
export AWS_PROFILE=ml-worker AWS_REGION=us-west-2
log() { echo "$(date -u +%FT%TZ) ${SPEC} $*" | tee -a "${sup}/stop-at-step.log"; }
log "waiting for step ${STOP_STEP} validation + checkpoint"
until cat "${sup}"/pod-logs/*.log 2>/dev/null | grep -a " step:${STOP_STEP} - " | grep -q "val-core/math/acc/mean@16" \
  && aws s3 ls "${CKPT}/global_step_${STOP_STEP}/_REMOTE_COMPLETE.json" >/dev/null 2>&1; do sleep 30; done
# W&B uploads asynchronously: give the run up to 15 min to put step STOP_STEP on the server before the job is cancelled
# (cancelling right after the log line lost step 120 for two runs); whatever is still missing is backfilled below.
WANDB_RUN_ID="q3-opd-${SPEC}-${RUN_TAG}-s42"
wandb_has_step() {
  (set -a; source .env; set +a; .venv/bin/python -c "
import sys, wandb
rows = wandb.Api(timeout=60).run('rl-distill/DAPO/${WANDB_RUN_ID}').history(keys=['actor/distillation/k1'], samples=100000, pandas=False)
sys.exit(0 if any(r['_step'] == ${STOP_STEP} for r in rows) else 1)" >/dev/null 2>&1)
}
for _ in $(seq 1 30); do wandb_has_step && break; sleep 30; done
job="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["job_id"])' "${sup}/state.json")"
[ -n "${job}" ] || { log "FATAL: no job_id in ${sup}/state.json"; exit 2; }
touch "${sup}/STOP"
tmux kill-session -t "=rl-${name}" 2>/dev/null || true
tmux kill-session -t "=ckpt-janitor-${name}" 2>/dev/null || true
env -u AWS_PROFILE scale-train cancel job "${job}" >/dev/null 2>&1 || true
log "STOPPED at step ${STOP_STEP}: supervisor + janitor killed, job ${job} cancel requested ($(env -u AWS_PROFILE scale-train get job "${job}" 2>&1 | grep -o 'status: [A-Z_]*'))"
# backfill BEFORE selecting: the selection reads validation from W&B
(set -a; source .env; set +a; .venv/bin/python rl-distill-scripts/scale_train/wandb_backfill_step_from_pod_log.py \
  --run-id "${WANDB_RUN_ID}" --step "${STOP_STEP}" --pod-log "${sup}"/pod-logs/*.log) 2>&1 | grep -E "WANDB_STEP_" | while read -r l; do log "${l}"; done
(set -a; source .env; set +a; .venv/bin/python rl-distill-scripts/scale_train/republish_best_hf.py \
  --artifact-s3 "${ART}" --ckpt-s3 "${CKPT}" --wandb-run "rl-distill/DAPO/${WANDB_RUN_ID}" --max-step "${STOP_STEP}" \
  --stop-reason "manual_stop_at_${STOP_STEP}" --final-step "${STOP_STEP}") 2>&1 | while read -r l; do log "${l}"; done
exit "${PIPESTATUS[0]}"
