#!/usr/bin/env bash
# Follow-up launcher for the FLOP-matched OPD sweep (4B medium RL step-k teachers -> 1.7B base, RUN_TAG pgk1-base-v1):
# waits until NEED (default 2) of the WATCH runs have COMPLETED (supervisor state COMPLETED_RECEIPTS), then starts the
# NEXT runs with the same settings (250 steps, val every 5, ckpt every 10, no early stopping, borrowing false, priority high).
#   tmux new -d -s q3-opd-sweep-followup "bash rl-distill-scripts/scale_train/run_qwen3_opd_sweep_followup.sh"
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
WATCH="${WATCH:-20 50 100 150}"; NEXT="${NEXT:-200 250}"; NEED="${NEED:-2}"; RUN_TAG="${RUN_TAG:-pgk1-base-v1}"
LOG=".scale_train_supervisors/q3-opd-sweep-followup-${RUN_TAG}.log"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "${LOG}"; }
log "waiting for ${NEED} of teacher steps {${WATCH}} to complete before launching {${NEXT}}"
while :; do
  done_n=0; done_list=""
  for st in ${WATCH}; do
    f=".scale_train_supervisors/q3-opd-q4bm-s${st}-to-q1p7b-base-${RUN_TAG}/state.json"
    # done = supervisor saw the completion receipts, or (ScaleTrain API down) the run-file's S3 receipt exists
    if grep -q '"status": "COMPLETED_RECEIPTS"' "${f}" 2>/dev/null \
      || AWS_PROFILE=ml-worker timeout 60 aws s3 ls "s3://scale-ml/genai/rl-distill/qwen3-opd-full-checkpoints/q4bm-s${st}-to-q1p7b-base-${RUN_TAG}/run_complete.json" >/dev/null 2>&1; then
      done_n=$((done_n + 1)); done_list+=" s${st}"
    fi
  done
  (( done_n >= NEED )) && break
  sleep 120
done
log "completed:${done_list} -> launching ${NEXT}"
for st in ${NEXT}; do
  SPEC=q4bm-s${st}-to-q1p7b-base RUN_TAG="${RUN_TAG}" TOTAL_TRAINING_STEPS=250 TEST_FREQ=5 SAVE_FREQ=10 ALLOW_BORROWING=false \
    bash rl-distill-scripts/scale_train/start_qwen3_onpolicy_distill_supervisor.sh 2>&1 | tee -a "${LOG}"
done
log "FOLLOWUP_DONE"
