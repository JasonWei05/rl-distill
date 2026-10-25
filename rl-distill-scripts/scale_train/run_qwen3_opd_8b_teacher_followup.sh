#!/usr/bin/env bash
# Launch the 8B-teacher on-policy distillations of a band once that band's 8B RL run has completed: waits for the RL
# completion receipt (run_complete.json), takes best_step from the run's run_outcome.json (only once its final_step matches
# the receipt, so a stale first-run outcome is never used), checks the step's HF export, then starts
# SPEC=q8b<band>-s<best>-to-{q1p7b,q4b}-base (250 steps, no early stopping, validation every 5, borrowing false).
#   BANDS="medium hard" bash run_qwen3_opd_8b_teacher_followup.sh     # run in tmux q3-opd-8b-followup
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
B=s3://scale-ml/genai/rl-distill
s3() { AWS_PROFILE=ml-worker AWS_REGION=us-west-2 aws s3 "$@"; }
declare -A TAG=([easy]=q8be [medium]=q8bm [hard]=q8bh)
pending=(${BANDS:-medium hard})
while [ ${#pending[@]} -gt 0 ]; do
  left=()
  for band in "${pending[@]}"; do
    C=${B}/qwen3-8b-base-rl-full-checkpoints/qwen30b-${band}-es5; A=${B}/qwen3-8b-base-rl/qwen3-8b-${band}-es5
    receipt_step=$(s3 cp ${C}/run_complete.json - 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin)["checkpoint_step"])' 2>/dev/null || true)
    read -r final_step best < <(s3 cp ${A}/run_outcome.json - 2>/dev/null \
      | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["final_step"], d["best_step"])' 2>/dev/null || echo "- -")
    if [ -z "${receipt_step}" ] || [ "${final_step}" != "${receipt_step}" ]; then
      left+=("${band}"); continue
    fi
    s3 ls ${C}/global_step_${best}/actor/huggingface/model.safetensors >/dev/null \
      || { echo "$(date -u +%FT%TZ) FATAL: ${band}: no HF export at global_step_${best}" >&2; exit 1; }
    echo "$(date -u +%FT%TZ) ${band}: 8B RL complete at step ${receipt_step}, best step ${best} -> launching students"
    for student in q1p7b q4b; do
      SPEC=${TAG[$band]}-s${best}-to-${student}-base RUN_TAG=pgk1-base-v1 TOTAL_TRAINING_STEPS=250 TEST_FREQ=5 SAVE_FREQ=10 \
        ALLOW_BORROWING=false MAX_RELAUNCHES=200 bash start_qwen3_onpolicy_distill_supervisor.sh
    done
  done
  pending=(${left[@]+"${left[@]}"})
  if [ ${#pending[@]} -gt 0 ]; then sleep 600; fi
done
echo "$(date -u +%FT%TZ) all bands launched"
