#!/usr/bin/env bash
# Qwen3-4B-Base -> Qwen3-30B-A3B-Base top-128 distillation on ScaleTrain: one 8-GPU job (whole p5 node), borrowing on,
# priority high. BAND=medium|hard picks the q4b-base-<band> trace bundle. See run_qwen3_4b_to_30b_distill_st.sh.
#   BAND=medium bash launch_qwen3_4b_to_30b_distill_st.sh --image <uri>
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
BAND="${BAND:-medium}"
case "${BAND}" in medium) short=med ;; hard) short=hard ;; *) echo "FATAL: BAND must be medium or hard" >&2; exit 2 ;; esac
ENV_VARS="TEACHER_SPEC=q4b-base-${BAND}"
[ -n "${EXTRA_ENV_VARS:-}" ] && ENV_VARS+=",${EXTRA_ENV_VARS}"   # e.g. LR=...,TOTAL_TRAINING_STEPS=... (changes the S3 prefix + W&B name)
echo "LAUNCH_ENV ${ENV_VARS}"
BORROW_FLAG="--allow-borrowing"; [ "${ALLOW_BORROWING:-true}" = false ] && BORROW_FLAG=""   # ALLOW_BORROWING=false -> dedicated (non-preemptible) capacity
exec bash launch_st_with_code.sh --gpus-per-instance 8 --priority high ${BORROW_FLAG} --active-deadline-hours 120 \
  --run-file run_qwen3_4b_to_30b_distill_st.sh --job-name "${JOB_NAME:-q3-dist-4b30b-${short}}" --env-vars "${ENV_VARS}" "$@"
