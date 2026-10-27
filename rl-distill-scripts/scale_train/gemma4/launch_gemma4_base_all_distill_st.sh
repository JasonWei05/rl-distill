#!/usr/bin/env bash
# One ScaleTrain job of the Gemma 4 base -> base pre-distillation (run_gemma4_base_all_distill_st.sh): STAGES=1 traces of the untrained
# 12B base over the easy + medium + hard band union, STAGES=2 the tail-bucket top-128 forward-KL distillation into the E4B base.
# 4 GPUs, priority high, dedicated capacity by default (ALLOW_BORROWING=true for borrowed).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
TEACHER_SPEC="${TEACHER_SPEC:-12b-base-all}"; STUDENT="${STUDENT:-e4b}"; STAGES="${STAGES:?STAGES=1 (traces) or STAGES=2 (distill)}"
ENV_VARS="TEACHER_SPEC=${TEACHER_SPEC},STUDENT=${STUDENT},STAGES=${STAGES}"
[ -n "${EXTRA_ENV_VARS:-}" ] && ENV_VARS+=",${EXTRA_ENV_VARS}"
echo "LAUNCH_ENV ${ENV_VARS}"
BORROW_FLAG=""; [ "${ALLOW_BORROWING:-false}" = true ] && BORROW_FLAG="--allow-borrowing"
case "${STAGES}" in 1) tag=trace ;; 2) tag=dist-${STUDENT} ;; *) tag=s${STAGES// /} ;; esac
exec bash ../launch_st_with_code.sh --gpus-per-instance "${GPUS:-4}" --priority high ${BORROW_FLAG} --active-deadline-hours 120 \
  --run-file gemma4/run_gemma4_base_all_distill_st.sh --job-name "${JOB_NAME:-g4-${TEACHER_SPEC}-${tag}}" --env-vars "${ENV_VARS}" "$@"
