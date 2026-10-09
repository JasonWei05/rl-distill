#!/usr/bin/env bash
# Best Qwen3-30B-A3B medium RL checkpoint -> Qwen3-4B-Base: traces + distillation + medium-val evals in one 4-GPU ScaleTrain job
# (borrowing on, priority high). See run_qwen3_30bm_to_4b_distill_st.sh.
#   bash launch_qwen3_30bm_to_4b_distill_st.sh --image <uri>
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
ENV_VARS="TEACHER_SPEC=${TEACHER_SPEC:-q30bm-medium},STUDENT=${STUDENT:-q4b}"
[ -n "${EXTRA_ENV_VARS:-}" ] && ENV_VARS+=",${EXTRA_ENV_VARS}"
echo "LAUNCH_ENV ${ENV_VARS}"
BORROW_FLAG="--allow-borrowing"; [ "${ALLOW_BORROWING:-true}" = false ] && BORROW_FLAG=""   # ALLOW_BORROWING=false -> dedicated capacity
exec bash launch_st_with_code.sh --gpus-per-instance "${GPUS:-4}" --priority high ${BORROW_FLAG} --active-deadline-hours 120 \
  --run-file run_qwen3_30bm_to_4b_distill_st.sh --job-name "${JOB_NAME:-q3-30bm-dist-4b}" --env-vars "${ENV_VARS}" "$@"
