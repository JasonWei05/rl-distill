#!/usr/bin/env bash
# Best Qwen3-30B-A3B medium RL checkpoint -> Qwen3-4B-Base: traces + distillation + medium-val evals in one 4-GPU ScaleTrain job
# (borrowing on, priority high). See run_qwen3_30bm_to_4b_distill_st.sh.
#   bash launch_qwen3_30bm_to_4b_distill_st.sh --image <uri>
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
ENV_VARS="TEACHER_SPEC=${TEACHER_SPEC:-q30bm-medium}"
[ -n "${EXTRA_ENV_VARS:-}" ] && ENV_VARS+=",${EXTRA_ENV_VARS}"
echo "LAUNCH_ENV ${ENV_VARS}"
exec bash launch_st_with_code.sh --gpus-per-instance 4 --priority high --allow-borrowing --active-deadline-hours 120 \
  --run-file run_qwen3_30bm_to_4b_distill_st.sh --job-name "${JOB_NAME:-q3-30bm-dist-4b}" --env-vars "${ENV_VARS}" "$@"
