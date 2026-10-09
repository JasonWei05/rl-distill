#!/usr/bin/env bash
# Qwen math suite (easy/medium/hard val, GSM8K, MATH500, AIME25/26) for 4B-class checkpoints in one 4-GPU ScaleTrain job (dedicated capacity by default).
# See run_qwen3_math_suite_st.sh.
#   bash launch_qwen3_math_suite_st.sh --image <uri>
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
ENV_VARS="AIME_N=${AIME_N:-32}"
[ -n "${EXTRA_ENV_VARS:-}" ] && ENV_VARS+=",${EXTRA_ENV_VARS}"
echo "LAUNCH_ENV ${ENV_VARS}"
BORROW_FLAG=""; [ "${ALLOW_BORROWING:-false}" = true ] && BORROW_FLAG="--allow-borrowing"
exec bash launch_st_with_code.sh --gpus-per-instance "${GPUS:-4}" --priority high ${BORROW_FLAG} --active-deadline-hours 48 \
  --run-file run_qwen3_math_suite_st.sh --job-name "${JOB_NAME:-q3-math-suite}" --env-vars "${ENV_VARS}" "$@"
