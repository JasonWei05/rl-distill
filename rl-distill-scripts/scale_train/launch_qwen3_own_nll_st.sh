#!/usr/bin/env bash
# Own-sample NLL (entropy estimate) of the 4B base -> 30B base forward-KL students and the two bases, one ScaleTrain job.
# See run_qwen3_own_nll_st.sh.
#   ALLOW_BORROWING=true GPUS=4 bash launch_qwen3_own_nll_st.sh --image <uri>
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
ENV_VARS="PER_QUESTION=${PER_QUESTION:-8},EVAL_TP=${EVAL_TP:-2}"
echo "LAUNCH_ENV ${ENV_VARS}"
BORROW_FLAG=""; [ "${ALLOW_BORROWING:-false}" = true ] && BORROW_FLAG="--allow-borrowing"
exec bash launch_st_with_code.sh --gpus-per-instance "${GPUS:-4}" --priority high ${BORROW_FLAG} --active-deadline-hours 12 \
  --run-file run_qwen3_own_nll_st.sh --job-name "${JOB_NAME:-q3-own-nll}" --env-vars "${ENV_VARS}" "$@"
