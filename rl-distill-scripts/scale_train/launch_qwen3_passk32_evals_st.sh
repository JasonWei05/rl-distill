#!/usr/bin/env bash
# Qwen3 pass@32 evals (30B base, 4B base, 4B->30B distilled students) in one 8-GPU ScaleTrain job (dedicated capacity by default).
# See run_qwen3_passk32_evals_st.sh.
#   bash launch_qwen3_passk32_evals_st.sh --image <uri>
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
ENV_VARS="EVAL_N=${EVAL_N:-32}"
[ -n "${EXTRA_ENV_VARS:-}" ] && ENV_VARS+=",${EXTRA_ENV_VARS}"
echo "LAUNCH_ENV ${ENV_VARS}"
BORROW_FLAG=""; [ "${ALLOW_BORROWING:-false}" = true ] && BORROW_FLAG="--allow-borrowing"
exec bash launch_st_with_code.sh --gpus-per-instance "${GPUS:-8}" --priority high ${BORROW_FLAG} --active-deadline-hours 48 \
  --run-file run_qwen3_passk32_evals_st.sh --job-name "${JOB_NAME:-q3-passk32-evals}" --env-vars "${ENV_VARS}" "$@"
