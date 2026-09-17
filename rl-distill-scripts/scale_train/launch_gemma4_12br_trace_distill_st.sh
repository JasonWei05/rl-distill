#!/usr/bin/env bash
# §9.0j(a) on ScaleTrain: trace collection from the resumed untrained-12B medium RL teacher (step 140) + distillation into E4B base,
# one 4-GPU job (borrowing on, priority high). See run_gemma4_12br_trace_and_distill_st.sh.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
ENV_VARS="TEACHER_SPEC=12br-medium,STUDENT=e4b,LR=${LR:-1e-6},TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-1000}"
echo "LAUNCH_ENV ${ENV_VARS}"
exec bash launch_st_with_code.sh --gpus-per-instance "${GPUS_PER_INSTANCE:-4}" --priority high --allow-borrowing --active-deadline-hours 240 \
  --run-file run_gemma4_12br_trace_and_distill_st.sh --job-name "${JOB_NAME:-g4-12br-trace-distill}" --env-vars "${ENV_VARS}" "$@"
