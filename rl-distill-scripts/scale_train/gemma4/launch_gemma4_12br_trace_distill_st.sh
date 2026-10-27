#!/usr/bin/env bash
# §9.0j(a) on ScaleTrain: trace collection from the resumed untrained-12B medium RL teacher (step 140) + distillation into E4B base,
# one 4-GPU job (borrowing on, priority high). See run_gemma4_12br_trace_and_distill_st.sh.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
TEACHER_SPEC="${TEACHER_SPEC:-12br-medium}"   # 12br-medium (§9.0j) or 12bo-medium (RL-on-bkt50 teacher, §9.0k)
ENV_VARS="TEACHER_SPEC=${TEACHER_SPEC},STUDENT=e4b,LR=${LR:-1e-6},TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-1000}"
[ -n "${BEST_STEP_OVERRIDE:-}" ] && ENV_VARS+=",BEST_STEP_OVERRIDE=${BEST_STEP_OVERRIDE}"   # 26bd/26bw teachers: best step decided at RL stop time
[ -n "${EXTRA_ENV_VARS:-}" ] && ENV_VARS+=",${EXTRA_ENV_VARS}"   # e.g. REMOTE_CHECKPOINT_S3_URI=...-n8,EXP_NAME=...-st8 for an 8-GPU copy (distill ckpts are world-size specific)
echo "LAUNCH_ENV ${ENV_VARS}"
BORROW_FLAG="--allow-borrowing"; [ "${ALLOW_BORROWING:-true}" = false ] && BORROW_FLAG=""   # ALLOW_BORROWING=false -> dedicated (non-preemptible) capacity
exec bash ../launch_st_with_code.sh --gpus-per-instance "${GPUS_PER_INSTANCE:-4}" --priority high ${BORROW_FLAG} --active-deadline-hours 240 \
  --run-file gemma4/run_gemma4_12br_trace_and_distill_st.sh --job-name "${JOB_NAME:-g4-${TEACHER_SPEC%-medium}-trace-distill-lr${LR:-1e-6}}" --env-vars "${ENV_VARS}" "$@"
