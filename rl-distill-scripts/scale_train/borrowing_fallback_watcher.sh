#!/usr/bin/env bash
# Borrowing fallback for dedicated-capacity (ALLOW_BORROWING=false) supervised jobs: if a supervisor's current job has been QUEUED
# for more than QUEUE_LIMIT_MIN minutes since submission, stop that supervisor, cancel exactly that job, and restart the same
# supervisor with ALLOW_BORROWING=true (new state dir <name>-<date>-brw). Every supervisor relaunches after preemption /
# failure / cancel and the run-files resume from S3 (RL: newest complete checkpoint; distill: newest 250-step checkpoint +
# finished evals; pass@32: finished evals), so preemptions on borrowed capacity cost at most the work since the last save.
#   bash borrowing_fallback_watcher.sh        # loops; one-shot: ONCE=1; log: .scale_train_supervisors/borrowing-fallback.log
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
QUEUE_LIMIT_MIN="${QUEUE_LIMIT_MIN:-60}"
SUP_ROOT=.scale_train_supervisors; LOG="${SUP_ROOT}/borrowing-fallback.log"
RETRY="MAX_RELAUNCHES=200 FAILURE_BACKOFF_SECONDS=3600 MAX_FAILURE_BACKOFF_SECONDS=3600"
S=rl-distill-scripts/scale_train
# name | supervisor state dir (borrowing false) | restart command (borrowing true)
JOBS=(
  "q3-30bvh-dist-4b|${SUP_ROOT}/q3-30bvh-dist-4b-20261013-r2|TEACHER_SPEC=q30bvh-very_hard GPUS=8 EXTRA_ENV_VARS=TRACE_TP=1,MAX_NUM_SEQS=48,GPU_MEMORY_UTILIZATION=0.92,TRACE_PIN_ENGINE_IDENTITY=1,TRACE_VALIDATION_WORKERS=16 ${RETRY} bash ${S}/start_qwen3_30bm_to_4b_distill_supervisor.sh"
  "q3-4bvh-dist-1p7b|${SUP_ROOT}/q3-4bvh-dist-1p7b-20261013-r2|TEACHER_SPEC=q4bvh-very_hard STUDENT=q1p7b GPUS=8 EXTRA_ENV_VARS=TRACE_VALIDATION_WORKERS=16 ${RETRY} bash ${S}/start_qwen3_30bm_to_4b_distill_supervisor.sh"
  "q3-4bm-dist-1p7b|${SUP_ROOT}/q3-4bm-dist-1p7b-20261009|TEACHER_SPEC=q4bm-medium STUDENT=q1p7b GPUS=8 EXTRA_ENV_VARS=TRACE_VALIDATION_WORKERS=16 ${RETRY} bash ${S}/start_qwen3_30bm_to_4b_distill_supervisor.sh"
  "q3-4bh-dist-1p7b|${SUP_ROOT}/q3-4bh-dist-1p7b-20261009|TEACHER_SPEC=q4bh-hard STUDENT=q1p7b GPUS=8 EXTRA_ENV_VARS=TRACE_VALIDATION_WORKERS=16 ${RETRY} bash ${S}/start_qwen3_30bm_to_4b_distill_supervisor.sh"
  "q3-30bm-dist-1p7b|${SUP_ROOT}/q3-30bm-dist-1p7b-20261009|TEACHER_SPEC=q30bm-medium STUDENT=q1p7b GPUS=2 ${RETRY} bash ${S}/start_qwen3_30bm_to_4b_distill_supervisor.sh"
  "q3-30bh-dist-1p7b|${SUP_ROOT}/q3-30bh-dist-1p7b-20261009|TEACHER_SPEC=q30bh-hard STUDENT=q1p7b GPUS=2 ${RETRY} bash ${S}/start_qwen3_30bm_to_4b_distill_supervisor.sh"
  "q3-1p7b-rl-med|${SUP_ROOT}/q3-1p7b-rl-med-20261009|QWEN_SIZE=1.7b BAND=medium TOTAL_TRAINING_STEPS=600 ${RETRY} bash ${S}/start_qwen3_30b_a3b_medium_rl_supervisor.sh"
  "q3-1p7b-rl-hard|${SUP_ROOT}/q3-1p7b-rl-hard-20261009|QWEN_SIZE=1.7b BAND=hard TOTAL_TRAINING_STEPS=600 ${RETRY} bash ${S}/start_qwen3_30b_a3b_medium_rl_supervisor.sh"
  "q3-ms-30bh-1p7b-s500|${SUP_ROOT}/q3-ms-30bh-1p7b-s500-20261009|MODEL_NAME=q30bh-to-q1p7b-s500 MODEL_SRC=s3:s3://scale-ml/genai/rl-distill/qwen3-30b-rl-distill-ckpts-v1/q30bh-hard-to-q1p7b-bs128-s1000-lr2e-6/global_step_500/huggingface JOB_TAG=30bh-1p7b-s500 GPUS=1 ${RETRY} bash ${S}/start_qwen3_math_suite_supervisor.sh"
  "q3-ms-30bm-1p7b-s1000|${SUP_ROOT}/q3-ms-30bm-1p7b-s1000-20261009|MODEL_NAME=q30bm-to-q1p7b-s1000 MODEL_SRC=s3:s3://scale-ml/genai/rl-distill/qwen3-30b-rl-distill-ckpts-v1/q30bm-medium-to-q1p7b-bs128-s1000-lr2e-6/global_step_1000/huggingface JOB_TAG=30bm-1p7b-s1000 GPUS=1 EXTRA_ENV_VARS=SINGLE_PASS=1 ${RETRY} bash ${S}/start_qwen3_math_suite_supervisor.sh"
  "q3-math-suite|${SUP_ROOT}/q3-math-suite-20261009|${RETRY} bash ${S}/start_qwen3_math_suite_supervisor.sh"
  "q3-passk32-evals|${SUP_ROOT}/q3-passk32-evals-20261009|${RETRY} bash ${S}/start_qwen3_passk32_evals_supervisor.sh"
  "q3-30bh-dist-4b|${SUP_ROOT}/q3-30bh-dist-4b-20261009-tp1|TEACHER_SPEC=q30bh-hard GPUS=8 EXTRA_ENV_VARS=TRACE_TP=1,MAX_NUM_SEQS=48,GPU_MEMORY_UTILIZATION=0.92,TRACE_PIN_ENGINE_IDENTITY=1,TRACE_VALIDATION_WORKERS=16 ${RETRY} bash ${S}/start_qwen3_30bm_to_4b_distill_supervisor.sh"
)
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "${LOG}"; }
check_once() {
  local entry name sup cmd last job st sub_ts age_min
  for entry in "${JOBS[@]}"; do
    IFS='|' read -r name sup cmd <<<"${entry}"
    [ -d "${sup}" ] && [ ! -e "${sup}/STOP" ] && [ ! -e "${sup}/BORROW_SWITCHED" ] || continue
    last="$(grep -aE ' status=[A-Z_]+| submitted' "${sup}/supervisor.out" 2>/dev/null | tail -1)"
    st="$(grep -oE 'status=[A-Z_]+' <<<"${last}" | cut -d= -f2)"
    [ "${st}" = QUEUED ] || continue
    job="$(grep -oE 'job_id=job_[a-z0-9]+' <<<"${last}" | cut -d= -f2)"
    sub_ts="$(grep -aE "job_id=${job} submitted" "${sup}/supervisor.out" | tail -1 | cut -d' ' -f1)"
    [ -n "${sub_ts}" ] || continue
    age_min=$(( ( $(date +%s) - $(date -d "${sub_ts}" +%s) ) / 60 ))
    (( age_min > QUEUE_LIMIT_MIN )) || continue
    log "${name}: ${job} QUEUED for ${age_min} min (> ${QUEUE_LIMIT_MIN}) on borrowing false -> switching to borrowing true"
    touch "${sup}/STOP" "${sup}/BORROW_SWITCHED"
    tmux kill-session -t "rl-${name}" 2>/dev/null || true
    log "${name}: cancel ${job}: $(env -u AWS_PROFILE scale-train cancel job "${job}" 2>&1 | grep -ioE 'cancel[a-z]*|error[^,]{0,80}' | sort -u | tr '\n' ' ')"
    log "${name}: $(env ALLOW_BORROWING=true SUP_TAG=-brw bash -c "${cmd}" 2>&1 | tail -1)"
  done
}
if [ -n "${ONCE:-}" ]; then check_once; exit 0; fi
while true; do check_once; sleep 300; done
