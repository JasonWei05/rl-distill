#!/usr/bin/env bash
# Supervised launch of launch_gemma4_base_all_distill_st.sh (one stage): relaunched after preemption / failure / cancel until the stage's
# durable S3 output exists -- STAGES=1: the trace bundle's COMPLETE.json; STAGES=2: the step-TOTAL_TRAINING_STEPS checkpoint marker.
#   STAGES=1 bash start_gemma4_base_all_distill_supervisor.sh      # 12B-base traces (tmux rl-g4-12bbase-trace)
#   STAGES=2 bash start_gemma4_base_all_distill_supervisor.sh      # E4B distillation (tmux rl-g4-12bbase-dist-e4b)
#   stop: touch <state>/STOP; tmux kill-session -t "=rl-<name>"
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
IMG="${ST_IMAGE:-692474966980.dkr.ecr.us-west-2.amazonaws.com/scale_train/shared/training/tmp:20260829-002920.cd13c11a-e0f3-4f65-a74b-0e0c5d72d58a}"
TEACHER_SPEC="${TEACHER_SPEC:-12b-base-all}"; STUDENT="${STUDENT:-e4b}"; STAGES="${STAGES:?STAGES=1 or 2}"; ALLOW_BORROWING="${ALLOW_BORROWING:-false}"
case "${TEACHER_SPEC}" in 12b-base-all) FAMILY=gemma4-12b-base; short=12bbase ;; *) echo "FATAL: TEACHER_SPEC must be 12b-base-all" >&2; exit 2 ;; esac
B=s3://scale-ml/genai/rl-distill
case "${STAGES}" in
  1) name=g4-${short}-trace; MARKER="${B}/${FAMILY}-traces-topk128-v1/${TEACHER_SPEC}/COMPLETE.json" ;;
  2) name=g4-${short}-dist-${STUDENT}
     MARKER="${B}/${FAMILY}-distill-ckpts-v1/${TEACHER_SPEC}-to-${STUDENT}-bs${TRAIN_BATCH_SIZE:-128}-s${TOTAL_TRAINING_STEPS:-1000}-lr${LR:-2e-6}/global_step_${TOTAL_TRAINING_STEPS:-1000}/_REMOTE_COMPLETE.json" ;;
  *) echo "FATAL: STAGES must be 1 or 2" >&2; exit 2 ;;
esac
sup=".scale_train_supervisors/${name}-$(date -u +%Y%m%d)${SUP_TAG:-}"; mkdir -p "${sup}/pod-logs"; rm -f "${sup}/STOP"
# pin every relaunch to the commit the run starts on (launch_st_with_code.sh ships CODE_REF, default HEAD)
CODE_REF="$(git rev-parse --short=8 "${CODE_REF:-HEAD}")"
tmux kill-session -t "=rl-${name}" 2>/dev/null || true
tmux new-session -d -s "rl-${name}" "env -u AWS_PROFILE python3 rl-distill-scripts/scale_train/supervise_borrowing_job.py --name ${name} ${INITIAL_JOB_ID:+--initial-job-id ${INITIAL_JOB_ID}} \
  --relaunch-on-cancel --relaunch-on-failure --max-relaunches ${MAX_RELAUNCHES:-40} --quick-cancel-seconds 300 --max-quick-cancels 3 --failure-backoff-seconds ${FAILURE_BACKOFF_SECONDS:-900} --max-failure-backoff-seconds ${MAX_FAILURE_BACKOFF_SECONDS:-3600} \
  --launch-log ${sup}/launch.log --monitor-log ${sup}/monitor.log --state-file ${sup}/state.json --stop-file ${sup}/STOP --pod-log-dir ${sup}/pod-logs \
  --completion-marker-s3-uri ${MARKER} \
  -- env CODE_REF=${CODE_REF} TEACHER_SPEC=${TEACHER_SPEC} STUDENT=${STUDENT} STAGES=${STAGES} ALLOW_BORROWING=${ALLOW_BORROWING} JOB_NAME=${name} EXTRA_ENV_VARS=${EXTRA_ENV_VARS:-} \
  bash rl-distill-scripts/scale_train/gemma4/launch_gemma4_base_all_distill_st.sh --image ${IMG} 2>&1 | tee -a ${sup}/supervisor.out"
echo "started supervisor rl-${name} (state ${sup}, done when ${MARKER} exists, code ${CODE_REF})"
