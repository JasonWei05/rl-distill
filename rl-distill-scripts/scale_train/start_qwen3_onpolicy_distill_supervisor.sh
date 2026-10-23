#!/usr/bin/env bash
# Supervised launch of launch_qwen3_onpolicy_distill.sh (one SPEC): relaunched after preemption / FAILED / cancel until the
# completion receipt (step TOTAL_TRAINING_STEPS) + best_hf exist; every relaunch restores the newest complete S3 checkpoint.
# Also starts the rolling-checkpoint janitor (pods cannot delete S3 objects). Borrowing off, priority high by default.
#   SPEC=q4bm-to-q1p7b bash start_qwen3_onpolicy_distill_supervisor.sh   # stop: touch <state>/STOP; tmux kill-session -t rl-q3-opd-<SPEC>
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
IMG="${ST_IMAGE:-692474966980.dkr.ecr.us-west-2.amazonaws.com/scale_train/shared/training/tmp:20260829-002920.cd13c11a-e0f3-4f65-a74b-0e0c5d72d58a}"
SPEC="${SPEC:?SPEC required}"; RUN_TAG="${RUN_TAG:-pgk1-v1}"; TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-120}"
case "${SPEC##*-to-}" in q1p7b|q1p7b-base) GPUS=2 ;; q4b|q4b-base) GPUS=4 ;; *) echo "FATAL: unknown student in SPEC=${SPEC}" >&2; exit 2 ;; esac
S3_BASE="${S3_BASE:-s3://scale-ml/genai/rl-distill/qwen3-opd}"
CKPT_URI="${S3_BASE}-full-checkpoints/${SPEC}-${RUN_TAG}"
name=q3-opd-${SPEC}
sup=".scale_train_supervisors/${name}-${RUN_TAG}${SUP_TAG:-}"; mkdir -p "${sup}/pod-logs"; rm -f "${sup}/STOP"
tmux kill-session -t "rl-${name}" 2>/dev/null || true
tmux new-session -d -s "rl-${name}" "env -u AWS_PROFILE python3 rl-distill-scripts/scale_train/supervise_borrowing_job.py --name ${name} \
  --relaunch-on-cancel --relaunch-on-failure --max-relaunches ${MAX_RELAUNCHES:-40} --quick-cancel-seconds 300 --max-quick-cancels 3 --failure-backoff-seconds ${FAILURE_BACKOFF_SECONDS:-900} --max-failure-backoff-seconds ${MAX_FAILURE_BACKOFF_SECONDS:-0} \
  --launch-log ${sup}/launch.log --monitor-log ${sup}/monitor.log --state-file ${sup}/state.json --stop-file ${sup}/STOP --pod-log-dir ${sup}/pod-logs \
  --completion-s3-uri ${CKPT_URI} --max-completion-step ${TOTAL_TRAINING_STEPS} --expected-completion-world-size ${GPUS} \
  --completion-best-hf-s3-uri ${S3_BASE}/${SPEC}-${RUN_TAG} \
  -- env SPEC=${SPEC} RUN_TAG=${RUN_TAG} S3_BASE=${S3_BASE} TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS} ALLOW_BORROWING=${ALLOW_BORROWING:-false} \
  TEST_FREQ=${TEST_FREQ:-10} SAVE_FREQ=${SAVE_FREQ:-10} \
  bash rl-distill-scripts/scale_train/launch_qwen3_onpolicy_distill.sh --image ${IMG} 2>&1 | tee -a ${sup}/supervisor.out"
tmux kill-session -t "ckpt-janitor-${name}" 2>/dev/null || true
tmux new-session -d -s "ckpt-janitor-${name}" "while [ ! -e ${sup}/STOP ] && tmux has-session -t rl-${name} 2>/dev/null; do \
  AWS_PROFILE=ml-worker AWS_REGION=us-west-2 .venv/bin/python rl-distill-scripts/full_checkpoint_s3.py cleanup-rolling --s3-uri ${CKPT_URI} 2>&1 \
  | grep -v -E 'deleted=0|no completed remote checkpoint|NoSuchKey' | sed \"s/^/\$(date -u +%FT%TZ) /\" >> ${sup}/rolling-janitor.log; sleep 60; done"
echo "started supervisor rl-${name} + ckpt-janitor-${name} (state ${sup}, checkpoints ${CKPT_URI})"
