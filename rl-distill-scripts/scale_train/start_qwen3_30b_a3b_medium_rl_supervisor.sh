#!/usr/bin/env bash
# Supervised launch of launch_qwen3_30b_a3b_medium_rl.sh: one 8-GPU borrowing job (priority high), relaunched after
# preemption / FAILED / cancel until the RL completion receipt + best_hf exist; every relaunch restores the newest complete
# S3 checkpoint (the rolling slot is refreshed after every step). Also starts the rolling-checkpoint janitor: pods cannot
# delete S3 objects, so superseded rolling steps (~370 GB each) are deleted from this box with the ml-worker identity.
#   bash start_qwen3_30b_a3b_medium_rl_supervisor.sh       # stop: touch <state>/STOP; tmux kill-session -t rl-q3-30b-rl-med
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
IMG="${ST_IMAGE:-692474966980.dkr.ecr.us-west-2.amazonaws.com/scale_train/shared/training/tmp:20260829-002920.cd13c11a-e0f3-4f65-a74b-0e0c5d72d58a}"
RUN_TAG="${RUN_TAG:-es5}"; TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-400}"
S3_BASE="${S3_BASE:-s3://scale-ml/genai/rl-distill/qwen3-30b-a3b-base-rl}"
CKPT_URI="${S3_BASE}-full-checkpoints/qwen30b-medium-${RUN_TAG}"
name="q3-30b-rl-med"; sup=".scale_train_supervisors/${name}-$(date -u +%Y%m%d)${SUP_TAG:-}"; mkdir -p "${sup}/pod-logs"; rm -f "${sup}/STOP"
tmux kill-session -t "rl-${name}" 2>/dev/null || true
tmux new-session -d -s "rl-${name}" "env -u AWS_PROFILE python3 rl-distill-scripts/scale_train/supervise_borrowing_job.py --name ${name} \
  --relaunch-on-cancel --relaunch-on-failure --max-relaunches 40 --quick-cancel-seconds 300 --max-quick-cancels 3 --failure-backoff-seconds 900 \
  --launch-log ${sup}/launch.log --monitor-log ${sup}/monitor.log --state-file ${sup}/state.json --stop-file ${sup}/STOP --pod-log-dir ${sup}/pod-logs \
  --completion-s3-uri ${CKPT_URI} --max-completion-step ${TOTAL_TRAINING_STEPS} --expected-completion-world-size 8 \
  --completion-best-hf-s3-uri ${S3_BASE}/qwen3-30b-a3b-medium-${RUN_TAG} \
  -- env RUN_TAG=${RUN_TAG} S3_BASE=${S3_BASE} TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS} bash rl-distill-scripts/scale_train/launch_qwen3_30b_a3b_medium_rl.sh --image ${IMG} 2>&1 | tee -a ${sup}/supervisor.out"
# cleanup-rolling never deletes the committed rolling step (or anything newer); it only removes superseded ones.
tmux kill-session -t "ckpt-janitor-${name}" 2>/dev/null || true
tmux new-session -d -s "ckpt-janitor-${name}" "while [ ! -e ${sup}/STOP ]; do \
  AWS_PROFILE=ml-worker AWS_REGION=us-west-2 .venv/bin/python rl-distill-scripts/full_checkpoint_s3.py cleanup-rolling --s3-uri ${CKPT_URI} 2>&1 \
  | grep -v -E 'deleted=0|no completed remote checkpoint|NoSuchKey' | sed \"s/^/\$(date -u +%FT%TZ) /\" >> ${sup}/rolling-janitor.log; sleep 60; done"
echo "started supervisor rl-${name} + ckpt-janitor-${name} (state ${sup}, checkpoints ${CKPT_URI})"
