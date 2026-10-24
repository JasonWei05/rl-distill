#!/usr/bin/env bash
# Supervised launch of launch_qwen3_30b_a3b_medium_rl.sh: one 8-GPU borrowing job (priority high), relaunched after
# preemption / FAILED / cancel until the RL completion receipt + best_hf exist; every relaunch restores the newest complete
# S3 checkpoint (the rolling slot is refreshed after every step). Also starts the rolling-checkpoint janitor: pods cannot
# delete S3 objects, so superseded rolling steps (~370 GB each) are deleted from this box with the ml-worker identity.
#   bash start_qwen3_30b_a3b_medium_rl_supervisor.sh       # stop: touch <state>/STOP; tmux kill-session -t rl-q3-30b-rl-med
#   QWEN_SIZE=4b bash start_qwen3_30b_a3b_medium_rl_supervisor.sh   # Qwen3-4B-Base, 4 GPUs (sessions rl-q3-4b-rl-med, ...)
#   BAND=hard [QWEN_SIZE=4b] bash start_qwen3_30b_a3b_medium_rl_supervisor.sh   # hard band (sessions rl-q3-<size>-rl-hard)
#   QWEN_SIZE=1.7b ALLOW_BORROWING=false ...   # Qwen3-1.7B-Base, 2 GPUs, dedicated capacity (sessions rl-q3-1p7b-rl-<band>)
#   ST_CLUSTER=gke ...   # GKE a3-mega capacity instead of EKS p5 (same S3 checkpoints)
#   ROLLING_CHECKPOINT_ENABLED=False ASYNC_CHECKPOINT_UPLOAD=True ...   # permanent saves only, background S3 upload (30B: ~34% -> ~10% ckpt time)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
IMG="${ST_IMAGE:-692474966980.dkr.ecr.us-west-2.amazonaws.com/scale_train/shared/training/tmp:20260829-002920.cd13c11a-e0f3-4f65-a74b-0e0c5d72d58a}"
RUN_TAG="${RUN_TAG:-es5}"; TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-400}"; QWEN_SIZE="${QWEN_SIZE:-30b-a3b}"; BAND="${BAND:-medium}"
case "${BAND}" in easy) BAND_SHORT=easy ;; medium) BAND_SHORT=med ;; hard) BAND_SHORT=hard ;;
  very_hard) BAND_SHORT=vhard ;; *) echo "FATAL: BAND must be easy, medium, hard or very_hard" >&2; exit 2 ;; esac
case "${QWEN_SIZE}" in
  30b-a3b) SLUG=qwen3-30b-a3b; name=q3-30b-rl-${BAND_SHORT}; GPUS=8 ;;
  8b) SLUG=qwen3-8b; name=q3-8b-rl-${BAND_SHORT}; GPUS=4 ;;
  4b) SLUG=qwen3-4b; name=q3-4b-rl-${BAND_SHORT}; GPUS=4 ;;
  1.7b) SLUG=qwen3-1.7b; name=q3-1p7b-rl-${BAND_SHORT}; GPUS=2 ;;
  *) echo "FATAL: QWEN_SIZE must be 30b-a3b, 8b, 4b or 1.7b" >&2; exit 2 ;;
esac
S3_BASE="${S3_BASE:-s3://scale-ml/genai/rl-distill/${SLUG}-base-rl}"
CKPT_URI="${S3_BASE}-full-checkpoints/qwen30b-${BAND}-${RUN_TAG}"
sup=".scale_train_supervisors/${name}-$(date -u +%Y%m%d)${SUP_TAG:-}"; mkdir -p "${sup}/pod-logs"; rm -f "${sup}/STOP"
tmux kill-session -t "rl-${name}" 2>/dev/null || true
tmux new-session -d -s "rl-${name}" "env -u AWS_PROFILE python3 rl-distill-scripts/scale_train/supervise_borrowing_job.py --name ${name} \
  --relaunch-on-cancel --relaunch-on-failure --max-relaunches ${MAX_RELAUNCHES:-40} --quick-cancel-seconds 300 --max-quick-cancels 3 --failure-backoff-seconds ${FAILURE_BACKOFF_SECONDS:-900} --max-failure-backoff-seconds ${MAX_FAILURE_BACKOFF_SECONDS:-0} \
  --launch-log ${sup}/launch.log --monitor-log ${sup}/monitor.log --state-file ${sup}/state.json --stop-file ${sup}/STOP --pod-log-dir ${sup}/pod-logs \
  --completion-s3-uri ${CKPT_URI} --max-completion-step ${TOTAL_TRAINING_STEPS} --expected-completion-world-size ${GPUS} \
  --completion-best-hf-s3-uri ${S3_BASE}/${SLUG}-${BAND}-${RUN_TAG} \
  -- env QWEN_SIZE=${QWEN_SIZE} BAND=${BAND} RUN_TAG=${RUN_TAG} S3_BASE=${S3_BASE} TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS} ROLLING_CHECKPOINT_ENABLED=${ROLLING_CHECKPOINT_ENABLED:-True} ASYNC_CHECKPOINT_UPLOAD=${ASYNC_CHECKPOINT_UPLOAD:-False} ALLOW_BORROWING=${ALLOW_BORROWING:-true} MBS=${MBS:-4} PADDED_TOKENS=${PADDED_TOKENS:-8192} EARLY_STOPPING_RESET_PATIENCE_AT_STEP=${EARLY_STOPPING_RESET_PATIENCE_AT_STEP:-} bash rl-distill-scripts/scale_train/launch_qwen3_30b_a3b_medium_rl.sh --image ${IMG}${ST_CLUSTER:+ --cluster ${ST_CLUSTER}} 2>&1 | tee -a ${sup}/supervisor.out"
# cleanup-rolling never deletes the committed rolling step (or anything newer); it only removes superseded ones.
tmux kill-session -t "ckpt-janitor-${name}" 2>/dev/null || true
tmux new-session -d -s "ckpt-janitor-${name}" "while [ ! -e ${sup}/STOP ]; do \
  AWS_PROFILE=ml-worker AWS_REGION=us-west-2 .venv/bin/python rl-distill-scripts/full_checkpoint_s3.py cleanup-rolling --s3-uri ${CKPT_URI} 2>&1 \
  | grep -v -E 'deleted=0|no completed remote checkpoint|NoSuchKey' | sed \"s/^/\$(date -u +%FT%TZ) /\" >> ${sup}/rolling-janitor.log; sleep 60; done"
echo "started supervisor rl-${name} + ckpt-janitor-${name} (state ${sup}, checkpoints ${CKPT_URI})"
