#!/usr/bin/env bash
# Start one supervise_borrowing_job.py per (size, band) for a seed sweep: launches the ScaleTrain job (4 GPUs, borrowing,
# high) and relaunches it after FAILED/ERROR until the durable completion markers exist. One tmux session per run.
#   SEED=43 SIZES="e2b e4b" BANDS="easy medium hard" bash start_gemma4_rl_seed_supervisors.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
SEED="${SEED:?}"; SIZES="${SIZES:-e2b e4b}"; BANDS="${BANDS:-easy medium hard}"; GPUS="${GPUS:-4}"
IMG="${ST_IMAGE:-692474966980.dkr.ecr.us-west-2.amazonaws.com/scale_train/shared/training/tmp:20260829-002920.cd13c11a-e0f3-4f65-a74b-0e0c5d72d58a}"
S3_BASE="${S3_BASE:-s3://scale-ml/genai/rl-distill/gemma4-difficulty-s${SEED}-20260910}"
for size in ${SIZES}; do for band in ${BANDS}; do
  key="${size}-${band}"; name="g4-${size}-s${SEED}-${band:0:4}"; sup=".scale_train_supervisors/${name}-$(date -u +%Y%m%d)"
  mkdir -p "${sup}/pod-logs"
  tmux kill-session -t "rl-${name}" 2>/dev/null || true
  tmux new-session -d -s "rl-${name}" "env -u AWS_PROFILE python3 rl-distill-scripts/scale_train/supervise_borrowing_job.py --name ${name} \
    --launch-log ${sup}/launch.log --monitor-log ${sup}/monitor.log --state-file ${sup}/state.json --stop-file ${sup}/STOP --pod-log-dir ${sup}/pod-logs \
    --completion-s3-uri ${S3_BASE}-full-checkpoints/${key} --max-completion-step 400 --expected-completion-world-size ${GPUS} \
    --completion-best-hf-s3-uri ${S3_BASE}/gemma4-${key} \
    -- env SIZE=${size} BAND=${band} SEED=${SEED} GPUS=${GPUS} S3_BASE=${S3_BASE} bash rl-distill-scripts/scale_train/launch_gemma4_rl_band_seed.sh --image ${IMG} 2>&1 | tee -a ${sup}/supervisor.out"
  echo "started supervisor rl-${name} (state ${sup})"; sleep 20   # stagger the launches (each uploads/reuses the code tarball)
done; done
