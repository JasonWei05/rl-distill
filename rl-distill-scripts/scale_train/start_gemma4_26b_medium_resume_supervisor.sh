#!/usr/bin/env bash
# Supervised launch of launch_gemma4_26b_medium_resume.sh: one 8-GPU borrowing job (priority high), relaunched after
# FAILED/ERROR/cancel until the RL completion receipt + best_hf exist; every relaunch restores the newest rolling S3 checkpoint.
#   bash start_gemma4_26b_medium_resume_supervisor.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
IMG="${ST_IMAGE:-692474966980.dkr.ecr.us-west-2.amazonaws.com/scale_train/shared/training/tmp:20260829-002920.cd13c11a-e0f3-4f65-a74b-0e0c5d72d58a}"
RUN_TAG="${RUN_TAG:-26b-bands-es5}"; CUDNN_SDPA="${CUDNN_SDPA:-1}"
S3_BASE="${S3_BASE:-s3://scale-ml/genai/rl-distill/gemma4-difficulty-s42-20260819}"
name="g4-26b-med-resume"; sup=".scale_train_supervisors/${name}-$(date -u +%Y%m%d)${SUP_TAG:-}"; mkdir -p "${sup}/pod-logs"; rm -f "${sup}/STOP"
tmux kill-session -t "rl-${name}" 2>/dev/null || true
tmux new-session -d -s "rl-${name}" "env -u AWS_PROFILE python3 rl-distill-scripts/scale_train/supervise_borrowing_job.py --name ${name} \
  --relaunch-on-cancel --relaunch-on-failure --max-relaunches 40 --quick-cancel-seconds 300 --max-quick-cancels 3 --failure-backoff-seconds 900 \
  --launch-log ${sup}/launch.log --monitor-log ${sup}/monitor.log --state-file ${sup}/state.json --stop-file ${sup}/STOP --pod-log-dir ${sup}/pod-logs \
  --completion-s3-uri ${S3_BASE}-full-checkpoints/26b-a4b-medium --max-completion-step 400 --expected-completion-world-size 8 \
  --completion-best-hf-s3-uri ${S3_BASE}/gemma4-26b-a4b-medium \
  -- env RUN_TAG=${RUN_TAG} CUDNN_SDPA=${CUDNN_SDPA} S3_BASE=${S3_BASE} bash rl-distill-scripts/scale_train/launch_gemma4_26b_medium_resume.sh --image ${IMG} 2>&1 | tee -a ${sup}/supervisor.out"
echo "started supervisor rl-${name} (state ${sup})"
