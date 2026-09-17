#!/usr/bin/env bash
# Supervised launch of launch_gemma4_12b_onpolicy_bkt50_rl_medium.sh (rides out preemptions, relaunches on cancel/failure).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
S3_BASE="${S3_BASE:-s3://scale-ml/genai/rl-distill/gemma4-12b-from-onpolicy-bkt50-rl}"; RUN_TAG="${RUN_TAG:-from-onpolicy-bkt50-es5}"
IMG="${ST_IMAGE:-692474966980.dkr.ecr.us-west-2.amazonaws.com/scale_train/shared/training/tmp:20260829-002920.cd13c11a-e0f3-4f65-a74b-0e0c5d72d58a}"
name="${NAME:-g4-12b-bkt50-rl-med}"; sup=".scale_train_supervisors/${name}-$(date -u +%Y%m%d)${SUP_TAG:-}"; mkdir -p "${sup}/pod-logs"; rm -f "${sup}/STOP"
tmux new-session -d -s "rl-${name}" "env -u AWS_PROFILE python3 rl-distill-scripts/scale_train/supervise_borrowing_job.py --name ${name} \
  --relaunch-on-cancel --relaunch-on-failure --max-relaunches 30 --quick-cancel-seconds 300 --max-quick-cancels 2 --failure-backoff-seconds 900 \
  --launch-log ${sup}/launch.log --monitor-log ${sup}/monitor.log --state-file ${sup}/state.json --stop-file ${sup}/STOP --pod-log-dir ${sup}/pod-logs \
  --completion-s3-uri ${S3_BASE}-full-checkpoints/12b-medium-${RUN_TAG} --max-completion-step 400 --expected-completion-world-size 4 \
  --completion-best-hf-s3-uri ${S3_BASE}/gemma4-12b-medium-${RUN_TAG} \
  -- env S3_BASE=${S3_BASE} RUN_TAG=${RUN_TAG} EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE:-5} EARLY_STOPPING_MIGRATE_PATIENCE_FROM=${EARLY_STOPPING_MIGRATE_PATIENCE_FROM:-} bash rl-distill-scripts/scale_train/launch_gemma4_12b_onpolicy_bkt50_rl_medium.sh --image ${IMG} 2>&1 | tee -a ${sup}/supervisor.out"
echo "started supervisor rl-${name} (state ${sup})"
