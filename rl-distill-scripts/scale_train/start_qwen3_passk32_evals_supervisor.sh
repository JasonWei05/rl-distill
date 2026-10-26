#!/usr/bin/env bash
# Supervised launch of launch_qwen3_passk32_evals_st.sh: relaunched after preemption / failure / cancel (hourly retry on infra
# failures); every relaunch skips the evals already on S3. Done = all 12 summaries under
# s3://scale-ml/genai/rl-distill/qwen3-passk32-evals-v1/<model>/<band>/summary.json.
#   bash start_qwen3_passk32_evals_supervisor.sh      # stop: touch <state>/STOP; tmux kill-session -t rl-q3-passk32-evals
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
IMG="${ST_IMAGE:-692474966980.dkr.ecr.us-west-2.amazonaws.com/scale_train/shared/training/tmp:20260829-002920.cd13c11a-e0f3-4f65-a74b-0e0c5d72d58a}"
name=q3-passk32-evals; E=s3://scale-ml/genai/rl-distill/qwen3-passk32-evals-v1
MARKERS=""
for mb in q30b-base/medium q30b-base/hard q4b-base/medium q4b-base/hard; do MARKERS+=" --completion-marker-s3-uri ${E}/${mb}/summary.json"; done
for b in medium hard; do for s in 250 500 750 1000; do MARKERS+=" --completion-marker-s3-uri ${E}/q4b30b-${b}-s${s}/${b}/summary.json"; done; done
sup=".scale_train_supervisors/${name}-$(date -u +%Y%m%d)${SUP_TAG:-}"; mkdir -p "${sup}/pod-logs"; rm -f "${sup}/STOP"
tmux kill-session -t "=rl-${name}" 2>/dev/null || true
tmux new-session -d -s "rl-${name}" "env -u AWS_PROFILE python3 rl-distill-scripts/scale_train/supervise_borrowing_job.py --name ${name} \
  --relaunch-on-cancel --relaunch-on-failure --max-relaunches ${MAX_RELAUNCHES:-200} --quick-cancel-seconds 300 --max-quick-cancels 3 --failure-backoff-seconds ${FAILURE_BACKOFF_SECONDS:-3600} --max-failure-backoff-seconds ${MAX_FAILURE_BACKOFF_SECONDS:-3600} \
  --launch-log ${sup}/launch.log --monitor-log ${sup}/monitor.log --state-file ${sup}/state.json --stop-file ${sup}/STOP --pod-log-dir ${sup}/pod-logs${MARKERS}${INITIAL_JOB_ID:+ --initial-job-id ${INITIAL_JOB_ID}} \
  -- env JOB_NAME=${name} GPUS=${GPUS:-8} ALLOW_BORROWING=${ALLOW_BORROWING:-false} EXTRA_ENV_VARS=${EXTRA_ENV_VARS:-} bash rl-distill-scripts/scale_train/launch_qwen3_passk32_evals_st.sh --image ${IMG} 2>&1 | tee -a ${sup}/supervisor.out"
echo "started supervisor rl-${name} (state ${sup})"
