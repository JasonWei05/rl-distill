#!/usr/bin/env bash
# Supervised launch of launch_qwen3_30bm_to_4b_distill_st.sh: relaunched after preemption / failure / cancel; every relaunch resumes
# (trace shards and COMPLETE bundle on S3, newest complete distill checkpoint, finished evals skipped). A COMPLETED job = done; results
# under s3://scale-ml/genai/rl-distill/qwen3-30b-rl-distill-evals-v1/q30bm-medium-to-q4b-bs128-s1000-lr2e-6/step_*/summary.json.
#   bash start_qwen3_30bm_to_4b_distill_supervisor.sh      # stop: touch <state>/STOP; tmux kill-session -t rl-q3-30bm-dist-4b
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
IMG="${ST_IMAGE:-692474966980.dkr.ecr.us-west-2.amazonaws.com/scale_train/shared/training/tmp:20260829-002920.cd13c11a-e0f3-4f65-a74b-0e0c5d72d58a}"
name=q3-30bm-dist-4b; sup=".scale_train_supervisors/${name}-$(date -u +%Y%m%d)${SUP_TAG:-}"; mkdir -p "${sup}/pod-logs"; rm -f "${sup}/STOP"
tmux kill-session -t "rl-${name}" 2>/dev/null || true
tmux new-session -d -s "rl-${name}" "env -u AWS_PROFILE python3 rl-distill-scripts/scale_train/supervise_borrowing_job.py --name ${name} \
  --relaunch-on-cancel --relaunch-on-failure --max-relaunches ${MAX_RELAUNCHES:-40} --quick-cancel-seconds 300 --max-quick-cancels 3 --failure-backoff-seconds ${FAILURE_BACKOFF_SECONDS:-900} --max-failure-backoff-seconds ${MAX_FAILURE_BACKOFF_SECONDS:-0} \
  --launch-log ${sup}/launch.log --monitor-log ${sup}/monitor.log --state-file ${sup}/state.json --stop-file ${sup}/STOP --pod-log-dir ${sup}/pod-logs \
  -- env JOB_NAME=${name} EXTRA_ENV_VARS=${EXTRA_ENV_VARS:-} bash rl-distill-scripts/scale_train/launch_qwen3_30bm_to_4b_distill_st.sh --image ${IMG} 2>&1 | tee -a ${sup}/supervisor.out"
echo "started supervisor rl-${name} (state ${sup})"
