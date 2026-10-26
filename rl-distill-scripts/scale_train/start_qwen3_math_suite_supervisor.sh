#!/usr/bin/env bash
# Supervised launch of launch_qwen3_math_suite_st.sh: relaunched after preemption / failure / cancel (hourly retry on infra
# failures); every relaunch skips the (model, set) evals already on S3. Done = every (model, set) summary under
# s3://scale-ml/genai/rl-distill/qwen3-math-suite-v1/<model>/<set>/summary.json (models: qwen3_math_suite_models.sh).
#   bash start_qwen3_math_suite_supervisor.sh      # stop: touch <state>/STOP; tmux kill-session -t rl-q3-math-suite
#   SUITE=q1p7b GPUS=4 bash start_qwen3_math_suite_supervisor.sh      # a registry suite (job / tmux name q3-ms-<SUITE>)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
IMG="${ST_IMAGE:-692474966980.dkr.ecr.us-west-2.amazonaws.com/scale_train/shared/training/tmp:20260829-002920.cd13c11a-e0f3-4f65-a74b-0e0c5d72d58a}"
# MODEL_NAME=<tag> MODEL_SRC=<s3:...|hf:...> JOB_TAG=<short>: one model on GPUS=1 (job / tmux name q3-ms-<JOB_TAG>)
source rl-distill-scripts/scale_train/qwen3_math_suite_models.sh
name=q3-math-suite; E=s3://scale-ml/genai/rl-distill/qwen3-math-suite-v1
[ -n "${SUITE:-}" ] && name=q3-ms-${SUITE}
names=""; for m in $(suite_models "${SUITE:-v1}"); do names+=" ${m%%=*}"; done
if [ -n "${MODEL_NAME:-}" ]; then name=q3-ms-${JOB_TAG:?JOB_TAG required with MODEL_NAME}; names="${MODEL_NAME}"; fi
MARKERS=""
for m in ${names}; do
  for s in ${SUITE_SETS}; do MARKERS+=" --completion-marker-s3-uri ${E}/${m}/${s}/summary.json"; done
done
sup=".scale_train_supervisors/${name}-$(date -u +%Y%m%d)${SUP_TAG:-}"; mkdir -p "${sup}/pod-logs"; rm -f "${sup}/STOP"
tmux kill-session -t "=rl-${name}" 2>/dev/null || true
tmux new-session -d -s "rl-${name}" "env -u AWS_PROFILE python3 rl-distill-scripts/scale_train/supervise_borrowing_job.py --name ${name} \
  --relaunch-on-cancel --relaunch-on-failure --max-relaunches ${MAX_RELAUNCHES:-200} --quick-cancel-seconds 300 --max-quick-cancels 3 --failure-backoff-seconds ${FAILURE_BACKOFF_SECONDS:-3600} --max-failure-backoff-seconds ${MAX_FAILURE_BACKOFF_SECONDS:-3600} \
  --launch-log ${sup}/launch.log --monitor-log ${sup}/monitor.log --state-file ${sup}/state.json --stop-file ${sup}/STOP --pod-log-dir ${sup}/pod-logs${MARKERS}${INITIAL_JOB_ID:+ --initial-job-id ${INITIAL_JOB_ID}} \
  -- env JOB_NAME=${name} GPUS=${GPUS:-4} SUITE=${SUITE:-} MODEL_NAME=${MODEL_NAME:-} MODEL_SRC=${MODEL_SRC:-} ALLOW_BORROWING=${ALLOW_BORROWING:-false} EXTRA_ENV_VARS=${EXTRA_ENV_VARS:-} bash rl-distill-scripts/scale_train/launch_qwen3_math_suite_st.sh --image ${IMG} 2>&1 | tee -a ${sup}/supervisor.out"
echo "started supervisor rl-${name} (state ${sup})"
