#!/usr/bin/env bash
# Supervised ScaleTrain trace collection from the untrained Qwen3-4B-Base teacher: one 1-GPU borrowing job per band
# (run_qwen3_4b_base_traces_st.sh), relaunched after preemption / failure / cancel; each relaunch restores the finished
# shards from S3 and continues. Bundles: s3://scale-ml/genai/rl-distill/qwen3-4b-base-traces-topk128-v1/q4b-base-<band>/
#   bash start_qwen3_4b_base_traces_supervisors.sh            # BANDS="medium hard" by default
#   stop one: touch .scale_train_supervisors/q4b-tr-<band>-<date>/STOP; tmux kill-session -t tr-q4b-tr-<band>
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
IMG="${ST_IMAGE:-692474966980.dkr.ecr.us-west-2.amazonaws.com/scale_train/shared/training/tmp:20260829-002920.cd13c11a-e0f3-4f65-a74b-0e0c5d72d58a}"
for band in ${BANDS:-medium hard}; do
  short=$([ "${band}" = medium ] && echo med || echo "${band}")
  name="q4b-tr-${short}"; sup=".scale_train_supervisors/${name}-$(date -u +%Y%m%d)"; mkdir -p "${sup}/pod-logs"; rm -f "${sup}/STOP"
  tmux kill-session -t "tr-${name}" 2>/dev/null || true
  tmux new-session -d -s "tr-${name}" "env -u AWS_PROFILE python3 rl-distill-scripts/scale_train/supervise_borrowing_job.py --name ${name} \
    --relaunch-on-cancel --relaunch-on-failure --max-relaunches 30 --quick-cancel-seconds 300 --max-quick-cancels 3 --failure-backoff-seconds 600 \
    --launch-log ${sup}/launch.log --monitor-log ${sup}/monitor.log --state-file ${sup}/state.json --stop-file ${sup}/STOP --pod-log-dir ${sup}/pod-logs \
    -- bash -c 'cd rl-distill-scripts/scale_train && bash launch_st_with_code.sh --gpus-per-instance 1 --priority high --allow-borrowing \
       --active-deadline-hours 72 --run-file run_qwen3_4b_base_traces_st.sh --job-name ${name} --image ${IMG} \
       --env-vars TRACE_SPEC=q4b-base-${band}' 2>&1 | tee -a ${sup}/supervisor.out"
  echo "started supervisor tr-${name} (state ${sup})"
done
