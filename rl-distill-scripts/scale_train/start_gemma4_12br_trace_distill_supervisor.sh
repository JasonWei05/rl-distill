#!/usr/bin/env bash
# Supervised launch of launch_gemma4_12br_trace_distill_st.sh (rides out preemptions; relaunches on cancel/failure).
# No --completion-s3-uri: the supervisor's durable-completion receipts follow the RL protocol (gemma4_rl_run_complete_v1), which the
# distill trainer does not write; without them a COMPLETED job is taken as finished (Kueue preemptions still show as QUEUED).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
IMG="${ST_IMAGE:-692474966980.dkr.ecr.us-west-2.amazonaws.com/scale_train/shared/training/tmp:20260829-002920.cd13c11a-e0f3-4f65-a74b-0e0c5d72d58a}"
name="${NAME:-g4-12br-trace-distill}"; sup=".scale_train_supervisors/${name}-$(date -u +%Y%m%d)"; mkdir -p "${sup}/pod-logs"; rm -f "${sup}/STOP"
tmux new-session -d -s "rl-${name}" "env -u AWS_PROFILE python3 rl-distill-scripts/scale_train/supervise_borrowing_job.py --name ${name} \
  --relaunch-on-cancel --relaunch-on-failure --max-relaunches 30 --quick-cancel-seconds 300 --max-quick-cancels 2 --failure-backoff-seconds 900 \
  --launch-log ${sup}/launch.log --monitor-log ${sup}/monitor.log --state-file ${sup}/state.json --stop-file ${sup}/STOP --pod-log-dir ${sup}/pod-logs \
  -- bash rl-distill-scripts/scale_train/launch_gemma4_12br_trace_distill_st.sh --image ${IMG} 2>&1 | tee -a ${sup}/supervisor.out"
echo "started supervisor rl-${name} (state ${sup})"
