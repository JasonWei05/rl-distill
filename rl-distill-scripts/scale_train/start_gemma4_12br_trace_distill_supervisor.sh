#!/usr/bin/env bash
# Supervised launch of launch_gemma4_12br_trace_distill_st.sh (rides out preemptions; relaunches on cancel/failure).
# No RL-protocol completion receipts: the distill trainer writes its own run_complete.json (different fields) and the RL-style receipt step
# broke a finished run (2026-09-17 11:57Z). A COMPLETED job is taken as finished; a preemption race misread as COMPLETED (01:59Z the same day)
# needs a manual relaunch -- check the S3 prefix for global_step_1000/_REMOTE_COMPLETE.json.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
IMG="${ST_IMAGE:-692474966980.dkr.ecr.us-west-2.amazonaws.com/scale_train/shared/training/tmp:20260829-002920.cd13c11a-e0f3-4f65-a74b-0e0c5d72d58a}"
CKPT_S3="${CKPT_S3:-s3://scale-ml/genai/rl-distill/gemma4-12br-distill-ckpts-v1/12br-medium-to-e4b-bs128-s1000-lr1e-6}"
LR="${LR:-1e-6}"; ALLOW_BORROWING="${ALLOW_BORROWING:-true}"
name="${NAME:-g4-12br-trace-distill-lr${LR}}"; sup=".scale_train_supervisors/${name}-$(date -u +%Y%m%d)${SUP_TAG:-}"; mkdir -p "${sup}/pod-logs"; rm -f "${sup}/STOP"
tmux new-session -d -s "rl-${name}" "env -u AWS_PROFILE python3 rl-distill-scripts/scale_train/supervise_borrowing_job.py --name ${name} \
  --relaunch-on-cancel --relaunch-on-failure --max-relaunches 30 --quick-cancel-seconds 300 --max-quick-cancels 2 --failure-backoff-seconds 900 \
  --launch-log ${sup}/launch.log --monitor-log ${sup}/monitor.log --state-file ${sup}/state.json --stop-file ${sup}/STOP --pod-log-dir ${sup}/pod-logs \
  -- env LR=${LR} ALLOW_BORROWING=${ALLOW_BORROWING} bash rl-distill-scripts/scale_train/launch_gemma4_12br_trace_distill_st.sh --image ${IMG} 2>&1 | tee -a ${sup}/supervisor.out"
echo "started supervisor rl-${name} (state ${sup})"
