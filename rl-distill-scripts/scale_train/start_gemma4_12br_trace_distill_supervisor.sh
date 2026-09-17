#!/usr/bin/env bash
# Supervised launch of launch_gemma4_12br_trace_distill_st.sh (rides out preemptions; relaunches on cancel/failure).
# Completion = the RL-protocol receipt the run-file writes after the step-1000 permanent checkpoint (full_checkpoint_s3.py complete-run).
# A COMPLETED job without it (2026-09-17 01:59Z: a preemption race was misread as COMPLETED) is relaunched and resumes from S3.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
IMG="${ST_IMAGE:-692474966980.dkr.ecr.us-west-2.amazonaws.com/scale_train/shared/training/tmp:20260829-002920.cd13c11a-e0f3-4f65-a74b-0e0c5d72d58a}"
CKPT_S3="${CKPT_S3:-s3://scale-ml/genai/rl-distill/gemma4-12br-distill-ckpts-v1/12br-medium-to-e4b-bs128-s1000-lr1e-6}"
name="${NAME:-g4-12br-trace-distill}"; sup=".scale_train_supervisors/${name}-$(date -u +%Y%m%d)"; mkdir -p "${sup}/pod-logs"; rm -f "${sup}/STOP"
tmux new-session -d -s "rl-${name}" "env -u AWS_PROFILE python3 rl-distill-scripts/scale_train/supervise_borrowing_job.py --name ${name} \
  --relaunch-on-cancel --relaunch-on-failure --max-relaunches 30 --quick-cancel-seconds 300 --max-quick-cancels 2 --failure-backoff-seconds 900 \
  --launch-log ${sup}/launch.log --monitor-log ${sup}/monitor.log --state-file ${sup}/state.json --stop-file ${sup}/STOP --pod-log-dir ${sup}/pod-logs \
  --completion-s3-uri ${CKPT_S3} --expected-completion-step 1000 --expected-completion-world-size 4 \
  -- bash rl-distill-scripts/scale_train/launch_gemma4_12br_trace_distill_st.sh --image ${IMG} 2>&1 | tee -a ${sup}/supervisor.out"
echo "started supervisor rl-${name} (state ${sup})"
