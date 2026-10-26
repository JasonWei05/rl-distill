#!/usr/bin/env bash
# Supervised launch of launch_qwen3_4b_to_30b_distill_st.sh (rides out preemptions; every relaunch resumes from the newest
# complete S3 checkpoint) + the rolling-checkpoint janitor (pods cannot delete S3 objects; ~370 GB per stale rolling step).
# A COMPLETED job is taken as finished (the distill trainer writes its own run_complete.json, no RL-style receipt);
# verify <prefix>/global_step_1000/_REMOTE_COMPLETE.json by hand.
#   BAND=medium bash start_qwen3_4b_to_30b_distill_supervisor.sh     # stop: touch <state>/STOP; tmux kill-session -t rl-q3-dist-4b30b-med
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
IMG="${ST_IMAGE:-692474966980.dkr.ecr.us-west-2.amazonaws.com/scale_train/shared/training/tmp:20260829-002920.cd13c11a-e0f3-4f65-a74b-0e0c5d72d58a}"
BAND="${BAND:-medium}"
case "${BAND}" in medium) short=med ;; hard) short=hard ;; *) echo "FATAL: BAND must be medium or hard" >&2; exit 2 ;; esac
# done = the step-1000 checkpoint committed on S3 (a COMPLETED status without it is a misreported preemption -> relaunch)
MARKER="s3://scale-ml/genai/rl-distill/qwen3-4b-base-distill-ckpts-v1/q4b-base-${BAND}-to-q30b-bs128-s1000-lr2e-6/global_step_1000/_REMOTE_COMPLETE.json"
name="q3-dist-4b30b-${short}"; sup=".scale_train_supervisors/${name}-$(date -u +%Y%m%d)${SUP_TAG:-}"; mkdir -p "${sup}/pod-logs"; rm -f "${sup}/STOP"
tmux kill-session -t "=rl-${name}" 2>/dev/null || true
tmux new-session -d -s "rl-${name}" "env -u AWS_PROFILE python3 rl-distill-scripts/scale_train/supervise_borrowing_job.py --name ${name} \
  --relaunch-on-cancel --relaunch-on-failure --max-relaunches ${MAX_RELAUNCHES:-30} --quick-cancel-seconds 300 --max-quick-cancels 3 --failure-backoff-seconds ${FAILURE_BACKOFF_SECONDS:-900} --max-failure-backoff-seconds ${MAX_FAILURE_BACKOFF_SECONDS:-0} \
  --launch-log ${sup}/launch.log --monitor-log ${sup}/monitor.log --state-file ${sup}/state.json --stop-file ${sup}/STOP --pod-log-dir ${sup}/pod-logs --completion-marker-s3-uri ${MARKER}${INITIAL_JOB_ID:+ --initial-job-id ${INITIAL_JOB_ID}} \
  -- env BAND=${BAND} JOB_NAME=${name} ALLOW_BORROWING=${ALLOW_BORROWING:-true} EXTRA_ENV_VARS=${EXTRA_ENV_VARS:-} bash rl-distill-scripts/scale_train/launch_qwen3_4b_to_30b_distill_st.sh --image ${IMG} 2>&1 | tee -a ${sup}/supervisor.out"
# one janitor for the whole checkpoint root (both bands); idempotent
if ! tmux has-session -t =ckpt-janitor-q3-dist-4b30b 2>/dev/null; then
  tmux new-session -d -s ckpt-janitor-q3-dist-4b30b "CKPT_S3_ROOTS=s3://scale-ml/genai/rl-distill/qwen3-4b-base-distill-ckpts-v1 PRUNE_INTERVAL_MIN=30 \
    bash rl-distill-scripts/scale_train/prune_rolling_checkpoints.sh loop 2>&1 | tee -a .scale_train_supervisors/ckpt-janitor-q3-dist-4b30b.log"
fi
echo "started supervisor rl-${name} (state ${sup}) + ckpt-janitor-q3-dist-4b30b"
