#!/usr/bin/env bash
# Supervised ScaleTrain resume of the untrained-12B DeepScaleR *medium* DAPO run on FOUR GPUs, from the world_size-4 reshard of
# step 130 (uploaded to …-full-checkpoints/12b-medium-local4 by rl-distill-scripts/reshard_fsdp2_checkpoint.py + full_checkpoint_s3.py upload).
# Same recipe as launch_gemma4_12b_medium_resume.sh (patience 4 migrated from 1, fast update layout, rolling S3 saves every 5 steps,
# permanent every 10, no Hub pushes), just GPUS_PER_INSTANCE=4 and the -local4 prefixes. The supervisor rides out borrowing preemptions
# (Kueue re-queues the Job; the run-file restores the newest S3 checkpoint) and relaunches on external cancel / failure.
#
#   bash rl-distill-scripts/scale_train/start_gemma4_12b_medium_resume_local4_supervisor.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
S3_BASE="${S3_BASE:-s3://scale-ml/genai/rl-distill/gemma4-difficulty-s42-20260819}"
name="${NAME:-g4-12b-med-resume-local4}"
sup=".scale_train_supervisors/${name}-$(date -u +%Y%m%d)"
mkdir -p "${sup}"; rm -f "${sup}/STOP"
tmux new-session -d -s "rl-${name}" "env -u AWS_PROFILE python3 rl-distill-scripts/scale_train/supervise_borrowing_job.py --name ${name} \
  --relaunch-on-cancel --relaunch-on-failure --max-relaunches 30 --quick-cancel-seconds 300 --max-quick-cancels 2 --failure-backoff-seconds 900 \
  --launch-log ${sup}/launch.log --monitor-log ${sup}/monitor.log --state-file ${sup}/state.json --stop-file ${sup}/STOP --pod-log-dir ${sup}/pod-logs \
  --completion-s3-uri ${S3_BASE}-full-checkpoints/12b-medium-local4 --max-completion-step 400 --expected-completion-world-size 4 \
  --completion-best-hf-s3-uri ${S3_BASE}/gemma4-12b-medium-local4 \
  -- env GPUS_PER_INSTANCE=4 CKPT_SUFFIX=-local4 S3_BASE=${S3_BASE} bash rl-distill-scripts/scale_train/launch_gemma4_12b_medium_resume.sh 2>&1 | tee -a ${sup}/supervisor.out"
echo "started supervisor rl-${name} (state ${sup})"
