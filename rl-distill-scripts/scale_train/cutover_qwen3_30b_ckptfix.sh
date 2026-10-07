#!/usr/bin/env bash
# One-shot cutover of a running Qwen3-30B-A3B RL run (start_qwen3_30b_a3b_medium_rl_supervisor.sh) to the cheaper checkpoint
# layout: waits until the running job has fully uploaded its permanent step TARGET_STEP, stops that supervisor + janitor,
# cancels exactly its ScaleTrain job, then restarts the supervisor with ROLLING_CHECKPOINT_ENABLED=False,
# ASYNC_CHECKPOINT_UPLOAD=True and TOTAL_TRAINING_STEPS=500; the new pod restores step TARGET_STEP (Adam + data cursor + ES state).
#   BAND=medium OLD_SUP=q3-30b-rl-med-20261006 TARGET_STEP=350 setsid nohup bash cutover_qwen3_30b_ckptfix.sh > log 2>&1 &
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
: "${BAND:?}" "${OLD_SUP:?}" "${TARGET_STEP:?}"
short=$([ "${BAND}" = medium ] && echo med || echo hard); name="q3-30b-rl-${short}"; sup=".scale_train_supervisors/${OLD_SUP}"
log() { echo "$(date -u +%FT%TZ) [cutover ${name}] $*"; }
log "waiting for permanent step ${TARGET_STEP} upload"
until grep -aqF "[FullCheckpointS3] completed step=${TARGET_STEP}" $(ls -t "${sup}"/pod-logs/* | head -1); do sleep 30; done
sleep 90   # let the post-upload rolling retirement finish
job_id=$(python3 -c "import json; print(json.load(open('${sup}/state.json'))['job_id'])")
log "step ${TARGET_STEP} committed; stopping supervisor and cancelling ${job_id}"
touch "${sup}/STOP"
tmux kill-session -t "rl-${name}" 2>/dev/null || true
tmux kill-session -t "ckpt-janitor-${name}" 2>/dev/null || true
env -u AWS_PROFILE scale-train cancel job "${job_id}" 2>&1 | grep -ioE "cancel[a-z]*|error[^,]*" | sort -u | sed "s/^/  /" || true
sleep 120
log "relaunching (500 steps, permanent-only saves, async upload)"
env BAND="${BAND}" QWEN_SIZE=30b-a3b TOTAL_TRAINING_STEPS=500 ROLLING_CHECKPOINT_ENABLED=False ASYNC_CHECKPOINT_UPLOAD=True SUP_TAG=-ckptfix \
  bash rl-distill-scripts/scale_train/start_qwen3_30b_a3b_medium_rl_supervisor.sh
log "done"
