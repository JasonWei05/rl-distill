#!/usr/bin/env bash
# Queue for the 9 Qwen on-policy distillation runs: starts start_qwen3_onpolicy_distill_supervisor.sh per SPEC in order,
# keeping at most MAX_GPUS GPUs in flight (a SPEC is in flight while its supervisor tmux session rl-q3-opd-<SPEC> lives;
# the supervisor exits once the run's completion receipt + best_hf are on S3). Specs already started are skipped.
#   tmux new -d -s q3-opd-queue "bash rl-distill-scripts/scale_train/run_qwen3_onpolicy_distill_queue.sh"   # stop: touch $STATE/STOP
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
MAX_GPUS="${MAX_GPUS:-8}"; RUN_TAG="${RUN_TAG:-pgk1-v1}"
SPECS="${SPECS:-q4bm-to-q1p7b q4bh-to-q1p7b q4bvh-to-q1p7b q30bm-to-q1p7b q30bh-to-q1p7b q30bvh-to-q1p7b q30bm-to-q4b q30bh-to-q4b q30bvh-to-q4b}"
STATE=".scale_train_supervisors/q3-opd-queue-${RUN_TAG}"; mkdir -p "${STATE}"
gpus_of() { case "${1##*-to-}" in q1p7b) echo 2 ;; q4b) echo 4 ;; esac; }
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "${STATE}/queue.log"; }
while [ ! -e "${STATE}/STOP" ]; do
  used=0; pending=""
  for s in ${SPECS}; do
    if tmux has-session -t "=rl-q3-opd-${s}" 2>/dev/null; then used=$((used + $(gpus_of "${s}")))
    elif [ ! -e "${STATE}/started.${s}" ]; then pending+=" ${s}"; fi
  done
  [ -n "${pending}" ] || { log "QUEUE_DONE all specs started"; break; }
  next="${pending# }"; next="${next%% *}"; need="$(gpus_of "${next}")"
  if (( used + need <= MAX_GPUS )); then
    log "START ${next} (gpus ${need}, in flight ${used} -> $((used + need)))"
    SPEC="${next}" RUN_TAG="${RUN_TAG}" bash rl-distill-scripts/scale_train/start_qwen3_onpolicy_distill_supervisor.sh >> "${STATE}/queue.log" 2>&1
    touch "${STATE}/started.${next}"
    sleep 30; continue
  fi
  sleep 120
done
