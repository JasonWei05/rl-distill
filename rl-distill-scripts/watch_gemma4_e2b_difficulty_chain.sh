#!/usr/bin/env bash
# Watchdog for run_gemma4_e2b_difficulty_sequential_local.sh: if the chain exits before printing
# SEQUENTIAL_GEMMA4_E2B_DIFFICULTY_RUNS_DONE, relaunch it (the launcher skips completed bands and
# the trainer resumes the interrupted band from its latest local checkpoint via resume_mode=auto).
# Gives up after MAX_RESTARTS consecutive failures that made no checkpoint progress.
# Several chains may run side by side on one node (e.g. medium on GPUs 0,1 and hard on 2,3): the
# watchdog only tracks launcher processes whose DIFFICULTY_SEQUENCE matches its own, and forwards
# its CUDA_VISIBLE_DEVICES / VERL_VLLM_PORT_BASE / DIFFICULTY_SEQUENCE to relaunches.
#   nohup bash rl-distill-scripts/watch_gemma4_e2b_difficulty_chain.sh > ~/verl/logs/g4_e2b_watchdog.log 2>&1 &
#   DIFFICULTY_SEQUENCE=hard CUDA_VISIBLE_DEVICES=2,3 VERL_VLLM_PORT_BASE=54000 \
#     nohup bash rl-distill-scripts/watch_gemma4_e2b_difficulty_chain.sh > ~/verl/logs/g4_e2b_hard_watchdog.log 2>&1 &
set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${PROJECT_ROOT}"
LAUNCHER=rl-distill-scripts/run_gemma4_e2b_difficulty_sequential_local.sh
export DIFFICULTY_SEQUENCE="${DIFFICULTY_SEQUENCE:-easy medium hard}"
SEQ_TAG="$(tr ' ' '-' <<<"${DIFFICULTY_SEQUENCE}")"
if [ "${SEQ_TAG}" = "easy-medium-hard" ]; then
  CHAIN_LOG="${CHAIN_LOG:-${HOME}/verl/logs/g4_e2b_sequential.log}"
else
  CHAIN_LOG="${CHAIN_LOG:-${HOME}/verl/logs/g4_e2b_${SEQ_TAG}_sequential.log}"
fi
ROOT="${SEQUENTIAL_ROOT:-${HOME}/gemma4-e2b-difficulty-s42}"
MAX_RESTARTS="${MAX_RESTARTS:-3}"
restarts=0
progress_marker() {
  for b in ${DIFFICULTY_SEQUENCE}; do cat "${ROOT}/${b}/ckpts/latest_checkpointed_iteration.txt" 2>/dev/null; done | tr '\n' ','
}
# A launcher belongs to this chain iff its environment carries the same DIFFICULTY_SEQUENCE.
chain_running() {
  local pid
  for pid in $(pgrep -f "bash ${LAUNCHER}"); do
    if tr '\0' '\n' < "/proc/${pid}/environ" 2>/dev/null | grep -qx "DIFFICULTY_SEQUENCE=${DIFFICULTY_SEQUENCE}"; then return 0; fi
  done
  return 1
}
last_progress="$(progress_marker)"
while true; do
  # Any new checkpoint resets the consecutive-failure budget.
  cur="$(progress_marker)"
  if [ "${cur}" != "${last_progress}" ]; then last_progress="${cur}"; restarts=0; fi
  if chain_running; then sleep 120; continue; fi
  if grep -q "SEQUENTIAL_GEMMA4_E2B_DIFFICULTY_RUNS_DONE" "${CHAIN_LOG}" 2>/dev/null; then
    echo "[$(date +%F' '%T)] WATCHDOG_CHAIN_COMPLETE"; exit 0
  fi
  before="$(progress_marker)"
  if [ "${restarts}" -ge "${MAX_RESTARTS}" ]; then
    echo "[$(date +%F' '%T)] WATCHDOG_GIVING_UP after ${restarts} restarts without checkpoint progress" >&2; exit 1
  fi
  restarts=$((restarts+1))
  ts="$(date +%Y%m%d-%H%M%S)"
  [ -f "${CHAIN_LOG}" ] && mv "${CHAIN_LOG}" "${CHAIN_LOG%.log}.failed-${ts}.log"
  echo "[$(date +%F' '%T)] WATCHDOG_RESTART #${restarts} (progress before: ${before:-none})"
  # Do not `ray stop` here: with several chains on one node that would kill the other chains' Ray
  # clusters too. Kill only this chain's leftover trainer processes (matched by RAY_TEMP_DIR).
  for b in ${DIFFICULTY_SEQUENCE}; do pkill -f "ray_g4_e2b_${b}" 2>/dev/null || true; done
  sleep 20
  nohup bash "${LAUNCHER}" > "${CHAIN_LOG}" 2>&1 &
  sleep 600
done
