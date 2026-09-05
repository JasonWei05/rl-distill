#!/usr/bin/env bash
# Run the Gemma 4 E2B MEDIUM and HARD difficulty bands side by side on one 4-GPU node, 2 GPUs each
# (easy is assumed to be running elsewhere). Each band is its own single-band instance of
# run_gemma4_e2b_difficulty_sequential_local.sh (same recipe, HF push every 10 steps, early stopping)
# plus its own watchdog. The two instances are isolated by CUDA_VISIBLE_DEVICES, per-band Ray temp
# dirs, and disjoint vLLM port bases.
#
#   nohup bash rl-distill-scripts/run_gemma4_e2b_medium_hard_parallel_local.sh > ~/verl/logs/g4_e2b_medium_hard_launch.log 2>&1 &
#
# Logs: ~/verl/logs/g4_e2b_{medium,hard}_sequential.log (chain), ~/verl/logs/g4_e2b_{medium,hard}_watchdog.log,
#       ~/gemma4-e2b-difficulty-s42/logs/{medium,hard}.log (trainer).
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${PROJECT_ROOT}"
LOGS="${HOME}/verl/logs"; mkdir -p "${LOGS}"
LAUNCHER=rl-distill-scripts/run_gemma4_e2b_difficulty_sequential_local.sh
WATCHDOG=rl-distill-scripts/watch_gemma4_e2b_difficulty_chain.sh

MEDIUM_GPUS="${MEDIUM_GPUS:-0,1}"; HARD_GPUS="${HARD_GPUS:-2,3}"
MEDIUM_PORT_BASE="${MEDIUM_PORT_BASE:-52000}"; HARD_PORT_BASE="${HARD_PORT_BASE:-54000}"
# Seconds between starting the two chains, so their model/dataset downloads and Ray bring-up are staggered.
STAGGER_SECONDS="${STAGGER_SECONDS:-120}"

# START_CHAINS=0 starts only the watchdogs (e.g. after verifying a manual launch by hand);
# START_WATCHDOGS=0 starts only the chains (verify a fresh config before arming auto-restart).
START_CHAINS="${START_CHAINS:-1}"; START_WATCHDOGS="${START_WATCHDOGS:-1}"

launch_band() {  # band gpus port_base
  local band="$1" gpus="$2" port="$3"
  if [ "${START_CHAINS}" = 1 ]; then
    echo "[$(date +%F' '%T)] LAUNCH band=${band} gpus=${gpus} vllm_port_base=${port}"
    DIFFICULTY_SEQUENCE="${band}" CUDA_VISIBLE_DEVICES="${gpus}" VERL_VLLM_PORT_BASE="${port}" \
      nohup bash "${LAUNCHER}" >> "${LOGS}/g4_e2b_${band}_sequential.log" 2>&1 &
    echo "  chain pid=$!"
    # The watchdog polls the chain via /proc environ; give the chain a moment to appear first.
    sleep 5
  fi
  if [ "${START_WATCHDOGS}" = 1 ]; then
    DIFFICULTY_SEQUENCE="${band}" CUDA_VISIBLE_DEVICES="${gpus}" VERL_VLLM_PORT_BASE="${port}" \
      CHAIN_LOG="${LOGS}/g4_e2b_${band}_sequential.log" \
      nohup bash "${WATCHDOG}" >> "${LOGS}/g4_e2b_${band}_watchdog.log" 2>&1 &
    echo "  watchdog band=${band} pid=$!"
  fi
}

launch_band medium "${MEDIUM_GPUS}" "${MEDIUM_PORT_BASE}"
[ "${START_CHAINS}" = 1 ] && sleep "${STAGGER_SECONDS}"
launch_band hard "${HARD_GPUS}" "${HARD_PORT_BASE}"
echo "MEDIUM_HARD_PARALLEL_LAUNCHED"
