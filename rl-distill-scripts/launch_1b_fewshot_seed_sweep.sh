#!/usr/bin/env bash
set -euo pipefail
# Launch 3 LOCAL Gemma-3-1B PT few-shot-math DAPO RL runs with different data seeds, each on its
# own GPU pair + an isolated local Ray (distinct _temp_dir, no dashboard) so all 3 coexist without
# touching other users' Ray. Default partition: seed42->2,3 ; seed43->4,5 ; seed44->6,7.
#
# vLLM port isolation: verl (vllm_async_server._set_vllm_port_floor) computes the vLLM port floor
# from VERL_VLLM_PORT_BASE (default 52000) + replica_rank*100, then patches vLLM's get_open_port to
# allocate sequentially from there — it IGNORES/overwrites VLLM_PORT. So the 3 runs MUST get
# distinct VERL_VLLM_PORT_BASE values, spaced well beyond the ~100-port span one run consumes.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

SEEDS=(${SEEDS:-42 43 44})
PAIRS=(${PAIRS:-"2,3" "4,5" "6,7"})
PORT_BASES=(52000 53000 54000)   # distinct vLLM port floors, 1000 apart (one run spans ~base..base+200)
LOGDIR="${LOGDIR:-${HOME}/verl/logs/1b_fewshot_sweep}"; mkdir -p "${LOGDIR}"

# ---- pre-flight: guarantee a clean slate for exactly this sweep (never touches other users) ----
for i in 0 1 2; do
  seed="${SEEDS[$i]}"; base="${PORT_BASES[$i]}"
  pkill -9 -f "ray_1b_seed${seed}" 2>/dev/null || true                       # ray infra+actors (unique temp-dir tag)
  for off in 0 100 200; do                                                    # free this run's likely floor ports
    holders="$(ss -ltnHp 2>/dev/null | grep -F ":$((base + off)) " | grep -oP 'pid=\K[0-9]+' | sort -u || true)"
    [ -n "${holders}" ] && kill -9 ${holders} 2>/dev/null || true
  done
  rm -rf "/tmp/ray_1b_seed${seed}" 2>/dev/null || true
done
sleep 4

for i in 0 1 2; do
  seed="${SEEDS[$i]}"; pair="${PAIRS[$i]}"
  ng=$(( $(tr -cd ',' <<<"${pair}" | wc -c) + 1 ))
  echo "[seed ${seed}] GPUs ${pair} (${ng} gpu) vllm_port_base ${PORT_BASES[$i]} -> ${LOGDIR}/seed${seed}.log"
  CUDA_VISIBLE_DEVICES="${pair}" \
  N_GPUS_PER_NODE="${ng}" \
  VERL_VLLM_PORT_BASE="${PORT_BASES[$i]}" \
  RAY_TMP="/tmp/ray_1b_seed${seed}" \
  DATA_SEED="${seed}" SAVE_FREQ=25 \
  EXP_NAME="gemma3-1b-pt-fewshot-math-seed${seed}" \
  CKPTS_DIR="${HOME}/verl/ckpts/DAPO/gemma3-1b-pt-fewshot-math-seed${seed}" \
  HF_PUSH_REPO="JWei05/DAPO-Gemma3-1B-PT-FewShotMath-seed${seed}" \
    nohup bash rl-distill-scripts/gemma3_1b_pt_fewshot_math_rl.sh > "${LOGDIR}/seed${seed}.log" 2>&1 &
  echo "[seed ${seed}] driver pid $!"
  sleep 45   # stagger: let each run's local Ray grab free ports + warm the shared HF cache
done
echo "launched ${#SEEDS[@]} runs. logs: ${LOGDIR}/seed{42,43,44}.log"
