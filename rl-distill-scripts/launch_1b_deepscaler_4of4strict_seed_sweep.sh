#!/usr/bin/env bash
set -euo pipefail
# Launch 2 LOCAL Gemma-3-1B-PT DeepScaleR-4/4 DAPO RL runs on the *strict-graded* 4/4 split
# (data seeds 42,43), each on its own GPU pair + an isolated local Ray (distinct _temp_dir, no
# dashboard) so both coexist without touching other users' Ray. 12-shot prompt.
#
# Differences vs launch_1b_deepscaler_4of4_seed_sweep.sh:
#   * Data = the STRICT 4/4 split (deepscaler_4of4strict_rl_{train,val200_x16}.parquet, cut from
#     deepscaler_acc4of4_strict.parquet = 9,923 questions the 4B-IT solved 4/4 under strict boxed-only
#     grading). LOCAL files only (not on HF), built by the split step in the summary; error if missing.
#   * Strict grader is the math_verify default now (VERL_MATH_VERIFY_STRICT_BOXED), so no flag needed.
#   * Heavier wandb logging: 100 random VAL traces per eval + 100 random TRAIN rollouts per step
#     (train/generations table, fork addition in dapo_ray_trainer._log_train_generations_to_wandb).
#
#   bash rl-distill-scripts/launch_1b_deepscaler_4of4strict_seed_sweep.sh
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
[ -f .venv/bin/activate ] && source .venv/bin/activate
DATA_DIR="${DATA_DIR:-${HOME}/verl/data}"
TRAIN="${DATA_DIR}/deepscaler_4of4strict_rl_train.parquet"
VAL_FILE="${DATA_DIR}/deepscaler_4of4strict_rl_val200_x16.parquet"
VAL="['${VAL_FILE}']"
for f in "${TRAIN}" "${VAL_FILE}"; do
  [ -f "${f}" ] || { echo "ERROR: missing strict 4/4 split file: ${f}" >&2; exit 1; }
done

SEEDS=(${SEEDS:-42 43})
PAIRS=(${PAIRS:-"2,3" "4,5"})
PORT_BASES=(52000 53000)          # distinct vLLM port floors (see launch_1b_fewshot_seed_sweep.sh note)
TPL="${PROJECT_ROOT}/rl-distill-scripts/data/gemma3_it_fewshot_math.jinja"       # 12-shot prompt
LOGDIR="${LOGDIR:-${HOME}/verl/logs/1b_ds4of4strict_sweep}"; mkdir -p "${LOGDIR}"

# pre-flight: clean slate for exactly these runs (unique ray tag + our own port floors; no other users)
for i in 0 1; do
  seed="${SEEDS[$i]}"; base="${PORT_BASES[$i]}"
  pkill -9 -f "ray_1b_ds4of4s_seed${seed}" 2>/dev/null || true
  for off in 0 100 200; do
    holders="$(ss -ltnHp 2>/dev/null | grep -F ":$((base + off)) " | grep -oP 'pid=\K[0-9]+' | sort -u || true)"
    [ -n "${holders}" ] && kill -9 ${holders} 2>/dev/null || true
  done
  rm -rf "/tmp/ray_1b_ds4of4s_seed${seed}" 2>/dev/null || true
done
sleep 4

for i in 0 1; do
  seed="${SEEDS[$i]}"; pair="${PAIRS[$i]}"
  ng=$(( $(tr -cd ',' <<<"${pair}" | wc -c) + 1 ))
  echo "[seed ${seed}] GPUs ${pair} (${ng} gpu) vllm_port_base ${PORT_BASES[$i]} 12-shot strict-4/4 -> ${LOGDIR}/seed${seed}.log"
  CUDA_VISIBLE_DEVICES="${pair}" \
  N_GPUS_PER_NODE="${ng}" \
  VERL_VLLM_PORT_BASE="${PORT_BASES[$i]}" \
  RAY_TMP="/tmp/ray_1b_ds4of4s_seed${seed}" \
  DATA_SEED="${seed}" SAVE_FREQ=25 \
  MAX_RESPONSE_LENGTH=8192 OVERLONG_BUFFER_LEN=2048 \
  LOG_VAL_GENERATIONS=100 LOG_TRAIN_GENERATIONS=100 \
  TRAIN_FILE="${TRAIN}" VAL_FILES="${VAL}" \
  GEMMA3_CHAT_TEMPLATE_FILE="${TPL}" \
  EXP_NAME="DAPO-gemma3-1b-pt-DeepScaleR-4of4strict-seed${seed}" \
  CKPTS_DIR="${HOME}/verl/ckpts/DAPO/gemma3-1b-pt-deepscaler-4of4strict-seed${seed}" \
  HF_PUSH_REPO="JWei05/DAPO-Gemma3-1B-PT-DeepScaleR-4of4strict-seed${seed}" \
    nohup bash rl-distill-scripts/gemma3_1b_pt_fewshot_math_rl.sh > "${LOGDIR}/seed${seed}.log" 2>&1 &
  echo "[seed ${seed}] driver pid $!"
  sleep 150   # stagger: let seed 1's local Ray FULLY start before seed 2 (45s caused a GCS-startup
              # timeout collision under EFS load — see PROGRESS_LOG 2026-07-22)
done
echo "launched ${#SEEDS[@]} runs -> ${LOGDIR}/"
