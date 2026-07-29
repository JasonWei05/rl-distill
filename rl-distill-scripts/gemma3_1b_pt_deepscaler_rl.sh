#!/usr/bin/env bash
set -euo pipefail
# Gemma 3 1B PT — DeepScaleR DAPO RL, LOCAL single-machine run on 2 GPUs (default 2,4).
# Identical recipe to gemma3_1b_pt_fewshot_math_rl.sh (unified 12-shot prompt, 20k response + 4k
# overlong buffer, val=train sampling, SAVE_FREQ=25 + HF push, wandb) — only the dataset changes:
# trains on the DeepScaleR train split (40.1k) with a 200-question x16 val (pass@16/maj@16/mean@16).
#
#   bash rl-distill-scripts/gemma3_1b_pt_deepscaler_rl.sh
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${HERE}/.." && pwd)"
DATA_DIR="${DATA_DIR:-${HOME}/verl/data}"

# Activate the FSDP2 venv so the data prep (huggingface_hub) runs under the right interpreter.
if [ -f "${PROJECT_ROOT}/.venv/bin/activate" ]; then source "${PROJECT_ROOT}/.venv/bin/activate"; fi
# Data (idempotent; downloads from JWei05/DeepScaleR-RL if not already local).
DATA_DIR="${DATA_DIR}" bash "${HERE}/data/prepare_deepscaler_rl_data.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,4}"
export HF_PUSH_REPO="${HF_PUSH_REPO:-JWei05/DAPO-Gemma3-1B-PT-DeepScaleR}"
export EXP_NAME="${EXP_NAME:-DAPO-gemma3-1b-pt-DeepScaleR-$(date +%Y%m%d-%H%M)}"
export TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/deepscaler_rl_train.parquet}"
export VAL_FILES="${VAL_FILES:-['${DATA_DIR}/deepscaler_rl_val200_x16.parquet']}"
# Unified 12-shot prompt (the core script's default; set explicitly for clarity).
export GEMMA3_CHAT_TEMPLATE_FILE="${GEMMA3_CHAT_TEMPLATE_FILE:-${PROJECT_ROOT}/rl-distill-scripts/data/gemma3_it_fewshot_math.jinja}"

exec bash "${HERE}/gemma3_1b_pt_fewshot_math_rl.sh" "$@"
