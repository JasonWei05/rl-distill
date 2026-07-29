#!/usr/bin/env bash
set -euo pipefail
# Build ALL data for the few-shot math RL runs from HF (works in a fresh container or on devbox).
# Idempotent: skips files that already exist. Data is plain verl format; the few-shot prompt is
# applied at train/val time via the chat template (data/gemma3_it_fewshot_math.jinja).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${DATA_DIR:-${HOME}/verl/data}"
PY="${PYTHON:-python3}"
mkdir -p "${DATA_DIR}"

# 1. DAPO-Math-17k source (to split off a 100-row val)
DAPO_SRC="${DATA_DIR}/dapo_17k_source_train.parquet"
[ -f "${DAPO_SRC}" ] || wget -qO "${DAPO_SRC}" \
    "https://huggingface.co/datasets/JWei05/DAPO-17.4k/resolve/main/data/train.parquet?download=true"

# 2. base val parquets that need HF conversion (reuse the repo's tested converters)
[ -f "${DATA_DIR}/math__olympiadbench.parquet" ] || \
    "${PY}" "${SCRIPT_DIR}/convert_olympiadbench.py" --output_path "${DATA_DIR}/math__olympiadbench.parquet"
[ -f "${DATA_DIR}/math__minervamath.parquet" ] || \
    "${PY}" "${SCRIPT_DIR}/convert_minervamath.py" --output_path "${DATA_DIR}/math__minervamath.parquet"
[ -f "${DATA_DIR}/math__aime2025_30.parquet" ] || \
    "${PY}" "${SCRIPT_DIR}/convert_aime.py" --dataset MathArena/aime_2025 --year 2025 \
        --output_path "${DATA_DIR}/math__aime2025_30.parquet"
[ -f "${DATA_DIR}/math__aime2026_30.parquet" ] || \
    "${PY}" "${SCRIPT_DIR}/convert_aime.py" --dataset MathArena/aime_2026 --year 2026 \
        --output_path "${DATA_DIR}/math__aime2026_30.parquet"

# 3. DAPO split (val 100) + MATH500/GSM8K/BeyondAIME from HF + all repeats
"${PY}" "${SCRIPT_DIR}/build_math_rl_data.py" --data-dir "${DATA_DIR}" --dapo-source "${DAPO_SRC}"

echo "prepare_fewshot_math_rl_data: DONE -> ${DATA_DIR}"
