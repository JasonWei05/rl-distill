#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Output directory for prepared data
DATA_DIR="${DATA_DIR:-${HOME}/verl/data}"
OVERWRITE="${OVERWRITE:-0}"
mkdir -p "${DATA_DIR}"

TRAIN_FILE="${DATA_DIR}/dapo-math-17k.parquet"
DAPO_OPENMATH2_TRAIN_FILE="${DATA_DIR}/dapo_openmath2_mix_train.parquet"
DAPO_OPENMATH2_VAL_FILE="${DATA_DIR}/dapo_openmath2_mix_val.parquet"

# Intermediate files (downloaded, used to produce val sets)
AIME2024_8X_FILE="${DATA_DIR}/math__aime2024_repeated_8x_240.parquet"
MATH500_FILE="${DATA_DIR}/math__math_500.parquet"
AIME2025_FILE="${DATA_DIR}/math__aime2025_30.parquet"
AIME2026_FILE="${DATA_DIR}/math__aime2026_30.parquet"
OLYMPIADBENCH_FILE="${DATA_DIR}/math__olympiadbench.parquet"
MINERVAMATH_FILE="${DATA_DIR}/math__minervamath.parquet"

# Final val files (referenced by training script)
AIME2024_32X_FILE="${DATA_DIR}/math__aime2024_repeated_32x_960.parquet"
AIME2025_32X_FILE="${DATA_DIR}/math__aime2025_repeated_32x_960.parquet"
AIME2026_32X_FILE="${DATA_DIR}/math__aime2026_repeated_32x_960.parquet"
MATH500_2X_FILE="${DATA_DIR}/math__math_500_repeated_2x_1000.parquet"
OLYMPIADBENCH_2X_FILE="${DATA_DIR}/math__olympiadbench_repeated_2x.parquet"
MINERVAMATH_4X_FILE="${DATA_DIR}/math__minervamath_repeated_4x.parquet"
GSM8K_FILE="${DATA_DIR}/math__gsm8k_test.parquet"

echo "=== Dataset Preparation ==="
echo "Data directory: ${DATA_DIR}"
echo ""

# --- Train data ---
if [[ ! -f "${TRAIN_FILE}" || "${OVERWRITE}" -eq 1 ]]; then
  echo "Downloading dapo-math-17k training set..."
  wget -O "${TRAIN_FILE}" \
    "https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k/resolve/main/data/dapo-math-17k.parquet?download=true"
else
  echo "Training file found: ${TRAIN_FILE}"
fi

# --- DAPO/OpenMathInstruct2 train/val split ---
DATA_DIR="${DATA_DIR}" OVERWRITE="${OVERWRITE}" bash "${SCRIPT_DIR}/prepare_dapo_openmath2_mix_split.sh"

# --- AIME 2024 ---
if [[ ! -f "${AIME2024_8X_FILE}" || "${OVERWRITE}" -eq 1 ]]; then
  echo "Downloading AIME 2024 8x dataset..."
  wget -O "${AIME2024_8X_FILE}" \
    "https://huggingface.co/datasets/LLM360/guru-RL-92k/resolve/main/offline_eval/math__aime_repeated_8x_240.parquet"
fi
if [[ ! -f "${AIME2024_32X_FILE}" || "${OVERWRITE}" -eq 1 ]]; then
  echo "Creating AIME 2024 32x..."
  python3 "${SCRIPT_DIR}/filter_test_dataset_keys.py" --input_file "${AIME2024_8X_FILE}"
  python3 "${SCRIPT_DIR}/duplicate_rows.py" \
    --input_path "${AIME2024_8X_FILE}" \
    --save_path "${AIME2024_32X_FILE}" \
    --repeat_times 4
fi

# --- AIME 2025 ---
if [[ ! -f "${AIME2025_FILE}" || "${OVERWRITE}" -eq 1 ]]; then
  echo "Converting AIME 2025 from MathArena/aime_2025..."
  python3 "${SCRIPT_DIR}/convert_aime.py" --dataset MathArena/aime_2025 --year 2025 --output_path "${AIME2025_FILE}"
fi
if [[ ! -f "${AIME2025_32X_FILE}" || "${OVERWRITE}" -eq 1 ]]; then
  echo "Creating AIME 2025 32x..."
  python3 "${SCRIPT_DIR}/duplicate_rows.py" \
    --input_path "${AIME2025_FILE}" \
    --save_path "${AIME2025_32X_FILE}" \
    --repeat_times 32
fi

# --- AIME 2026 ---
if [[ ! -f "${AIME2026_FILE}" || "${OVERWRITE}" -eq 1 ]]; then
  echo "Converting AIME 2026 from MathArena/aime_2026..."
  python3 "${SCRIPT_DIR}/convert_aime.py" --dataset MathArena/aime_2026 --year 2026 --output_path "${AIME2026_FILE}"
fi
if [[ ! -f "${AIME2026_32X_FILE}" || "${OVERWRITE}" -eq 1 ]]; then
  echo "Creating AIME 2026 32x..."
  python3 "${SCRIPT_DIR}/duplicate_rows.py" \
    --input_path "${AIME2026_FILE}" \
    --save_path "${AIME2026_32X_FILE}" \
    --repeat_times 32
fi

# --- MATH500 ---
if [[ ! -f "${MATH500_FILE}" || "${OVERWRITE}" -eq 1 ]]; then
  echo "Downloading MATH500 dataset..."
  wget -O "${MATH500_FILE}" \
    "https://huggingface.co/datasets/LLM360/guru-RL-92k/resolve/main/offline_eval/math__math_500.parquet"
fi
if [[ ! -f "${MATH500_2X_FILE}" || "${OVERWRITE}" -eq 1 ]]; then
  echo "Creating MATH500 2x..."
  python3 "${SCRIPT_DIR}/filter_test_dataset_keys.py" --input_file "${MATH500_FILE}"
  python3 "${SCRIPT_DIR}/duplicate_rows.py" \
    --input_path "${MATH500_FILE}" \
    --save_path "${MATH500_2X_FILE}" \
    --repeat_times 2
fi

# --- OlympiadBench ---
if [[ ! -f "${OLYMPIADBENCH_FILE}" || "${OVERWRITE}" -eq 1 ]]; then
  echo "Converting OlympiadBench from math-ai/OlympiadBench..."
  python3 "${SCRIPT_DIR}/convert_olympiadbench.py" --output_path "${OLYMPIADBENCH_FILE}"
fi
if [[ ! -f "${OLYMPIADBENCH_2X_FILE}" || "${OVERWRITE}" -eq 1 ]]; then
  echo "Creating OlympiadBench 2x..."
  python3 "${SCRIPT_DIR}/duplicate_rows.py" \
    --input_path "${OLYMPIADBENCH_FILE}" \
    --save_path "${OLYMPIADBENCH_2X_FILE}" \
    --repeat_times 2
fi

# --- MinervaMAth ---
if [[ ! -f "${MINERVAMATH_FILE}" || "${OVERWRITE}" -eq 1 ]]; then
  echo "Converting MinervaMAth from math-ai/minervamath..."
  python3 "${SCRIPT_DIR}/convert_minervamath.py" --output_path "${MINERVAMATH_FILE}"
fi
if [[ ! -f "${MINERVAMATH_4X_FILE}" || "${OVERWRITE}" -eq 1 ]]; then
  echo "Creating MinervaMAth 4x..."
  python3 "${SCRIPT_DIR}/duplicate_rows.py" \
    --input_path "${MINERVAMATH_FILE}" \
    --save_path "${MINERVAMATH_4X_FILE}" \
    --repeat_times 4
fi

# --- GSM8K ---
if [[ ! -f "${GSM8K_FILE}" || "${OVERWRITE}" -eq 1 ]]; then
  echo "Converting GSM8K from openai/gsm8k..."
  python3 "${SCRIPT_DIR}/convert_gsm8k.py" --output_path "${GSM8K_FILE}"
fi

# --- Set data_source labels on final val files ---
echo ""
echo "Setting data_source labels on val files..."
python3 -c "
import pandas as pd, os
updates = [
    ('${AIME2024_32X_FILE}', 'aime2024'),
    ('${AIME2025_32X_FILE}', 'aime2025'),
    ('${AIME2026_32X_FILE}', 'aime2026'),
    ('${MATH500_2X_FILE}', 'math500'),
    ('${OLYMPIADBENCH_2X_FILE}', 'olympiadbench'),
    ('${MINERVAMATH_4X_FILE}', 'minervamath'),
    ('${GSM8K_FILE}', 'gsm8k'),
]
for path, ds in updates:
    if not os.path.exists(path):
        print(f'SKIP (not found): {path}')
        continue
    df = pd.read_parquet(path)
    df['data_source'] = ds
    df.to_parquet(path, index=False)
    print(f'  {os.path.basename(path)}: data_source -> {ds}')
"

echo ""
echo "=== Summary ==="
echo "Train file:        ${TRAIN_FILE}"
echo "OpenMath2 train:   ${DAPO_OPENMATH2_TRAIN_FILE}"
echo "OpenMath2 val:     ${DAPO_OPENMATH2_VAL_FILE}"
echo "AIME 2024 32x:     ${AIME2024_32X_FILE}"
echo "AIME 2025 32x:     ${AIME2025_32X_FILE}"
echo "AIME 2026 32x:     ${AIME2026_32X_FILE}"
echo "MATH500 2x:        ${MATH500_2X_FILE}"
echo "OlympiadBench 2x:  ${OLYMPIADBENCH_2X_FILE}"
echo "MinervaMAth 4x:    ${MINERVAMATH_4X_FILE}"
echo "GSM8K test:        ${GSM8K_FILE}"
