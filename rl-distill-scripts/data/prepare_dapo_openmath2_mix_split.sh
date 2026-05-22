#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${DATA_DIR:-${HOME}/verl/data}"
OVERWRITE="${OVERWRITE:-0}"
mkdir -p "${DATA_DIR}"

TRAIN_FILE="${DATA_DIR}/dapo_openmath2_mix_train.parquet"
VAL_FILE="${DATA_DIR}/dapo_openmath2_mix_val.parquet"
SOURCE_FILE="${DATA_DIR}/dapo_openmath2_mix.parquet"
REPO_URL="https://huggingface.co/datasets/JWei05/DAPO-OpenMathInstruct2-34k/resolve/main"

if [[ ! -f "${TRAIN_FILE}" || "${OVERWRITE}" -eq 1 ]]; then
  echo "Downloading DAPO/OpenMathInstruct2 train split..."
  wget -O "${TRAIN_FILE}" "${REPO_URL}/data/train.parquet?download=true"
else
  echo "DAPO/OpenMathInstruct2 train split found: ${TRAIN_FILE}"
fi

if [[ ! -f "${VAL_FILE}" || "${OVERWRITE}" -eq 1 ]]; then
  echo "Downloading DAPO/OpenMathInstruct2 validation split..."
  wget -O "${VAL_FILE}" "${REPO_URL}/data/validation.parquet?download=true"
else
  echo "DAPO/OpenMathInstruct2 validation split found: ${VAL_FILE}"
fi

if [[ ! -s "${TRAIN_FILE}" || ! -s "${VAL_FILE}" ]]; then
  if [[ -f "${SOURCE_FILE}" ]]; then
    echo "Download failed or produced empty split files; regenerating from ${SOURCE_FILE}..."
    python3 "${SCRIPT_DIR}/split_dapo_openmath2_mix.py" \
      --input "${SOURCE_FILE}" \
      --output_train "${TRAIN_FILE}" \
      --output_val "${VAL_FILE}"
  else
    echo "Missing split files and source file: ${SOURCE_FILE}" >&2
    exit 1
  fi
fi

echo "DAPO/OpenMathInstruct2 train: ${TRAIN_FILE}"
echo "DAPO/OpenMathInstruct2 val:   ${VAL_FILE}"
