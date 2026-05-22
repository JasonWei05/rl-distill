#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-/home/tiger/verl/data/dapo_gemma3_4b_pt_teacher_v7_independent/train_n16}"
LOG_DIR="${OUTPUT_DIR}/logs"

echo "output_dir=${OUTPUT_DIR}"
echo "parquets=$(find "${OUTPUT_DIR}" -maxdepth 1 -type f -name 'shard_*.parquet' 2>/dev/null | wc -l)"
echo "processes=$(ps -ef | grep generate_teacher_data.py | grep -v grep | wc -l)"

if [[ -d "${LOG_DIR}" ]]; then
  grep -RInE 'Traceback|RuntimeError|ValueError|Exception|Generating|Processed prompts|Saved to|Shard [0-9]+:' "${LOG_DIR}" 2>/dev/null | tail -120 || true
else
  echo "missing log dir"
fi
