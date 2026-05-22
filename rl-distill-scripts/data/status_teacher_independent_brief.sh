#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-/home/tiger/verl/data/dapo_gemma3_4b_pt_teacher_v7_independent/train_n16}"
LOG_DIR="${OUTPUT_DIR}/logs"

echo "output_dir=${OUTPUT_DIR}"
echo "parquets=$(find "${OUTPUT_DIR}" -maxdepth 1 -type f -name 'shard_*.parquet' 2>/dev/null | wc -l)"
echo "processes=$(ps -ef | grep generate_teacher_data.py | grep -v grep | wc -l)"

if [[ -d "${LOG_DIR}" ]]; then
  for log in "${LOG_DIR}"/shard_*.log; do
    [[ -f "${log}" ]] || continue
    name="$(basename "${log}")"
    if grep -q "Saved to" "${log}"; then
      status="$(grep "Saved to" "${log}" | tail -1)"
    elif grep -q "Processed prompts" "${log}"; then
      status="$(grep -ao 'Processed prompts:[^\r]*' "${log}" | tail -1 | cut -c1-180)"
    elif grep -q "Generating" "${log}"; then
      status="$(grep "Generating" "${log}" | tail -1)"
    elif grep -q "Traceback\\|RuntimeError\\|ValueError" "${log}"; then
      status="$(grep -E "Traceback|RuntimeError|ValueError" "${log}" | tail -1)"
    else
      status="starting"
    fi
    echo "${name}: ${status}"
  done
fi
