#!/usr/bin/env bash
set -euo pipefail

NODE_ID="${1:?usage: status_teacher_dp8_torchrun_node.sh NODE_ID}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/tiger/verl/data/dapo_gemma3_4b_pt_teacher_v6_torchrun_dp8/train_n16}"
LOG="${OUTPUT_DIR}/logs/node${NODE_ID}_torchrun.log"

echo "log=${LOG}"
if [[ -f "${LOG}" ]]; then
  tail -n 220 "${LOG}"
else
  echo "missing log"
fi

echo "parquets=$(find "${OUTPUT_DIR}" -maxdepth 1 -type f -name '*.parquet' 2>/dev/null | wc -l)"
echo "processes:"
ps -ef | grep -E 'torchrun|generate_teacher_data.py' | grep -v grep || true
