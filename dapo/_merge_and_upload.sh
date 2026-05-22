#!/usr/bin/env bash
# After all 8 shard parquets exist, merge them and upload to HF.
set -euo pipefail
cd /mlx_devbox/users/jason.wei/playground/rl-distill
source .venv/bin/activate
set -a; source .env; set +a

OUT_DIR="${HOME}/verl/data/teacher_gen"
MERGED="${HOME}/verl/data/teacher_27b_step40_n4.parquet"

n=$(ls "${OUT_DIR}"/shard_*.parquet 2>/dev/null | wc -l)
echo "shard parquets found: ${n}/8"
if [ "${n}" -lt 8 ]; then
    echo "not all shards done yet; aborting"
    exit 1
fi

echo ""
echo "=== merging ==="
python3 rl-distill-scripts/data/merge_teacher_shards.py --input_dir "${OUT_DIR}" --output "${MERGED}"

echo ""
echo "=== merged parquet ==="
ls -la "${MERGED}"

echo ""
echo "=== uploading to HF ==="
python3 dapo/_upload_sft_dataset.py
