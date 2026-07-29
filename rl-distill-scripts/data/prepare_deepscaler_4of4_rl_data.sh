#!/usr/bin/env bash
set -euo pipefail
# Download the DeepScaleR 4/4-subset RL train/val parquets into DATA_DIR (idempotent). Built from the
# questions Gemma-3-4B-IT solved 4/4 (see PROGRESS_LOG.md), uploaded to JWei05/DeepScaleR-4of4-RL so
# the local 1B seeds and the ScaleTrain 4B run use the EXACT same split. Same plain verl format; the
# 12-shot prompt is applied at train/val time via the chat template.
DATA_DIR="${DATA_DIR:-${HOME}/verl/data}"
PY="${PYTHON:-python3}"
REPO="${DEEPSCALER_4OF4_REPO:-JWei05/DeepScaleR-4of4-RL}"
mkdir -p "${DATA_DIR}"
"${PY}" - "${REPO}" "${DATA_DIR}" <<'PYEOF'
import os, shutil, sys
repo, ddir = sys.argv[1], sys.argv[2]
files = ["deepscaler_4of4_rl_train.parquet", "deepscaler_4of4_rl_val200.parquet", "deepscaler_4of4_rl_val200_x16.parquet"]
missing = [f for f in files if not os.path.exists(os.path.join(ddir, f))]
if not missing:
    print("  all present, skip download"); sys.exit(0)
from huggingface_hub import hf_hub_download  # only needed when a file is actually missing
for f in missing:
    p = hf_hub_download(repo_id=repo, filename=f, repo_type="dataset")
    shutil.copy(p, os.path.join(ddir, f)); print("  downloaded", f)
PYEOF
echo "prepare_deepscaler_4of4_rl_data: DONE -> ${DATA_DIR}"
