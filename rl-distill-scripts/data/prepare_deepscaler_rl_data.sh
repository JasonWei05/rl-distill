#!/usr/bin/env bash
set -euo pipefail
# Download the DeepScaleR RL train/val parquets into DATA_DIR (idempotent). These were built by
# build_deepscaler_rl_data.py and uploaded to JWei05/DeepScaleR-RL so the local 1B run and the
# ScaleTrain 4B run train/val on the EXACT same split. Same plain verl format as the DAPO few-shot
# data; the 12-shot prompt is applied at train/val time via the chat template.
DATA_DIR="${DATA_DIR:-${HOME}/verl/data}"
PY="${PYTHON:-python3}"
REPO="${DEEPSCALER_REPO:-JWei05/DeepScaleR-RL}"
mkdir -p "${DATA_DIR}"
"${PY}" - "${REPO}" "${DATA_DIR}" <<'PYEOF'
import os, shutil, sys
repo, ddir = sys.argv[1], sys.argv[2]
files = ["deepscaler_rl_train.parquet", "deepscaler_rl_val200.parquet", "deepscaler_rl_val200_x16.parquet"]
missing = [f for f in files if not os.path.exists(os.path.join(ddir, f))]
if not missing:
    print("  all present, skip download"); sys.exit(0)
from huggingface_hub import hf_hub_download  # only needed when a file is actually missing
for f in missing:
    p = hf_hub_download(repo_id=repo, filename=f, repo_type="dataset")
    shutil.copy(p, os.path.join(ddir, f))
    print("  downloaded", f)
PYEOF
echo "prepare_deepscaler_rl_data: DONE -> ${DATA_DIR}"
