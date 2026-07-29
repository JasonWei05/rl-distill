#!/usr/bin/env bash
set -euo pipefail
# Download the DeepScaleR STRICT-4/4 RL train/val parquets into DATA_DIR (idempotent). Cut from
# deepscaler_acc4of4_strict.parquet (9,923 questions Gemma-3-4B-IT solved 4/4 under the STRICT
# boxed-only grader), shuffle seed 42 -> train 9,723 / val 200x16. Uploaded to
# JWei05/DeepScaleR-4of4-strict-RL so the local sweep and ScaleTrain use the EXACT same split.
DATA_DIR="${DATA_DIR:-${HOME}/verl/data}"
PY="${PYTHON:-python3}"
REPO="${DEEPSCALER_4OF4STRICT_REPO:-JWei05/DeepScaleR-4of4-strict-RL}"
mkdir -p "${DATA_DIR}"
"${PY}" - "${REPO}" "${DATA_DIR}" <<'PYEOF'
import os, shutil, sys
repo, ddir = sys.argv[1], sys.argv[2]
files = ["deepscaler_4of4strict_rl_train.parquet", "deepscaler_4of4strict_rl_val200_x16.parquet"]
missing = [f for f in files if not os.path.exists(os.path.join(ddir, f))]
if not missing:
    print("  all present, skip download"); sys.exit(0)
from huggingface_hub import hf_hub_download  # only needed when a file is actually missing
for f in missing:
    p = hf_hub_download(repo_id=repo, filename=f, repo_type="dataset")
    shutil.copy(p, os.path.join(ddir, f)); print("  downloaded", f)
PYEOF
echo "prepare_deepscaler_4of4strict_rl_data: DONE -> ${DATA_DIR}"
