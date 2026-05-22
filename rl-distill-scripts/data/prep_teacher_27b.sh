#!/usr/bin/env bash
# Snapshot-download JWei05/dapo-gemma3-27b-it/step_000040/* into a flat local dir
# so the on-policy distillation teacher can be loaded as a normal HF model path.
# Idempotent — safe to re-run.
set -euo pipefail

REVISION="${REVISION:-step_000040}"
DEST="${DEST:-${HOME}/verl/models/dapo-gemma3-27b-it-${REVISION}}"

if [ -f "${DEST}/config.json" ]; then
    echo "[prep] already present at ${DEST}"
    exit 0
fi

mkdir -p "${DEST}"
echo "[prep] downloading JWei05/dapo-gemma3-27b-it/${REVISION}/* -> ${DEST}"
python3 - <<PY
import os, shutil, glob
from huggingface_hub import snapshot_download
src = snapshot_download(
    repo_id="JWei05/dapo-gemma3-27b-it",
    allow_patterns=["${REVISION}/*"],
)
sub = os.path.join(src, "${REVISION}")
for f in glob.glob(os.path.join(sub, "*")):
    dst = os.path.join("${DEST}", os.path.basename(f))
    if not os.path.exists(dst):
        os.symlink(f, dst)
print("ok ->", "${DEST}")
PY

ls -la "${DEST}" | head -20
