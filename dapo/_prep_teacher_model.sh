#!/usr/bin/env bash
# Pre-download the step_000040 subfolder of JWei05/dapo-gemma3-27b-it to /tmp.
# vLLM's LLM(revision=...) treats revision as a git branch/tag, so it can't
# target a subfolder. Solution: download locally, pass the local path.
set -euo pipefail
cd /mlx_devbox/users/jason.wei/playground/rl-distill
source .venv/bin/activate
set -a; source .env; set +a
# hf_transfer isn't installed in the shared venv; fall back to plain requests.
export HF_HUB_ENABLE_HF_TRANSFER=0

DST=/tmp/teacher_27b_step40
if [ -f "${DST}/config.json" ] && ls "${DST}"/model-*.safetensors 2>/dev/null | head -1 > /dev/null; then
    echo "model already present at ${DST}"
    du -sh "${DST}"
    exit 0
fi

mkdir -p "${DST}"
python3 - <<PY
import os
from huggingface_hub import snapshot_download
d = snapshot_download(
    repo_id="JWei05/dapo-gemma3-27b-it",
    repo_type="model",
    allow_patterns="step_000040/*",
    local_dir="/tmp/teacher_dl",
)
# The safetensors etc. live under /tmp/teacher_dl/step_000040/.
# Move them up to DST so a bare HF path works.
import shutil
src = os.path.join("/tmp/teacher_dl", "step_000040")
for name in os.listdir(src):
    s = os.path.join(src, name); t = os.path.join("${DST}", name)
    if os.path.exists(t): continue
    shutil.move(s, t)
print("files in ${DST}:")
for n in sorted(os.listdir("${DST}")):
    print(" ", n)
PY
du -sh "${DST}"
