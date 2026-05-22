#!/usr/bin/env bash
set -euo pipefail

# Dense Gemma3 4B PT SFT checkpoint, trained with the same DAPO settings as
# gemma3_4b_it_fsdp2_20k.sh. The HF repo stores the model under step_000250.
if [ -z "${MODEL_PATH:-}" ]; then
    if [ -f .env ]; then
        set -a
        source .env
        set +a
    fi
    DENSE_SFT_REPO="${DENSE_SFT_REPO:-JWei05/gemma3-4b-pt-sft-nemotron-cascade2-16k}"
    DENSE_SFT_REVISION="${DENSE_SFT_REVISION:-main}"
    DENSE_SFT_CACHE_DIR="${DENSE_SFT_CACHE_DIR:-/tmp/hf-dense-sft-rl-cache}"
    DENSE_SNAPSHOT_DIR="$(python3 -c "
from huggingface_hub import snapshot_download
print(snapshot_download(
    repo_id='${DENSE_SFT_REPO}',
    revision='${DENSE_SFT_REVISION}',
    repo_type='model',
    cache_dir='${DENSE_SFT_CACHE_DIR}',
    allow_patterns=['step_000250/*'],
))
")"
    export MODEL_PATH="${DENSE_SNAPSHOT_DIR}/step_000250"
fi

GEMMA3_PROCESSOR_BASE_REPO="${GEMMA3_PROCESSOR_BASE_REPO:-google/gemma-3-4b-pt}" \
    bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_ops/ensure_gemma3_processor_files.sh" "${MODEL_PATH}"

export EXP_NAME="${EXP_NAME:-DAPO-Gemma3-4B-PT-SFT-NC2-16k-FSDP2-RL-$(date +%Y%m%d-%H%M)}"
export HF_PUSH_REPO="${HF_PUSH_REPO:-JWei05/dapo-gemma3-4b-pt-sft-nc2-16k}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/gemma3_4b_it_fsdp2_20k.sh" \
    +ray_kwargs.ray_init.address="\"${RAY_ADDRESS:-auto}\"" \
    "$@"
