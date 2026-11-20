#!/usr/bin/env bash
# ScaleTrain run-file (launch_st_with_code.sh --run-file ../synth/run_synth_st.sh): runs the plan ${S3}/plans/${TAG}.json
# (plan.py) on the pod's GPUs with pack.py under one CUDA MPS daemon, then writes ${S3}/pods/${TAG}.COMPLETE.
#   env: TAG, GPUS, S3 (s3://scale-ml/genai/rl-distill/synth), SYNC_SECONDS (300)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "${ROOT}"
export PATH="${ROOT}/.venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
: "${TAG:?}" "${GPUS:?}"; S3="${S3:-s3://scale-ml/genai/rl-distill/synth}"; W=/tmp/synth
mkdir -p "${W}"
export WANDB_ENTITY=rl-distill WANDB_MODE=online TORCHINDUCTOR_CACHE_DIR="${W}/inductor" HOME="${W}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # many processes per GPU: return freed memory, less fragmentation
python -c "import torch, yaml, safetensors, tokenizers, wandb; assert torch.cuda.device_count() == ${GPUS}; print('SYNTH_ENV_OK', torch.__version__)"
export CUDA_MPS_PIPE_DIRECTORY="${W}/mps_pipe" CUDA_MPS_LOG_DIRECTORY="${W}/mps_log"
mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"
nvidia-cuda-mps-control -d && echo "MPS_ON"
( while sleep 60; do echo "PODSTAT $(date -u +%T) load=$(cut -d' ' -f1 /proc/loadavg) gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr -d ' ' | paste -sd';')"; done ) &
aws s3 cp --only-show-errors "${S3}/plans/${TAG}.json" "${W}/plan.json"
python rl-distill-scripts/synth/pack.py --plan "${W}/plan.json" --gpus "${GPUS}" --root "${W}" --s3 "${S3}" --sync-seconds "${SYNC_SECONDS:-300}"
echo "{\"done\": \"$(date -u +%FT%TZ)\"}" | aws s3 cp - "${S3}/pods/${TAG}.COMPLETE"; echo "SYNTH_POD_DONE ${TAG}"
