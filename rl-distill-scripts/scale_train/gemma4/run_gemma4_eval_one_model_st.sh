#!/usr/bin/env bash
# ScaleTrain run-file: the §7 math suite for ONE registry model on the pod's GPUs (data-parallel over the eval datasets,
# GPU_COUNT = all visible GPUs, i.e. 2 for the 2-GPU eval jobs). Wraps run_gemma4_rl_distill_eval_one_model.sh, which
# materializes the model from its registry source (s3_hf_export), runs run_gemma4_math_4gpu.py with one shard per GPU,
# mirrors results to RESULT_S3_ROOT/<tag>/, and writes RUN_COMPLETE.json. Env: MODEL_TAG (required), EVAL_PHASES (default math).
#   bash launch_st_with_code.sh --gpus-per-instance 2 --priority high --run-file gemma4/run_gemma4_eval_one_model_st.sh \
#     --job-name g4ev-<tag> --env-vars "MODEL_TAG=<registry tag>"
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; cd "${PROJECT_ROOT}"
export PATH="${PROJECT_ROOT}/.venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH:-}"
echo "ST_EVAL_START $(date -u +%FT%TZ) host=$(hostname) tag=${MODEL_TAG:?MODEL_TAG required}"
if [ -n "${CODE_S3_URI:-}" ]; then
  AWS_BIN="$(command -v aws || echo "${PROJECT_ROOT}/.venv/bin/aws")"
  "${AWS_BIN}" s3 cp --only-show-errors "${CODE_S3_URI}" /tmp/rl-distill-code.tar.gz
  tar -xzf /tmp/rl-distill-code.tar.gz --unlink-first --recursive-unlink -C "${PROJECT_ROOT}"
  echo "CODE_REFRESHED $(sha256sum /tmp/rl-distill-code.tar.gz | cut -c1-16) files=$(tar -tzf /tmp/rl-distill-code.tar.gz | wc -l)"
fi
if [ -f .env ]; then set -a; source .env; set +a; fi
: "${HF_TOKEN:?HF_TOKEN missing (forward it with --dotenv-keys)}"
export PATH="${HOME}/.local/bin:/root/.local/bin:${PATH}"
command -v uv >/dev/null 2>&1 || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="${HOME}/.local/bin:${PATH}"; }
export VENV="${VENV:-/tmp/.venv-gemma4}" GEMMA4_CUDA_VARIANT="${GEMMA4_CUDA_VARIANT:-cu129}"
if [ ! -x "${VENV}/bin/python" ]; then VENV="${VENV}" bash rl-distill-scripts/setup_env_gemma4.sh; fi
export PATH="${VENV}/bin:${PATH}"; source "${VENV}/bin/activate"
if [ -x /usr/local/cuda/bin/nvcc ]; then export CUDA_HOME="${CUDA_HOME:-$(readlink -f /usr/local/cuda)}"; fi
python -c "import torch, vllm" || { echo "FATAL: venv import check failed" >&2; exit 2; }
command -v aws >/dev/null || { echo "FATAL: aws CLI missing" >&2; exit 2; }
export HF_HOME="${HF_HOME:-/tmp/hf_cache}" HF_HUB_CACHE="${HF_HUB_CACHE:-/tmp/hf_cache/hub}" TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache}"
n_gpus="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
echo "ST_EVAL config tag=${MODEL_TAG} gpus=${n_gpus} phases=${EVAL_PHASES:-math}"
# One dataset shard per GPU (DP over datasets); same knobs as the local §9.0j/§9.0l evals.
env -u CUDA_VISIBLE_DEVICES MODEL_TAG="${MODEL_TAG}" EVAL_PHASES="${EVAL_PHASES:-math}" GPU_COUNT="${n_gpus}" \
  PACKED_PHYSICAL_GPU_IDS="$(seq -s, 0 $((n_gpus - 1)))" PREPARE_SHARED_ASSETS=true \
  MODEL_WORK_ROOT="/tmp/gemma4_eval/${MODEL_TAG}" RESULT_ROOT_OVERRIDE="/tmp/gemma4_eval/${MODEL_TAG}/results" \
  EVAL_KV_CACHE_GIB=16 EVAL_GPU_MEMORY_UTILIZATION=0.40 EVAL_PREDICTIVE_TOPK_WIDTH=0 \
  MATH_QUESTIONS_PER_BATCH=64 MATH_REQUEST_BATCH_SIZE=1024 MATH_RESUME_TRACES=1 EVAL_S3_ENABLE=true \
  bash rl-distill-scripts/scale_train/gemma4/run_gemma4_rl_distill_eval_one_model.sh
echo "ST_EVAL_DONE tag=${MODEL_TAG} $(date -u +%FT%TZ)"
