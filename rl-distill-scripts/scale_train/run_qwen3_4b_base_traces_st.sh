#!/usr/bin/env bash
# ScaleTrain pod entry: top-128 off-policy traces from the UNTRAINED Qwen3-4B-Base teacher (pre-training control for a
# Qwen3-4B -> Qwen3-30B-A3B distillation; Qwen analogue of the e4b-base-* bundles). TRACE_SPEC=q4b-base-medium|q4b-base-hard:
# the band's 3,000 train questions x 16 samples + 300 validation questions x 1, temp 1 / top-p 1 / top-k -1, 8192 response
# tokens, 0-shot native ChatML + boxed instruction, token ids + top-128 teacher logprobs per position
# (run_gemma4_bestckpt_trace_collection.sh -> generate_gemma4_distill_traces.py). Every visible GPU is a DP worker (TP 1).
# Preemption: every finished shard is mirrored to S3 and restored on restart; a complete bundle (COMPLETE.json) is skipped.
#   bash launch_st_with_code.sh --gpus-per-instance 1 --priority high --allow-borrowing \
#     --run-file run_qwen3_4b_base_traces_st.sh --job-name q4b-tr-med --env-vars TRACE_SPEC=q4b-base-medium
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "${PROJECT_ROOT}"
export PATH="${PROJECT_ROOT}/.venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH:-}"
TRACE_SPEC="${TRACE_SPEC:?TRACE_SPEC=q4b-base-medium|q4b-base-hard is required}"
echo "Q4B_TRACES_START $(date -u +%FT%TZ) host=$(hostname) spec=${TRACE_SPEC}"
if [ -n "${CODE_S3_URI:-}" ]; then
  AWS_BIN="$(command -v aws || echo "${PROJECT_ROOT}/.venv/bin/aws")"
  "${AWS_BIN}" s3 cp --only-show-errors "${CODE_S3_URI}" /tmp/rl-distill-code.tar.gz
  tar -xzf /tmp/rl-distill-code.tar.gz --unlink-first --recursive-unlink -C "${PROJECT_ROOT}"
  echo "CODE_REFRESHED $(sha256sum /tmp/rl-distill-code.tar.gz | cut -c1-16)"
fi
if [ -f .env ]; then set -a; source .env; set +a; fi
: "${HF_TOKEN:?HF_TOKEN missing (forward it with --dotenv-keys)}"
export PATH="${HOME}/.local/bin:/root/.local/bin:${PATH}"
command -v uv >/dev/null 2>&1 || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="${HOME}/.local/bin:${PATH}"; }
export VENV="${VENV:-/tmp/.venv-gemma4}" GEMMA4_CUDA_VARIANT="${GEMMA4_CUDA_VARIANT:-cu129}"   # p5 fleet: CUDA 12.8 driver
[ -x "${VENV}/bin/python" ] || VENV="${VENV}" bash rl-distill-scripts/setup_env_gemma4.sh
command -v aws >/dev/null || { echo "FATAL: aws CLI missing (shards are mirrored to S3)" >&2; exit 2; }
TRACE_S3_BASE="${TRACE_S3_BASE:-s3://scale-ml/genai/rl-distill/qwen3-4b-base-traces-topk128-v1}"
if aws s3 ls "${TRACE_S3_BASE}/${TRACE_SPEC}/COMPLETE.json" >/dev/null 2>&1; then
  echo "Q4B_TRACES_ALREADY_COMPLETE ${TRACE_S3_BASE}/${TRACE_SPEC}"; exit 0
fi
n_gpus="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
unset CUDA_VISIBLE_DEVICES
# 32 prompts x 16 = 512 requests per shard (seeds depend only on (uid, sample index), so sharding never changes a sample)
TRACE_SPEC="${TRACE_SPEC}" TRACE_GPU_IDS="$(seq -s, 0 $((n_gpus - 1)))" TENSOR_PARALLEL_SIZE=1 TRACE_S3_BASE="${TRACE_S3_BASE}" \
  PROMPTS_PER_SHARD="${PROMPTS_PER_SHARD:-32}" MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}" \
  TRAIN_SAMPLES_PER_QUESTION="${TRAIN_SAMPLES_PER_QUESTION:-16}" VALIDATION_SAMPLES_PER_QUESTION=1 TRACE_S3_MIRROR_ENABLE=true \
  bash rl-distill-scripts/scale_train/run_gemma4_bestckpt_trace_collection.sh
echo "Q4B_TRACES_DONE $(date -u +%FT%TZ) spec=${TRACE_SPEC}"
