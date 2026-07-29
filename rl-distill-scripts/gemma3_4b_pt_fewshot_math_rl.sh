#!/usr/bin/env bash
set -euo pipefail
# Gemma 3 4B PT — DAPO math RL with the unified few-shot prompt (train + val use the same prompt).
# Thin wrapper over gemma3_pt_fewshot_math_rl.sh. All vars below are env-overridable.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL_TAG=${MODEL_TAG:-"gemma3-4b-pt"}
export MODEL_REPO=${MODEL_REPO:-"google/gemma-3-4b-pt"}
export MODEL_PATH=${MODEL_PATH:-"google/gemma-3-4b-pt"}   # resolved from HF cache (gated; HF_TOKEN in .env)
export HF_PUSH_REPO=${HF_PUSH_REPO:-"JWei05/DAPO-Gemma3-4B-PT-FewShotMath"}

# 4B RL wants a few GPUs. Point CUDA_VISIBLE_DEVICES/N_GPUS_PER_NODE at free GPUs before launch.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-4}
export OFFLOAD=${OFFLOAD:-True}

exec bash "${HERE}/gemma3_pt_fewshot_math_rl.sh" "$@"
