#!/usr/bin/env bash
# End-to-end correctness gate for the upcycled Gemma 3 4B PT 2E MoE stack.
#
# This is intentionally bounded: it exercises the real 8xH100 TP=4/EP=2
# Megatron actor, native vLLM rollout, R2 routing replay, initial
# MCore->vLLM sync, old-log-prob computation, and an actor update, without
# allowing an unverified 20k-token rollout to consume a production run.
#
# Usage:
#   bash rl-distill-scripts/gemma3_4b_pt_moe_megatron_correctness_1node_h100.sh
#
# Set TOTAL_TRAINING_STEPS=1 for a single-round gate. The default of two also
# validates the post-update weight resynchronization before returning.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Gates A--C in GEMMA3_MOE_RL_TRAINING.md use the canonical view to prove
# exact initialization. This integration gate intentionally uses the ordinary
# sparse checkpoint: it exercises the same dispatch, actor update, and
# post-update vLLM weight synchronization as the production run.
export UPCYCLED_MOE_DIR="${UPCYCLED_MOE_DIR:-/tmp/gemma3-4b-pt-moe-2e-upcycled}"
if [ ! -f "${UPCYCLED_MOE_DIR}/config.json" ]; then
    echo "Missing upcycled checkpoint: ${UPCYCLED_MOE_DIR}" >&2
    exit 2
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"
export TRAIN_PROMPT_BSZ="${TRAIN_PROMPT_BSZ:-4}"
export N_RESP_PER_PROMPT="${N_RESP_PER_PROMPT:-2}"
export TRAIN_PROMPT_MINI_BSZ="${TRAIN_PROMPT_MINI_BSZ:-4}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-2}"
# The production launcher has a 20-step warmup.  A 1--2 step correctness gate
# must instead use a scheduler horizon that is strictly longer than its warmup
# (Megatron validates this at actor initialization).
export CORRECTNESS_LR_WARMUP_STEPS="${CORRECTNESS_LR_WARMUP_STEPS:-0}"
export CORRECTNESS_LR_DECAY_STEPS="${CORRECTNESS_LR_DECAY_STEPS:-${TOTAL_TRAINING_STEPS}}"
if ! [[ "${CORRECTNESS_LR_WARMUP_STEPS}" =~ ^[0-9]+$ ]]; then
    echo "CORRECTNESS_LR_WARMUP_STEPS must be a non-negative integer." >&2
    exit 2
fi
if ! [[ "${CORRECTNESS_LR_DECAY_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "CORRECTNESS_LR_DECAY_STEPS must be a positive integer." >&2
    exit 2
fi
if [ "${CORRECTNESS_LR_DECAY_STEPS}" -le "${CORRECTNESS_LR_WARMUP_STEPS}" ]; then
    echo "CORRECTNESS_LR_DECAY_STEPS must be greater than CORRECTNESS_LR_WARMUP_STEPS." >&2
    exit 2
fi
export ACTOR_LR_WARMUP_STEPS="${CORRECTNESS_LR_WARMUP_STEPS}"
export ENABLE_OVERLONG_BUFFER="${ENABLE_OVERLONG_BUFFER:-False}"
# DAPO validates this even when the overlong buffer is disabled.
export OVERLONG_BUFFER_LEN="${OVERLONG_BUFFER_LEN:-128}"
export SAVE_FREQ="${SAVE_FREQ:--1}"
export TEST_FREQ="${TEST_FREQ:--1}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
export RESUME_MODE="${RESUME_MODE:-disable}"
export HF_PUSH_ENABLE="False"
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.6}"
# Generic Transformers execution was the source of the non-terminating MoE
# rollout.  The editable vLLM plugin provides a native Gemma3-MoE model; pin
# Triton so dense and MoE use the same attention implementation.
export ROLLOUT_MODEL_IMPL="${ROLLOUT_MODEL_IMPL:-native}"
export ROLLOUT_ATTENTION_BACKEND="${ROLLOUT_ATTENTION_BACKEND:-TRITON_ATTN}"
export EXP_NAME="${EXP_NAME:-DAPO-Gemma3-4B-PT-MoE-2E-correctness-$(date +%Y%m%d-%H%M%S)}"
export CKPTS_DIR="${CKPTS_DIR:-/tmp/verl/ckpts/DAPO/${EXP_NAME}}"
# Ray nests several socket paths below this directory; keep it short enough for
# the AF_UNIX 107-byte limit instead of deriving it from the descriptive run name.
export RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/ray_moe_correct_$(date +%H%M%S)}"

exec bash "${SCRIPT_DIR}/gemma3_4b_pt_moe_2e_megatron_rl_1node_h100.sh" \
    actor_rollout_ref.actor.optim.lr_decay_steps="${CORRECTNESS_LR_DECAY_STEPS}" \
    'trainer.logger=["console"]' \
    "$@"
