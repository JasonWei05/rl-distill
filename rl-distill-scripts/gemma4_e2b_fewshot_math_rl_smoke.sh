#!/usr/bin/env bash
set -euo pipefail
# Gemma 4 E2B PT — few-shot math DAPO RL **SMOKE**: does verl RL work on the gemma-4 class of models?
#
# Uses the .venv-gemma4 stack (torch 2.11/cu130, transformers 5.14.1, vllm 0.25.1) — the only venv that
# supports gemma-4 (model_type `gemma4`, `Gemma4ForConditionalGeneration`, a text/vision/audio VLM whose
# text backbone is `gemma4_text` / `Gemma4TextDecoderLayer`). Runs locally on 2 free H100s with an
# isolated local Ray. Tiny/short/no-save — just enough to exercise the full loop:
#   rollout (vLLM) -> reward -> FSDP2 actor update -> weight resync to vLLM -> next rollout.
#
# gemma-4 vs gemma-3 deltas handled here:
#   * VENV               -> .venv-gemma4
#   * WRAP_LAYER_CLS     -> Gemma4TextDecoderLayer (FSDP2 transformer-layer wrap)
#   * attn_implementation-> sdpa, use_remove_padding=False (this venv has NO flash-attn; verl defaults to
#                           flash_attention_2 + varlen remove-padding, which would hard-fail)
#   * LD_LIBRARY_PATH    -> cu13 nvrtc; VLLM_ATTENTION_BACKEND=TRITON_ATTN (flashinfer JIT unavailable)
#   * update_weights_bucket_megabytes=8192: gemma-4 has a per-layer embedding table
#     (embed_tokens_per_layer, 262144x8960 bf16 ~= 4.5GB) that overflows the default 2GB FSDP->vLLM
#     weight-sync bucket (bucketed_weight_transfer asserts a single tensor must fit one bucket)
#   * HF served offline from the /tmp cache populated during the gemma-4 eval
#
#   bash rl-distill-scripts/gemma4_e2b_fewshot_math_rl_smoke.sh
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${HERE}/.." && pwd)"

export VENV=${VENV:-"${PROJECT_ROOT}/.venv-gemma4"}
export MODEL_TAG=${MODEL_TAG:-"gemma4-e2b"}
export MODEL_REPO=${MODEL_REPO:-"google/gemma-4-E2B"}
export MODEL_PATH=${MODEL_PATH:-"google/gemma-4-E2B"}
export WRAP_LAYER_CLS=${WRAP_LAYER_CLS:-"Gemma4TextDecoderLayer"}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5}
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-2}
export NNODES=1
export RAY_ADDRESS=${RAY_ADDRESS:-local}                 # isolated local Ray, not the shared cluster
RAY_TMP=${RAY_TMP:-/tmp/ray_gemma4_e2b_smoke}
RAY_NUM_CPUS=${RAY_NUM_CPUS:-32}
export RAY_RAYLET_START_WAIT_TIME_S=${RAY_RAYLET_START_WAIT_TIME_S:-300}

# gemma-4 runtime: cu13 nvrtc on the load path + Triton vLLM attention; model from the /tmp eval cache.
GEMMA4_LD=${GEMMA4_LD:-"${VENV}/lib/python3.12/site-packages/nvidia/cu13/lib"}
export LD_LIBRARY_PATH="${GEMMA4_LD}:${LD_LIBRARY_PATH:-}"
HF_HOME_G4=${HF_HOME_G4:-/tmp/hf_gemma4}

# Smoke sizing: small + short + no HF push + no checkpoint + no pre-train val.
export HF_PUSH_ENABLE=${HF_PUSH_ENABLE:-False}
export OFFLOAD=${OFFLOAD:-False}
export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-3}
export SAVE_FREQ=${SAVE_FREQ:-1000}
export TEST_FREQ=${TEST_FREQ:-1000}
export TRAIN_PROMPT_BSZ=${TRAIN_PROMPT_BSZ:-8}
export GEN_PROMPT_BSZ=${GEN_PROMPT_BSZ:-8}
export N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT:-4}
export TRAIN_PROMPT_MINI_BSZ=${TRAIN_PROMPT_MINI_BSZ:-4}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
export OVERLONG_BUFFER_LEN=${OVERLONG_BUFFER_LEN:-512}   # must be <= max_resp_len (default 4096 > 2048 smoke)
export ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.7}
export EXP_NAME=${EXP_NAME:-"SMOKE-gemma4-e2b-$(date +%m%d-%H%M)"}

exec bash "${HERE}/gemma3_pt_fewshot_math_rl.sh" \
  +ray_kwargs.ray_init._temp_dir="${RAY_TMP}" \
  +ray_kwargs.ray_init.include_dashboard=False \
  +ray_kwargs.ray_init.num_cpus="${RAY_NUM_CPUS}" \
  +ray_kwargs.ray_init._system_config.agent_register_timeout_ms="${RAY_AGENT_REGISTER_TIMEOUT_MS:-300000}" \
  +ray_kwargs.ray_init._system_config.worker_register_timeout_seconds="${RAY_WORKER_REGISTER_TIMEOUT_S:-600}" \
  +ray_kwargs.ray_init.runtime_env.env_vars.HF_HOME="${HF_HOME_G4}" \
  +ray_kwargs.ray_init.runtime_env.env_vars.TRANSFORMERS_OFFLINE="'1'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.HF_HUB_OFFLINE="'1'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
  +ray_kwargs.ray_init.runtime_env.env_vars.VLLM_ATTENTION_BACKEND="TRITON_ATTN" \
  actor_rollout_ref.model.use_remove_padding=False \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  ++actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes="${UPDATE_WEIGHTS_BUCKET_MB:-8192}" \
  "$@"
