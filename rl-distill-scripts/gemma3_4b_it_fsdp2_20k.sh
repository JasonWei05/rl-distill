#!/usr/bin/env bash
set -euo pipefail

# Load API keys from .env
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "${PROJECT_ROOT}/.env" ]; then
    set -a
    source "${PROJECT_ROOT}/.env"
    set +a
fi
if [ -f "${PROJECT_ROOT}/.venv/bin/activate" ]; then
    source "${PROJECT_ROOT}/.venv/bin/activate"
fi

export VLLM_USE_V1=1

project_name='DAPO'
exp_name=${EXP_NAME:-"DAPO-Gemma3-4B-IT-$(date +%Y%m%d-%H%M)"}

adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

max_prompt_length=${MAX_PROMPT_LENGTH:-$((1024 * 2))}
max_response_length=${MAX_RESPONSE_LENGTH:-$((1024 * 20))}
enable_overlong_buffer=${ENABLE_OVERLONG_BUFFER:-True}
overlong_buffer_len=${OVERLONG_BUFFER_LEN:-$((1024 * 4))}
overlong_penalty_factor=${OVERLONG_PENALTY_FACTOR:-1.0}

loss_agg_mode="token-mean"

enable_filter_groups=${ENABLE_FILTER_GROUPS:-False}
filter_groups_metric=${FILTER_GROUPS_METRIC:-acc}
max_num_gen_batches=${MAX_NUM_GEN_BATCHES:-10}
train_prompt_bsz=${TRAIN_PROMPT_BSZ:-512}
gen_prompt_bsz=${GEN_PROMPT_BSZ:-${train_prompt_bsz}}
n_resp_per_prompt=${N_RESP_PER_PROMPT:-16}
train_prompt_mini_bsz=${TRAIN_PROMPT_MINI_BSZ:-32}

# Ray
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}
NNODES=${NNODES:-1}
# Paths
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/gemma-3-4b-it"}
CKPTS_DIR=${CKPTS_DIR:-"/tmp/verl/ckpts/${project_name}/${exp_name}"}
HF_PUSH_REPO=${HF_PUSH_REPO:-"JWei05/dapo-gemma3-4b-it"}
HF_PUSH_ENABLE=${HF_PUSH_ENABLE:-True}
HF_PUSH_DELETE_LOCAL_AFTER=${HF_PUSH_DELETE_LOCAL_AFTER:-True}
ACTOR_CKPT_SAVE_CONTENTS=${ACTOR_CKPT_SAVE_CONTENTS:-"[model,optimizer,extra,hf_model]"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/dapo_openmath2_mix_train.parquet"}
VAL_FILES="['${RAY_DATA_HOME}/data/dapo_openmath2_mix_val_compat.parquet','${RAY_DATA_HOME}/data/math__aime2024_repeated_32x_960_compat.parquet','${RAY_DATA_HOME}/data/math__aime2025_repeated_32x_960_compat.parquet','${RAY_DATA_HOME}/data/math__aime2026_repeated_32x_960_compat.parquet','${RAY_DATA_HOME}/data/math__math_500_repeated_2x_1000_compat.parquet','${RAY_DATA_HOME}/data/math__olympiadbench_repeated_2x_compat.parquet','${RAY_DATA_HOME}/data/math__minervamath_repeated_4x_compat.parquet','${RAY_DATA_HOME}/data/math__gsm8k_test_compat.parquet']"

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.7

# Performance Related Parameter
sp_size=${SP_SIZE:-1}
use_dynamic_bsz=${USE_DYNAMIC_BSZ:-True}
actor_ppo_max_token_len=${ACTOR_PPO_MAX_TOKEN_LEN:-$(((max_prompt_length + max_response_length) / sp_size))}
infer_ppo_max_token_len=${INFER_PPO_MAX_TOKEN_LEN:-$(((max_prompt_length + max_response_length) / sp_size))}
offload=${OFFLOAD:-True}
gen_tp=${GEN_TP:-1}
enable_chunked_prefill=${ENABLE_CHUNKED_PREFILL:-True}
enable_activation_offload=${ENABLE_ACTIVATION_OFFLOAD:-False}
enable_tiled_mlp=${ENABLE_TILED_MLP:-False}
tiled_mlp_num_shards=${TILED_MLP_NUM_SHARDS:-4}
rollout_gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.80}
rollout_block_size=${ROLLOUT_BLOCK_SIZE:-32}
rollout_attention_backend=${ROLLOUT_ATTENTION_BACKEND:-}
rollout_attention_backend_args=()
if [ -n "${rollout_attention_backend}" ]; then
    rollout_attention_backend_args+=(+actor_rollout_ref.rollout.engine_kwargs.vllm.attention_backend="${rollout_attention_backend}")
fi

export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-lo}
NCCL_SOCKET_FAMILY=${NCCL_SOCKET_FAMILY:-AF_INET}
GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-lo}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME#=}
export NCCL_SOCKET_FAMILY=${NCCL_SOCKET_FAMILY#=}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME#=}
NVML_SHIM_DIR=${NVML_SHIM_DIR:-${VERL_EXTRA_LD_LIBRARY_PATH:-/tmp/nvidia-nvml-535}}
if [ ! -e "${NVML_SHIM_DIR}/libnvidia-ml.so.1" ] && [ -f /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.535.129.03 ]; then
    mkdir -p "${NVML_SHIM_DIR}"
    ln -sf /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.535.129.03 "${NVML_SHIM_DIR}/libnvidia-ml.so.1"
fi
PYTHON_VERSION_TAG=${PYTHON_VERSION_TAG:-python3.12}
SITE_PACKAGES="${PROJECT_ROOT}/.venv/lib/${PYTHON_VERSION_TAG}/site-packages"
LD_LIBRARY_PATH_PREFIX="${NVML_SHIM_DIR}"
if [ -d "${SITE_PACKAGES}" ]; then
    export CUDNN_HOME="${CUDNN_HOME:-${SITE_PACKAGES}/nvidia/cudnn}"
    export NVRTC_HOME="${NVRTC_HOME:-${SITE_PACKAGES}/nvidia/cuda_nvrtc}"
    export CURAND_HOME="${CURAND_HOME:-${SITE_PACKAGES}/nvidia/curand}"
    export CUBLAS_HOME="${CUBLAS_HOME:-${SITE_PACKAGES}/nvidia/cublas}"
    export CUDART_HOME="${CUDART_HOME:-${SITE_PACKAGES}/nvidia/cuda_runtime}"
    LD_LIBRARY_PATH_PREFIX="${LD_LIBRARY_PATH_PREFIX}:${CUDNN_HOME}/lib:${NVRTC_HOME}/lib:${CURAND_HOME}/lib:${CUBLAS_HOME}/lib:${CUDART_HOME}/lib"
fi
export CUDNN_HOME="${CUDNN_HOME:-}"
export NVRTC_HOME="${NVRTC_HOME:-}"
export CURAND_HOME="${CURAND_HOME:-}"
export CUBLAS_HOME="${CUBLAS_HOME:-}"
export CUDART_HOME="${CUDART_HOME:-}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH_PREFIX}:${LD_LIBRARY_PATH:-}"
export VERL_SKIP_VLLM_MM_WEIGHT_RELOAD="${VERL_SKIP_VLLM_MM_WEIGHT_RELOAD:-1}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"

python3 -m dapo.main_dapo \
    +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}" \
    +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_SOCKET_FAMILY="${NCCL_SOCKET_FAMILY}" \
    +ray_kwargs.ray_init.runtime_env.env_vars.GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME}" \
    +ray_kwargs.ray_init.runtime_env.env_vars.TORCH_CUDA_ARCH_LIST="\"${TORCH_CUDA_ARCH_LIST}\"" \
    +ray_kwargs.ray_init.runtime_env.env_vars.CUDNN_HOME="${CUDNN_HOME}" \
    +ray_kwargs.ray_init.runtime_env.env_vars.NVRTC_HOME="${NVRTC_HOME}" \
    +ray_kwargs.ray_init.runtime_env.env_vars.CURAND_HOME="${CURAND_HOME}" \
    +ray_kwargs.ray_init.runtime_env.env_vars.CUBLAS_HOME="${CUBLAS_HOME}" \
    +ray_kwargs.ray_init.runtime_env.env_vars.CUDART_HOME="${CUDART_HOME}" \
    +ray_kwargs.ray_init.runtime_env.env_vars.LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
    +ray_kwargs.ray_init.runtime_env.env_vars.PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-}" \
    +ray_kwargs.ray_init.runtime_env.env_vars.PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-}" \
    +ray_kwargs.ray_init.runtime_env.env_vars.VERL_SKIP_VLLM_MM_WEIGHT_RELOAD="\"${VERL_SKIP_VLLM_MM_WEIGHT_RELOAD}\"" \
    +ray_kwargs.ray_init.runtime_env.env_vars.VERL_VLLM_PORT_BASE="\"${VERL_VLLM_PORT_BASE:-52000}\"" \
    +ray_kwargs.ray_init.runtime_env.env_vars.VERL_VLLM_PORT_STRIDE="\"${VERL_VLLM_PORT_STRIDE:-100}\"" \
    +ray_kwargs.ray_init.runtime_env.env_vars.VERL_VLLM_NODE_PORT_STRIDE="\"${VERL_VLLM_NODE_PORT_STRIDE:-10}\"" \
    +ray_kwargs.ray_init.runtime_env.env_vars.VERL_VLLM_RANDOM_PORTS="\"${VERL_VLLM_RANDOM_PORTS:-1}\"" \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILES}" \
    data.prompt_key=prompt \
    data.shuffle=True \
    data.seed=42 \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.train_batch_size=${train_prompt_bsz} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.enable_activation_offload=${enable_activation_offload} \
    actor_rollout_ref.model.tiled_mlp.enabled=${enable_tiled_mlp} \
    actor_rollout_ref.model.tiled_mlp.num_shards=${tiled_mlp_num_shards} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=20 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_memory_utilization} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.block_size=${rollout_block_size} \
    "${rollout_attention_backend_args[@]}" \
    actor_rollout_ref.rollout.enable_chunked_prefill=${enable_chunked_prefill} \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    '+actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=["Gemma3DecoderLayer"]' \
    reward_model.reward_manager=dapo \
    reward.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    reward.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    reward.reward_kwargs.overlong_buffer_cfg.log=True \
    reward.reward_kwargs.max_resp_len=${max_response_length} \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=False \
    trainer.test_freq=2 \
    trainer.save_freq=${SAVE_FREQ:-20} \
    trainer.max_actor_ckpt_to_keep=${MAX_ACTOR_CKPT_TO_KEEP:-2} \
    +trainer.hf_push.enable=${HF_PUSH_ENABLE} \
    +trainer.hf_push.repo_id="${HF_PUSH_REPO}" \
    +trainer.hf_push.private=False \
    +trainer.hf_push.delete_local_after=${HF_PUSH_DELETE_LOCAL_AFTER} \
    +trainer.hf_push.max_to_keep=5 \
    actor_rollout_ref.actor.checkpoint.save_contents="${ACTOR_CKPT_SAVE_CONTENTS}" \
    trainer.total_epochs=100 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode="${RESUME_MODE:-auto}" $@
