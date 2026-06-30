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
exp_name=${EXP_NAME:-"DAPO-Gemma3-12B-Distilled-TopK128-DAPO17k-$(date +%Y%m%d-%H%M)"}

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
train_prompt_bsz=${TRAIN_PROMPT_BSZ:-64}
gen_prompt_bsz=${GEN_PROMPT_BSZ:-${train_prompt_bsz}}
n_resp_per_prompt=${N_RESP_PER_PROMPT:-16}
train_prompt_mini_bsz=${TRAIN_PROMPT_MINI_BSZ:-32}

# Ray
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}
NNODES=${NNODES:-2}
# Paths
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
MODEL_REPO=${MODEL_REPO:-"JWei05/DAPO-Gemma3-12B-PT-TopK128Distill-From-Gemma3-4B-PT-DAPO-17.4k-LR2e6-linear500-1node"}
MODEL_SUBFOLDER=${MODEL_SUBFOLDER:-"step_000500"}
BASE_MODEL_REPO=${BASE_MODEL_REPO:-"google/gemma-3-12b-it"}
MODEL_LOCAL_DIR=${MODEL_LOCAL_DIR:-"/tmp/hf_models/gemma3-12b-topk128-distill-step000500"}
PREPARE_MODEL=${PREPARE_MODEL:-1}
MODEL_PATH=${MODEL_PATH:-}
CKPTS_DIR=${CKPTS_DIR:-"/tmp/verl/ckpts/${project_name}/${exp_name}"}
HF_PUSH_REPO=${HF_PUSH_REPO:-"JWei05/DAPO-Gemma3-12B-TopK128Distill-RL-DAPO17k"}
HF_PUSH_ENABLE=${HF_PUSH_ENABLE:-True}
HF_PUSH_DELETE_LOCAL_AFTER=${HF_PUSH_DELETE_LOCAL_AFTER:-True}
ACTOR_CKPT_SAVE_CONTENTS=${ACTOR_CKPT_SAVE_CONTENTS:-"[model,optimizer,extra,hf_model]"}
DATA_SEED=${DATA_SEED:-42}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/dapo_17k_train.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/dapo_17k_test.parquet"}
VAL_FILES=${VAL_FILES:-"['${TEST_FILE}']"}
GEMMA3_CHAT_TEMPLATE_FILE=${GEMMA3_CHAT_TEMPLATE_FILE:-"${PROJECT_ROOT}/rl-distill-scripts/data/gemma3_it_chat_template.jinja"}

if [ -z "${MODEL_PATH}" ]; then
    if [ "${PREPARE_MODEL}" = "1" ]; then
        python3 "${PROJECT_ROOT}/rl-distill-scripts/data/download_hf_subfolder.py" \
            --repo-id "${MODEL_REPO}" \
            --subfolder "${MODEL_SUBFOLDER}" \
            --metadata-repo "${BASE_MODEL_REPO}" \
            --output-dir "${MODEL_LOCAL_DIR}"
        MODEL_PATH="${MODEL_LOCAL_DIR}"
    else
        MODEL_PATH="${MODEL_REPO}"
    fi
fi

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.7
val_n=${VAL_N:-1}

# Performance Related Parameter
sp_size=${SP_SIZE:-1}
use_dynamic_bsz=${USE_DYNAMIC_BSZ:-True}
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) / sp_size))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) / sp_size))
offload=${OFFLOAD:-True}
gen_tp=${GEN_TP:-1}
enable_chunked_prefill=${ENABLE_CHUNKED_PREFILL:-True}
rollout_gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.78}
rollout_block_size=${ROLLOUT_BLOCK_SIZE:-32}
actor_fsdp_size=${ACTOR_FSDP_SIZE:--1}
actor_lr=${ACTOR_LR:-5e-7}
actor_lr_warmup_steps=${ACTOR_LR_WARMUP_STEPS:-50}
save_freq=${SAVE_FREQ:-20}
test_freq=${TEST_FREQ:-5}
total_training_steps=${TOTAL_TRAINING_STEPS:-}
max_actor_ckpt_to_keep=${MAX_ACTOR_CKPT_TO_KEEP:-4}
hf_push_max_to_keep=${HF_PUSH_MAX_TO_KEEP:-8}
val_before_train=${VAL_BEFORE_TRAIN:-True}

total_training_steps_args=()
if [ -n "${total_training_steps}" ]; then
    total_training_steps_args+=(trainer.total_training_steps="${total_training_steps}")
fi

export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-${NETWORK_INTERFACE_NAME:-eth0}}
export NCCL_SOCKET_FAMILY=${NCCL_SOCKET_FAMILY:-AF_INET}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-${NETWORK_INTERFACE_NAME:-eth0}}

python3 -m dapo.main_dapo \
    +ray_kwargs.ray_init.address="'${RAY_ADDRESS}'" \
    +ray_kwargs.ray_init.runtime_env=null \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILES}" \
    data.prompt_key=prompt \
    data.shuffle=True \
    data.seed=${DATA_SEED} \
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
    actor_rollout_ref.model.custom_chat_template="@${GEMMA3_CHAT_TEMPLATE_FILE}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=${actor_lr} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=${actor_lr_warmup_steps} \
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
    actor_rollout_ref.rollout.enable_chunked_prefill=${enable_chunked_prefill} \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=${val_n} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${actor_fsdp_size} \
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
    trainer.val_before_train=${val_before_train} \
    trainer.test_freq=${test_freq} \
    trainer.save_freq=${save_freq} \
    trainer.max_actor_ckpt_to_keep=${max_actor_ckpt_to_keep} \
    +trainer.hf_push.enable=${HF_PUSH_ENABLE} \
    +trainer.hf_push.repo_id="${HF_PUSH_REPO}" \
    +trainer.hf_push.private=False \
    +trainer.hf_push.delete_local_after=${HF_PUSH_DELETE_LOCAL_AFTER} \
    +trainer.hf_push.max_to_keep=${hf_push_max_to_keep} \
    actor_rollout_ref.actor.checkpoint.save_contents="${ACTOR_CKPT_SAVE_CONTENTS}" \
    trainer.total_epochs=100 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode="${RESUME_MODE:-auto}" \
    "${total_training_steps_args[@]}" \
    "$@"
