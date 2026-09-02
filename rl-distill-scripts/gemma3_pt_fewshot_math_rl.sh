#!/usr/bin/env bash
set -euo pipefail

# Core DAPO RL launcher for Gemma 3 PT math training with the UNIFIED FEW-SHOT PROMPT.
#
# The prompt (12-shot interleaved MATH+GSM8K, `\boxed{}` contract) is applied via the custom
# chat template `data/gemma3_it_fewshot_math.jinja`, so it is IDENTICAL for training rollouts
# and every validation set (verl applies custom_chat_template to both). Verified to render
# byte-identical to the eval prompt that produced GSM8K 37.4 / MATH 24.4 on 4B PT.
#
# Model-specific settings come from the thin wrappers (gemma3_{1b,4b}_pt_fewshot_math_rl.sh);
# everything is env-overridable. Data is built by data/build_math_rl_data.py (plain verl format).

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "${PROJECT_ROOT}/.env" ]; then set -a; source "${PROJECT_ROOT}/.env"; set +a; fi
# VENV override lets a local-disk venv copy be used instead of the EFS .venv — on this shared box the
# EFS venv makes Ray's dashboard-agent import too slow, tripping the raylet's fixed ~80s port-file wait.
VENV="${VENV:-${PROJECT_ROOT}/.venv}"
if [ -f "${VENV}/bin/activate" ]; then source "${VENV}/bin/activate"; fi

export VLLM_USE_V1=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

project_name=${PROJECT_NAME:-'DAPO'}
MODEL_TAG=${MODEL_TAG:-gemma3-pt}
exp_name=${EXP_NAME:-"DAPO-${MODEL_TAG}-FewShotMath-$(date +%Y%m%d-%H%M)"}

adv_estimator=grpo
use_kl_in_reward=False; kl_coef=0.0; use_kl_loss=False; kl_loss_coef=0.0
clip_ratio_low=0.2; clip_ratio_high=0.28

# Matches the most-recent dense-PT DAPO recipe: 20k response + 4k soft-overlong buffer (factor 1.0).
# max_prompt is 4k (NOT the recipe's 2k) because the few-shot prefix is ~1250 tok and the longest
# question (OlympiadBench) is ~1200 tok -> ~2461 max; 2k would left-truncate the few-shot exemplars.
max_prompt_length=${MAX_PROMPT_LENGTH:-$((1024 * 4))}
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

RAY_ADDRESS=${RAY_ADDRESS:-"auto"}
NNODES=${NNODES:-1}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-4}
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}

# Model (set by wrapper)
MODEL_REPO=${MODEL_REPO:-"google/gemma-3-4b-pt"}
MODEL_LOCAL_DIR=${MODEL_LOCAL_DIR:-"${RAY_DATA_HOME}/models/gemma-3-4b-pt"}
MODEL_PATH=${MODEL_PATH:-"${MODEL_LOCAL_DIR}"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
HF_PUSH_REPO=${HF_PUSH_REPO:-"JWei05/DAPO-Gemma3-PT-FewShotMath"}
HF_PUSH_ENABLE=${HF_PUSH_ENABLE:-True}
HF_PUSH_DELETE_LOCAL_AFTER=${HF_PUSH_DELETE_LOCAL_AFTER:-False}
HF_PUSH_REQUIRED=${HF_PUSH_REQUIRED:-True}
ACTOR_CKPT_SAVE_CONTENTS=${ACTOR_CKPT_SAVE_CONTENTS:-"[model,optimizer,extra,hf_model]"}
DATA_SEED=${DATA_SEED:-42}
# FSDP2 transformer-layer wrap class. Gemma 3 -> Gemma3DecoderLayer (default); Gemma 4 (VLM) has a
# separate text backbone -> Gemma4TextDecoderLayer. Set by the wrapper for the gemma-4 stack.
WRAP_LAYER_CLS=${WRAP_LAYER_CLS:-Gemma3DecoderLayer}

# THE UNIFIED FEW-SHOT PROMPT — identical for train + val (see header).
GEMMA3_CHAT_TEMPLATE_FILE=${GEMMA3_CHAT_TEMPLATE_FILE:-"${PROJECT_ROOT}/rl-distill-scripts/data/gemma3_it_fewshot_math.jinja"}

# Data: plain verl parquets built by data/build_math_rl_data.py
DATA_DIR=${DATA_DIR:-"${RAY_DATA_HOME}/data"}
TRAIN_FILE=${TRAIN_FILE:-"${DATA_DIR}/dapo_rl_train.parquet"}
# Val: only the held-out DAPO val (100 questions x16 -> pass@k / maj@k / mean@k grouped by uid).
# (The other math benchmarks are built by build_math_rl_data.py and can be added back via VAL_FILES.)
VAL_FILES=${VAL_FILES:-"['${DATA_DIR}/dapo_rl_val100_x16.parquet']"}

# Algorithm / sampling — validation uses the SAME sampling as training (temp 1.0, top_p 1.0, top_k -1)
temperature=1.0; top_p=1.0; top_k=-1
val_top_p=${VAL_TOP_P:-1.0}
val_n=${VAL_N:-1}

sp_size=${SP_SIZE:-1}
use_dynamic_bsz=${USE_DYNAMIC_BSZ:-True}
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) / sp_size))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) / sp_size))
offload=${OFFLOAD:-True}
gen_tp=${GEN_TP:-1}
enable_chunked_prefill=${ENABLE_CHUNKED_PREFILL:-True}
rollout_gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.78}
actor_fsdp_size=${ACTOR_FSDP_SIZE:--1}
actor_lr=${ACTOR_LR:-1e-6}
actor_lr_warmup_steps=${ACTOR_LR_WARMUP_STEPS:-20}
save_freq=${SAVE_FREQ:-25}
test_freq=${TEST_FREQ:-10}
total_training_steps=${TOTAL_TRAINING_STEPS:-}
max_actor_ckpt_to_keep=${MAX_ACTOR_CKPT_TO_KEEP:-4}
hf_push_max_to_keep=${HF_PUSH_MAX_TO_KEEP:-8}
hf_push_freq=${HF_PUSH_FREQ:-${save_freq}}   # push HF checkpoint every N steps (aligned to save_freq)
val_before_train=${VAL_BEFORE_TRAIN:-True}
# Generation traces: sample N val generations -> wandb Table (browsable each eval). Standard
# practice is a sample, not all. LOG_TRAIN_GENERATIONS>0 also uploads N random *train* rollouts to
# a wandb Table every train step (fork addition, fresh per-step table). To dump *every* train/val
# trace to disk (JSONL) instead, set ROLLOUT_DATA_DIR / VALIDATION_DATA_DIR to a path (default off).
log_val_generations=${LOG_VAL_GENERATIONS:-64}
log_train_generations=${LOG_TRAIN_GENERATIONS:-0}
rollout_data_dir=${ROLLOUT_DATA_DIR:-null}
validation_data_dir=${VALIDATION_DATA_DIR:-null}

total_training_steps_args=()
if [ -n "${total_training_steps}" ]; then
    total_training_steps_args+=(trainer.total_training_steps="${total_training_steps}")
fi

# Do not put credentials in Hydra overrides: resolved configs, process listings,
# and failure logs would expose the literal values. The DAPO entry point reads
# exported driver credentials and injects them into Ray's job runtime environment
# programmatically for both local Ray and an existing cluster (`address=auto`).
ray_secret_args=()
if [ "${FORWARD_SECRETS_IN_RAY_CONFIG:-False}" = "True" ]; then
    echo "Refusing to put WANDB_API_KEY/HF_TOKEN in command-line Ray config" >&2
    exit 2
fi

export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-${NETWORK_INTERFACE_NAME:-lo}}
export NCCL_SOCKET_FAMILY=${NCCL_SOCKET_FAMILY:-AF_INET}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-${NETWORK_INTERFACE_NAME:-lo}}

python3 -m dapo.main_dapo \
    +ray_kwargs.ray_init.address="'${RAY_ADDRESS}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.WANDB_BASE_URL="${WANDB_BASE_URL:-}" \
    +ray_kwargs.ray_init.runtime_env.env_vars.VERL_VLLM_PORT_BASE="'${VERL_VLLM_PORT_BASE:-52000}'" \
    "${ray_secret_args[@]}" \
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
    actor_rollout_ref.rollout.enable_chunked_prefill=${enable_chunked_prefill} \
    actor_rollout_ref.rollout.enable_prefix_caching=True \
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
    "+actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[\"${WRAP_LAYER_CLS}\"]" \
    reward_model.reward_manager=dapo \
    reward.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    reward.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    reward.reward_kwargs.overlong_buffer_cfg.log=True \
    reward.reward_kwargs.max_resp_len=${max_response_length} \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=${val_before_train} \
    trainer.log_val_generations=${log_val_generations} \
    +trainer.log_train_generations=${log_train_generations} \
    trainer.rollout_data_dir=${rollout_data_dir} \
    trainer.validation_data_dir=${validation_data_dir} \
    trainer.test_freq=${test_freq} \
    trainer.save_freq=${save_freq} \
    trainer.max_actor_ckpt_to_keep=${max_actor_ckpt_to_keep} \
    +trainer.hf_push.enable=${HF_PUSH_ENABLE} \
    +trainer.hf_push.required=${HF_PUSH_REQUIRED} \
    +trainer.hf_push.repo_id="${HF_PUSH_REPO}" \
    +trainer.hf_push.private=False \
    +trainer.hf_push.delete_local_after=${HF_PUSH_DELETE_LOCAL_AFTER} \
    +trainer.hf_push.max_to_keep=${hf_push_max_to_keep} \
    +trainer.hf_push.freq=${hf_push_freq} \
    actor_rollout_ref.actor.checkpoint.save_contents="${ACTOR_CKPT_SAVE_CONTENTS}" \
    trainer.total_epochs=100 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode="${RESUME_MODE:-auto}" \
    "${total_training_steps_args[@]}" \
    "$@"
