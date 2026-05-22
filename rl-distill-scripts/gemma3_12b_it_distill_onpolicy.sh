#!/usr/bin/env bash
# On-policy distillation: Gemma3-12B-IT student <- trained DAPO-Gemma3-27B teacher.
#
# Same recipe as gemma3_4b_it_distill_onpolicy.sh; only the student model and
# memory-pressure knobs differ. See that file's header for the loss / schedule.
set -xeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "${PROJECT_ROOT}/.env" ]; then
    set -a; source "${PROJECT_ROOT}/.env"; set +a
fi

TEACHER_REV="${TEACHER_REV:-step_000040}"
TEACHER_PATH="${TEACHER_PATH:-${HOME}/verl/models/dapo-gemma3-27b-it-${TEACHER_REV}}"
if [ ! -f "${TEACHER_PATH}/config.json" ]; then
    REVISION="${TEACHER_REV}" DEST="${TEACHER_PATH}" \
        bash "${PROJECT_ROOT}/rl-distill-scripts/data/prep_teacher_27b.sh"
fi

export VLLM_USE_V1=1

project_name='distill_onpolicy'
exp_name=${EXP_NAME:-"onpolicy-gemma3-12b-$(date +%Y%m%d-%H%M)"}

RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/gemma-3-12b-it"}
CKPTS_DIR=${CKPTS_DIR:-"/tmp/verl/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/dapo-math-17k.parquet"}
VAL_FILES="['${RAY_DATA_HOME}/data/math__aime2024_repeated_32x_960.parquet','${RAY_DATA_HOME}/data/math__aime2025_repeated_32x_960.parquet','${RAY_DATA_HOME}/data/math__aime2026_repeated_32x_960.parquet','${RAY_DATA_HOME}/data/math__math_500_repeated_2x_1000.parquet','${RAY_DATA_HOME}/data/math__olympiadbench_repeated_2x.parquet','${RAY_DATA_HOME}/data/math__minervamath_repeated_4x.parquet','${RAY_DATA_HOME}/data/math__gsm8k_test.parquet']"

export HF_PUSH_ENABLE="${HF_PUSH_ENABLE:-true}"
export HF_PUSH_REPO="${HF_PUSH_REPO:-JWei05/gemma3-12b-it-on-policy-distilled-from-dapo27b}"
export HF_PUSH_PRIVATE="${HF_PUSH_PRIVATE:-false}"
export HF_PUSH_MAX_TO_KEEP="${HF_PUSH_MAX_TO_KEEP:-10}"

train_prompt_bsz=128
gen_prompt_bsz=${train_prompt_bsz}
n_resp_per_prompt=1
train_prompt_mini_bsz=128

max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 20))

loss_agg_mode="token-mean"

sp_size=1
use_dynamic_bsz=True
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) / sp_size))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) / sp_size))
offload=True
gen_tp=2     # 12B student rollout: TP=2 to keep KV cache headroom for teacher
enable_chunked_prefill=True

temperature=0.7
top_p=0.95
top_k=-1
val_top_p=0.95

# 12B + 27B teacher colocated on 1 node (8×B200) is tight — give teacher more TP
# and lower the rollout/teacher mem fractions to leave room for FSDP grads/opt.
TEACHER_TP=${TEACHER_TP:-4}

export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_SOCKET_IFNAME=lo
export NCCL_SOCKET_FAMILY=AF_INET
export GLOO_SOCKET_IFNAME=lo

NNODES=${NNODES:-1}

python3 -m dapo.main_dapo \
    +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_SOCKET_IFNAME=lo \
    +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_SOCKET_FAMILY=AF_INET \
    +ray_kwargs.ray_init.runtime_env.env_vars.GLOO_SOCKET_IFNAME=lo \
    +ray_kwargs.ray_init.runtime_env.env_vars.WANDB_API_KEY="${WANDB_API_KEY}" \
    +ray_kwargs.ray_init.runtime_env.env_vars.HF_TOKEN="${HF_TOKEN}" \
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
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    algorithm.filter_groups.enable=False \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=20 \
    actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
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
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    '+actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=["Gemma3DecoderLayer"]' \
    +distillation.enabled=true \
    +distillation.num_workers=8 \
    +distillation.distillation_loss.loss_mode=k3 \
    +distillation.distillation_loss.use_task_rewards=false \
    +distillation.distillation_loss.use_policy_gradient=false \
    +distillation.distillation_loss.distillation_loss_coef=1.0 \
    +distillation.teacher_model.model_path="${TEACHER_PATH}" \
    +distillation.teacher_model.enable_resource_pool=false \
    +distillation.teacher_model.n_gpus_per_node=8 \
    +distillation.teacher_model.nnodes=0 \
    +distillation.teacher_model.inference.tensor_model_parallel_size=${TEACHER_TP} \
    +distillation.teacher_model.inference.gpu_memory_utilization=0.30 \
    +distillation.teacher_model.inference.enforce_eager=true \
    +distillation.teacher_model.inference.max_model_len=$((max_prompt_length + max_response_length)) \
    reward_model.reward_manager=dapo \
    reward.reward_kwargs.overlong_buffer_cfg.enable=False \
    reward.reward_kwargs.max_resp_len=${max_response_length} \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=False \
    trainer.test_freq=2 \
    trainer.save_freq=50 \
    trainer.max_actor_ckpt_to_keep=2 \
    +trainer.hf_push.enable=True \
    +trainer.hf_push.repo_id="${HF_PUSH_REPO}" \
    +trainer.hf_push.private=False \
    +trainer.hf_push.delete_local_after=True \
    +trainer.hf_push.max_to_keep=10 \
    'actor_rollout_ref.actor.checkpoint.save_contents=[hf_model]' \
    trainer.total_epochs=2 \
    trainer.total_training_steps=200 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto $@
