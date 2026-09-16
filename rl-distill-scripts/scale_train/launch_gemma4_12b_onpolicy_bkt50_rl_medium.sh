#!/usr/bin/env bash
# DAPO RL (medium band) on the tail-bucket on-policy-distilled 12B student at step 50 (lowest offline reverse KL to the E4B base
# teacher, DISTILLATION_EXPERIMENTS.md §9.0f/§9.0i), as a 4-GPU ScaleTrain job. Recipe = launch_gemma4_12b_distilled_rl_medium.sh
# (§9.0: lr 1e-6, 64×16, patience 5, cap 400, S3-only saving) with the fast update layout validated on 4 GPUs in §9.0h
# (mbs 4 / 8192-token cap / phase-level OFFLOAD + vLLM sleep). Init weights come straight from S3 via GEMMA4_INIT_MODEL_S3_URI.
#
#   bash launch_gemma4_12b_onpolicy_bkt50_rl_medium.sh      # 4 GPUs, borrowing, priority high
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
INIT_S3="${INIT_S3:-s3://scale-ml/genai/rl-distill/gemma4-12b-from-e4bbase-distill-onpolicy-full-checkpoints/12b-medium-onpolicy-studenttop128-bucket-from-e4bbase-distill/global_step_50/actor/huggingface/}"
RUN_TAG="${RUN_TAG:-from-onpolicy-bkt50-es5}"
S3_BASE="${S3_BASE:-s3://scale-ml/genai/rl-distill/gemma4-12b-from-onpolicy-bkt50-rl}"
GPUS_PER_INSTANCE="${GPUS_PER_INSTANCE:-4}"
ENV_VARS="GEMMA4_MODEL=google/gemma-4-12B,GEMMA4_MODEL_REVISION=023679ed352de9bb66cc873c9009ce3482585c08,GEMMA4_INIT_MODEL_S3_URI=${INIT_S3}"
ENV_VARS+=",DIFFICULTY_DATASET_SOURCE=gemma4_26b_bands,DIFFICULTY_DATASET=medium,DIFFICULTY_DATASET_REPO=JWei05/DeepScaleR-Easy-Medium-Hard-Gemma-26B-PT-10k,DIFFICULTY_DATASET_REVISION=a0ba3c3dc07c7bc27e901670ceb1a0b0ceeaa8db"
ENV_VARS+=",DATA_SEED=42,RUN_NAME_SUFFIX=${RUN_TAG},RUN_SLOT=gemma4-12b-medium-${RUN_TAG},VERL_VLLM_PORT_BASE=58000"
ENV_VARS+=",ACTOR_FSDP_SIZE=-1,ACTOR_LR=1e-6,ACTOR_LR_WARMUP_STEPS=20,EARLY_STOPPING_ENABLED=True,EARLY_STOPPING_INCLUDE_INITIAL_VALIDATION=True,EARLY_STOPPING_METRIC=val-core/math/acc/mean@16,EARLY_STOPPING_MIN_DELTA=0.0,EARLY_STOPPING_MODE=max,EARLY_STOPPING_PATIENCE=5"
ENV_VARS+=",ENABLE_OVERLONG_BUFFER=True,OVERLONG_BUFFER_LEN=2048,OVERLONG_PENALTY_FACTOR=1.0,GEN_PROMPT_BSZ=64,GEN_TP=1,LOG_TRAIN_GENERATIONS=100,LOG_VAL_GENERATIONS=100,MAX_ACTOR_CKPT_TO_KEEP=2,MAX_MODEL_LEN=12288,MAX_PROMPT_LENGTH=4096,MAX_RESPONSE_LENGTH=8192,N_RESP_PER_PROMPT=16"
ENV_VARS+=",ROLLOUT_ENFORCE_EAGER=False,ROUTER_REPLAY_MODE=disabled,ROUTER_Z_LOSS_COEF=0.0,SAVE_FREQ=10,SP_SIZE=1,TEST_FREQ=10,TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-400},TRAIN_PROMPT_BSZ=64,TRAIN_PROMPT_MINI_BSZ=32,VAL_BEFORE_TRAIN=True,VAL_N=1,VLLM_DISABLE_COMPILE_CACHE=0"
ENV_VARS+=",MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU:-4},MAX_PADDED_TOKENS_PER_MICROBATCH=${MAX_PADDED_TOKENS_PER_MICROBATCH:-8192}"
ENV_VARS+=",FSDP_CPU_OFFLOAD_POLICY=${FSDP_CPU_OFFLOAD_POLICY:-False},OFFLOAD=${OFFLOAD:-True},VLLM_SLEEP_MODE=${VLLM_SLEEP_MODE:-True}"
ENV_VARS+=",ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.45},VLLM_KV_CACHE_MEMORY_BYTES=${VLLM_KV_CACHE_MEMORY_BYTES:-10737418240},VERL_SKIP_VLLM_MM_WEIGHT_RELOAD=1"
ENV_VARS+=",HF_PUSH_ENABLE=False,HF_PUSH_REQUIRED=False,HF_PUSH_FREQ=10,HF_PUSH_MAX_TO_KEEP=8"
ENV_VARS+=",ROLLING_CHECKPOINT_ENABLED=True,ROLLING_CHECKPOINT_FREQ=5,FULL_CHECKPOINT_S3_URI=${S3_BASE}-full-checkpoints/12b-medium-${RUN_TAG},RUN_ARTIFACT_S3_URI=${S3_BASE}/gemma4-12b-medium-${RUN_TAG}"
ENV_VARS+=",WANDB_RUN_ID=g4ds26b-12b-medium-${RUN_TAG}-s42-v1,WANDB_RESUME=allow"
echo "LAUNCH_ENV ${ENV_VARS}"
exec bash launch_st_with_code.sh --gpus-per-instance "${GPUS_PER_INSTANCE}" --priority high --allow-borrowing --active-deadline-hours 240 \
  --run-file run_gemma4_pt_deepscaler_4of4strict_rl.sh --job-name "${JOB_NAME:-g4-12b-bkt50-rl-med}" --env-vars "${ENV_VARS}" "$@"
