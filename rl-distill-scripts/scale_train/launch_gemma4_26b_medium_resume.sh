#!/usr/bin/env bash
# RESUME of the untrained-26B-A4B DeepScaleR *medium* DAPO run (W&B g4ds26b-26b-a4b-medium-s42-v1: 0.656 @ 90 on ScaleTrain, then continued
# off-cluster to 0.6725 @ 140 and crashed at step 146 without ever plateauing). The newest FULL checkpoint (weights + Adam + dataloader
# cursor + early-stopping state) anywhere reachable is the S3 permanent step 90 (steps 120/130/140 exist only as weight-only HF exports),
# so this restores step 90 and re-does 91+. Recipe unchanged (GRPO n=16, bsz 64 / mini 32, lr 1e-6, 8k responses, seed 42, R3 replay,
# 400-step cap); early-stopping patience raised 2 -> 5 via EARLY_STOPPING_MIGRATE_PATIENCE_FROM (history kept). Memory layout = the fast
# one-node 26B layout + cuDNN SDPA ON (§9.0m A/B). New W&B run id (the original has steps <= 146 logged; resuming it would drop 91-146).
#   bash launch_gemma4_26b_medium_resume.sh            # whole node (8 GPUs, the checkpoint is world_size_8), borrowing on, priority high
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
RUN_TAG="${RUN_TAG:-26b-bands-es5}"     # original suffix -> run name DAPO-gemma4-26b-a4b-pt-DeepScaleR-gemma26b-medium-seed42-26b-bands-es5
S3_BASE="${S3_BASE:-s3://scale-ml/genai/rl-distill/gemma4-difficulty-s42-20260819}"
CUDNN_SDPA="${CUDNN_SDPA:-1}"   # 2026-09-19 A/B on this run: cuDNN ON is ~7% faster on 26B-A4B (118 vs 125-129 s update); E4B is the opposite
ENV_VARS="GEMMA4_MODEL=google/gemma-4-26B-A4B,GEMMA4_MODEL_REVISION=24548b62aa021d562695c04aaf7758a1ea47990b"
ENV_VARS+=",DIFFICULTY_DATASET_SOURCE=gemma4_26b_bands,DIFFICULTY_DATASET=medium,DIFFICULTY_DATASET_REPO=JWei05/DeepScaleR-Easy-Medium-Hard-Gemma-26B-PT-10k,DIFFICULTY_DATASET_REVISION=a0ba3c3dc07c7bc27e901670ceb1a0b0ceeaa8db"
ENV_VARS+=",DATA_SEED=42,RUN_NAME_SUFFIX=${RUN_TAG},RUN_SLOT=gemma4-26b-a4b-medium,VERL_VLLM_PORT_BASE=54000"
# sweep recipe (identical for E4B and 12B)
ENV_VARS+=",ACTOR_FSDP_SIZE=-1,ACTOR_LR=1e-6,ACTOR_LR_WARMUP_STEPS=20,EARLY_STOPPING_ENABLED=True,EARLY_STOPPING_INCLUDE_INITIAL_VALIDATION=True,EARLY_STOPPING_METRIC=val-core/math/acc/mean@16,EARLY_STOPPING_MIN_DELTA=0.0,EARLY_STOPPING_MODE=max,EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE:-5},EARLY_STOPPING_MIGRATE_PATIENCE_FROM=${EARLY_STOPPING_MIGRATE_PATIENCE_FROM:-2}"
ENV_VARS+=",ENABLE_OVERLONG_BUFFER=True,GEN_PROMPT_BSZ=64,GEN_TP=1,HF_PUSH_FREQ=10,HF_PUSH_MAX_TO_KEEP=8,LOG_TRAIN_GENERATIONS=100,LOG_VAL_GENERATIONS=100,MAX_ACTOR_CKPT_TO_KEEP=6,MAX_MODEL_LEN=12288,MAX_PROMPT_LENGTH=4096,MAX_RESPONSE_LENGTH=8192,N_RESP_PER_PROMPT=16,OVERLONG_BUFFER_LEN=2048,OVERLONG_PENALTY_FACTOR=1.0"
ENV_VARS+=",ROLLOUT_ENFORCE_EAGER=False,ROUTER_REPLAY_MODE=R3,ROUTER_Z_LOSS_COEF=0.0,SAVE_FREQ=10,SP_SIZE=1,TEST_FREQ=10,TOTAL_TRAINING_STEPS=400,TRAIN_PROMPT_BSZ=64,TRAIN_PROMPT_MINI_BSZ=32,VAL_BEFORE_TRAIN=True,VAL_N=1,VLLM_DISABLE_COMPILE_CACHE=0"
# fast one-node 26B-A4B layout (RESUME_GEMMA4_26B_A4B_LOCAL.md) + cuDNN SDPA switch
ENV_VARS+=",FSDP_CPU_OFFLOAD_POLICY=False,OFFLOAD=True,VLLM_SLEEP_MODE=True,MAX_PADDED_TOKENS_PER_MICROBATCH=8192,MICRO_BATCH_SIZE_PER_GPU=4,ROLLOUT_GPU_MEMORY_UTILIZATION=0.10,VERL_SKIP_VLLM_MM_WEIGHT_RELOAD=1,VLLM_KV_CACHE_MEMORY_BYTES=3221225472"
ENV_VARS+=",VERL_GEMMA4_CUDNN_SDPA=${CUDNN_SDPA},VERL_GEMMA4_EVAL_CUDNN_SDPA=${CUDNN_SDPA},TORCH_CUDNN_SDPA_ENABLED=${CUDNN_SDPA}"
# S3-only artifacts (user policy since 2026-09-15): permanent S3 checkpoints carry the HF snapshot, best_hf/ published at the end
ENV_VARS+=",HF_PUSH_ENABLE=${HF_PUSH_ENABLE:-False},HF_PUSH_REQUIRED=False"
# resumable checkpoints every 5 steps + durable completion markers
ENV_VARS+=",ROLLING_CHECKPOINT_ENABLED=True,ROLLING_CHECKPOINT_FREQ=5,FULL_CHECKPOINT_S3_URI=${S3_BASE}-full-checkpoints/26b-a4b-medium,RUN_ARTIFACT_S3_URI=${S3_BASE}/gemma4-26b-a4b-medium"
ENV_VARS+=",WANDB_RUN_ID=${WANDB_RUN_ID:-g4ds26b-26b-a4b-medium-s42-resume90-v1},WANDB_RESUME=allow"
echo "LAUNCH_ENV ${ENV_VARS}"
exec bash launch_st_with_code.sh --gpus-per-instance 8 --priority high --allow-borrowing --active-deadline-hours 240 \
  --run-file run_gemma4_pt_deepscaler_4of4strict_rl.sh --job-name "g4-26b-med-resume" --env-vars "${ENV_VARS}" "$@"
