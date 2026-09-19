#!/usr/bin/env bash
# DAPO RL on the DeepScaleR *medium* band starting from the 26B-A4B student distilled off-policy from the E4B base
# (JWei05/Distill-gemma4-e4b-base-medium-to-26b-base/step_001000, §9.1). Recipe = the 12B distilled-init RL run
# (launch_gemma4_12b_distilled_rl_medium.sh, §9.0): GRPO n=16, bsz 64 / mini 32, lr 1e-6, 8k responses, early stopping
# patience 5 on val-core/math/acc/mean@16, 400 steps max, rolling resumable checkpoints every 5 steps + permanent every 10.
# Memory layout = the fast one-node 26B-A4B layout from RESUME_GEMMA4_26B_A4B_LOCAL.md (2026-09-03: 439 s/step vs 2157 s):
# 8 GPUs FSDP2 DP8, mbs 4 / 8192-token padded cap, FSDP_CPU_OFFLOAD_POLICY=False + phase-level OFFLOAD=True, vLLM sleep mode,
# GEN_TP=1, rollout util 0.10 + 3 GiB KV, R3 router replay. NEW (2026-09-19, §9.0l): VERL_GEMMA4_CUDNN_SDPA=0 — the cuDNN SDPA
# backend the engine forces on for Gemma 4 train forwards cost 3x on E4B; first test on 26B-A4B (set CUDNN_SDPA=1 to revert).
#   bash launch_gemma4_26b_distilled_rl_medium.sh            # whole node (p5.48xlarge, 8 GPUs), borrowing on, priority high
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
INIT_REPO="${INIT_REPO:-JWei05/Distill-gemma4-e4b-base-medium-to-26b-base}"
INIT_REVISION="${INIT_REVISION:-faacc50ec67a5688beef8770ee489be0bfe8bc52}"   # commit that pushed step_001000 (2026-09-10)
INIT_SUBFOLDER="${INIT_SUBFOLDER:-step_001000}"
RUN_TAG="${RUN_TAG:-from-e4bbase-distill-es5}"     # -> run name DAPO-gemma4-26b-a4b-PT-DeepScaleR-gemma26b-medium-seed42-${RUN_TAG}
S3_BASE="${S3_BASE:-s3://scale-ml/genai/rl-distill/gemma4-26b-from-e4bbase-distill-rl}"
CUDNN_SDPA="${CUDNN_SDPA:-0}"
ENV_VARS="GEMMA4_MODEL=google/gemma-4-26B-A4B,GEMMA4_MODEL_REVISION=24548b62aa021d562695c04aaf7758a1ea47990b"
ENV_VARS+=",GEMMA4_INIT_MODEL_REPO=${INIT_REPO},GEMMA4_INIT_MODEL_REVISION=${INIT_REVISION},GEMMA4_INIT_MODEL_SUBFOLDER=${INIT_SUBFOLDER}"
ENV_VARS+=",DIFFICULTY_DATASET_SOURCE=gemma4_26b_bands,DIFFICULTY_DATASET=medium,DIFFICULTY_DATASET_REPO=JWei05/DeepScaleR-Easy-Medium-Hard-Gemma-26B-PT-10k,DIFFICULTY_DATASET_REVISION=a0ba3c3dc07c7bc27e901670ceb1a0b0ceeaa8db"
ENV_VARS+=",DATA_SEED=42,RUN_NAME_SUFFIX=${RUN_TAG},RUN_SLOT=gemma4-26b-a4b-medium-${RUN_TAG},VERL_VLLM_PORT_BASE=54000"
# sweep recipe (identical for E4B and 12B)
ENV_VARS+=",ACTOR_FSDP_SIZE=-1,ACTOR_LR=1e-6,ACTOR_LR_WARMUP_STEPS=20,EARLY_STOPPING_ENABLED=True,EARLY_STOPPING_INCLUDE_INITIAL_VALIDATION=True,EARLY_STOPPING_METRIC=val-core/math/acc/mean@16,EARLY_STOPPING_MIN_DELTA=0.0,EARLY_STOPPING_MODE=max,EARLY_STOPPING_PATIENCE=5"
ENV_VARS+=",ENABLE_OVERLONG_BUFFER=True,GEN_PROMPT_BSZ=64,GEN_TP=1,HF_PUSH_FREQ=10,HF_PUSH_MAX_TO_KEEP=8,LOG_TRAIN_GENERATIONS=100,LOG_VAL_GENERATIONS=100,MAX_ACTOR_CKPT_TO_KEEP=6,MAX_MODEL_LEN=12288,MAX_PROMPT_LENGTH=4096,MAX_RESPONSE_LENGTH=8192,N_RESP_PER_PROMPT=16,OVERLONG_BUFFER_LEN=2048,OVERLONG_PENALTY_FACTOR=1.0"
ENV_VARS+=",ROLLOUT_ENFORCE_EAGER=False,ROUTER_REPLAY_MODE=R3,ROUTER_Z_LOSS_COEF=0.0,SAVE_FREQ=10,SP_SIZE=1,TEST_FREQ=10,TOTAL_TRAINING_STEPS=400,TRAIN_PROMPT_BSZ=64,TRAIN_PROMPT_MINI_BSZ=32,VAL_BEFORE_TRAIN=True,VAL_N=1,VLLM_DISABLE_COMPILE_CACHE=0"
# fast one-node 26B-A4B layout (RESUME_GEMMA4_26B_A4B_LOCAL.md) + cuDNN SDPA switch
ENV_VARS+=",FSDP_CPU_OFFLOAD_POLICY=False,OFFLOAD=True,VLLM_SLEEP_MODE=True,MAX_PADDED_TOKENS_PER_MICROBATCH=8192,MICRO_BATCH_SIZE_PER_GPU=4,ROLLOUT_GPU_MEMORY_UTILIZATION=0.10,VERL_SKIP_VLLM_MM_WEIGHT_RELOAD=1,VLLM_KV_CACHE_MEMORY_BYTES=3221225472"
ENV_VARS+=",VERL_GEMMA4_CUDNN_SDPA=${CUDNN_SDPA},VERL_GEMMA4_EVAL_CUDNN_SDPA=${CUDNN_SDPA},TORCH_CUDNN_SDPA_ENABLED=${CUDNN_SDPA}"
# S3-only artifacts (user policy since 2026-09-15): permanent S3 checkpoints carry the HF snapshot, best_hf/ published at the end
ENV_VARS+=",HF_PUSH_ENABLE=${HF_PUSH_ENABLE:-False},HF_PUSH_REQUIRED=False"
# resumable checkpoints every 5 steps + durable completion markers
ENV_VARS+=",ROLLING_CHECKPOINT_ENABLED=True,ROLLING_CHECKPOINT_FREQ=5,FULL_CHECKPOINT_S3_URI=${S3_BASE}-full-checkpoints/26b-a4b-medium-${RUN_TAG},RUN_ARTIFACT_S3_URI=${S3_BASE}/gemma4-26b-a4b-medium-${RUN_TAG}"
ENV_VARS+=",WANDB_RUN_ID=g4ds26b-26b-a4b-medium-${RUN_TAG}-s42-v1,WANDB_RESUME=allow"
echo "LAUNCH_ENV ${ENV_VARS}"
exec bash launch_st_with_code.sh --gpus-per-instance 8 --priority high --allow-borrowing --active-deadline-hours 240 \
  --run-file run_gemma4_pt_deepscaler_4of4strict_rl.sh --job-name "g4-26b-distill-rl-med" --env-vars "${ENV_VARS}" "$@"
