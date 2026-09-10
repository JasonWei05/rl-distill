#!/usr/bin/env bash
# One DAPO RL run of a small Gemma 4 base (E2B/E4B) on one DeepScaleR difficulty band with the difficulty-sweep recipe
# (seed 42 sweep, 2026-08: GRPO n=16, bsz 64 / mini 32, lr 1e-6, 8k responses, early stopping patience 5 on
# val-core/math/acc/mean@16, 400 steps max) plus resumable checkpoints: ROLLING_CHECKPOINT_FREQ=5 (model + Adam + LR/RNG +
# dataloader cursor -> S3 rolling slot), permanent + HF push every 10. A preempted borrowing pod restores the newest S3
# checkpoint at restart and resumes.
#   SIZE=e2b|e4b BAND=easy|medium|hard SEED=43 [GPUS=4] bash launch_gemma4_rl_band_seed.sh [extra launch_st_job args]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
SIZE="${SIZE:?e2b or e4b}"; BAND="${BAND:?easy|medium|hard}"; SEED="${SEED:?data seed}"; GPUS="${GPUS:-4}"
# Per-model memory layout = the seed-42 packed-queue values (run_gemma4_e2b_e4b_difficulty_queue.sh): the run-file defaults
# (0.5 GiB KV, util 0.65, no packing) are too small for E4B at max_model_len 12288 ("0.66 GiB KV cache is needed").
case "${SIZE}" in
  e2b) MODEL=google/gemma-4-E2B; REV=d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f
       MEM="MICRO_BATCH_SIZE_PER_GPU=8,MAX_PADDED_TOKENS_PER_MICROBATCH=12288,ROLLOUT_GPU_MEMORY_UTILIZATION=0.25,VLLM_KV_CACHE_MEMORY_BYTES=536870912" ;;
  e4b) MODEL=google/gemma-4-E4B; REV=411aa17b749aa952df1359d2dcea73917a544d9a
       MEM="MICRO_BATCH_SIZE_PER_GPU=1,MAX_PADDED_TOKENS_PER_MICROBATCH=4096,ROLLOUT_GPU_MEMORY_UTILIZATION=0.25,VLLM_KV_CACHE_MEMORY_BYTES=1073741824" ;;
  *) echo "FATAL: SIZE must be e2b or e4b" >&2; exit 2 ;;
esac
case "${BAND}" in easy|medium|hard) ;; *) echo "FATAL: BAND must be easy|medium|hard" >&2; exit 2 ;; esac
SUFFIX="${RUN_NAME_SUFFIX:-26b-bands-es5}"          # HF repo JWei05/DAPO-gemma4-${SIZE}-PT-DeepScaleR-gemma26b-${BAND}-seed${SEED}-${SUFFIX}
S3_BASE="${S3_BASE:-s3://scale-ml/genai/rl-distill/gemma4-difficulty-s${SEED}-20260910}"
KEY="${SIZE}-${BAND}"
ENV_VARS="GEMMA4_MODEL=${MODEL},GEMMA4_MODEL_REVISION=${REV},DATA_SEED=${SEED},RUN_NAME_SUFFIX=${SUFFIX},RUN_SLOT=gemma4-${KEY}-s${SEED}"
ENV_VARS+=",DIFFICULTY_DATASET_SOURCE=gemma4_26b_bands,DIFFICULTY_DATASET=${BAND},DIFFICULTY_DATASET_REPO=JWei05/DeepScaleR-Easy-Medium-Hard-Gemma-26B-PT-10k,DIFFICULTY_DATASET_REVISION=a0ba3c3dc07c7bc27e901670ceb1a0b0ceeaa8db"
# sweep recipe (the E2B/E4B seed-42 launch env, verbatim)
ENV_VARS+=",ACTOR_FSDP_SIZE=-1,ACTOR_LR=1e-6,ACTOR_LR_WARMUP_STEPS=20,EARLY_STOPPING_ENABLED=True,EARLY_STOPPING_INCLUDE_INITIAL_VALIDATION=True,EARLY_STOPPING_METRIC=val-core/math/acc/mean@16,EARLY_STOPPING_MIN_DELTA=0.0,EARLY_STOPPING_MODE=max,EARLY_STOPPING_PATIENCE=5"
ENV_VARS+=",ENABLE_OVERLONG_BUFFER=True,FSDP_CPU_OFFLOAD_POLICY=False,GEN_PROMPT_BSZ=64,GEN_TP=1,HF_PUSH_FREQ=10,HF_PUSH_MAX_TO_KEEP=8,LOG_TRAIN_GENERATIONS=100,LOG_VAL_GENERATIONS=100,MAX_ACTOR_CKPT_TO_KEEP=6,MAX_MODEL_LEN=12288,MAX_PROMPT_LENGTH=4096,MAX_RESPONSE_LENGTH=8192,N_RESP_PER_PROMPT=16,OFFLOAD=False,OVERLONG_BUFFER_LEN=2048,OVERLONG_PENALTY_FACTOR=1.0"
ENV_VARS+=",${MEM}"
ENV_VARS+=",ROLLOUT_ENFORCE_EAGER=False,ROUTER_REPLAY_MODE=disabled,ROUTER_Z_LOSS_COEF=0.0,SAVE_FREQ=10,SP_SIZE=1,TEST_FREQ=10,TOTAL_TRAINING_STEPS=400,TRAIN_PROMPT_BSZ=64,TRAIN_PROMPT_MINI_BSZ=32,VAL_BEFORE_TRAIN=True,VAL_N=1,VLLM_DISABLE_COMPILE_CACHE=0"
# resumable checkpoints every 5 steps + durable completion markers
ENV_VARS+=",ROLLING_CHECKPOINT_ENABLED=True,ROLLING_CHECKPOINT_FREQ=${ROLLING_CHECKPOINT_FREQ:-5},FULL_CHECKPOINT_S3_URI=${S3_BASE}-full-checkpoints/${KEY},RUN_ARTIFACT_S3_URI=${S3_BASE}/gemma4-${KEY}"
ENV_VARS+=",WANDB_RUN_ID=g4ds26b-${KEY}-s${SEED}-v1,WANDB_RESUME=allow"
echo "LAUNCH_ENV ${ENV_VARS}"
exec bash launch_st_with_code.sh --gpus-per-instance "${GPUS}" --priority high --allow-borrowing --active-deadline-hours 240 \
  --run-file run_gemma4_pt_deepscaler_4of4strict_rl.sh --job-name "g4-${SIZE}-s${SEED}-${BAND:0:4}" --env-vars "${ENV_VARS}" "$@"
