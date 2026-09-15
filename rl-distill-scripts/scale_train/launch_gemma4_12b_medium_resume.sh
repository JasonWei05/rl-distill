#!/usr/bin/env bash
# Resume the untrained-12B DAPO run on the DeepScaleR *medium* band (seed 42; §1 / §8 teacher `12b-medium`) from its last
# checkpoint on S3 (permanent step 130: fp32 weights + Adam + LR/RNG + dataloader cursor + early-stopping history) on ONE
# 8-GPU ScaleTrain node, with
#   * early stopping = 4 non-improving validations IN A ROW (the original run stopped at 130 on patience 1: 0.51875 vs best
#     0.5208 @ 120). EARLY_STOPPING_MIGRATE_PATIENCE_FROM=1 keeps the saved best/miss history (misses=1) and only recomputes
#     the trigger flag, so the next validation is miss 2 of 4;
#   * the fast update layout measured on the 26B resume (2026-09-03, 4.9x/step): MICRO_BATCH_SIZE_PER_GPU=4 under an 8192
#     padded-token cap, phase-level OFFLOAD=True instead of the per-layer FSDP2 CPU offload policy, vLLM sleep mode during the
#     update; gradient-neutral (token-mean loss is normalized by the global token count). Every knob is overridable.
#   * S3-only artifacts (no Hub pushes); HF_PUSH_REPO stays the original repo name because the completion receipt records it.
# BEFORE LAUNCH the finished run's completion markers must be moved aside (the run-file exits RUN_ALREADY_COMPLETE otherwise):
#   see move_gemma4_12b_medium_completion_markers.sh (same directory).
#   bash launch_gemma4_12b_medium_resume.sh                       # whole node, borrowing on, priority high
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
S3_BASE="${S3_BASE:-s3://scale-ml/genai/rl-distill/gemma4-difficulty-s42-20260819}"
RUN_TAG="${RUN_TAG:-26b-bands-es5}"     # original experiment: DAPO-gemma4-12b-pt-DeepScaleR-gemma26b-medium-seed42-26b-bands-es5
ENV_VARS="GEMMA4_MODEL=google/gemma-4-12B,GEMMA4_MODEL_REVISION=023679ed352de9bb66cc873c9009ce3482585c08"
ENV_VARS+=",DIFFICULTY_DATASET_SOURCE=gemma4_26b_bands,DIFFICULTY_DATASET=medium,DIFFICULTY_DATASET_REPO=JWei05/DeepScaleR-Easy-Medium-Hard-Gemma-26B-PT-10k,DIFFICULTY_DATASET_REVISION=a0ba3c3dc07c7bc27e901670ceb1a0b0ceeaa8db"
ENV_VARS+=",DATA_SEED=42,RUN_NAME_SUFFIX=${RUN_TAG},RUN_SLOT=gemma4-12b-medium-s42-resume,VERL_VLLM_PORT_BASE=56000"
ENV_VARS+=",ACTOR_FSDP_SIZE=-1,ACTOR_LR=1e-6,ACTOR_LR_WARMUP_STEPS=20"
ENV_VARS+=",EARLY_STOPPING_ENABLED=True,EARLY_STOPPING_INCLUDE_INITIAL_VALIDATION=True,EARLY_STOPPING_METRIC=val-core/math/acc/mean@16,EARLY_STOPPING_MIN_DELTA=0.0,EARLY_STOPPING_MODE=max"
ENV_VARS+=",EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE:-4},EARLY_STOPPING_MIGRATE_PATIENCE_FROM=${EARLY_STOPPING_MIGRATE_PATIENCE_FROM:-1}"
ENV_VARS+=",ENABLE_OVERLONG_BUFFER=True,GEN_PROMPT_BSZ=64,GEN_TP=1,LOG_TRAIN_GENERATIONS=100,LOG_VAL_GENERATIONS=100,MAX_ACTOR_CKPT_TO_KEEP=6,MAX_MODEL_LEN=12288,MAX_PROMPT_LENGTH=4096,MAX_RESPONSE_LENGTH=8192"
ENV_VARS+=",ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-False},ROUTER_REPLAY_MODE=disabled,ROUTER_Z_LOSS_COEF=0.0,SAVE_FREQ=10,SP_SIZE=1,TEST_FREQ=10,TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-400},TRAIN_PROMPT_BSZ=64,TRAIN_PROMPT_MINI_BSZ=32,VAL_BEFORE_TRAIN=True,VAL_N=1,VLLM_DISABLE_COMPILE_CACHE=0"
# fast update layout (26B-measured); the original 12B recipe was MICRO_BATCH 1 / cap 4096 / FSDP_CPU_OFFLOAD_POLICY=True / OFFLOAD=False / no sleep
ENV_VARS+=",MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU:-4},MAX_PADDED_TOKENS_PER_MICROBATCH=${MAX_PADDED_TOKENS_PER_MICROBATCH:-8192}"
ENV_VARS+=",FSDP_CPU_OFFLOAD_POLICY=${FSDP_CPU_OFFLOAD_POLICY:-False},OFFLOAD=${OFFLOAD:-True},VLLM_SLEEP_MODE=${VLLM_SLEEP_MODE:-True}"
# rollout memory: with the trainer state offloaded between phases the engine can take more KV than the recipe's 5 GiB
ENV_VARS+=",ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.45},VLLM_KV_CACHE_MEMORY_BYTES=${VLLM_KV_CACHE_MEMORY_BYTES:-10737418240},VERL_SKIP_VLLM_MM_WEIGHT_RELOAD=1"
# S3-only; same prefixes as the original run so restore-latest picks up global_step_130 (+ data.pt + validation_early_stopping.json)
ENV_VARS+=",HF_PUSH_ENABLE=${HF_PUSH_ENABLE:-False},HF_PUSH_REQUIRED=False,HF_PUSH_REPO=JWei05/DAPO-gemma4-12b-PT-DeepScaleR-gemma26b-medium-seed42-${RUN_TAG},HF_PUSH_FREQ=10,HF_PUSH_MAX_TO_KEEP=8"
ENV_VARS+=",ROLLING_CHECKPOINT_ENABLED=True,ROLLING_CHECKPOINT_FREQ=${ROLLING_CHECKPOINT_FREQ:-5},FULL_CHECKPOINT_S3_URI=${S3_BASE}-full-checkpoints/12b-medium,RUN_ARTIFACT_S3_URI=${S3_BASE}/gemma4-12b-medium"
ENV_VARS+=",WANDB_RUN_ID=g4ds26b-12b-medium-s42-v1,WANDB_RESUME=allow"
echo "LAUNCH_ENV ${ENV_VARS}"
exec bash launch_st_with_code.sh --gpus-per-instance 8 --priority high --allow-borrowing --active-deadline-hours 240 \
  --run-file run_gemma4_pt_deepscaler_4of4strict_rl.sh --job-name "${JOB_NAME:-g4-12b-med-resume}" --env-vars "${ENV_VARS}" "$@"
