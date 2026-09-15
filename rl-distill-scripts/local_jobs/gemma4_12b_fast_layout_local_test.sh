#!/usr/bin/env bash
# Local 4-GPU smoke/timing test of the fast update layout for the untrained 12B (fresh start from the base model, 2 steps,
# no S3/HF/W&B). Purpose: confirm MICRO_BATCH_SIZE_PER_GPU=4 + 8192 cap + OFFLOAD=True + vLLM sleep fit and measure
# update_actor vs the recipe layout before the ScaleTrain resume. NOTE: 4 GPUs hold 2x the per-GPU fp32 state of the
# 8-GPU node (48 vs 24 GB), so memory here is a pessimistic bound; the step-130 checkpoint itself (world_size 8) cannot
# be loaded here.
#   GPUS=0,1,2,3 MICRO_BATCH_SIZE_PER_GPU=4 MAX_PADDED_TOKENS_PER_MICROBATCH=8192 bash rl-distill-scripts/local_jobs/gemma4_12b_fast_layout_local_test.sh
set -euo pipefail
REPO=/mnt/efs/jasonwei/rl-distill; cd "$REPO"; if [ -f .env ]; then set -a; source .env; set +a; fi
export CUDA_VISIBLE_DEVICES="${GPUS:-0,1,2,3}"
TAG="${TAG:-mbs${MICRO_BATCH_SIZE_PER_GPU:-4}-cap${MAX_PADDED_TOKENS_PER_MICROBATCH:-8192}-offpol${FSDP_CPU_OFFLOAD_POLICY:-False}}"
export CKPTS_DIR="/tmp/gemma4_12b_local_test/${TAG}/ckpts" DATA_DIR="/tmp/gemma4_12b_local_test/data" HF_HOME="${HF_HOME:-/tmp/hf_cache}" RAY_DATA_HOME="/tmp/gemma4_12b_local_test/verl"
export RAY_TEMP_DIR="/tmp/ray_12b_local_${TAG}" VENV="${VENV:-/tmp/.venv-gemma4}"
export NCCL_SOCKET_IFNAME=lo NCCL_SOCKET_FAMILY=AF_INET GLOO_SOCKET_IFNAME=lo
export RAY_ADDRESS=local
export GEMMA4_MODEL=google/gemma-4-12B GEMMA4_MODEL_REVISION=023679ed352de9bb66cc873c9009ce3482585c08
export DIFFICULTY_DATASET_SOURCE=gemma4_26b_bands DIFFICULTY_DATASET=medium DIFFICULTY_DATASET_REPO=JWei05/DeepScaleR-Easy-Medium-Hard-Gemma-26B-PT-10k DIFFICULTY_DATASET_REVISION=a0ba3c3dc07c7bc27e901670ceb1a0b0ceeaa8db
export DATA_SEED=42 RUN_NAME_SUFFIX="local-fastlayout-${TAG}" RUN_SLOT="gemma4-12b-local-${TAG}" VERL_VLLM_PORT_BASE="${VERL_VLLM_PORT_BASE:-57000}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-2}" TRAIN_PROMPT_BSZ=64 GEN_PROMPT_BSZ=64 N_RESP_PER_PROMPT=16 TRAIN_PROMPT_MINI_BSZ=32
export ACTOR_LR=1e-6 ACTOR_LR_WARMUP_STEPS=20 MAX_PROMPT_LENGTH=4096 MAX_RESPONSE_LENGTH=8192 MAX_MODEL_LEN=12288
export ENABLE_OVERLONG_BUFFER=True SP_SIZE=1 GEN_TP=1 ACTOR_FSDP_SIZE=-1 ROUTER_REPLAY_MODE=disabled ROUTER_Z_LOSS_COEF=0.0 VERL_SKIP_VLLM_MM_WEIGHT_RELOAD=1
export MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-4}" MAX_PADDED_TOKENS_PER_MICROBATCH="${MAX_PADDED_TOKENS_PER_MICROBATCH:-8192}"
export FSDP_CPU_OFFLOAD_POLICY="${FSDP_CPU_OFFLOAD_POLICY:-False}" OFFLOAD="${OFFLOAD:-True}" VLLM_SLEEP_MODE="${VLLM_SLEEP_MODE:-True}"
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.45}" VLLM_KV_CACHE_MEMORY_BYTES="${VLLM_KV_CACHE_MEMORY_BYTES:-10737418240}" ROLLOUT_ENFORCE_EAGER="${ROLLOUT_ENFORCE_EAGER:-False}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export EARLY_STOPPING_ENABLED=False VAL_BEFORE_TRAIN=False TEST_FREQ=1000 SAVE_FREQ=1000 ROLLING_CHECKPOINT_ENABLED=False HF_PUSH_ENABLE=False HF_PUSH_REQUIRED=False
export LOG_TRAIN_GENERATIONS=10 LOG_VAL_GENERATIONS=10 WANDB_MODE=offline
exec bash rl-distill-scripts/scale_train/run_gemma4_pt_deepscaler_4of4strict_rl.sh "$@"
