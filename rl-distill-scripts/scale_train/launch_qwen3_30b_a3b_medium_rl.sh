#!/usr/bin/env bash
# DAPO RL of Qwen3-30B-A3B-Base on the Qwen3-30B-defined DeepScaleR *medium* band
# (JWei05/DeepScaleR-Easy-Medium-Hard-Qwen3-30B-A3B-Base-10k, 30B k/8 in {2,3}; 3,000 train / 300 validation x16).
# Recipe = the Gemma 4 26B-A4B RL runs (launch_gemma4_26b_distilled_rl_medium.sh): GRPO n=16, bsz 64 / mini 32, lr 1e-6
# (20 warmup), token-mean loss, clip 0.2/0.28, 8k responses + 2k soft overlong buffer, token TIS (threshold 2), temp 1
# train = val, validation every 10 steps (mean@16), early stopping patience 5 on val-core/math/acc/mean@16, 400 steps max.
# Qwen deltas: 0-shot native ChatML + boxed instruction (the teacher's sampling prompt), <|im_end|>/<|im_start|> stop ids,
# Qwen3MoeDecoderLayer wrap, R3 router replay (vLLM's routes replayed in the trainer, as for 26B-A4B), GEN_TP=2 (61 GB of bf16
# weights per vLLM copy at TP 1 leaves no headroom on an 80 GB H100 at weight sync; the 26B-A4B at 52 GB already hit it).
# Early stopping counts a validation as a miss only if it is more than 1 point (0.01) below the best so far; 5 misses in
# a row stop the run (a score within 1 point of the best breaks the streak).
# Preemption: a fully resumable checkpoint (sharded fp32 model + Adam + LR/RNG + dataloader cursor) is published to a
# single rolling S3 slot every 5 steps and replaces the previous one (the run-file restores the newest complete
# checkpoint on every pod start); every 10 steps a permanent checkpoint with the HF export is kept for best-step selection.
#   bash launch_qwen3_30b_a3b_medium_rl.sh --image <uri>        # whole node (p5.48xlarge, 8 GPUs), borrowing on, priority high
# Use start_qwen3_30b_a3b_medium_rl_supervisor.sh to run it under the relaunching supervisor + rolling-checkpoint janitor.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
RUN_TAG="${RUN_TAG:-es5}"   # -> run name DAPO-qwen3-30b-a3b-base-pt-DeepScaleR-qwen30b-medium-seed42-${RUN_TAG}
S3_BASE="${S3_BASE:-s3://scale-ml/genai/rl-distill/qwen3-30b-a3b-base-rl}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-400}"
ENV_VARS="GEMMA4_MODEL=Qwen/Qwen3-30B-A3B-Base,GEMMA4_MODEL_REVISION=1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9"
ENV_VARS+=",DIFFICULTY_DATASET_SOURCE=qwen3_30b_bands,DIFFICULTY_DATASET=medium"
ENV_VARS+=",DATA_SEED=42,RUN_NAME_SUFFIX=${RUN_TAG},RUN_SLOT=qwen3-30b-a3b-medium-${RUN_TAG},VERL_VLLM_PORT_BASE=54000"
# sweep recipe (identical to the Gemma 4 runs)
ENV_VARS+=",ACTOR_FSDP_SIZE=-1,ACTOR_LR=1e-6,ACTOR_LR_WARMUP_STEPS=20,EARLY_STOPPING_ENABLED=True,EARLY_STOPPING_INCLUDE_INITIAL_VALIDATION=True,EARLY_STOPPING_METRIC=val-core/math/acc/mean@16,EARLY_STOPPING_MIN_DELTA=0.0,EARLY_STOPPING_MODE=max,EARLY_STOPPING_PATIENCE=5,EARLY_STOPPING_MISS_TOLERANCE=0.01"
ENV_VARS+=",ENABLE_OVERLONG_BUFFER=True,GEN_PROMPT_BSZ=64,HF_PUSH_FREQ=10,HF_PUSH_MAX_TO_KEEP=8,LOG_TRAIN_GENERATIONS=100,LOG_VAL_GENERATIONS=100,MAX_MODEL_LEN=12288,MAX_PROMPT_LENGTH=4096,MAX_RESPONSE_LENGTH=8192,N_RESP_PER_PROMPT=16,OVERLONG_BUFFER_LEN=2048,OVERLONG_PENALTY_FACTOR=1.0"
ENV_VARS+=",ROLLOUT_ENFORCE_EAGER=False,ROUTER_REPLAY_MODE=R3,ROUTER_Z_LOSS_COEF=0.0,SAVE_FREQ=10,SP_SIZE=1,TEST_FREQ=10,TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS},TRAIN_PROMPT_BSZ=64,TRAIN_PROMPT_MINI_BSZ=32,VAL_BEFORE_TRAIN=True,VAL_N=1,VLLM_DISABLE_COMPILE_CACHE=0"
# one-node MoE layout of the 26B-A4B runs (FSDP2 DP8, phase-level offload, vLLM sleep mode, mbs 4 / 8192-token cap), with
# vLLM at TP 2 (4 engines; ~28.5 GiB weights per GPU) and a 12 GiB KV cache per GPU (Qwen3-30B-A3B: 96 KiB/token, 48 KiB at TP 2)
ENV_VARS+=",GEN_TP=2,FSDP_CPU_OFFLOAD_POLICY=False,OFFLOAD=True,VLLM_SLEEP_MODE=True,MAX_PADDED_TOKENS_PER_MICROBATCH=8192,MICRO_BATCH_SIZE_PER_GPU=4,ROLLOUT_GPU_MEMORY_UTILIZATION=0.10,VLLM_KV_CACHE_MEMORY_BYTES=12884901888"
ENV_VARS+=",PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# S3-only artifacts: permanent S3 checkpoints carry the HF snapshot, best_hf/ is published at the end
ENV_VARS+=",HF_PUSH_ENABLE=False,HF_PUSH_REQUIRED=False,MAX_ACTOR_CKPT_TO_KEEP=2"
# resumable checkpoint every 5 steps (rolling slot) + permanent every 10 + durable completion markers
ENV_VARS+=",ROLLING_CHECKPOINT_ENABLED=True,ROLLING_CHECKPOINT_FREQ=5,FULL_CHECKPOINT_S3_URI=${S3_BASE}-full-checkpoints/qwen30b-medium-${RUN_TAG},RUN_ARTIFACT_S3_URI=${S3_BASE}/qwen3-30b-a3b-medium-${RUN_TAG}"
ENV_VARS+=",WANDB_RUN_ID=q3-30b-a3b-medium-${RUN_TAG}-s42-v1,WANDB_RESUME=allow"
echo "LAUNCH_ENV ${ENV_VARS}"
exec bash launch_st_with_code.sh --gpus-per-instance 8 --priority high --allow-borrowing --active-deadline-hours 240 \
  --run-file run_gemma4_pt_deepscaler_4of4strict_rl.sh --job-name "${JOB_NAME:-q3-30b-rl-med}" --env-vars "${ENV_VARS}" "$@"
