#!/usr/bin/env bash
# On-policy distillation of the E4B-base-distilled 12B student toward the E4B base teacher on the DeepScaleR
# *medium* band (continuation of the off-policy top-128 forward-KL run, DISTILLATION_EXPERIMENTS.md §9).
#
# Each step: the 12B student samples ONE response per prompt for 128 medium prompts (12-shot prompt, T=1.0,
# top_p=1.0, top_k=-1, stop on <end_of_turn>/<start_of_turn>, 8k max), a colocated vLLM copy of the E4B base
# returns its top-128 (token, logprob) at every response position, and the actor takes ONE update on the
# 128 sequences with verl's `reverse_kl_topk` loss (sum over the teacher's top-128 of q_s (log q_s - log p_t),
# token-mean, backpropagated through the student logits — no policy-gradient term, no task reward).
# Off-policy counterpart: 128 teacher traces/step, teacher top-128 forward KL, lr 2e-6, 1000 steps.
#
# Runtime = the Gemma 4 RL run-file (run_gemma4_pt_deepscaler_4of4strict_rl.sh) with ONPOLICY_DISTILL_ENABLE=True,
# so validation (val-core/math/acc/mean@16 on the medium val300 ×16), rolling S3 checkpoints, HF pushes and the
# borrowing supervisor all work unchanged. 8 GPUs: 12B student FSDP2 DP8 (CPU-offload policy, 4096-token
# micro-batches) + student vLLM (util 0.35 — n=1 needs far less KV than the RL n=16) + E4B teacher vLLM
# (TP=2 → 4 replicas, util 0.20, level-1 sleep between scoring calls).
#
#   bash launch_gemma4_12b_distilled_onpolicy_medium.sh           # whole node, borrowing on, priority high
#   TOTAL_TRAINING_STEPS=20 SAVE_FREQ=10 RUN_TAG=onpolicy-smoke bash launch_gemma4_12b_distilled_onpolicy_medium.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
INIT_REPO="${INIT_REPO:-JWei05/Distill-gemma4-e4b-base-medium-to-12b-base}"
INIT_REVISION="${INIT_REVISION:-92368d1f020e685436113d93dc1f75c64033570f}"   # commit that pushed step_001000
INIT_SUBFOLDER="${INIT_SUBFOLDER:-step_001000}"
TEACHER_REPO="${TEACHER_REPO:-google/gemma-4-E4B}"
TEACHER_REVISION="${TEACHER_REVISION:-411aa17b749aa952df1359d2dcea73917a544d9a}"   # same pin as the trace generation
RUN_TAG="${RUN_TAG:-onpolicy-rkl128-from-e4bbase-distill}"
S3_BASE="${S3_BASE:-s3://scale-ml/genai/rl-distill/gemma4-12b-from-e4bbase-distill-onpolicy}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-1000}"
SAVE_FREQ="${SAVE_FREQ:-50}"
ACTOR_LR="${ACTOR_LR:-2e-6}"

ENV_VARS="GEMMA4_MODEL=google/gemma-4-12B,GEMMA4_MODEL_REVISION=023679ed352de9bb66cc873c9009ce3482585c08"
ENV_VARS+=",GEMMA4_INIT_MODEL_REPO=${INIT_REPO},GEMMA4_INIT_MODEL_REVISION=${INIT_REVISION},GEMMA4_INIT_MODEL_SUBFOLDER=${INIT_SUBFOLDER}"
ENV_VARS+=",DIFFICULTY_DATASET_SOURCE=gemma4_26b_bands,DIFFICULTY_DATASET=medium,DIFFICULTY_DATASET_REPO=JWei05/DeepScaleR-Easy-Medium-Hard-Gemma-26B-PT-10k,DIFFICULTY_DATASET_REVISION=a0ba3c3dc07c7bc27e901670ceb1a0b0ceeaa8db"
ENV_VARS+=",DATA_SEED=42,RUN_NAME_SUFFIX=${RUN_TAG},RUN_SLOT=gemma4-12b-medium-${RUN_TAG},VERL_VLLM_PORT_BASE=55000"
ENV_VARS+=",EXP_NAME=OnPolicyDistill-gemma4-12b-from-e4b-base-medium-${RUN_TAG},HF_PUSH_REPO=JWei05/OnPolicyDistill-gemma4-e4b-base-medium-to-12b-${RUN_TAG}"
# on-policy distillation objective (teacher + loss); everything else stays the RL contract
ENV_VARS+=",ONPOLICY_DISTILL_ENABLE=True,ONPOLICY_DISTILL_TEACHER_REPO=${TEACHER_REPO},ONPOLICY_DISTILL_TEACHER_REVISION=${TEACHER_REVISION}"
ENV_VARS+=",ONPOLICY_DISTILL_LOSS_MODE=${ONPOLICY_DISTILL_LOSS_MODE:-reverse_kl_topk},ONPOLICY_DISTILL_TOPK=128,ONPOLICY_DISTILL_TEACHER_TP=${ONPOLICY_DISTILL_TEACHER_TP:-2}"
ENV_VARS+=",ONPOLICY_DISTILL_TEACHER_GPU_MEM_UTIL=${ONPOLICY_DISTILL_TEACHER_GPU_MEM_UTIL:-0.20},ONPOLICY_DISTILL_TEACHER_SLEEP=${ONPOLICY_DISTILL_TEACHER_SLEEP:-True}"
ENV_VARS+=",ONPOLICY_DISTILL_USE_TASK_REWARDS=False,ONPOLICY_DISTILL_USE_POLICY_GRADIENT=False,ONPOLICY_DISTILL_LOSS_COEF=1.0"
# batch contract mirroring the off-policy recipe: 128 sequences per step, one update per step (pure on-policy)
ENV_VARS+=",N_RESP_PER_PROMPT=1,TRAIN_PROMPT_BSZ=128,TRAIN_PROMPT_MINI_BSZ=128,GEN_PROMPT_BSZ=128,ENABLE_FILTER_GROUPS=False"
ENV_VARS+=",ACTOR_FSDP_SIZE=-1,ACTOR_LR=${ACTOR_LR},ACTOR_LR_WARMUP_STEPS=20,EARLY_STOPPING_ENABLED=False"
ENV_VARS+=",ENABLE_OVERLONG_BUFFER=False,GEN_TP=1,HF_PUSH_FREQ=${SAVE_FREQ},HF_PUSH_MAX_TO_KEEP=24,LOG_TRAIN_GENERATIONS=100,LOG_VAL_GENERATIONS=100,MAX_ACTOR_CKPT_TO_KEEP=2,MAX_MODEL_LEN=12288,MAX_PROMPT_LENGTH=4096,MAX_RESPONSE_LENGTH=8192"
ENV_VARS+=",ROLLOUT_ENFORCE_EAGER=False,ROUTER_REPLAY_MODE=disabled,ROUTER_Z_LOSS_COEF=0.0,SAVE_FREQ=${SAVE_FREQ},SP_SIZE=1,TEST_FREQ=10,TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS},VAL_BEFORE_TRAIN=True,VAL_N=1,VLLM_DISABLE_COMPILE_CACHE=0"
# 12B memory layout (from the sweep's 12B launch) with a smaller student KV budget (128 sequences/step, not 1024).
# FSDP_CPU_OFFLOAD_POLICY=True is the known-good 12B layout with a resident student engine (RL: update_actor 593 s of a
# 789 s step for 1024 sequences; here 128 sequences -> ~75 s). Faster variant to canary later: FSDP_CPU_OFFLOAD_POLICY=False
# OFFLOAD=True VLLM_SLEEP_MODE=True (the 26B recipe: engines sleep + param/optimizer offload between phases).
ENV_VARS+=",FSDP_CPU_OFFLOAD_POLICY=${FSDP_CPU_OFFLOAD_POLICY:-True},VLLM_SLEEP_MODE=${VLLM_SLEEP_MODE:-False},OFFLOAD=${OFFLOAD:-False},MAX_PADDED_TOKENS_PER_MICROBATCH=4096,MICRO_BATCH_SIZE_PER_GPU=1,ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.35},VERL_SKIP_VLLM_MM_WEIGHT_RELOAD=1,VLLM_KV_CACHE_MEMORY_BYTES=${VLLM_KV_CACHE_MEMORY_BYTES:-4294967296}"
# resumable checkpoints every 10 steps + durable completion markers
ENV_VARS+=",ROLLING_CHECKPOINT_ENABLED=True,ROLLING_CHECKPOINT_FREQ=10,FULL_CHECKPOINT_S3_URI=${S3_BASE}-full-checkpoints/12b-medium-${RUN_TAG},RUN_ARTIFACT_S3_URI=${S3_BASE}/gemma4-12b-medium-${RUN_TAG}"
ENV_VARS+=",WANDB_RUN_ID=g4-onpolicy-12b-medium-${RUN_TAG}-s42-v1,WANDB_RESUME=allow"
echo "LAUNCH_ENV ${ENV_VARS}"
exec bash launch_st_with_code.sh --gpus-per-instance 8 --priority high --allow-borrowing --active-deadline-hours 240 \
  --run-file run_gemma4_pt_deepscaler_4of4strict_rl.sh --job-name "${JOB_NAME:-g4-12b-onpolicy-med}" --env-vars "${ENV_VARS}" "$@"
