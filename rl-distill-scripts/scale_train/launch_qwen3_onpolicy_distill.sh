#!/usr/bin/env bash
# On-policy distillation (policy-gradient form) of the off-policy-distilled Qwen3 students toward their RL teachers.
#   SPEC=<teacher><band>-to-<student>: q4bm|q4bh|q4bvh|q30bm|q30bh|q30bvh - to - q1p7b|q4b (q4b* teachers only -> q1p7b)
#   q8be|q8bm|q8bh (Qwen3-8B RL teachers) exist only for base students: SPEC=q8b<band>-s<STEP>-to-<q1p7b|q4b>-base
#   q1p7be|q1p7bm|q1p7bh = Qwen3-1.7B RL teachers (same-size control: SPEC=q1p7b<band>-s<STEP>-to-q1p7b-base)
#   qb1p7b<e|m|h> / qb4b<e|m|h> = the UNTRAINED base model as teacher (pinned Hub revision; base->base pre-distillation on that
#   band's prompts), e.g. SPEC=qb1p7bm-to-q4b-base, SPEC=qb4bh-to-q1p7b-base
#   student = that off-policy distillation's global_step_1000 HF export; teacher = the teacher's RL best_hf; band = its band
# Recipe (Nemotron-Cascade 2 OPD + MOPD advantage clip; PROGRESS_LOG 2026-10-20):
#   per-token advantage A_t = sg[log pi_teacher(y_t) - log pi_train(y_t)] (k1 + policy gradient, no task reward, no
#   normalisation), clipped to [-5, 5]; IcePop token IS mask [0.5, 2] between the trainer and vLLM; one update per batch
#   (128 prompts x 1 sample, mini = train batch, so the PPO ratio clip is inactive); lr 2e-6, 20-step warmup, constant;
#   token-mean loss; temp 1 / top-p 1 / top-k off; 120 steps (the first 4 runs were launched with 150 and stopped at 120).
# Everything else is the Qwen RL contract (launch_qwen3_30b_a3b_medium_rl.sh): 0-shot ChatML prompt, 8k responses,
# validation every 10 steps (band val300 x16, mean@16, before training too), permanent S3 checkpoint + HF export every 10,
# rolling resumable checkpoint every 5, best_hf at the end; early stopping off (fixed 120 steps).
# vLLM TP 1 for the student and the 4B teacher (one engine per GPU); the 30B-A3B teacher runs at TP 2. All engines sleep between phases.
#   SPEC=q4bm-to-q1p7b bash launch_qwen3_onpolicy_distill.sh --image <uri>
# Teacher = an intermediate RL checkpoint, student = the untrained base (FLOP-matched sweep):
#   SPEC=q4bm-s<STEP>-to-q1p7b-base -> teacher qwen3-4b-base-rl-full-checkpoints/qwen30b-medium-es5/global_step_<STEP>, student
#   Qwen3-1.7B-Base itself (no off-policy init). TEST_FREQ / SAVE_FREQ override the validation / permanent-checkpoint cadence.
# EARLY_STOPPING_PATIENCE=<evals> [EARLY_STOPPING_MIN_DELTA=<abs>] turns validation early stopping on (no miss tolerance), e.g. for
# extending a finished run; EARLY_STOPPING_LEGACY_STATE_B64 seeds its history when the restored checkpoint predates it.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
SPEC="${SPEC:?SPEC=<teacher><band>-to-<student>, e.g. q4bm-to-q1p7b}"
TEACHER_TAG="${SPEC%%-to-*}"; STUDENT="${SPEC##*-to-}"
TEACHER_STEP=""
if [[ "${TEACHER_TAG}" == *-s[0-9]* ]]; then TEACHER_STEP="${TEACHER_TAG##*-s}"; TEACHER_TAG="${TEACHER_TAG%-s*}"; fi
case "${TEACHER_TAG}" in
  q4be) T_SIZE=4b; BAND=easy ;; q4bm) T_SIZE=4b; BAND=medium ;; q4bh) T_SIZE=4b; BAND=hard ;; q4bvh) T_SIZE=4b; BAND=very_hard ;;
  q8be) T_SIZE=8b; BAND=easy ;; q8bm) T_SIZE=8b; BAND=medium ;; q8bh) T_SIZE=8b; BAND=hard ;;
  q1p7be) T_SIZE=1p7b; BAND=easy ;; q1p7bm) T_SIZE=1p7b; BAND=medium ;; q1p7bh) T_SIZE=1p7b; BAND=hard ;;
  qb1p7be|qb1p7bm|qb1p7bh) T_SIZE=base1p7b ;; qb4be|qb4bm|qb4bh) T_SIZE=base4b ;;
  q30bm) T_SIZE=30b; BAND=medium ;; q30bh) T_SIZE=30b; BAND=hard ;; q30bvh) T_SIZE=30b; BAND=very_hard ;;
  *) echo "FATAL: unknown teacher in SPEC=${SPEC}" >&2; exit 2 ;;
esac
case "${T_SIZE}" in base*)
  case "${TEACHER_TAG: -1}" in e) BAND=easy ;; m) BAND=medium ;; h) BAND=hard ;; esac
  [ -z "${TEACHER_STEP}" ] || { echo "FATAL: base teachers take no -s<STEP>" >&2; exit 2; } ;;
esac
B=s3://scale-ml/genai/rl-distill
TEACHER_REPO=""; TEACHER_REVISION=""
if [ "${T_SIZE}" = base1p7b ]; then
  TEACHER_S3=""; TEACHER_REPO=Qwen/Qwen3-1.7B-Base; TEACHER_REVISION=ea980cb0a6c2ae4b936e82123acc929f1cec04c1; DIST=""
  TEACHER_UTIL=0.15; TEACHER_TP=1   # 3.4 GiB bf16 weights + KV
elif [ "${T_SIZE}" = base4b ]; then
  TEACHER_S3=""; TEACHER_REPO=Qwen/Qwen3-4B-Base; TEACHER_REVISION=906bfd4b4dc7f14ee4320094d8b41684abff8539; DIST=""
  TEACHER_UTIL=0.20; TEACHER_TP=1
elif [ "${T_SIZE}" = 1p7b ]; then
  TEACHER_S3="${B}/qwen3-1.7b-base-rl/qwen3-1.7b-${BAND}-es5/best_hf"; DIST=""
  [ -z "${TEACHER_STEP}" ] || TEACHER_S3="${B}/qwen3-1.7b-base-rl-full-checkpoints/qwen30b-${BAND}-es5/global_step_${TEACHER_STEP}/actor/huggingface"
  TEACHER_UTIL=0.15; TEACHER_TP=1
elif [ "${T_SIZE}" = 4b ]; then
  TEACHER_S3="${B}/qwen3-4b-base-rl/qwen3-4b-${BAND}-es5/best_hf"; DIST="${B}/qwen3-4b-rl-distill-ckpts-v1"
  [ -z "${TEACHER_STEP}" ] || TEACHER_S3="${B}/qwen3-4b-base-rl-full-checkpoints/qwen30b-${BAND}-es5/global_step_${TEACHER_STEP}/actor/huggingface"
  TEACHER_UTIL=0.20; TEACHER_TP=1   # 8 GiB bf16 weights + KV
elif [ "${T_SIZE}" = 8b ]; then
  TEACHER_S3="${B}/qwen3-8b-base-rl/qwen3-8b-${BAND}-es5/best_hf"; DIST=""
  [ -z "${TEACHER_STEP}" ] || TEACHER_S3="${B}/qwen3-8b-base-rl-full-checkpoints/qwen30b-${BAND}-es5/global_step_${TEACHER_STEP}/actor/huggingface"
  TEACHER_UTIL=0.32; TEACHER_TP=1   # 16.4 GiB bf16 weights + ~9 GiB KV, one engine per GPU
else
  TEACHER_S3="${B}/qwen3-30b-a3b-base-rl/qwen3-30b-a3b-${BAND}-es5/best_hf"; DIST="${B}/qwen3-30b-rl-distill-ckpts-v1"
  [ -z "${TEACHER_STEP}" ] || TEACHER_S3="${B}/qwen3-30b-a3b-base-rl-full-checkpoints/qwen30b-${BAND}-es5/global_step_${TEACHER_STEP}/actor/huggingface"
  TEACHER_UTIL=0.45; TEACHER_TP=2   # 61 GiB bf16 weights -> ~30.5 GiB per GPU at TP 2 + ~5 GiB KV
fi
STUDENT_INIT=distilled; [[ "${STUDENT}" == *-base ]] && { STUDENT_INIT=base; STUDENT="${STUDENT%-base}"; }
case "${T_SIZE}" in 8b|1p7b|base*) [ "${STUDENT_INIT}" = base ] || { echo "FATAL: ${T_SIZE} teachers have no off-policy students; use a -base student" >&2; exit 2; } ;; esac
case "${STUDENT}" in
  q1p7b) MODEL=Qwen/Qwen3-1.7B-Base; REVISION=ea980cb0a6c2ae4b936e82123acc929f1cec04c1; GPUS=2 ;;
  q4b) MODEL=Qwen/Qwen3-4B-Base; REVISION=906bfd4b4dc7f14ee4320094d8b41684abff8539; GPUS=4
       [ "${T_SIZE}" = 30b ] || [ "${STUDENT_INIT}" = base ] || { echo "FATAL: distilled q4b students exist only for 30B teachers" >&2; exit 2; } ;;
  *) echo "FATAL: unknown student in SPEC=${SPEC}" >&2; exit 2 ;;
esac
STUDENT_S3="${DIST}/${TEACHER_TAG}-${BAND}-to-${STUDENT}-bs128-s1000-lr2e-6/global_step_1000/huggingface"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-120}"
RUN_TAG="${RUN_TAG:-pgk1-v1}"
S3_BASE="${S3_BASE:-${B}/qwen3-opd}"
# RL memory layout of the student size (FSDP2, one TP-1 vLLM engine per GPU, sleep mode); with the 30B teacher the actor's
# params + optimizer are also offloaded outside the update (OFFLOAD; cheap for a 1.7B / 4B student) for headroom.
ACTOR_OFFLOAD=False; [ "${T_SIZE}" = 30b ] && ACTOR_OFFLOAD=True
LAYOUT="GEN_TP=1,FSDP_CPU_OFFLOAD_POLICY=False,OFFLOAD=${OFFLOAD:-${ACTOR_OFFLOAD}},VLLM_SLEEP_MODE=True,MAX_PADDED_TOKENS_PER_MICROBATCH=8192,MICRO_BATCH_SIZE_PER_GPU=4"
LAYOUT+=",ROLLOUT_GPU_MEMORY_UTILIZATION=0.10,VLLM_KV_CACHE_MEMORY_BYTES=${VLLM_KV_CACHE_MEMORY_BYTES:-8589934592}"   # n=1: 8 GiB KV is plenty

ENV_VARS="GEMMA4_MODEL=${MODEL},GEMMA4_MODEL_REVISION=${REVISION}"
[ "${STUDENT_INIT}" = base ] || ENV_VARS+=",GEMMA4_INIT_MODEL_S3_URI=${STUDENT_S3}"
ENV_VARS+=",DIFFICULTY_DATASET_SOURCE=qwen3_30b_bands,DIFFICULTY_DATASET=${BAND}"
ENV_VARS+=",DATA_SEED=42,RUN_NAME_SUFFIX=${RUN_TAG},RUN_SLOT=qwen3-opd-${SPEC}-${RUN_TAG},VERL_VLLM_PORT_BASE=54000"
ENV_VARS+=",EXP_NAME=OPD-qwen3-${SPEC}-${RUN_TAG}"
# on-policy distillation objective
if [ -n "${TEACHER_REPO}" ]; then
  ENV_VARS+=",ONPOLICY_DISTILL_ENABLE=True,ONPOLICY_DISTILL_TEACHER_REPO=${TEACHER_REPO},ONPOLICY_DISTILL_TEACHER_REVISION=${TEACHER_REVISION},ONPOLICY_DISTILL_LOSS_MODE=k1"
else
  ENV_VARS+=",ONPOLICY_DISTILL_ENABLE=True,ONPOLICY_DISTILL_TEACHER_S3_URI=${TEACHER_S3},ONPOLICY_DISTILL_LOSS_MODE=k1"
fi
ENV_VARS+=",ONPOLICY_DISTILL_USE_POLICY_GRADIENT=True,ONPOLICY_DISTILL_USE_TASK_REWARDS=False,ONPOLICY_DISTILL_LOSS_COEF=1.0"
ENV_VARS+=",ONPOLICY_DISTILL_LOSS_MAX_CLAMP=5.0,ONPOLICY_DISTILL_LOG_PROB_MIN_CLAMP=null,ONPOLICY_DISTILL_TOPK=1"
ENV_VARS+=",ONPOLICY_DISTILL_TEACHER_TP=${TEACHER_TP},ONPOLICY_DISTILL_TEACHER_GPU_MEM_UTIL=${TEACHER_GPU_MEM_UTIL:-${TEACHER_UTIL}},ONPOLICY_DISTILL_TEACHER_SLEEP=True"
ENV_VARS+=",ROLLOUT_IS_THRESHOLD=0.5_2.0"
# batch / optimiser: 128 prompts x 1 sample, one update per step
ENV_VARS+=",N_RESP_PER_PROMPT=1,TRAIN_PROMPT_BSZ=128,TRAIN_PROMPT_MINI_BSZ=128,GEN_PROMPT_BSZ=128,ENABLE_FILTER_GROUPS=False,ENABLE_OVERLONG_BUFFER=False"
ENV_VARS+=",ACTOR_FSDP_SIZE=-1,ACTOR_LR=2e-6,ACTOR_LR_WARMUP_STEPS=20"
if [ -n "${EARLY_STOPPING_PATIENCE:-}" ]; then
  ENV_VARS+=",EARLY_STOPPING_ENABLED=True,EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE},EARLY_STOPPING_MIN_DELTA=${EARLY_STOPPING_MIN_DELTA:-0.0}"
  [ -z "${EARLY_STOPPING_LEGACY_STATE_B64:-}" ] || ENV_VARS+=",EARLY_STOPPING_LEGACY_STATE_B64=${EARLY_STOPPING_LEGACY_STATE_B64}"
else
  ENV_VARS+=",EARLY_STOPPING_ENABLED=False"
fi
# RL contract: lengths, validation, checkpoints
ENV_VARS+=",HF_PUSH_FREQ=10,HF_PUSH_MAX_TO_KEEP=8,LOG_TRAIN_GENERATIONS=100,LOG_VAL_GENERATIONS=100,MAX_MODEL_LEN=12288,MAX_PROMPT_LENGTH=4096,MAX_RESPONSE_LENGTH=8192"
ENV_VARS+=",ROLLOUT_ENFORCE_EAGER=False,ROUTER_REPLAY_MODE=disabled,ROUTER_Z_LOSS_COEF=0.0,SAVE_FREQ=${SAVE_FREQ:-10},SP_SIZE=1,TEST_FREQ=${TEST_FREQ:-10},TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS},VAL_BEFORE_TRAIN=True,VAL_N=1,VLLM_DISABLE_COMPILE_CACHE=0"
ENV_VARS+=",${LAYOUT}"
ENV_VARS+=",PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
ENV_VARS+=",HF_PUSH_ENABLE=False,HF_PUSH_REQUIRED=False,MAX_ACTOR_CKPT_TO_KEEP=2"
ENV_VARS+=",ROLLING_CHECKPOINT_ENABLED=True,ROLLING_CHECKPOINT_FREQ=5,ASYNC_CHECKPOINT_UPLOAD=False,FULL_CHECKPOINT_S3_URI=${S3_BASE}-full-checkpoints/${SPEC}-${RUN_TAG},RUN_ARTIFACT_S3_URI=${S3_BASE}/${SPEC}-${RUN_TAG}"
ENV_VARS+=",WANDB_RUN_ID=q3-opd-${SPEC}-${RUN_TAG}-s42,WANDB_RESUME=allow"
# The pgk1-v1 batch runs (and relaunches with) the code it started on: 5c6d47d5, before the distillation global_batch_info
# ordering fix in verl/trainer/distillation/losses.py. New batches: CODE_REF=HEAD (or a later pin) + a new RUN_TAG.
[ "${RUN_TAG}" = pgk1-v1 ] && export CODE_REF="${CODE_REF:-5c6d47d5}"
echo "LAUNCH_ENV ${ENV_VARS} CODE_REF=${CODE_REF:-HEAD}"
BORROW_FLAG=""; [ "${ALLOW_BORROWING:-false}" = true ] && BORROW_FLAG="--allow-borrowing"
exec bash launch_st_with_code.sh --gpus-per-instance "${GPUS}" --priority high ${BORROW_FLAG} --active-deadline-hours 120 \
  --run-file run_gemma4_pt_deepscaler_4of4strict_rl.sh --job-name "${JOB_NAME:-q3-opd-${SPEC}}" --env-vars "${ENV_VARS}" "$@"
