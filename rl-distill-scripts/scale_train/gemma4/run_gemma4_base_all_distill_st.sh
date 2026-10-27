#!/usr/bin/env bash
# ScaleTrain pod entry (4 GPUs): base -> base off-policy pre-distillation for Gemma 4 (pre-training vs RL ablation of the medium
# E4B / 12B case, PROGRESS_LOG 2026-10-27), the Gemma counterpart of the Qwen3 q1p7b-base-all / q4b-base-all runs
# (scale_train/run_qwen3_30bm_to_4b_distill_st.sh):
#   stage 1  top-128 traces of the UNTRAINED Gemma 4 12B PT base (spec 12b-base-all) on the union of the easy + medium + hard 26B-PT
#            band train splits, 8 samples per question (+ 1 per validation question), temp 1, 8192 tokens, the 12-shot Gemma
#            prompt; 4 TP-1 vLLM engines (DP 4); every finished shard mirrored to S3; skipped once the bundle has COMPLETE.json
#   stage 2  off-policy top-128 forward-KL distillation WITH the tail bucket into the Gemma 4 E4B base on 4 GPUs (FSDP2 DP 4):
#            batch 128, lr 2e-6 (100 warmup, linear -> 2e-7), 1000 steps, val KL every 10; permanent S3 checkpoint (fp32 model +
#            Adam + LR/RNG + data cursor + HF export) every 250 steps; skipped once step 1000 is on S3
#   stage 3  x32 band-validation evals (eval/eval_gemma4_models_passk.py: id_easy / id_medium / id_hard, 300 q x 32, 12-shot, temp 1,
#            8192 tokens, strict boxed grader = the RL reward) of the E4B base, the 12B base teacher and the student's step 250 / 500 /
#            750 / 1000 exports: pass@k for k in {1,2,4,8,16,32}, mean@32, and the own-sample NLL per token (sampled-token logprobs);
#            ALL_DONE.json under the eval prefix marks completion
# STAGES selects the stages (default "1 2"): the trace job runs STAGES=1, the distillation job STAGES=2 once the bundle is complete,
# the eval job STAGES=3 once step 1000 is on S3.
# Preemption: every stage resumes (trace shards on S3; the trainer restores the newest complete S3 checkpoint).
#   launched by start_gemma4_base_all_distill_supervisor.sh
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; cd "${PROJECT_ROOT}"
# the pod's login shell drops the image PATH: put the baked venv (aws CLI) and the system dirs back
export PATH="${PROJECT_ROOT}/.venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH:-}"
echo "G4BASE_START $(date -u +%FT%TZ) host=$(hostname)"
if [ -n "${CODE_S3_URI:-}" ]; then
  AWS_BIN="$(command -v aws || echo "${PROJECT_ROOT}/.venv/bin/aws")"
  "${AWS_BIN}" s3 cp --only-show-errors "${CODE_S3_URI}" /tmp/rl-distill-code.tar.gz
  tar -xzf /tmp/rl-distill-code.tar.gz --unlink-first --recursive-unlink -C "${PROJECT_ROOT}"
  echo "CODE_REFRESHED $(sha256sum /tmp/rl-distill-code.tar.gz | cut -c1-16)"
fi
TEACHER_SPEC="${TEACHER_SPEC:-12b-base-all}"; STUDENT="${STUDENT:-e4b}"; STAGES="${STAGES:-1 2}"
case "${TEACHER_SPEC}" in
  12b-base-all) BAND=all; FAMILY=gemma4-12b-base; SPQ=8 ;;
  *) echo "FATAL: TEACHER_SPEC must be 12b-base-all" >&2; exit 2 ;;
esac
case "${STUDENT}" in e4b) min_gpus=4 ;; *) echo "FATAL: STUDENT must be e4b" >&2; exit 2 ;; esac
if [ -f .env ]; then set -a; source .env; set +a; fi
: "${HF_TOKEN:?HF_TOKEN missing (forward it with --dotenv-keys)}"
: "${WANDB_API_KEY:?WANDB_API_KEY missing (forward it with --dotenv-keys)}"
export PATH="${HOME}/.local/bin:/root/.local/bin:${PATH}"
command -v uv >/dev/null 2>&1 || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="${HOME}/.local/bin:${PATH}"; }
export VENV="${VENV:-/tmp/.venv-gemma4}" GEMMA4_CUDA_VARIANT="${GEMMA4_CUDA_VARIANT:-cu129}"   # p5 fleet: CUDA 12.8 driver
[ -x "${VENV}/bin/python" ] || VENV="${VENV}" bash rl-distill-scripts/setup_env_gemma4.sh
export PATH="${VENV}/bin:${PATH}"; source "${VENV}/bin/activate"
if [ -x /usr/local/cuda/bin/nvcc ]; then export CUDA_HOME="${CUDA_HOME:-$(readlink -f /usr/local/cuda)}"; export PATH="${CUDA_HOME}/bin:${PATH}"; fi
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
python -c "import torch, transformers, verl" || { echo "FATAL: venv import check failed" >&2; exit 2; }
command -v aws >/dev/null || { echo "FATAL: aws CLI missing (traces and checkpoints live on S3)" >&2; exit 2; }
export HF_HOME="${HF_HOME:-/tmp/hf_cache}" HF_HUB_CACHE="${HF_HUB_CACHE:-/tmp/hf_cache/hub}" WANDB_DIR="${WANDB_DIR:-/tmp/wandb}" TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache}"
# ScaleTrain pods on ml-gpu-batch (EKS) are IPv6-only, interface eth0
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-${NETWORK_INTERFACE_NAME:-eth0}}" NCCL_SOCKET_FAMILY="${NCCL_SOCKET_FAMILY:-AF_INET6}" GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${NETWORK_INTERFACE_NAME:-eth0}}"
n_gpus="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
(( n_gpus >= min_gpus )) || { echo "FATAL: needs ${min_gpus} GPUs (4 TP-1 trace engines / FSDP2 DP4 for the ${STUDENT} student), got ${n_gpus}" >&2; exit 2; }
GPU_IDS_CSV="$(seq -s, 0 $((n_gpus - 1)))"
unset CUDA_VISIBLE_DEVICES
TRACE_S3_BASE="${TRACE_S3_BASE:-s3://scale-ml/genai/rl-distill/${FAMILY}-traces-topk128-v1}"

# --- stage 1: traces -----------------------------------------------------------------------------------------------------------
if [[ " ${STAGES} " == *" 1 "* ]]; then
  if aws s3 ls "${TRACE_S3_BASE}/${TEACHER_SPEC}/COMPLETE.json" >/dev/null 2>&1; then
    echo "G4BASE stage1: S3 bundle ${TRACE_S3_BASE}/${TEACHER_SPEC} already complete -> skipping"
  else
    echo "G4BASE stage1: trace collection start $(date -u +%FT%TZ)"
    TRACE_SPEC="${TEACHER_SPEC}" TEACHER_SOURCE=hf TRACE_GPU_IDS="${GPU_IDS_CSV}" TENSOR_PARALLEL_SIZE="${TRACE_TP:-1}" TRACE_S3_BASE="${TRACE_S3_BASE}" \
      PROMPTS_PER_SHARD="${PROMPTS_PER_SHARD:-32}" MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}" \
      TRAIN_SAMPLES_PER_QUESTION="${SPQ}" VALIDATION_SAMPLES_PER_QUESTION=1 TRACE_S3_MIRROR_ENABLE=true \
      bash rl-distill-scripts/scale_train/run_gemma4_bestckpt_trace_collection.sh
    echo "G4BASE stage1: done $(date -u +%FT%TZ)"
  fi
fi

# --- stage 2: distillation -----------------------------------------------------------------------------------------------------
if [[ " ${STAGES} " == *" 2 "* ]]; then
  aws s3 ls "${TRACE_S3_BASE}/${TEACHER_SPEC}/COMPLETE.json" >/dev/null 2>&1 \
    || { echo "FATAL: stage 2 needs the complete trace bundle ${TRACE_S3_BASE}/${TEACHER_SPEC} (run STAGES=1 first)" >&2; exit 2; }
  # the bundle's question counts (band union) drive the training view
  aws s3 cp --only-show-errors "${TRACE_S3_BASE}/${TEACHER_SPEC}/dataset_index.json" /tmp/g4base_dataset_index.json
  read -r n_train n_val < <("${VENV}/bin/python" -c "
import json, sys
s = json.load(open(sys.argv[1]))['splits']
print(s['train']['question_count'], s['validation']['question_count'])" /tmp/g4base_dataset_index.json)
  export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}" LR="${LR:-2e-6}" TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-1000}" TOTAL_EPOCHS="${TOTAL_EPOCHS:-100}"
  export LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-100}" LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-linear}" MIN_LR_RATIO="${MIN_LR_RATIO:-0.1}" TEST_FREQ="${TEST_FREQ:-10}" SAVE_FREQ="${SAVE_FREQ:-250}"
  export TRAIN_SAMPLES_PER_QUESTION="${SPQ}"
  # base -> base: tail-bucket top-k forward KL so the student keeps the base teacher's tail mass; train on every bundle question,
  # validate on 300 of the teacher's validation generations (all bands)
  export TOPK_KL_TAIL_BUCKET="${TOPK_KL_TAIL_BUCKET:-true}"
  export SOURCE_QUESTIONS="${SOURCE_QUESTIONS:-${n_train}}" TRAIN_QUESTIONS="${TRAIN_QUESTIONS:-${n_train}}" VALIDATION_QUESTIONS="${VALIDATION_QUESTIONS:-300}"
  export VERL_FAIL_ON_NONFINITE_GRAD="${VERL_FAIL_ON_NONFINITE_GRAD:-0}"
  export CHECKPOINT_SAVE_CONTENTS="${CHECKPOINT_SAVE_CONTENTS:-[\"model\",\"optimizer\",\"extra\",\"hf_model\"]}" MAX_CKPT_TO_KEEP="${MAX_CKPT_TO_KEEP:-1}"
  export REMOTE_CHECKPOINT_ENABLE=true ROLLING_CHECKPOINT_FREQ="${ROLLING_CHECKPOINT_FREQ:-0}" ROLLING_HF_EXPORT=true ROLLING_HF_EXPORT_S3=true
  export REMOTE_CHECKPOINT_S3_URI="${REMOTE_CHECKPOINT_S3_URI:-s3://scale-ml/genai/rl-distill/${FAMILY}-distill-ckpts-v1/${TEACHER_SPEC}-to-${STUDENT}-bs${TRAIN_BATCH_SIZE}-s${TOTAL_TRAINING_STEPS}-lr${LR}}"
  export HF_PUSH_ENABLE=false HF_PUSH_DELETE_LOCAL=false PROJECT_NAME="${PROJECT_NAME:-${FAMILY}-distill-v1}" FSDP_OFFLOAD=false
  export EXP_NAME="${EXP_NAME:-${TEACHER_SPEC}-to-${STUDENT}-base-bs${TRAIN_BATCH_SIZE}-s${TOTAL_TRAINING_STEPS}-lr${LR}-g${n_gpus}}"
  if aws s3 ls "${REMOTE_CHECKPOINT_S3_URI}/global_step_${TOTAL_TRAINING_STEPS}/_REMOTE_COMPLETE.json" >/dev/null 2>&1; then
    echo "G4BASE stage2: step ${TOTAL_TRAINING_STEPS} already on S3 -> skipping"
  else
    free_gb="$(df -BG --output=avail /tmp | tail -1 | tr -dc 0-9)"
    (( free_gb >= ${REQUIRE_CKPT_DISK_GB:-300} )) || { echo "FATAL: /tmp has ${free_gb} GB free; need >= ${REQUIRE_CKPT_DISK_GB:-300} GB" >&2; exit 2; }
    echo "G4BASE stage2: distill start $(date -u +%FT%TZ) train_q=${TRAIN_QUESTIONS} val_q=${VALIDATION_QUESTIONS} s3=${REMOTE_CHECKPOINT_S3_URI} exp=${EXP_NAME} free_disk_gb=${free_gb}"
    TEACHER_SPEC="${TEACHER_SPEC}" STUDENT="${STUDENT}" DISTILL_GPU_IDS="${GPU_IDS_CSV}" bash rl-distill-scripts/scale_train/run_gemma4_distill_one.sh
    echo "G4BASE stage2: done $(date -u +%FT%TZ)"
  fi
fi
# --- stage 3: x32 band-validation evals ----------------------------------------------------------------------------------------
if [[ " ${STAGES} " == *" 3 "* ]]; then
  CKPT_ROOT="${REMOTE_CHECKPOINT_S3_URI:-s3://scale-ml/genai/rl-distill/${FAMILY}-distill-ckpts-v1/${TEACHER_SPEC}-to-${STUDENT}-bs${TRAIN_BATCH_SIZE:-128}-s${TOTAL_TRAINING_STEPS:-1000}-lr${LR:-2e-6}}"
  EVAL_S3="${EVAL_S3:-s3://scale-ml/genai/rl-distill/${FAMILY}-distill-evals-v1/${CKPT_ROOT##*/}}"
  DATA_ROOT=/tmp/g4base_val32/data
  python rl-distill-scripts/data/prepare_gemma4_rl_distill_eval_data.py --output-dir "${DATA_ROOT}" --overwrite \
    --samples-override "id_easy=32,id_medium=32,id_hard=32" --protocol gemma4_rl_distill_math_eval_v2_x32_allbands
  test -s "${DATA_ROOT}/math_eval_manifest.json"
  models=(--model e4b_base=base:e4b --model g12b_base=base:12b)
  for s in ${EVAL_STEPS:-250 500 750 1000}; do
    aws s3 ls "${CKPT_ROOT}/global_step_${s}/_REMOTE_COMPLETE.json" >/dev/null 2>&1 || { echo "FATAL: ${CKPT_ROOT}/global_step_${s} not complete" >&2; exit 2; }
    models+=(--model "e4bp_s${s}=s3:${CKPT_ROOT}/global_step_${s}")
  done
  echo "G4BASE stage3: evals start $(date -u +%FT%TZ) out=${EVAL_S3}"
  python rl-distill-scripts/eval/eval_gemma4_models_passk.py "${models[@]}" --s3-arch "${STUDENT}" --bands easy,medium,hard \
    --gpus "${GPU_IDS_CSV}" --parallelism dp --manifest "${DATA_ROOT}/math_eval_manifest.json" --out-root /tmp/g4base_val32/models_out \
    --s3-root "${EVAL_S3}" --predictive-topk-width 1
  echo "{\"done\": \"$(date -u +%FT%TZ)\"}" > /tmp/g4base_all_done.json
  aws s3 cp --only-show-errors /tmp/g4base_all_done.json "${EVAL_S3}/ALL_DONE.json"
  echo "G4BASE stage3: done $(date -u +%FT%TZ)"
fi
echo "G4BASE_DONE $(date -u +%FT%TZ) stages='${STAGES}' traces=${TRACE_S3_BASE}/${TEACHER_SPEC}"
