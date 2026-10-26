#!/usr/bin/env bash
# ScaleTrain pod entry (4 GPUs): best Qwen3-30B-A3B medium RL checkpoint (es5 run, val mean@16 0.7575 @ step 320) -> Qwen3-4B-Base,
# the RL-teacher distillation protocol of the Gemma study (run_gemma4_12br_trace_and_distill_st.sh / DISTILLATION_EXPERIMENTS.md §9):
#   stage 1  top-128 traces (spec q30bm-medium: 3000 train q x 16 + 300 val q x 1, temp 1, 8192 tokens, Qwen 0-shot ChatML + boxed
#            instruction), 2 vLLM engines x TP 2, every finished shard mirrored to S3; skipped once the S3 bundle has COMPLETE.json
#   stage 2  off-policy top-128 forward-KL distillation into Qwen3-4B-Base on all 4 GPUs (FSDP2): batch 128, lr 2e-6 (100 warmup,
#            linear -> 2e-7), 1000 steps, val KL every 10, 4 seqs/micro-batch (8192 padded-token cap), KL chunk 4096; permanent S3
#            checkpoint (fp32 model + Adam + LR/RNG + data cursor + HF export) every 250 steps; skipped once step 1000 is on S3
#   stage 3  eval of the step-250/500/750/1000 HF exports on the Qwen3-30B medium validation split (300 q, the RL val set) with the
#            Qwen band-eval protocol (n=16, temp 1, 8192 tokens, strict boxed math_verify = the RL reward): mean@16 / pass@k / pass^k
# Preemption: every stage resumes (trace shards on S3; the trainer restores the newest complete S3 checkpoint; finished evals skipped).
# STUDENT=q1p7b: the same stages 2-3 into Qwen3-1.7B-Base on 2 GPUs (stage 1 is skipped: the teacher bundles already exist).
# q1p7b-base-all / q4b-base-all (pre-training vs RL ablation, PROGRESS_LOG 2026-10-26): the UNTRAINED 1.7B / 4B base as teacher on the
# union of the easy + medium + hard train splits (9000 q x 8 samples + 900 val q x 1), students STUDENT=q4b / q1p7b; stage 3 evaluates
# steps 0 (= the student base) / 250 / 500 / 750 / 1000 on all three bands' validation splits at EVAL_N (32), with pass@k for every
# k <= n and the own-sample NLL per token (entropy estimate, score_own_sample_nll.py on 8 responses per question).
#   launched by launch_qwen3_30bm_to_4b_distill_st.sh / start_qwen3_30bm_to_4b_distill_supervisor.sh
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "${PROJECT_ROOT}"
export PATH="${PROJECT_ROOT}/.venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH:-}"
echo "Q30BM_START $(date -u +%FT%TZ) host=$(hostname)"
if [ -n "${CODE_S3_URI:-}" ]; then
  AWS_BIN="$(command -v aws || echo "${PROJECT_ROOT}/.venv/bin/aws")"
  "${AWS_BIN}" s3 cp --only-show-errors "${CODE_S3_URI}" /tmp/rl-distill-code.tar.gz
  tar -xzf /tmp/rl-distill-code.tar.gz --unlink-first --recursive-unlink -C "${PROJECT_ROOT}"
  echo "CODE_REFRESHED $(sha256sum /tmp/rl-distill-code.tar.gz | cut -c1-16)"
fi
TEACHER_SPEC="${TEACHER_SPEC:-q30bm-medium}"; STUDENT="${STUDENT:-q4b}"; SPQ_DEFAULT=16; TEACHER_SOURCE_DEFAULT=s3
# q30bh-hard: the same pipeline from the 30B *hard* RL run (best step 410), hard-band traces, evals on the hard validation split
# q4bm-medium / q4bh-hard: the 4B medium / hard RL teachers (best steps 510 / 560) -> STUDENT=q1p7b; traces on 1-GPU (TP 1) engines
case "${TEACHER_SPEC}" in
  q30bm-medium) BAND=medium; FAMILY=qwen3-30b-rl; TRACE_TP_DEFAULT=2 ;;
  q30bh-hard)   BAND=hard;   FAMILY=qwen3-30b-rl; TRACE_TP_DEFAULT=2 ;;
  q4bm-medium)  BAND=medium; FAMILY=qwen3-4b-rl;  TRACE_TP_DEFAULT=1 ;;
  q4bh-hard)    BAND=hard;   FAMILY=qwen3-4b-rl;  TRACE_TP_DEFAULT=1 ;;
  q30bvh-very_hard) BAND=very_hard; FAMILY=qwen3-30b-rl; TRACE_TP_DEFAULT=1 ;;   # TP-1 engines (8 per node), as the q30bh-hard cutover
  q4bvh-very_hard)  BAND=very_hard; FAMILY=qwen3-4b-rl;  TRACE_TP_DEFAULT=1 ;;
  q1p7b-base-all|q4b-base-all) BAND=all; FAMILY=qwen3-4b-base; TRACE_TP_DEFAULT=1; SPQ_DEFAULT=8; TEACHER_SOURCE_DEFAULT=hf ;;   # teacher = pinned Hub base
  *) echo "FATAL: TEACHER_SPEC must be q30bm-medium|q30bh-hard|q4bm-medium|q4bh-hard|q30bvh-very_hard|q4bvh-very_hard|q1p7b-base-all|q4b-base-all" >&2; exit 2 ;;
esac
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
command -v aws >/dev/null || { echo "FATAL: aws CLI missing (traces, checkpoints and evals live on S3)" >&2; exit 2; }
export HF_HOME="${HF_HOME:-/tmp/hf_cache}" HF_HUB_CACHE="${HF_HUB_CACHE:-/tmp/hf_cache/hub}" WANDB_DIR="${WANDB_DIR:-/tmp/wandb}" TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache}"
# ScaleTrain pods on ml-gpu-batch (EKS) are IPv6-only, interface eth0
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-${NETWORK_INTERFACE_NAME:-eth0}}" NCCL_SOCKET_FAMILY="${NCCL_SOCKET_FAMILY:-AF_INET6}" GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${NETWORK_INTERFACE_NAME:-eth0}}"
n_gpus="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
# q4b at DP2: 4B x 16 B fp32 params + grads + Adam = ~32 GB/GPU (DP4 was the old default for the 30B-teacher runs)
case "${STUDENT}" in q4b) min_gpus=2 ;; q1p7b) min_gpus=2 ;; *) echo "FATAL: STUDENT must be q4b|q1p7b" >&2; exit 2 ;; esac
(( n_gpus >= min_gpus )) || { echo "FATAL: needs ${min_gpus} GPUs (FSDP2 DP${min_gpus} for the ${STUDENT} student; 4 for trace collection), got ${n_gpus}" >&2; exit 2; }
GPU_IDS_CSV="$(seq -s, 0 $((n_gpus - 1)))"
unset CUDA_VISIBLE_DEVICES

# --- stage 1: traces -----------------------------------------------------------------------------------------------------------
TRACE_S3_BASE="${TRACE_S3_BASE:-s3://scale-ml/genai/rl-distill/${FAMILY}-traces-topk128-v1}"
if aws s3 ls "${TRACE_S3_BASE}/${TEACHER_SPEC}/COMPLETE.json" >/dev/null 2>&1; then
  echo "Q30BM stage1: S3 bundle ${TRACE_S3_BASE}/${TEACHER_SPEC} already complete -> skipping"
else
  echo "Q30BM stage1: trace collection start $(date -u +%FT%TZ)"
  TRACE_SPEC="${TEACHER_SPEC}" TEACHER_SOURCE="${TEACHER_SOURCE_DEFAULT}" TRACE_GPU_IDS="${GPU_IDS_CSV}" TENSOR_PARALLEL_SIZE="${TRACE_TP:-${TRACE_TP_DEFAULT}}" TRACE_S3_BASE="${TRACE_S3_BASE}" \
    PROMPTS_PER_SHARD="${PROMPTS_PER_SHARD:-32}" MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}" \
    TRAIN_SAMPLES_PER_QUESTION="${SPQ_DEFAULT}" VALIDATION_SAMPLES_PER_QUESTION=1 TRACE_S3_MIRROR_ENABLE=true \
    bash rl-distill-scripts/scale_train/run_gemma4_bestckpt_trace_collection.sh
  echo "Q30BM stage1: done $(date -u +%FT%TZ)"
fi

# --- stage 2: distillation -----------------------------------------------------------------------------------------------------
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}" LR="${LR:-2e-6}" TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-1000}" TOTAL_EPOCHS="${TOTAL_EPOCHS:-100}"
export LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-100}" LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-linear}" MIN_LR_RATIO="${MIN_LR_RATIO:-0.1}" TEST_FREQ="${TEST_FREQ:-10}" SAVE_FREQ="${SAVE_FREQ:-250}"
export TRAIN_SAMPLES_PER_QUESTION="${SPQ_DEFAULT}" MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-4}" MAX_PADDED_TOKENS_PER_MICROBATCH="${MAX_PADDED_TOKENS_PER_MICROBATCH:-8192}"
# band union: train on all 9000 questions x 8; validation KL on 300 of the teacher's 900 validation generations (all bands)
# band union (base -> base pre-distillation): tail-bucket top-k forward KL so the student keeps the base teacher's tail mass
[ "${BAND}" = all ] && export TOPK_KL_TAIL_BUCKET="${TOPK_KL_TAIL_BUCKET:-true}"
[ "${BAND}" = all ] && export SOURCE_QUESTIONS="${SOURCE_QUESTIONS:-9000}" TRAIN_QUESTIONS="${TRAIN_QUESTIONS:-9000}" VALIDATION_QUESTIONS="${VALIDATION_QUESTIONS:-300}"
export FULL_VOCAB_KL_CHUNK_SIZE="${FULL_VOCAB_KL_CHUNK_SIZE:-4096}" VERL_FAIL_ON_NONFINITE_GRAD="${VERL_FAIL_ON_NONFINITE_GRAD:-0}"
export CHECKPOINT_SAVE_CONTENTS="${CHECKPOINT_SAVE_CONTENTS:-[\"model\",\"optimizer\",\"extra\",\"hf_model\"]}" MAX_CKPT_TO_KEEP="${MAX_CKPT_TO_KEEP:-1}"
export REMOTE_CHECKPOINT_ENABLE=true ROLLING_CHECKPOINT_FREQ="${ROLLING_CHECKPOINT_FREQ:-0}" ROLLING_HF_EXPORT=true ROLLING_HF_EXPORT_S3=true
export REMOTE_CHECKPOINT_S3_URI="${REMOTE_CHECKPOINT_S3_URI:-s3://scale-ml/genai/rl-distill/${FAMILY}-distill-ckpts-v1/${TEACHER_SPEC}-to-${STUDENT}-bs${TRAIN_BATCH_SIZE}-s${TOTAL_TRAINING_STEPS}-lr${LR}}"
export HF_PUSH_ENABLE=false HF_PUSH_DELETE_LOCAL=false PROJECT_NAME="${PROJECT_NAME:-${FAMILY}-distill-v1}" FSDP_OFFLOAD=false ALLOW_UNDERSIZED_STUDENT_LAYOUT=true
export EXP_NAME="${EXP_NAME:-${TEACHER_SPEC}-to-${STUDENT}-base-bs${TRAIN_BATCH_SIZE}-s${TOTAL_TRAINING_STEPS}-lr${LR}-g${n_gpus}}"
if aws s3 ls "${REMOTE_CHECKPOINT_S3_URI}/global_step_${TOTAL_TRAINING_STEPS}/_REMOTE_COMPLETE.json" >/dev/null 2>&1; then
  echo "Q30BM stage2: step ${TOTAL_TRAINING_STEPS} already on S3 -> skipping"
else
  free_gb="$(df -BG --output=avail /tmp | tail -1 | tr -dc 0-9)"
  (( free_gb >= ${REQUIRE_CKPT_DISK_GB:-300} )) || { echo "FATAL: /tmp has ${free_gb} GB free; need >= ${REQUIRE_CKPT_DISK_GB:-300} GB" >&2; exit 2; }
  echo "Q30BM stage2: distill start $(date -u +%FT%TZ) s3=${REMOTE_CHECKPOINT_S3_URI} exp=${EXP_NAME} free_disk_gb=${free_gb}"
  TEACHER_SPEC="${TEACHER_SPEC}" STUDENT="${STUDENT}" DISTILL_GPU_IDS="${GPU_IDS_CSV}" bash rl-distill-scripts/scale_train/run_gemma4_distill_one.sh
  echo "Q30BM stage2: done $(date -u +%FT%TZ)"
fi

# --- stage 3: band-val evals of the saved checkpoints ----------------------------------------------------------------------------
EVAL_S3="${EVAL_S3:-s3://scale-ml/genai/rl-distill/${FAMILY}-distill-evals-v1/${TEACHER_SPEC}-to-${STUDENT}-bs${TRAIN_BATCH_SIZE}-s${TOTAL_TRAINING_STEPS}-lr${LR}}"
if [ "${BAND}" = all ]; then
  EVAL_BANDS="easy medium hard"; EVAL_STEPS="${EVAL_STEPS:-0 250 500 750 1000}"; N="${EVAL_N:-32}"; EVAL_OWN_NLL="${EVAL_OWN_NLL:-true}"
else
  EVAL_BANDS="${BAND}"; EVAL_STEPS="${EVAL_STEPS:-250 500 750 1000}"; N="${EVAL_N:-16}"; EVAL_OWN_NLL="${EVAL_OWN_NLL:-false}"
fi
case "${STUDENT}" in q4b) STUDENT_HF=Qwen/Qwen3-4B-Base@906bfd4b4dc7f14ee4320094d8b41684abff8539 ;; q1p7b) STUDENT_HF=Qwen/Qwen3-1.7B-Base@ea980cb0a6c2ae4b936e82123acc929f1cec04c1 ;; esac
WORK=/tmp/q30b_evals_${TEACHER_SPEC}; mkdir -p "${WORK}"
for band in ${EVAL_BANDS}; do
  "${VENV}/bin/python" - "${WORK}/${band}_validation.parquet" "${band}" <<'PY'
import shutil, sys
from huggingface_hub import hf_hub_download
# the Qwen3-30B band validation split = the RL validation set (raw problems; the generator appends the boxed instruction)
# very_hard was added in a later dataset commit (easy/medium/hard files are unchanged there too)
revision = {"very_hard": "9bec69ca0c432e6286cafdf8b26b4c6aa737084e"}.get(sys.argv[2], "be4dcd8b1fad7d5eb7bf721f0210baf78277aaa9")
p = hf_hub_download("JWei05/DeepScaleR-Easy-Medium-Hard-Qwen3-30B-A3B-Base-10k", f"{sys.argv[2]}/validation.parquet", repo_type="dataset",
                    revision=revision)
shutil.copy(p, sys.argv[1])
PY
done
# one band -> step_<s>/ (unchanged layout of the earlier runs); band union -> step_<s>/<band>/
eval_prefix() { if [ "${BAND}" = all ]; then echo "${EVAL_S3}/step_$1/$2"; else echo "${EVAL_S3}/step_$1"; fi; }
DEFAULT_INSTRUCTION='Please reason step by step, and put your final answer within \boxed{}.'
NL=$'\n'; STOP_STRINGS="<|im_end|>,${NL}Human:,${NL}user${NL}"; STOP_TOKEN_IDS=151645,151643,151644
tasks=()
for s in ${EVAL_STEPS}; do
  for band in ${EVAL_BANDS}; do
    if aws s3 ls "$(eval_prefix "${s}" "${band}")/summary.json" >/dev/null 2>&1; then echo "EVAL_ON_S3 step_${s}/${band}"; else tasks+=("${s}:${band}"); fi
  done
done
echo "Q30BM stage3: evals todo ${#tasks[@]} ($(date -u +%FT%TZ)) out=${EVAL_S3} n=${N} own_nll=${EVAL_OWN_NLL}"
pids=()
for g in $(seq 0 $((n_gpus - 1))); do
  (
    set -e
    for ((i = g; i < ${#tasks[@]}; i += n_gpus)); do
      s="${tasks[$i]%%:*}"; band="${tasks[$i]##*:}"; dest="$(eval_prefix "${s}" "${band}")"
      out="${WORK}/step_${s}/${band}"; model="${WORK}/models/step_${s}"; mkdir -p "${out}" "${model}"
      if [ "${s}" = 0 ]; then   # step 0 = the untrained student base (pinned Hub revision)
        ( flock 9; [ -f "${model}/.complete" ] || { "${VENV}/bin/python" -c "import sys; from huggingface_hub import snapshot_download; snapshot_download(sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3])" "${STUDENT_HF%@*}" "${STUDENT_HF#*@}" "${model}" >/dev/null && touch "${model}/.complete"; } ) 9>"${model}.lock"
      else
        ( flock 9; [ -f "${model}/.complete" ] || { aws s3 sync --only-show-errors "${REMOTE_CHECKPOINT_S3_URI}/global_step_${s}/huggingface/" "${model}/" && touch "${model}/.complete"; } ) 9>"${model}.lock"
      fi
      CUDA_VISIBLE_DEVICES="${g}" "${VENV}/bin/python" rl-distill-scripts/data/generate_deepscaler_4bpt_samples.py \
        --model "${model}" --input "${WORK}/${band}_validation.parquet" --chat-template rl-distill-scripts/data/qwen3_base_native_chat.jinja \
        --output "${out}/samples.parquet" --shard-id 0 --num-shards 1 --n "${N}" \
        --max-tokens 8192 --max-model-len 12288 --max-prompt-tokens 4096 \
        --temperature 1.0 --top-p 1.0 --top-k -1 --stop-strings "${STOP_STRINGS}" --stop-token-ids "${STOP_TOKEN_IDS}" \
        --append-boxed-instruction --boxed-instruction "${DEFAULT_INSTRUCTION}" --attention-backend TRITON_ATTN \
        --chunk-size 300 --max-num-seqs 512 --gpu-memory-utilization 0.9 > "${out}/log.txt" 2>&1 \
        || { echo "FATAL: eval step_${s}/${band} failed" >&2; tail -n 30 "${out}/log.txt" >&2; aws s3 cp --only-show-errors "${out}/log.txt" "${dest}/log.failed.txt"; exit 1; }
      PYTHONPATH="${PROJECT_ROOT}" "${VENV}/bin/python" rl-distill-scripts/data/grade_band_eval_samples.py "${out}/samples.parquet"
      if [ "${EVAL_OWN_NLL}" = true ]; then
        CUDA_VISIBLE_DEVICES="${g}" "${VENV}/bin/python" rl-distill-scripts/data/score_own_sample_nll.py --model "${model}" --samples "${out}/samples.parquet" \
          --chat-template rl-distill-scripts/data/qwen3_base_native_chat.jinja --per-question 8 --tensor-parallel-size 1 --output "${out}/nll.json" >> "${out}/log.txt" 2>&1 \
          || { echo "FATAL: own-NLL step_${s}/${band} failed" >&2; tail -n 30 "${out}/log.txt" >&2; aws s3 cp --only-show-errors "${out}/log.txt" "${dest}/log.failed.txt"; exit 1; }
      fi
      "${VENV}/bin/python" - "${out}/grades.json" "${out}/summary.json" "${s}" "${band}" "${out}/nll.json" <<'PY'
import json, os, sys
from math import comb
rows = json.load(open(sys.argv[1])); n = len(rows[0]["acc"])
c = [sum(r["acc"]) for r in rows]; q = len(rows)
pass_k = {k: 100 * sum(1 - comb(n - ci, k) / comb(n, k) for ci in c) / q for k in range(1, n + 1)}
pass_hat_k = {k: 100 * sum(comb(ci, k) / comb(n, k) for ci in c) / q for k in (2, 4, 8, 16, 32) if k <= n}
summary = {"step": int(sys.argv[3]), "band": sys.argv[4], "questions": q, "n": n, "mean_at_n": 100 * sum(c) / (q * n), "pass_at_k": pass_k,
           "pass_hat_k": pass_hat_k, "boxed_frac": sum(r["boxed"] for r in rows) / (q * n),
           "trunc_frac": sum(r["trunc"] for r in rows) / (q * n), "mean_response_len": sum(r["mean_len"] for r in rows) / q}
if os.path.exists(sys.argv[5]):
    summary["own_nll"] = json.load(open(sys.argv[5]))
json.dump(summary, open(sys.argv[2], "w"), indent=2)
print("EVAL_SUMMARY", json.dumps({k: v for k, v in summary.items() if k != "pass_at_k"}))
PY
      for f in log.txt samples.parquet grades.json nll.json summary.json; do [ -f "${out}/${f}" ] && aws s3 cp --only-show-errors "${out}/${f}" "${dest}/${f}"; done   # summary last: marks done
      echo "EVAL_DONE $(date -u +%FT%TZ) step_${s}/${band}"
    done
  ) &
  pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait "${p}" || rc=1; done
(( rc == 0 )) || { echo "FATAL: an eval task failed" >&2; exit 1; }
echo "Q30BM_DONE $(date -u +%FT%TZ) traces=${TRACE_S3_BASE}/${TEACHER_SPEC} ckpts=${REMOTE_CHECKPOINT_S3_URI} evals=${EVAL_S3}"
