#!/usr/bin/env bash
# End-to-end pipeline for one (teacher_dataset × student_base) distillation:
#   Phase A — download teacher parquet + split prompt/response samples
#   Phase B — torchrun SFT training, saving/pushing at SAVE_FREQ
#   Phase C — vLLM eval selected pushed checkpoints on the 8 RL validation sets
#
# Required env vars (caller sets):
#   TEACHER_REPO          HF dataset repo, e.g. JWei05/DAPO-Gemma3-27B-PT-RL-SFT-Data
#   TEACHER_PARQUET_NAME  filename inside the dataset, e.g. teacher_27b_step80_n2.parquet
#   or TEACHER_LOCAL_PARQUET for an already-generated local parquet.
#   STUDENT_HF_REPO       google/gemma-3-{1,4,12}b-pt
#   STUDENT_TAG           short tag: 1b | 4b | 12b
#   TEACHER_TAG           short tag: 4b | 12b | 27b
# Optional:
#   EXP_NAME              defaults to distill-${STUDENT_TAG}-from-${TEACHER_TAG}-<ts>
#   TOTAL_STEPS           default 1000
#   SAVE_FREQ             default 250
#   EVAL_STEPS            default "250 500 750 1000"
#   TEST_FREQ             default 5 (in-loop forward-KL val loss)
#   MICRO_BSZ             default 16
#   N_TRAIN_PROMPTS       default 32000
#   TRAIN_RESPONSES_PER_PROMPT default 4
#   N_VAL_PROMPTS         default 1000
#   VAL_RESPONSES_PER_PROMPT default 1
#   SPLIT_TAG             optional cache key for train/val split filenames
#   WANDB_PROXY_CPU_IP    optional: CPU node IPv6/host for SSH SOCKS proxy
#   WANDB_PROXY_CPU_PORT  optional: CPU node ssh port for SSH SOCKS proxy
#
set -xeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/.venv/bin/activate"
set -a; source "${PROJECT_ROOT}/.env"; set +a

export HF_HOME="${HF_HOME:-/tmp/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "${HF_HOME}"

if [ -f /proc/driver/nvidia/version ]; then
    NVIDIA_KERNEL_VERSION="$(sed -n 's/.*  \([0-9][0-9.]*\)  .*/\1/p' /proc/driver/nvidia/version | head -1 || true)"
    if [ -n "${NVIDIA_KERNEL_VERSION}" ] && [ -f "/usr/lib/x86_64-linux-gnu/libcuda.so.${NVIDIA_KERNEL_VERSION}" ]; then
        DRIVER_LIB_DIR="/tmp/nvidia-driver-libs-${NVIDIA_KERNEL_VERSION}"
        mkdir -p "${DRIVER_LIB_DIR}"
        ln -sf "/usr/lib/x86_64-linux-gnu/libcuda.so.${NVIDIA_KERNEL_VERSION}" "${DRIVER_LIB_DIR}/libcuda.so"
        ln -sf "/usr/lib/x86_64-linux-gnu/libcuda.so.${NVIDIA_KERNEL_VERSION}" "${DRIVER_LIB_DIR}/libcuda.so.1"
        if [ -f "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.${NVIDIA_KERNEL_VERSION}" ]; then
            ln -sf "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.${NVIDIA_KERNEL_VERSION}" "${DRIVER_LIB_DIR}/libnvidia-ml.so.1"
        fi
        export LD_LIBRARY_PATH="${DRIVER_LIB_DIR}:${LD_LIBRARY_PATH:-}"
        echo "[env] using NVIDIA user libraries from ${DRIVER_LIB_DIR}"
    fi
fi

: "${TEACHER_LOCAL_PARQUET:=${TEACHER_LOCAL_PARQUET:-}}"
if [ -z "${TEACHER_LOCAL_PARQUET}" ]; then
    : "${TEACHER_REPO:?TEACHER_REPO is required when TEACHER_LOCAL_PARQUET is unset}"
    : "${TEACHER_PARQUET_NAME:?TEACHER_PARQUET_NAME is required when TEACHER_LOCAL_PARQUET is unset}"
fi
: "${STUDENT_HF_REPO:?STUDENT_HF_REPO is required}"
: "${STUDENT_TAG:?STUDENT_TAG is required (1b|4b|12b)}"
: "${TEACHER_TAG:?TEACHER_TAG is required (4b|12b|27b)}"

TS="$(date +%Y%m%d-%H%M)"
export EXP_NAME="${EXP_NAME:-distill-${STUDENT_TAG}-from-${TEACHER_TAG}-${TS}}"
export PROJECT_NAME="distill"

RAY_DATA_HOME="${RAY_DATA_HOME:-${HOME}/verl}"
DATA_DIR="${RAY_DATA_HOME}/data"
EVAL_DATA_DIR="${EVAL_DATA_DIR:-${HOME}/verl/data}"
mkdir -p "${DATA_DIR}"

TOTAL_STEPS="${TOTAL_STEPS:-1000}"
export SAVE_FREQ="${SAVE_FREQ:-250}"
export TEST_FREQ="${TEST_FREQ:-5}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-true}"
export MAX_CKPT_TO_KEEP="${MAX_CKPT_TO_KEEP:-4}"
MICRO_BSZ="${MICRO_BSZ:-16}"
SEED="${SEED:-42}"
WARMUP_STEPS="${WARMUP_STEPS:-100}"
LR_MAX="${LR_MAX:-5e-6}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0.1}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
N_TRAIN_PROMPTS="${N_TRAIN_PROMPTS:-32000}"
N_VAL_PROMPTS="${N_VAL_PROMPTS:-1000}"
TRAIN_RESPONSES_PER_PROMPT="${TRAIN_RESPONSES_PER_PROMPT:-4}"
VAL_RESPONSES_PER_PROMPT="${VAL_RESPONSES_PER_PROMPT:-1}"
export EVAL_STEPS="${EVAL_STEPS:-250 500 750 1000}"

# ---- HF push: each save (steps 250, 500, 750) → JWei05/gemma3-{tag}-pt-sft-distill-from-{teacher_tag}/step_NNNNNN
export HF_PUSH_ENABLE=true
export HF_PUSH_REPO="${HF_PUSH_REPO:-JWei05/gemma3-${STUDENT_TAG}-pt-sft-distill-from-${TEACHER_TAG}}"
export HF_PUSH_PRIVATE=false
export HF_PUSH_MAX_TO_KEEP="${HF_PUSH_MAX_TO_KEEP:-4}"
export HF_PUSH_DELETE_LOCAL=true   # reclaim /tmp after push

CKPT_BASE="${CKPT_BASE:-/tmp/verl/ckpts}"
export CKPTS_DIR="${CKPT_BASE}/${PROJECT_NAME}/${EXP_NAME}"
mkdir -p "${CKPTS_DIR}"

PROXY_PID=""
if [ -n "${WANDB_PROXY_CPU_IP:-}" ] && [ -n "${WANDB_PROXY_CPU_PORT:-}" ]; then
    WANDB_PROXY_LOCAL_PORT="${WANDB_PROXY_LOCAL_PORT:-18080}"
    echo "[env] starting SSH SOCKS proxy for W&B/HF via ${WANDB_PROXY_CPU_IP}:${WANDB_PROXY_CPU_PORT} on 127.0.0.1:${WANDB_PROXY_LOCAL_PORT}"
    ssh -N -D "127.0.0.1:${WANDB_PROXY_LOCAL_PORT}" \
        -o ExitOnForwardFailure=yes \
        -o ServerAliveInterval=30 \
        -o ServerAliveCountMax=3 \
        -p "${WANDB_PROXY_CPU_PORT}" \
        "tiger@${WANDB_PROXY_CPU_IP}" &
    PROXY_PID="$!"
    trap 'if [ -n "${PROXY_PID}" ]; then kill "${PROXY_PID}" >/dev/null 2>&1 || true; fi' EXIT
    sleep 3
    export HTTP_PROXY="socks5h://127.0.0.1:${WANDB_PROXY_LOCAL_PORT}"
    export HTTPS_PROXY="${HTTP_PROXY}"
    export http_proxy="${HTTP_PROXY}"
    export https_proxy="${HTTPS_PROXY}"
fi

# ============================================================
# Phase A — fetch teacher parquet + split into train/val
# ============================================================
TEACHER_PARQUET_STEM="$(basename "${TEACHER_LOCAL_PARQUET:-${TEACHER_PARQUET_NAME}}" .parquet)"
SPLIT_TAG="${SPLIT_TAG:-${TEACHER_TAG}_${TEACHER_PARQUET_STEM}_p${N_TRAIN_PROMPTS}_r${TRAIN_RESPONSES_PER_PROMPT}_v${N_VAL_PROMPTS}_vr${VAL_RESPONSES_PER_PROMPT}_seed${SEED}}"
TRAIN_PARQUET="${DATA_DIR}/teacher_${SPLIT_TAG}_sft_train.parquet"
VAL_PARQUET="${DATA_DIR}/teacher_${SPLIT_TAG}_sft_val.parquet"
export TRAIN_FILE="${TRAIN_PARQUET}"
export VAL_FILE="${VAL_PARQUET}"

if [ ! -f "${TRAIN_PARQUET}" ] || [ ! -f "${VAL_PARQUET}" ]; then
    SPLIT_ARGS=(
        --n_train_prompts "${N_TRAIN_PROMPTS}"
        --n_val_prompts "${N_VAL_PROMPTS}"
        --train_responses_per_prompt "${TRAIN_RESPONSES_PER_PROMPT}"
        --val_responses_per_prompt "${VAL_RESPONSES_PER_PROMPT}"
        --output_train "${TRAIN_PARQUET}"
        --output_val "${VAL_PARQUET}"
        --seed "${SEED}"
    )
    if [ -n "${TEACHER_LOCAL_PARQUET}" ]; then
        echo "[phaseA] splitting local teacher data: ${TEACHER_LOCAL_PARQUET} (seed=${SEED}, train_prompts=${N_TRAIN_PROMPTS}, train_responses_per_prompt=${TRAIN_RESPONSES_PER_PROMPT}, val_prompts=${N_VAL_PROMPTS}, val_responses_per_prompt=${VAL_RESPONSES_PER_PROMPT})"
        SPLIT_ARGS=(--input "${TEACHER_LOCAL_PARQUET}" "${SPLIT_ARGS[@]}")
    else
        echo "[phaseA] downloading + splitting teacher data: ${TEACHER_REPO}/${TEACHER_PARQUET_NAME} (seed=${SEED}, train_prompts=${N_TRAIN_PROMPTS}, train_responses_per_prompt=${TRAIN_RESPONSES_PER_PROMPT}, val_prompts=${N_VAL_PROMPTS}, val_responses_per_prompt=${VAL_RESPONSES_PER_PROMPT})"
        SPLIT_ARGS=(--repo_id "${TEACHER_REPO}" --filename "${TEACHER_PARQUET_NAME}" "${SPLIT_ARGS[@]}")
    fi
    python3 "${SCRIPT_DIR}/data/split_sft_dataset.py" "${SPLIT_ARGS[@]}"
else
    echo "[phaseA] reusing cached splits: ${TRAIN_PARQUET}, ${VAL_PARQUET}"
fi

# ============================================================
# Phase A.5 — fetch student base model (idempotent)
# ============================================================
LOCAL_STUDENT_DIR="/tmp/verl/models/$(echo "${STUDENT_HF_REPO}" | tr '/' '_')"
# Gemma 3 PT repos have no chat_template — fetch from the IT counterpart and
# inline it into the PT tokenizer_config.json so transformers picks it up.
IT_REPO="$(echo "${STUDENT_HF_REPO}" | sed 's/-pt$/-it/')"
echo "[phaseA] preparing student ${STUDENT_HF_REPO} -> ${LOCAL_STUDENT_DIR}"
python3 - <<PYEOF
import json, os
from huggingface_hub import snapshot_download, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError

local = "${LOCAL_STUDENT_DIR}"
it = "${IT_REPO}"
pt = "${STUDENT_HF_REPO}"

if not os.path.exists(f"{local}/config.json"):
    snapshot_download(pt, local_dir=local,
        allow_patterns=["*.json", "*.jinja", "*.safetensors", "*.model", "*.txt"])

# Extract chat_template from IT (try chat_template.json first, fall back to tokenizer_config.json)
chat_template = None
try:
    src = hf_hub_download(repo_id=it, filename="chat_template.json",
                          local_dir=f"{local}/_it_aux")
    j = json.load(open(src))
    chat_template = j.get("chat_template")
    print(f"[dl] got chat_template from {it}/chat_template.json")
except EntryNotFoundError:
    src = hf_hub_download(repo_id=it, filename="tokenizer_config.json",
                          local_dir=f"{local}/_it_aux")
    j = json.load(open(src))
    chat_template = j.get("chat_template")
    print(f"[dl] got chat_template from {it}/tokenizer_config.json (1B-style)")

if not chat_template:
    raise RuntimeError(f"could not find chat_template in {it}")

# verl's extract_system_prompt_and_generation probes the template with two
# consecutive {role:user} dummy messages; Gemma 3's template raises on
# non-alternating roles. Strip that raise_exception so the introspection works.
patched = chat_template.replace(
    '{{ raise_exception("Conversation roles must alternate user/assistant/user/assistant/...") }}',
    '{# alternation check disabled for verl probing #}',
)
if patched == chat_template:
    print("[dl] WARN: did not find expected raise_exception in chat_template")
else:
    print("[dl] patched chat_template (removed alternation raise)")
chat_template = patched

# Inline into PT's tokenizer_config.json so transformers loads it natively.
tcfg_path = f"{local}/tokenizer_config.json"
tcfg = json.load(open(tcfg_path))
tcfg["chat_template"] = chat_template
json.dump(tcfg, open(tcfg_path, "w"), indent=2)
print(f"[dl] inlined chat_template into {tcfg_path} ({len(chat_template)} chars)")
PYEOF

# ============================================================
# Phase B — train
# ============================================================
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_SOCKET_IFNAME=lo
export NCCL_SOCKET_FAMILY=AF_INET
export GLOO_SOCKET_IFNAME=lo
export VLLM_WORKER_MULTIPROC_METHOD=spawn

cd "${SCRIPT_DIR}"
torchrun --standalone --nnodes=1 --nproc_per_node=8 main_distill_offpolicy.py \
    model.path="${LOCAL_STUDENT_DIR}" \
    model.enable_gradient_checkpointing=True \
    engine.strategy=fsdp2 \
    engine.fsdp_size=-1 \
    '+engine.wrap_policy.transformer_layer_cls_to_wrap=["Gemma3DecoderLayer"]' \
    optim.lr="${LR_MAX}" \
    optim.lr_warmup_steps="${WARMUP_STEPS}" \
    optim.lr_scheduler_type="${LR_SCHEDULER}" \
    optim.min_lr_ratio="${MIN_LR_RATIO}" \
    optim.total_training_steps="${TOTAL_STEPS}" \
    optim.weight_decay=0.1 \
    optim.betas=[0.9,0.98] \
    optim.clip_grad=1.0 \
    data.train_batch_size=128 \
    data.micro_batch_size_per_gpu="${MICRO_BSZ}" \
    data.max_length=22528 \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.total_training_steps="${TOTAL_STEPS}" \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    trainer.seed="${SEED}" \
    trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
    trainer.default_local_dir="${CKPTS_DIR}"

echo "[phaseB] training done"

# ============================================================
# Phase C — eval each pushed checkpoint from HF
# ============================================================
EVAL_OUT="${DATA_DIR}/eval_results/${EXP_NAME}"
mkdir -p "${EVAL_OUT}"

VAL_FILES=(
    "${EVAL_DATA_DIR}/dapo_openmath2_mix_val_compat.parquet"
    "${EVAL_DATA_DIR}/math__aime2024_repeated_32x_960_compat.parquet"
    "${EVAL_DATA_DIR}/math__aime2025_repeated_32x_960_compat.parquet"
    "${EVAL_DATA_DIR}/math__aime2026_repeated_32x_960_compat.parquet"
    "${EVAL_DATA_DIR}/math__math_500_repeated_2x_1000_compat.parquet"
    "${EVAL_DATA_DIR}/math__olympiadbench_repeated_2x_compat.parquet"
    "${EVAL_DATA_DIR}/math__minervamath_repeated_4x_compat.parquet"
    "${EVAL_DATA_DIR}/math__gsm8k_test_compat.parquet"
)

# Wait briefly to ensure async HF pushes from training have completed.
# (HFPusher.wait was already called in main_distill_offpolicy.fit's finally block,
#  so by the time torchrun exits the pushes are done; sleep is just a guard.)
sleep 5

# Determine which step subfolders to evaluate.
STEPS=()
for s in ${EVAL_STEPS}; do
    STEPS+=("$(printf 'step_%06d' "${s}")")
done
echo "[phaseC] evaluating ${#STEPS[@]} ckpts: ${STEPS[*]}"

# Gemma 3 1B needs FLASH_ATTN backend (FlashInfer has head_size=256 + B200 bug).
# Other sizes use vLLM default.
EVAL_EXTRA_ARGS=()
if [ "${STUDENT_TAG}" = "1b" ]; then
    EVAL_EXTRA_ARGS+=(--attention_backend FLASH_ATTN)
fi
if [ -n "${MM_ENCODER_ATTN_BACKEND:-}" ]; then
    EVAL_EXTRA_ARGS+=(--mm_encoder_attn_backend "${MM_ENCODER_ATTN_BACKEND}")
fi

# Run evals in batches of 2 parallel × TP=4 each (8 GPUs total per batch).
EVAL_TP=4
EVAL_PARALLEL=2
idx=0
while [ $idx -lt ${#STEPS[@]} ]; do
    BATCH_PIDS=()
    for slot in $(seq 0 $((EVAL_PARALLEL - 1))); do
        [ $idx -ge ${#STEPS[@]} ] && break
        SUB="${STEPS[$idx]}"
        GPU_START=$((slot * EVAL_TP))
        GPU_END=$((GPU_START + EVAL_TP - 1))
        GPUS="$(seq -s, ${GPU_START} ${GPU_END})"
        LOG="${EVAL_OUT}/eval_${SUB}.log"
        echo "[phaseC] batch=$((idx / EVAL_PARALLEL)) slot=${slot} ${SUB} GPUs=${GPUS} -> ${LOG}"
        CUDA_VISIBLE_DEVICES="${GPUS}" \
        nohup python3 "${PROJECT_ROOT}/dapo/_eval_model_on_math.py" \
            --repo_id "${HF_PUSH_REPO}" \
            --subfolder "${SUB}" \
            --base_hf_model "${STUDENT_HF_REPO}" \
            --val_files "${VAL_FILES[@]}" \
            --output_dir "${EVAL_OUT}" \
            --tp ${EVAL_TP} \
            --temperature 1.0 \
            --top_p 0.7 \
            --top_k -1 \
            --max_tokens 20480 \
            "${EVAL_EXTRA_ARGS[@]}" \
            > "${LOG}" 2>&1 &
        BATCH_PIDS+=("$!")
        idx=$((idx + 1))
        sleep 30  # stagger to avoid simultaneous shm/NCCL contention
    done
    echo "[phaseC] waiting on batch PIDs: ${BATCH_PIDS[*]}"
    wait "${BATCH_PIDS[@]}" || true
    echo "[phaseC] batch done"
done

echo ""
echo "[phaseC] all eval JSONs in ${EVAL_OUT}:"
ls -la "${EVAL_OUT}"/*.json 2>/dev/null || echo "(no JSONs found — check eval logs)"

# Summary table
python3 - <<PYEOF
import glob, json, os
out = "${EVAL_OUT}"
files = sorted(glob.glob(f"{out}/*__summary.json"))
# per_dataset keys are val-file basenames (e.g. math__aime2024_repeated_32x_960)
benches = [
    ("dapo_val", "dapo_openmath2_mix_val_compat"),
    ("aime2024", "math__aime2024_repeated_32x_960_compat"),
    ("aime2025", "math__aime2025_repeated_32x_960_compat"),
    ("aime2026", "math__aime2026_repeated_32x_960_compat"),
    ("math500",  "math__math_500_repeated_2x_1000_compat"),
    ("olympiad", "math__olympiadbench_repeated_2x_compat"),
    ("minerva",  "math__minervamath_repeated_4x_compat"),
    ("gsm8k",    "math__gsm8k_test_compat"),
]
print("\n=== ${EXP_NAME} eval summary (acc) ===")
print(f"{'step':>14s} | " + " | ".join(f"{b[0]:>9s}" for b in benches))
for f in sorted(files):
    try:
        d = json.load(open(f))
    except Exception as e:
        print(f"{os.path.basename(f)}: parse error {e}"); continue
    step = d.get("subfolder", os.path.basename(f).split("__")[1] if "__" in os.path.basename(f) else "?")
    row = []
    for short, key in benches:
        v = d.get("per_dataset", {}).get(key, {}).get("acc")
        row.append(f"{v:>9.4f}" if isinstance(v,(int,float)) else f"{'?':>9s}")
    print(f"{step:>14s} | " + " | ".join(row))
PYEOF

python3 "${SCRIPT_DIR}/update_eval_results_md.py" \
    --results-dir "${EVAL_OUT}" \
    --run-name "${EXP_NAME}" || true

echo "[done] ${EXP_NAME}"
