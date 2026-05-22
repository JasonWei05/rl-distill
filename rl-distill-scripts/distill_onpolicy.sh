#!/usr/bin/env bash
# On-policy distillation (Thinking-Machines style) — parameterized student/teacher.
#
# Each training step:
#   1. Student rolls out 1 response per prompt via vLLM at T=1.0, top_p=1.0, top_k=-1.
#   2. Colocated teacher computes its log_prob of each student-sampled token
#      via vLLM prompt_logprobs (temperature=1.0, no top-k).
#   3. Reverse-KL k1 estimator:    kl_t = student_lp_t - teacher_lp_t
#      Advantage = -kl_t = teacher_lp_t - student_lp_t
#      Update    via vanilla policy gradient (on-policy, no ratio clipping).
#   4. Optionally eval student on the 7 math benchmarks at TEST_FREQ.
#
# Required env:
#   STUDENT_HF_REPO   e.g. google/gemma-3-1b-pt
#   STUDENT_TAG       short tag used in exp / HF-repo names, e.g. 1b-pt
#   TEACHER_REPO      e.g. JWei05/dapo-gemma3-4b-pt
#   TEACHER_REV       subfolder revision, e.g. step_000060
#   TEACHER_TAG       short tag used in HF-repo name, e.g. dapo4b-step60
#
# Optional env (defaults shown):
#   EXP_NAME                 onpolicy-<student_tag>-from-<teacher_tag>-<ts>
#   TOTAL_STEPS              200
#   SAVE_FREQ                50
#   TEST_FREQ                50            (in-loop eval on 7 math parquets)
#   LOSS_MODE                k1                (single-sample reverse-KL: student_lp − teacher_lp)
#   USE_POLICY_GRADIENT      true              (Thinking Machines recipe: -KL as advantage, importance-sampled PG)
#   POLICY_LOSS_MODE         vanilla           (= importance_sampling PPO; with ppo_epochs=1 the ratio is 1)
#   TOPK                     32                (only used if LOSS_MODE=reverse_kl_topk / forward_kl_topk)
#   LR_MAX                   1e-5
#   LR_WARMUP_STEPS          20
#   LR_SCHEDULER             constant
#   TRAIN_BSZ                128       (prompts per step)
#   N_RESP_PER_PROMPT        1         (group size: student rollouts per prompt)
#   TEACHER_TP               2
#   GEN_TP                   1             (student rollout TP)
#   STUDENT_GPU_MEM_UTIL     0.55
#   TEACHER_GPU_MEM_UTIL     0.35
#   MAX_PROMPT_LEN           2048
#   MAX_RESP_LEN             20480
#   HF_PUSH_REPO             JWei05/gemma3-<student_tag>-onpolicy-distill-from-<teacher_tag>
#
set -xeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/.venv/bin/activate"
set -a; source "${PROJECT_ROOT}/.env"; set +a

# CRITICAL: route all HF downloads to /tmp (worker-local SSD, ~3 TB free).
# Default ~/.cache/huggingface lives on /home/tiger which is 125 GB and NFS-
# shared across workers — 4 parallel teacher downloads (4B+12B+27B) overflow it.
export HF_HOME="${HF_HOME:-/tmp/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "${HF_HOME}"

: "${STUDENT_HF_REPO:?STUDENT_HF_REPO is required}"
: "${STUDENT_TAG:?STUDENT_TAG is required}"
: "${TEACHER_REPO:?TEACHER_REPO is required}"
: "${TEACHER_REV:?TEACHER_REV is required}"
: "${TEACHER_TAG:?TEACHER_TAG is required}"

TS="$(date +%Y%m%d-%H%M)"
EXP_NAME="${EXP_NAME:-onpolicy-${STUDENT_TAG}-from-${TEACHER_TAG}-${TS}}"
PROJECT_NAME="distill_onpolicy"

TOTAL_STEPS="${TOTAL_STEPS:-200}"
SAVE_FREQ="${SAVE_FREQ:-50}"
TEST_FREQ="${TEST_FREQ:-5}"
LOSS_MODE="${LOSS_MODE:-k1}"
TOPK="${TOPK:-32}"
LOSS_MAX_CLAMP="${LOSS_MAX_CLAMP:-10}"   # symmetric clamp [-K, +K] on per-token KL → caps PG advantage magnitude
USE_POLICY_GRADIENT="${USE_POLICY_GRADIENT:-true}"
POLICY_LOSS_MODE="${POLICY_LOSS_MODE:-vanilla}"
LR_MAX="${LR_MAX:-1e-5}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-20}"
LR_SCHEDULER="${LR_SCHEDULER:-constant}"
TRAIN_BSZ="${TRAIN_BSZ:-128}"
N_RESP_PER_PROMPT="${N_RESP_PER_PROMPT:-1}"
TEACHER_TP="${TEACHER_TP:-2}"
GEN_TP="${GEN_TP:-1}"
STUDENT_GPU_MEM_UTIL="${STUDENT_GPU_MEM_UTIL:-0.40}"
TEACHER_GPU_MEM_UTIL="${TEACHER_GPU_MEM_UTIL:-0.30}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-2048}"
MAX_RESP_LEN="${MAX_RESP_LEN:-20480}"

RAY_DATA_HOME="${RAY_DATA_HOME:-${HOME}/verl}"
DATA_DIR="${RAY_DATA_HOME}/data"
TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/dapo_openmath2_mix.parquet}"
VAL_FILES="['${DATA_DIR}/math__aime2024_repeated_32x_960.parquet','${DATA_DIR}/math__aime2025_repeated_32x_960.parquet','${DATA_DIR}/math__aime2026_repeated_32x_960.parquet','${DATA_DIR}/math__math_500_repeated_2x_1000.parquet','${DATA_DIR}/math__olympiadbench_repeated_2x.parquet','${DATA_DIR}/math__minervamath_repeated_4x.parquet','${DATA_DIR}/math__gsm8k_test.parquet']"

CKPT_BASE="${CKPT_BASE:-/tmp/verl/ckpts}"
CKPTS_DIR="${CKPT_BASE}/${PROJECT_NAME}/${EXP_NAME}"
mkdir -p "${CKPTS_DIR}"

HF_PUSH_REPO="${HF_PUSH_REPO:-JWei05/gemma3-${STUDENT_TAG}-onpolicy-distill-from-${TEACHER_TAG}}"

# =====================================================================
# Phase A — student: fetch base weights + inject chat_template if PT
# =====================================================================
LOCAL_STUDENT_DIR="/tmp/verl/models/$(echo "${STUDENT_HF_REPO}" | tr '/' '_')"
echo "[phaseA] preparing student ${STUDENT_HF_REPO} -> ${LOCAL_STUDENT_DIR}"
python3 - <<PYEOF
import json, os
from huggingface_hub import snapshot_download, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError

local = "${LOCAL_STUDENT_DIR}"
pt = "${STUDENT_HF_REPO}"
is_pt = pt.endswith("-pt")
it = pt[:-3] + "-it" if is_pt else pt

if not os.path.exists(f"{local}/config.json"):
    snapshot_download(pt, local_dir=local,
        allow_patterns=["*.json", "*.jinja", "*.safetensors", "*.model", "*.txt"])

# If student is PT, it has no chat_template. Extract from the IT counterpart.
if is_pt:
    tcfg_path = f"{local}/tokenizer_config.json"
    tcfg = json.load(open(tcfg_path))
    if not tcfg.get("chat_template"):
        chat_template = None
        try:
            src = hf_hub_download(repo_id=it, filename="chat_template.json",
                                  local_dir=f"{local}/_it_aux")
            chat_template = json.load(open(src)).get("chat_template")
            print(f"[dl] got chat_template from {it}/chat_template.json")
        except EntryNotFoundError:
            src = hf_hub_download(repo_id=it, filename="tokenizer_config.json",
                                  local_dir=f"{local}/_it_aux")
            chat_template = json.load(open(src)).get("chat_template")
            print(f"[dl] got chat_template from {it}/tokenizer_config.json (1B-style)")
        if not chat_template:
            raise RuntimeError(f"could not find chat_template in {it}")
        # verl probes with two user turns; strip Gemma 3's alternation raise.
        patched = chat_template.replace(
            '{{ raise_exception("Conversation roles must alternate user/assistant/user/assistant/...") }}',
            '{# alternation check disabled for verl probing #}',
        )
        tcfg["chat_template"] = patched
        json.dump(tcfg, open(tcfg_path, "w"), indent=2)
        print(f"[dl] inlined chat_template into {tcfg_path} ({len(patched)} chars)")
    else:
        print("[dl] student already has chat_template, skipping")
PYEOF

# =====================================================================
# Phase B — teacher: snapshot <TEACHER_REPO>/<TEACHER_REV>/* locally
# =====================================================================
TEACHER_PATH="$(TEACHER_REPO="${TEACHER_REPO}" TEACHER_REV="${TEACHER_REV}" \
    bash "${SCRIPT_DIR}/data/prep_teacher.sh" | tail -1)"
echo "[phaseB] teacher at ${TEACHER_PATH}"

# Sanity: teacher must have a chat_template (either inlined in tokenizer_config.json
# or as a separate chat_template.jinja / chat_template.json file).
python3 - <<PYEOF
import json, os
t = "${TEACHER_PATH}"
tcfg = json.load(open(os.path.join(t, "tokenizer_config.json")))
ct = tcfg.get("chat_template")
if ct:
    print(f"[phaseB] teacher chat_template inlined in tokenizer_config.json ({len(ct)} chars)")
elif os.path.exists(os.path.join(t, "chat_template.jinja")):
    sz = os.path.getsize(os.path.join(t, "chat_template.jinja"))
    print(f"[phaseB] teacher chat_template.jinja present ({sz} bytes)")
elif os.path.exists(os.path.join(t, "chat_template.json")):
    print(f"[phaseB] teacher chat_template.json present")
else:
    raise SystemExit(f"[phaseB] teacher at {t} is missing chat_template (no inline / .jinja / .json)")
PYEOF

# =====================================================================
# Phase C — launch on-policy distillation
# =====================================================================
export VLLM_USE_V1=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
# Default lo for single-node; override to a real fabric interface (e.g. eth4) for multi-node.
NCCL_IFNAME="${NCCL_IFNAME:-lo}"
export NCCL_SOCKET_IFNAME="${NCCL_IFNAME}"
export NCCL_SOCKET_FAMILY=AF_INET
export GLOO_SOCKET_IFNAME="${NCCL_IFNAME}"

# Multi-node teacher resource pool (default: colocated on the same Ray cluster as student).
TEACHER_ENABLE_RESOURCE_POOL="${TEACHER_ENABLE_RESOURCE_POOL:-false}"
TEACHER_NNODES_OVERRIDE="${TEACHER_NNODES_OVERRIDE:-0}"

# Gemma 3 1B (text-only) on B200 hits a FlashInfer head_size=256 / block_size=16
# bug — force FLASH_ATTN on student rollout only. The teacher (4B+) is multimodal
# and rejects FLASH_ATTN ("partial multimodal token full attention not supported"),
# so leave the teacher on the vLLM default backend.
EXTRA_VLLM_OVERRIDES=()
if [ "${STUDENT_TAG}" = "1b-pt" ] || [ "${STUDENT_TAG}" = "1b-it" ] || [ "${STUDENT_TAG}" = "1b" ]; then
    EXTRA_VLLM_OVERRIDES+=(
        "+actor_rollout_ref.rollout.engine_kwargs.vllm.attention_backend=FLASH_ATTN"
    )
    echo "[phaseC] forcing FLASH_ATTN backend for 1B student rollout (B200 workaround)"
fi

SEQ_MAX=$((MAX_PROMPT_LEN + MAX_RESP_LEN))
TEACHER_MAX_MODEL_LEN=$((SEQ_MAX + 1))

NNODES="${NNODES:-1}"

python3 -m dapo.main_dapo \
    +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_SOCKET_IFNAME="${NCCL_IFNAME}" \
    +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_SOCKET_FAMILY=AF_INET \
    +ray_kwargs.ray_init.runtime_env.env_vars.GLOO_SOCKET_IFNAME="${NCCL_IFNAME}" \
    +ray_kwargs.ray_init.runtime_env.env_vars.WANDB_API_KEY="${WANDB_API_KEY}" \
    +ray_kwargs.ray_init.runtime_env.env_vars.HF_TOKEN="${HF_TOKEN}" \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILES}" \
    data.prompt_key=prompt \
    data.shuffle=True \
    data.seed=42 \
    data.truncation='left' \
    data.max_prompt_length=${MAX_PROMPT_LEN} \
    data.max_response_length=${MAX_RESP_LEN} \
    data.gen_batch_size=${TRAIN_BSZ} \
    data.train_batch_size=${TRAIN_BSZ} \
    actor_rollout_ref.rollout.n=${N_RESP_PER_PROMPT} \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${SEQ_MAX} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${SEQ_MAX} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${SEQ_MAX} \
    actor_rollout_ref.model.path="${LOCAL_STUDENT_DIR}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr="${LR_MAX}" \
    actor_rollout_ref.actor.optim.lr_warmup_steps="${LR_WARMUP_STEPS}" \
    actor_rollout_ref.actor.optim.lr_scheduler_type="${LR_SCHEDULER}" \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${TRAIN_BSZ} \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.clip_ratio_low=100.0 \
    actor_rollout_ref.actor.clip_ratio_high=100.0 \
    actor_rollout_ref.actor.clip_ratio_c=100.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=${STUDENT_GPU_MEM_UTIL} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=${SEQ_MAX} \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    '+actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=["Gemma3DecoderLayer"]' \
    ++distillation.enabled=true \
    ++distillation.num_workers=8 \
    ++distillation.distillation_loss.loss_mode="${LOSS_MODE}" \
    ++distillation.distillation_loss.topk="${TOPK}" \
    ++distillation.distillation_loss.loss_max_clamp="${LOSS_MAX_CLAMP}" \
    ++distillation.distillation_loss.use_task_rewards=false \
    ++distillation.distillation_loss.use_policy_gradient="${USE_POLICY_GRADIENT}" \
    ++distillation.distillation_loss.policy_loss_mode="${POLICY_LOSS_MODE}" \
    ++distillation.distillation_loss.distillation_loss_coef=1.0 \
    ++distillation.teacher_model.model_path="${TEACHER_PATH}" \
    ++distillation.teacher_model.enable_resource_pool=${TEACHER_ENABLE_RESOURCE_POOL} \
    ++distillation.teacher_model.n_gpus_per_node=8 \
    ++distillation.teacher_model.nnodes=${TEACHER_NNODES_OVERRIDE} \
    ++distillation.teacher_model.inference.tensor_model_parallel_size=${TEACHER_TP} \
    ++distillation.teacher_model.inference.gpu_memory_utilization=${TEACHER_GPU_MEM_UTIL} \
    ++distillation.teacher_model.inference.enforce_eager=true \
    ++distillation.teacher_model.inference.max_model_len=${TEACHER_MAX_MODEL_LEN} \
    ++distillation.teacher_model.inference.temperature=1.0 \
    "${EXTRA_VLLM_OVERRIDES[@]}" \
    reward_model.reward_manager=dapo \
    reward.reward_kwargs.max_resp_len=${MAX_RESP_LEN} \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=False \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.max_actor_ckpt_to_keep=2 \
    +trainer.hf_push.enable=True \
    +trainer.hf_push.repo_id="${HF_PUSH_REPO}" \
    +trainer.hf_push.private=False \
    +trainer.hf_push.delete_local_after=True \
    +trainer.hf_push.max_to_keep=10 \
    'actor_rollout_ref.actor.checkpoint.save_contents=[hf_model]' \
    trainer.total_epochs=2 \
    trainer.total_training_steps="${TOTAL_STEPS}" \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto $@
