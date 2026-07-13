#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

if [ -f "${PROJECT_ROOT}/.env" ]; then
    set -a
    source "${PROJECT_ROOT}/.env"
    set +a
fi

# Activate the Megatron venv. Prefer setup_megatron.sh's .venv-megatron when
# present; clusters whose .venv already is the Megatron venv keep working.
if [ -z "${MEGATRON_VENV:-}" ]; then
    if [ -d "${PROJECT_ROOT}/.venv-megatron" ]; then
        MEGATRON_VENV="${PROJECT_ROOT}/.venv-megatron"
    else
        MEGATRON_VENV="${PROJECT_ROOT}/.venv"
    fi
fi
unset PYTHONPATH
unset LD_LIBRARY_PATH
source "${MEGATRON_VENV}/bin/activate"

export VLLM_USE_V1=1
export VLLM_DO_NOT_TRACK=1
export DO_NOT_TRACK=1
export VLLM_NO_USAGE_STATS=1
export RAY_USAGE_STATS_ENABLED=0
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-/tmp/.config}"
export VLLM_CONFIG_ROOT="${VLLM_CONFIG_ROOT:-${XDG_CONFIG_HOME}/vllm}"
mkdir -p "${VLLM_CONFIG_ROOT}"
touch "${VLLM_CONFIG_ROOT}/do_not_track" 2>/dev/null || true
export TOKENIZERS_PARALLELISM=true
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-/tmp/hf-transformers-modules-gemma3-moe-${NUM_EXPERTS:-2}e}"
export GEMMA3_CHAT_TEMPLATE_FILE="${GEMMA3_CHAT_TEMPLATE_FILE:-${PROJECT_ROOT}/rl-distill-scripts/data/gemma3_it_chat_template.jinja}"
export VERL_MATH_VERIFY_FAST_INVALID="${VERL_MATH_VERIFY_FAST_INVALID:-False}"
export VERL_MATH_VERIFY_FAST_EQUIV="${VERL_MATH_VERIFY_FAST_EQUIV:-False}"
export VERL_MATH_VERIFY_TIMEOUT="${VERL_MATH_VERIFY_TIMEOUT:-30.0}"
export VERL_MATH_VERIFY_MAX_CHARS="${VERL_MATH_VERIFY_MAX_CHARS:-0}"
export VERL_MATH_VERIFY_POOL_WORKERS="${VERL_MATH_VERIFY_POOL_WORKERS:-4}"
export VERL_AGENT_LOOP_PREFER_GPU_NODES="${VERL_AGENT_LOOP_PREFER_GPU_NODES:-0}"

NUM_EXPERTS=${NUM_EXPERTS:-2}
if [ "${NUM_EXPERTS}" != "2" ] && [ "${NUM_EXPERTS}" != "4" ]; then
    echo "NUM_EXPERTS must be 2 or 4; got ${NUM_EXPERTS}" >&2
    exit 2
fi

project_name=${PROJECT_NAME:-DAPO}
exp_name=${EXP_NAME:-"DAPO-Gemma3-4B-PT-MoE-${NUM_EXPERTS}E-Megatron-RL-$(date +%Y%m%d-%H%M)"}

adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

max_prompt_length=${MAX_PROMPT_LENGTH:-$((1024 * 2))}
max_response_length=${MAX_RESPONSE_LENGTH:-$((1024 * 20))}
enable_overlong_buffer=${ENABLE_OVERLONG_BUFFER:-True}
overlong_buffer_len=${OVERLONG_BUFFER_LEN:-$((1024 * 4))}
overlong_penalty_factor=${OVERLONG_PENALTY_FACTOR:-1.0}

loss_agg_mode="token-mean"

enable_filter_groups=${ENABLE_FILTER_GROUPS:-False}
filter_groups_metric=${FILTER_GROUPS_METRIC:-acc}
max_num_gen_batches=${MAX_NUM_GEN_BATCHES:-10}
train_prompt_bsz=${TRAIN_PROMPT_BSZ:-512}
gen_prompt_bsz=${GEN_PROMPT_BSZ:-${train_prompt_bsz}}
n_resp_per_prompt=${N_RESP_PER_PROMPT:-16}
train_prompt_mini_bsz=${TRAIN_PROMPT_MINI_BSZ:-32}
test_freq=${TEST_FREQ:-2}
val_before_train=${VAL_BEFORE_TRAIN:-False}
reward_num_workers=${REWARD_NUM_WORKERS:-8}
agent_loop_num_workers=${AGENT_LOOP_NUM_WORKERS:-8}

# Ray. Launch this on the Ray head of the assigned 2-node H100 cluster.
RAY_ADDRESS=${RAY_ADDRESS:-auto}
NNODES=${NNODES:-2}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}

# Paths. Set MODEL_PATH/HF_MOE_LOCAL_DIR for a local conversion, or explicitly
# set HF_MOE_REPO for a Hub checkpoint. There is deliberately no implicit model
# fallback: accidentally training an old SFT checkpoint is not an upcycling run.
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
HF_MOE_REPO=${HF_MOE_REPO:-}
HF_MOE_REVISION=${HF_MOE_REVISION:-main}
HF_MOE_CACHE_DIR=${HF_MOE_CACHE_DIR:-"/tmp/hf-gemma3-moe-rl-cache"}
HF_MOE_LOCAL_DIR=${HF_MOE_LOCAL_DIR:-}
MODEL_PATH=${MODEL_PATH:-}
DIST_CKPT_PATH=${DIST_CKPT_PATH:-}
USE_DIST_CHECKPOINTING=${USE_DIST_CHECKPOINTING:-False}
CKPTS_DIR=${CKPTS_DIR:-"/tmp/verl/ckpts/${project_name}/${exp_name}"}
HF_PUSH_REPO=${HF_PUSH_REPO:-}
HF_PUSH_ENABLE=${HF_PUSH_ENABLE:-False}
HF_PUSH_PRIVATE=${HF_PUSH_PRIVATE:-False}
HF_PUSH_DELETE_LOCAL_AFTER=${HF_PUSH_DELETE_LOCAL_AFTER:-False}
HF_PUSH_MAX_TO_KEEP=${HF_PUSH_MAX_TO_KEEP:-10}
HF_PUSH_FREQ=${HF_PUSH_FREQ:-${SAVE_FREQ:-20}}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/dapo_openmath2_mix_train.parquet"}
if [ -n "${VAL_FILE:-}" ]; then
    VAL_FILES="['${VAL_FILE}']"
else
    VAL_FILES=${VAL_FILES:-"['${RAY_DATA_HOME}/data/dapo_openmath2_mix_val_compat.parquet','${RAY_DATA_HOME}/data/math__aime2024_repeated_32x_960_compat.parquet','${RAY_DATA_HOME}/data/math__aime2025_repeated_32x_960_compat.parquet','${RAY_DATA_HOME}/data/math__aime2026_repeated_32x_960_compat.parquet','${RAY_DATA_HOME}/data/math__math_500_repeated_2x_1000_compat.parquet','${RAY_DATA_HOME}/data/math__olympiadbench_repeated_2x_compat.parquet','${RAY_DATA_HOME}/data/math__minervamath_repeated_4x_compat.parquet','${RAY_DATA_HOME}/data/math__gsm8k_test_compat.parquet']"}
fi

HF_SNAPSHOT_DIR=""
if [ -z "${MODEL_PATH}" ]; then
    if [ -n "${HF_MOE_LOCAL_DIR}" ]; then
        HF_SNAPSHOT_DIR="${HF_MOE_LOCAL_DIR}"
    elif [ -n "${HF_MOE_REPO}" ]; then
        HF_SNAPSHOT_DIR="$(python3 - <<PY
from huggingface_hub import snapshot_download

print(snapshot_download(
    repo_id="${HF_MOE_REPO}",
    revision="${HF_MOE_REVISION}",
    repo_type="model",
    cache_dir="${HF_MOE_CACHE_DIR}",
    allow_patterns=[
        "config.json",
        "configuration_gemma3_moe.py",
        "modeling_gemma3_moe.py",
        "generation_config.json",
        "tokenizer*",
        "special_tokens_map.json",
        "added_tokens.json",
        "chat_template.jinja",
        "model.safetensors.index.json",
        "model-*.safetensors",
        "global_step_250/huggingface/*",
        "global_step_250/dist_ckpt/*",
        "global_step_250/transformer_config.json",
    ],
))
PY
)"
    else
        echo "Set MODEL_PATH, HF_MOE_LOCAL_DIR, or HF_MOE_REPO to a converted Gemma3-MoE checkpoint." >&2
        echo "See rl-distill-scripts/GEMMA3_MOE_RL_TRAINING.md for the dense upcycling command." >&2
        exit 2
    fi
    MODEL_PATH="${HF_SNAPSHOT_DIR}"
fi
if [ -z "${DIST_CKPT_PATH}" ] && [ -n "${HF_SNAPSHOT_DIR}" ] && [ -d "${HF_SNAPSHOT_DIR}/global_step_250/dist_ckpt" ]; then
    DIST_CKPT_PATH="${HF_SNAPSHOT_DIR}/global_step_250/dist_ckpt"
fi

if [ ! -f "${MODEL_PATH}/config.json" ]; then
    echo "MODEL_PATH must contain config.json; got ${MODEL_PATH}" >&2
    exit 2
fi

GEMMA3_PROCESSOR_BASE_REPO="${GEMMA3_PROCESSOR_BASE_REPO:-google/gemma-3-4b-pt}" \
    bash "${PROJECT_ROOT}/rl-distill-scripts/_ops/ensure_gemma3_processor_files.sh" "${MODEL_PATH}"

install -m 0644 "${PROJECT_ROOT}/rl-distill-scripts/gemma3_moe_hf/configuration_gemma3_moe.py" \
    "${MODEL_PATH}/configuration_gemma3_moe.py"
install -m 0644 "${PROJECT_ROOT}/rl-distill-scripts/gemma3_moe_hf/modeling_gemma3_moe.py" \
    "${MODEL_PATH}/modeling_gemma3_moe.py"

# Fail before allocating GPUs if a checkpoint violates the upcycling
# invariants or if a canonical test view is about to be trained.
python3 - "${MODEL_PATH}" "${NUM_EXPERTS}" <<'PY'
import json
import sys
from pathlib import Path

model_path = Path(sys.argv[1])
expected_experts = int(sys.argv[2])
with (model_path / "config.json").open() as handle:
    config = json.load(handle)

architectures = config.get("architectures", [])
if "Gemma3MoeForCausalLM" not in architectures:
    raise SystemExit(f"Expected Gemma3MoeForCausalLM in architectures; got {architectures}")
actual_experts = config.get("gemma3_moe_num_experts", config.get("num_experts"))
if int(actual_experts) != expected_experts:
    raise SystemExit(f"Checkpoint has {actual_experts} experts, launcher requested {expected_experts}")
if int(config.get("num_experts_per_tok", 1)) != 1:
    raise SystemExit("Only top-1 Gemma3 MoE checkpoints are supported")
if config.get("router_pre_softmax", False) is not False:
    raise SystemExit("router_pre_softmax must be false for dense-equivalent top-1 upcycling")
canonical_checkpoint = config.get("gemma3_moe_canonical_dense_init", False)
if canonical_checkpoint is not False:
    raise SystemExit(
        "This is a canonical correctness-only checkpoint. "
        "Use the normal sparse output for Gate D and training; the canonical view is for Gates A-C."
    )
print(f"Gemma3-MoE preflight: experts={actual_experts} topk=1 canonical_dense_init=False")
PY

if [ "${USE_DIST_CHECKPOINTING}" = "True" ] && [ -z "${DIST_CKPT_PATH}" ]; then
    echo "Set DIST_CKPT_PATH when USE_DIST_CHECKPOINTING=True." >&2
    exit 2
fi
if [ "${HF_PUSH_ENABLE}" = "True" ] && [ -z "${HF_PUSH_REPO}" ]; then
    echo "Set HF_PUSH_REPO when HF_PUSH_ENABLE=True." >&2
    exit 2
fi
if [ "${HF_PUSH_ENABLE}" = "True" ]; then
    python3 - <<'PY'
from huggingface_hub import get_token

if not get_token():
    raise SystemExit(
        "HF_PUSH_ENABLE=True but no Hugging Face token is available. "
        "Set HF_TOKEN in .env or run `hf auth login` before launching."
    )
PY
fi
save_freq=${SAVE_FREQ:-20}
for frequency_name in test_freq save_freq HF_PUSH_FREQ; do
    frequency_value=${!frequency_name}
    if ! [[ "${frequency_value}" =~ ^-?[0-9]+$ ]]; then
        echo "${frequency_name} must be an integer; got ${frequency_value}" >&2
        exit 2
    fi
done
if [ "${HF_PUSH_ENABLE}" = "True" ]; then
    if [ "${save_freq}" -le 0 ]; then
        echo "HF uploads occur after local saves, so SAVE_FREQ must be positive when HF push is enabled." >&2
        exit 2
    fi
    if [ "${HF_PUSH_FREQ}" -gt 0 ] && [ $((HF_PUSH_FREQ % save_freq)) -ne 0 ]; then
        echo "Warning: HF_PUSH_FREQ=${HF_PUSH_FREQ} is not a multiple of SAVE_FREQ=${save_freq}." >&2
        echo "Uploads only run on local-save steps, so the effective cadence will differ." >&2
    fi
fi

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1
val_top_p=0.7
actor_lr_warmup_steps=${ACTOR_LR_WARMUP_STEPS:-20}
if ! [[ "${actor_lr_warmup_steps}" =~ ^[0-9]+$ ]]; then
    echo "ACTOR_LR_WARMUP_STEPS must be a non-negative integer; got ${actor_lr_warmup_steps}" >&2
    exit 2
fi

# Performance related parameters
sp_size=${SP_SIZE:-1}
use_dynamic_bsz=${USE_DYNAMIC_BSZ:-True}
# Megatron's sequence-length balancer requires the token cap to be at least the
# longest individual sequence. Do not divide by SP here; a single rollout can be
# max_prompt_length + max_response_length tokens.
actor_ppo_max_token_len=${ACTOR_PPO_MAX_TOKEN_LEN:-$((max_prompt_length + max_response_length))}
infer_ppo_max_token_len=${INFER_PPO_MAX_TOKEN_LEN:-$((max_prompt_length + max_response_length))}
offload=${OFFLOAD:-True}
gen_tp=${GEN_TP:-1}
enable_chunked_prefill=${ENABLE_CHUNKED_PREFILL:-True}
rollout_gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.80}
rollout_block_size=${ROLLOUT_BLOCK_SIZE:-32}
rollout_free_cache_engine=${ROLLOUT_FREE_CACHE_ENGINE:-True}
rollout_max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-$((max_prompt_length + max_response_length))}
# Use the registered native Gemma3-MoE implementation by default.  The generic
# Transformers backend was the source of the 19k non-terminating rollout; it
# is retained only as an explicit debugging fallback.  Triton is pinned by
# default so dense and MoE native Gemma3 use the same attention path.
rollout_model_impl=${ROLLOUT_MODEL_IMPL:-native}
case "${rollout_model_impl}" in
    auto|native|transformers) ;;
    *)
        echo "ROLLOUT_MODEL_IMPL must be auto, native, or transformers; got ${rollout_model_impl}" >&2
        exit 2
        ;;
esac
if [ "${rollout_model_impl}" = "native" ]; then
    python3 - <<'PY'
from vllm import ModelRegistry
from vllm.plugins import load_general_plugins

load_general_plugins()
architecture = "Gemma3MoeForCausalLM"
if architecture not in ModelRegistry.get_supported_archs():
    raise SystemExit(
        "The native vLLM Gemma3-MoE plugin is not installed. "
        "Reinstall this checkout with: uv pip install --python .venv-megatron/bin/python -e ."
    )
print("Native vLLM Gemma3-MoE plugin registered")
PY
fi
rollout_attention_backend=${ROLLOUT_ATTENTION_BACKEND:-TRITON_ATTN}
rollout_attention_backend_args=()
if [ -n "${rollout_attention_backend}" ]; then
    rollout_attention_backend_args+=(+actor_rollout_ref.rollout.engine_kwargs.vllm.attention_backend="${rollout_attention_backend}")
fi
rollout_enforce_eager=${ROLLOUT_ENFORCE_EAGER:-True}

# MoE RL specifics.
MOE_AUX_LOSS_COEFF=${MOE_AUX_LOSS_COEFF:-1e-3}
# R2 records routes during the actor's old-log-prob pass and replays them in
# the update. R3 capture exists in the native model, but the colocated verl
# handoff has not passed the full correctness gate; keep it behind an explicit
# experimental opt-in.
ROUTER_REPLAY_MODE=${ROUTER_REPLAY_MODE:-R2}
if [ "${ROUTER_REPLAY_MODE}" != "disabled" ] && [ "${ROUTER_REPLAY_MODE}" != "R2" ] && [ "${ROUTER_REPLAY_MODE}" != "R3" ]; then
    echo "ROUTER_REPLAY_MODE must be disabled, R2, or R3; got ${ROUTER_REPLAY_MODE}" >&2
    exit 2
fi
if [ "${ROUTER_REPLAY_MODE}" = "R3" ] && [ "${ALLOW_EXPERIMENTAL_R3:-False}" != "True" ]; then
    echo "R3 is experimental for Gemma3-MoE. Use R2, or set ALLOW_EXPERIMENTAL_R3=True explicitly." >&2
    exit 2
fi
[ "${ROUTER_REPLAY_MODE}" = "R3" ] && ENABLE_ROLLOUT_ROUTING_REPLAY=True || ENABLE_ROLLOUT_ROUTING_REPLAY=False

if [ "${NUM_EXPERTS}" = "4" ]; then
    DEFAULT_ACTOR_TP=2
else
    DEFAULT_ACTOR_TP=1
fi

ACTOR_TP=${ACTOR_TP:-${DEFAULT_ACTOR_TP}}
ACTOR_PP=${ACTOR_PP:-1}
ACTOR_CP=${ACTOR_CP:-1}
ACTOR_EP=${ACTOR_EP:-${NUM_EXPERTS}}
REF_TP=${REF_TP:-${ACTOR_TP}}
REF_PP=${REF_PP:-${ACTOR_PP}}
REF_CP=${REF_CP:-${ACTOR_CP}}
REF_EP=${REF_EP:-${ACTOR_EP}}
CP_COMM_TYPE=${CP_COMM_TYPE:-a2a}
[ "${ACTOR_PP}" -gt 1 ] && ACTOR_VPP_OVERRIDE=${ACTOR_VPP:-2} || ACTOR_VPP_OVERRIDE=null
[ "${REF_PP}" -gt 1 ] && REF_VPP_OVERRIDE=${REF_VPP:-2} || REF_VPP_OVERRIDE=null

export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
NCCL_SOCKET_FAMILY="${NCCL_SOCKET_FAMILY:-AF_INET6}"
GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME#=}"
export NCCL_SOCKET_FAMILY="${NCCL_SOCKET_FAMILY#=}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME#=}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"

# vLLM V1 imports numba on these workers; use an optional local NumPy 2.2.x wheel
# target if the base environment has a newer NumPy than numba accepts.
VLLM_NUMBA_NUMPY_PATH=${VLLM_NUMBA_NUMPY_PATH:-/tmp/np226_py312}
if [ -d "${VLLM_NUMBA_NUMPY_PATH}" ]; then
    export PYTHONPATH="${VLLM_NUMBA_NUMPY_PATH}:${PYTHONPATH:-}"
fi

SITE_PACKAGES="$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
NVML_SHIM_DIR=${NVML_SHIM_DIR:-${VERL_EXTRA_LD_LIBRARY_PATH:-/tmp/nvidia-nvml-535}}
mkdir -p "${NVML_SHIM_DIR}"
if [ -n "${NVML_LIBRARY_PATH:-}" ] && [ -s "${NVML_LIBRARY_PATH}" ]; then
    ln -sf "${NVML_LIBRARY_PATH}" "${NVML_SHIM_DIR}/libnvidia-ml.so.1"
else
    for nvml_candidate in \
        /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1 \
        /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.105.08 \
        /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.126.20 \
        /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.570.133.20 \
        /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.535.129.03; do
        if [ -s "${nvml_candidate}" ]; then
            ln -sf "${nvml_candidate}" "${NVML_SHIM_DIR}/libnvidia-ml.so.1"
            break
        fi
    done
fi
export CUDNN_HOME="${CUDNN_HOME:-${SITE_PACKAGES}/nvidia/cudnn}"
export NVRTC_HOME="${NVRTC_HOME:-${SITE_PACKAGES}/nvidia/cuda_nvrtc}"
export CURAND_HOME="${CURAND_HOME:-${SITE_PACKAGES}/nvidia/curand}"
export NCCL_HOME="${NCCL_HOME:-${SITE_PACKAGES}/nvidia/nccl}"
export CUBLAS_HOME="${CUBLAS_HOME:-${SITE_PACKAGES}/nvidia/cublas}"
export CUDART_HOME="${CUDART_HOME:-${SITE_PACKAGES}/nvidia/cuda_runtime}"
export NVSHMEM_HOME="${NVSHMEM_HOME:-${SITE_PACKAGES}/nvidia/nvshmem}"
if [ -z "${CUDA_HOME:-}" ]; then
    for cuda_candidate in /tmp/cuda-no-recursion-real /tmp/cuda-12.9 /usr/local/cuda-12.9 /usr/local/cuda; do
        if [ -d "${cuda_candidate}" ]; then
            CUDA_HOME="${cuda_candidate}"
            break
        fi
    done
fi
export CUDA_HOME
export CUDA_PATH="${CUDA_PATH:-${CUDA_HOME}}"
export CUDA_INC_PATH="${CUDA_INC_PATH:-${CUDA_HOME}/include}"
export CUDACXX="${CUDACXX:-${CUDA_HOME}/bin/nvcc}"
export LD_LIBRARY_PATH="${NVML_SHIM_DIR}:${NCCL_HOME}/lib:${CUDNN_HOME}/lib:${NVRTC_HOME}/lib:${CURAND_HOME}/lib:${CUBLAS_HOME}/lib:${CUDART_HOME}/lib:${NVSHMEM_HOME}/lib:${LD_LIBRARY_PATH:-}"
export VERL_SKIP_VLLM_MM_WEIGHT_RELOAD="${VERL_SKIP_VLLM_MM_WEIGHT_RELOAD:-1}"

python3 -m dapo.main_dapo \
    --config-name=dapo_megatron_trainer \
    +ray_kwargs.ray_init.address="\"${RAY_ADDRESS}\"" \
    +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}" \
    +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_SOCKET_FAMILY="${NCCL_SOCKET_FAMILY}" \
    +ray_kwargs.ray_init.runtime_env.env_vars.GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME}" \
    +ray_kwargs.ray_init.runtime_env.env_vars.TORCH_CUDA_ARCH_LIST="\"${TORCH_CUDA_ARCH_LIST}\"" \
    +ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH="${PYTHONPATH:-}" \
    +ray_kwargs.ray_init.runtime_env.env_vars.HF_MODULES_CACHE="${HF_MODULES_CACHE}" \
    +ray_kwargs.ray_init.runtime_env.env_vars.CUDNN_HOME="${CUDNN_HOME}" \
    +ray_kwargs.ray_init.runtime_env.env_vars.NVRTC_HOME="${NVRTC_HOME}" \
    +ray_kwargs.ray_init.runtime_env.env_vars.CURAND_HOME="${CURAND_HOME}" \
    +ray_kwargs.ray_init.runtime_env.env_vars.CUBLAS_HOME="${CUBLAS_HOME}" \
    +ray_kwargs.ray_init.runtime_env.env_vars.CUDART_HOME="${CUDART_HOME}" \
    +ray_kwargs.ray_init.runtime_env.env_vars.NVSHMEM_HOME="${NVSHMEM_HOME}" \
    +ray_kwargs.ray_init.runtime_env.env_vars.CUDA_HOME="${CUDA_HOME}" \
    +ray_kwargs.ray_init.runtime_env.env_vars.CUDA_PATH="${CUDA_PATH}" \
    +ray_kwargs.ray_init.runtime_env.env_vars.CUDA_INC_PATH="${CUDA_INC_PATH}" \
    +ray_kwargs.ray_init.runtime_env.env_vars.CUDACXX="${CUDACXX}" \
    +ray_kwargs.ray_init.runtime_env.env_vars.LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
    +ray_kwargs.ray_init.runtime_env.env_vars.PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-}" \
    +ray_kwargs.ray_init.runtime_env.env_vars.PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-}" \
    +ray_kwargs.ray_init.runtime_env.env_vars.XDG_CONFIG_HOME="${XDG_CONFIG_HOME}" \
    +ray_kwargs.ray_init.runtime_env.env_vars.VLLM_CONFIG_ROOT="${VLLM_CONFIG_ROOT}" \
    +ray_kwargs.ray_init.runtime_env.env_vars.VLLM_DO_NOT_TRACK="\"${VLLM_DO_NOT_TRACK}\"" \
    +ray_kwargs.ray_init.runtime_env.env_vars.DO_NOT_TRACK="\"${DO_NOT_TRACK}\"" \
    +ray_kwargs.ray_init.runtime_env.env_vars.VLLM_NO_USAGE_STATS="\"${VLLM_NO_USAGE_STATS}\"" \
    +ray_kwargs.ray_init.runtime_env.env_vars.RAY_USAGE_STATS_ENABLED="\"${RAY_USAGE_STATS_ENABLED}\"" \
    +ray_kwargs.ray_init.runtime_env.env_vars.VLLM_TARGET_DEVICE="\"${VLLM_TARGET_DEVICE:-cuda}\"" \
    +ray_kwargs.ray_init.runtime_env.env_vars.VLLM_USE_V1="\"${VLLM_USE_V1:-1}\"" \
    +ray_kwargs.ray_init.runtime_env.env_vars.VLLM_FORCE_PLATFORM="\"${VLLM_FORCE_PLATFORM:-cuda}\"" \
    +ray_kwargs.ray_init.runtime_env.env_vars.VERL_MATH_VERIFY_FAST_INVALID="\"${VERL_MATH_VERIFY_FAST_INVALID}\"" \
    +ray_kwargs.ray_init.runtime_env.env_vars.VERL_MATH_VERIFY_FAST_EQUIV="\"${VERL_MATH_VERIFY_FAST_EQUIV}\"" \
    +ray_kwargs.ray_init.runtime_env.env_vars.VERL_MATH_VERIFY_TIMEOUT="\"${VERL_MATH_VERIFY_TIMEOUT}\"" \
    +ray_kwargs.ray_init.runtime_env.env_vars.VERL_MATH_VERIFY_MAX_CHARS="\"${VERL_MATH_VERIFY_MAX_CHARS}\"" \
    +ray_kwargs.ray_init.runtime_env.env_vars.VERL_MATH_VERIFY_POOL_WORKERS="\"${VERL_MATH_VERIFY_POOL_WORKERS}\"" \
    +ray_kwargs.ray_init.runtime_env.env_vars.VERL_AGENT_LOOP_PREFER_GPU_NODES="\"${VERL_AGENT_LOOP_PREFER_GPU_NODES}\"" \
    +ray_kwargs.ray_init.runtime_env.env_vars.NVML_SHIM_DIR="${NVML_SHIM_DIR}" \
    +ray_kwargs.ray_init.runtime_env.env_vars.CUDA_DEVICE_MAX_CONNECTIONS="\"${CUDA_DEVICE_MAX_CONNECTIONS}\"" \
    +ray_kwargs.ray_init.runtime_env.env_vars.VERL_SKIP_VLLM_MM_WEIGHT_RELOAD="\"${VERL_SKIP_VLLM_MM_WEIGHT_RELOAD}\"" \
    +ray_kwargs.ray_init.runtime_env.env_vars.VERL_VLLM_PORT_BASE="\"${VERL_VLLM_PORT_BASE:-52000}\"" \
    +ray_kwargs.ray_init.runtime_env.env_vars.VERL_VLLM_PORT_STRIDE="\"${VERL_VLLM_PORT_STRIDE:-100}\"" \
    +ray_kwargs.ray_init.runtime_env.env_vars.VERL_VLLM_NODE_PORT_STRIDE="\"${VERL_VLLM_NODE_PORT_STRIDE:-10}\"" \
    +ray_kwargs.ray_init.runtime_env.env_vars.VERL_VLLM_RANDOM_PORTS="\"${VERL_VLLM_RANDOM_PORTS:-1}\"" \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILES}" \
    data.prompt_key=prompt \
    data.shuffle=True \
    data.seed=42 \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.train_batch_size=${train_prompt_bsz} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.agent.num_workers=${agent_loop_num_workers} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.custom_chat_template="@${GEMMA3_CHAT_TEMPLATE_FILE}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=${actor_lr_warmup_steps} \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    +actor_rollout_ref.actor.optim.override_optimizer_config.adam_beta1=0.9 \
    +actor_rollout_ref.actor.optim.override_optimizer_config.adam_beta2=0.999 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.router_replay.mode="${ROUTER_REPLAY_MODE}" \
    actor_rollout_ref.actor.megatron.router_replay.mode="${ROUTER_REPLAY_MODE}" \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    actor_rollout_ref.actor.megatron.vanilla_mbridge=False \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=${USE_DIST_CHECKPOINTING} \
    actor_rollout_ref.actor.megatron.dist_checkpointing_path="${DIST_CKPT_PATH}" \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${ACTOR_TP} \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${ACTOR_PP} \
    actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size=${ACTOR_VPP_OVERRIDE} \
    actor_rollout_ref.actor.megatron.context_parallel_size=${ACTOR_CP} \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${ACTOR_EP} \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=1 \
    actor_rollout_ref.actor.megatron.param_offload=${offload} \
    actor_rollout_ref.actor.megatron.grad_offload=${offload} \
    actor_rollout_ref.actor.megatron.optimizer_offload=${offload} \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=False \
    +actor_rollout_ref.actor.megatron.override_transformer_config.masked_softmax_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_activation_fusion=False \
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_dropout_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=False \
    +actor_rollout_ref.actor.megatron.override_transformer_config.deallocate_pipeline_outputs=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.persist_layer_norm=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.cp_comm_type="${CP_COMM_TYPE}" \
    +actor_rollout_ref.actor.megatron.override_transformer_config.gemma3_moe_num_experts=${NUM_EXPERTS} \
    +actor_rollout_ref.actor.megatron.override_transformer_config.gemma3_moe_aux_loss_coeff=${MOE_AUX_LOSS_COEFF} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_memory_utilization} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.enforce_eager=${rollout_enforce_eager} \
    actor_rollout_ref.rollout.free_cache_engine=${rollout_free_cache_engine} \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.model_impl=${rollout_model_impl} \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.block_size=${rollout_block_size} \
    "${rollout_attention_backend_args[@]}" \
    actor_rollout_ref.rollout.enable_chunked_prefill=${enable_chunked_prefill} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${rollout_max_num_batched_tokens} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.enable_rollout_routing_replay=${ENABLE_ROLLOUT_ROUTING_REPLAY} \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.ref.megatron.use_mbridge=True \
    actor_rollout_ref.ref.megatron.vanilla_mbridge=False \
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=${USE_DIST_CHECKPOINTING} \
    actor_rollout_ref.ref.megatron.dist_checkpointing_path="${DIST_CKPT_PATH}" \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${REF_TP} \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${REF_PP} \
    actor_rollout_ref.ref.megatron.virtual_pipeline_model_parallel_size=${REF_VPP_OVERRIDE} \
    actor_rollout_ref.ref.megatron.context_parallel_size=${REF_CP} \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${REF_EP} \
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=1 \
    actor_rollout_ref.ref.megatron.param_offload=${offload} \
    ++actor_rollout_ref.ref.megatron.override_transformer_config.bias_activation_fusion=False \
    ++actor_rollout_ref.ref.megatron.override_transformer_config.gradient_accumulation_fusion=False \
    ++actor_rollout_ref.ref.megatron.override_transformer_config.cp_comm_type="${CP_COMM_TYPE}" \
    ++actor_rollout_ref.ref.megatron.override_transformer_config.gemma3_moe_num_experts=${NUM_EXPERTS} \
    ++actor_rollout_ref.ref.megatron.override_transformer_config.gemma3_moe_aux_loss_coeff=${MOE_AUX_LOSS_COEFF} \
    reward_model.reward_manager=dapo \
    reward.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    reward.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    reward.reward_kwargs.overlong_buffer_cfg.log=True \
    reward.reward_kwargs.max_resp_len=${max_response_length} \
    reward.num_workers=${reward_num_workers} \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=${GPUS_PER_NODE} \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=${val_before_train} \
    trainer.test_freq=${test_freq} \
    trainer.save_freq=${save_freq} \
    trainer.max_actor_ckpt_to_keep=${MAX_ACTOR_CKPT_TO_KEEP:-2} \
    +trainer.hf_push.enable=${HF_PUSH_ENABLE} \
    +trainer.hf_push.repo_id="${HF_PUSH_REPO}" \
    +trainer.hf_push.private=${HF_PUSH_PRIVATE} \
    +trainer.hf_push.delete_local_after=${HF_PUSH_DELETE_LOCAL_AFTER} \
    +trainer.hf_push.max_to_keep=${HF_PUSH_MAX_TO_KEEP} \
    +trainer.hf_push.freq=${HF_PUSH_FREQ} \
    'actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra,hf_model]' \
    'actor_rollout_ref.actor.checkpoint.load_contents=[model,optimizer,extra]' \
    trainer.total_epochs=100 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode="${RESUME_MODE:-auto}" "$@"
