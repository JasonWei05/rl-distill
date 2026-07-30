# Shared env for building/running the NeMo-RL gemma-4 repro on a local GPU box.
# Mirrors scale_train/run_nemorl_gemma4_e2b_repro.sh minus the pod-only parts.
# Assumes: CUDA-13 toolkit at /usr/local/cuda-13.0 (driver >= 580 -> no cuda-compat),
# system cuDNN dev headers, cmake >= 3.22, uv on PATH. See PROGRESS_LOG 2026-07-30
# for the full fresh-box recipe (apt cuda-toolkit-13-0 cudnn9-cuda-13
# libcudnn9-dev-cuda-13 librdmacm-dev + Kitware cmake).
_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export NEMO_RL_ROOT="${NEMO_RL_ROOT:-${_REPO_ROOT}/third_party/nemo-rl}"
export REPRO_DIR="${_REPO_ROOT}/rl-distill-scripts/nemo_rl_repro"

export PATH="$HOME/.local/bin:/usr/local/cuda-13.0/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-13.0
export MAX_JOBS="${MAX_JOBS:-64}"
export NVTE_BUILD_MAX_JOBS=8 NVTE_BUILD_THREADS_PER_JOB=2 NVTE_FRAMEWORK=pytorch
export NVTE_CUDA_ARCHS="${NVTE_CUDA_ARCHS:-90}" TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"  # H100
# CUDA 13 relocated CCCL/libcu++; host-compiler TUs including <cuda/std/...> need it explicitly.
if [ -d /usr/local/cuda-13.0/include/cccl ]; then
  export CPATH="/usr/local/cuda-13.0/include/cccl:${CPATH:-}"
  export CPLUS_INCLUDE_PATH="/usr/local/cuda-13.0/include/cccl:${CPLUS_INCLUDE_PATH:-}"
fi
export UV_HTTP_TIMEOUT=300
# Driver + worker venvs on local NVMe (never on shared FS).
export NEMO_RL_DRIVER_VENV="${NEMO_RL_DRIVER_VENV:-/tmp/nemo-rl-venv}"
export NEMO_RL_VENV_DIR="${NEMO_RL_VENV_DIR:-/tmp/nemo-rl-worker-venvs}"
# vllm 0.25 warmup spuriously probes DeepGEMM FP8 on this stack (bf16 model) and hard-fails.
export VLLM_USE_DEEP_GEMM=0
# HF cache on local disk (shared-FS caches have produced truncated config.json reads).
export HF_HOME="${HF_HOME:-/tmp/hf-home}"
# Do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True globally — it propagates
# into vLLM workers whose memory pool asserts at engine init (pytorch#147851).

# TransformerEngine (runtime + worker-venv builds) finds pip cudnn/nccl via these;
# globbed from the driver venv, so set only once the sync has run.
CUDNN_DIR=$(ls -d "${NEMO_RL_DRIVER_VENV}"/lib/python3.*/site-packages/nvidia/cudnn 2>/dev/null | head -1)
if [ -n "${CUDNN_DIR}" ]; then
  export CUDNN_HOME="${CUDNN_DIR}" CUDNN_PATH="${CUDNN_DIR}"
fi
NCCL_DIR=$(ls -d "${NEMO_RL_DRIVER_VENV}"/lib/python3.*/site-packages/nvidia/nccl 2>/dev/null | head -1)
if [ -n "${NCCL_DIR}" ]; then
  export CPATH="${NCCL_DIR}/include:${CPATH:-}"
  export CPLUS_INCLUDE_PATH="${NCCL_DIR}/include:${CPLUS_INCLUDE_PATH:-}"
  export LIBRARY_PATH="${NCCL_DIR}/lib:${LIBRARY_PATH:-}"
  export NCCL_INCLUDE_DIR="${NCCL_DIR}/include" NCCL_LIB_DIR="${NCCL_DIR}/lib"
fi
