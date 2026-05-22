#!/usr/bin/env bash
#
# Build a Megatron/mbridge-capable Python 3.12 environment for verl DAPO runs.
#
# Defaults are repo-local and can be overridden without editing this file:
#
#   bash setup_megatron.sh
#   VENV_DIR=/shared/path/.venv-megatron bash setup_megatron.sh
#   CUDA_HOME=/tmp/cuda-12.9 MAX_JOBS=64 bash setup_megatron.sh
#   RUN_HEAVY_BUILDS=0 RUN_SMOKE_TEST=0 bash setup_megatron.sh
#
# The full build compiles Apex, flash-attn, and TransformerEngine from source,
# so expect a long runtime and tens of GB of temporary build/cache data.

set -euo pipefail

log() {
    printf '\n[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$SCRIPT_DIR}"

PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv-megatron}"
VERL_PATH="${VERL_PATH:-$REPO_ROOT}"
CACHE_DIR="${CACHE_DIR:-${TMPDIR:-/tmp}/rl-distill-megatron-build}"
CUDA_HOME="${CUDA_HOME:-$(if [ -d /tmp/cuda-12.9 ]; then printf /tmp/cuda-12.9; else printf /usr/local/cuda; fi)}"
MAX_JOBS="${MAX_JOBS:-32}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-10.0}"
RUN_HEAVY_BUILDS="${RUN_HEAVY_BUILDS:-1}"
RUN_SMOKE_TEST="${RUN_SMOKE_TEST:-1}"

TORCH_VERSION="${TORCH_VERSION:-2.10.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.25.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.10.0}"
PYTORCH_CUDA_INDEX="${PYTORCH_CUDA_INDEX:-https://download.pytorch.org/whl/cu129}"
VLLM_VERSION="${VLLM_VERSION:-0.18.0}"
TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-5.3.0}"
MEGATRON_LM_REF="${MEGATRON_LM_REF:-core_v0.16.0}"
MBRIDGE_REF="${MBRIDGE_REF:-641a5a01de71080b2200d10e369090e40c9a351c}"
FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.8.3}"
TRANSFORMER_ENGINE_REF="${TRANSFORMER_ENGINE_REF:-release_v2.12}"
FLASH_LINEAR_ATTENTION_VERSION="${FLASH_LINEAR_ATTENTION_VERSION:-0.4.1}"
PEFT_VERSION="${PEFT_VERSION:-0.18.1}"
TRL_VERSION="${TRL_VERSION:-0.27.0}"
RAY_VERSION="${RAY_VERSION:-}"

export CUDA_HOME
if [ -d "$CUDA_HOME/bin" ]; then
    export PATH="$CUDA_HOME/bin:$PATH"
fi

export UV_CACHE_DIR="${CACHE_DIR}/uv_cache"
export PIP_CACHE_DIR="${CACHE_DIR}/pip_cache"
export TMPDIR="${CACHE_DIR}/build_tmp"
mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$TMPDIR"

# Devboxes often inject host package trees into PYTHONPATH. Leaving it set can
# make the fresh venv import packages from the wrong interpreter.
unset PYTHONPATH

command -v uv >/dev/null 2>&1 || die "uv not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
command -v nvcc >/dev/null 2>&1 || die "CUDA toolkit nvcc not found. Set CUDA_HOME to a CUDA 12.8+ toolkit."
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found; this setup expects an NVIDIA GPU host."
[ -f "$VERL_PATH/pyproject.toml" ] || die "VERL_PATH does not look like a verl checkout: $VERL_PATH"

NVCC_VER="$(nvcc --version | awk -F'release ' '/release/ {print $2}' | awk -F',' '{print $1}')"
GPU_CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')"
log "preflight: uv=$(uv --version | awk '{print $2}') nvcc=${NVCC_VER:-unknown} gpu_compute_cap=${GPU_CAP:-unknown}"
log "venv=$VENV_DIR cuda_home=$CUDA_HOME cache=$CACHE_DIR"

# Keep uv/pip commands away from any local pyproject workspace discovery.
cd "$CACHE_DIR"

log "creating Python $PYTHON_VERSION venv"
uv python install "$PYTHON_VERSION"
uv venv --python "$PYTHON_VERSION" "$VENV_DIR"

# Source-build packages invoke python -m pip through pyproject hooks, so pip
# must exist in the venv even though package installs below use uv.
uv pip install --python "$VENV_DIR/bin/python" pip

log "installing torch $TORCH_VERSION CUDA wheels"
uv pip install --python "$VENV_DIR/bin/python" \
    --index-url "$PYTORCH_CUDA_INDEX" \
    "torch==$TORCH_VERSION" \
    "torchvision==$TORCHVISION_VERSION" \
    "torchaudio==$TORCHAUDIO_VERSION"

log "installing vLLM and transformers"
uv pip install --python "$VENV_DIR/bin/python" "vllm==$VLLM_VERSION"
uv pip install --python "$VENV_DIR/bin/python" \
    "transformers==$TRANSFORMERS_VERSION" \
    pybind11 \
    ninja \
    nvidia-mathdx

log "installing Megatron-LM and mbridge"
uv pip install --python "$VENV_DIR/bin/python" --no-deps \
    "git+https://github.com/NVIDIA/Megatron-LM.git@$MEGATRON_LM_REF" \
    "git+https://github.com/ISEEKYAN/mbridge.git@$MBRIDGE_REF"

log "installing RL and verl runtime dependencies"
ray_spec="ray[default]"
if [ -n "$RAY_VERSION" ]; then
    ray_spec="ray[default]==$RAY_VERSION"
fi

uv pip install --python "$VENV_DIR/bin/python" \
    "flash-linear-attention==$FLASH_LINEAR_ATTENTION_VERSION" \
    "peft==$PEFT_VERSION" \
    "trl==$TRL_VERSION" \
    "$ray_spec" \
    accelerate \
    cachetools \
    codetiming \
    datasets \
    hydra-core \
    liger-kernel \
    mathruler \
    nvtx \
    pylatexenc \
    pytest-asyncio \
    qwen_vl_utils \
    tensordict \
    torchdata \
    wandb

SITE_PACKAGES="$("$VENV_DIR/bin/python" - <<'PY'
import sysconfig
print(sysconfig.get_paths()["purelib"])
PY
)"

NCCL_ROOT="${NCCL_ROOT:-$SITE_PACKAGES/nvidia/nccl}"
NVJITLINK_ROOT="${NVJITLINK_ROOT:-$SITE_PACKAGES/nvidia/nvjitlink}"
export NCCL_ROOT
export CPATH="$NCCL_ROOT/include:${CPATH:-}"
export LIBRARY_PATH="$NCCL_ROOT/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$NCCL_ROOT/lib:$NVJITLINK_ROOT/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST MAX_JOBS

if [ "$RUN_HEAVY_BUILDS" != "0" ]; then
    log "building Apex from source"
    "$VENV_DIR/bin/python" -m pip install -v --no-build-isolation \
        --config-settings "--build-option=--cpp_ext" \
        --config-settings "--build-option=--cuda_ext" \
        git+https://github.com/NVIDIA/apex.git

    log "building flash-attn $FLASH_ATTN_VERSION from source"
    FLASH_ATTENTION_FORCE_BUILD=TRUE \
        "$VENV_DIR/bin/python" -m pip install -v --no-build-isolation --no-cache-dir \
        "flash_attn==$FLASH_ATTN_VERSION"

    log "building TransformerEngine $TRANSFORMER_ENGINE_REF from source"
    NVTE_FRAMEWORK=pytorch NVTE_BUILD_THREADS_PER_JOB=2 \
        "$VENV_DIR/bin/python" -m pip install -v --no-build-isolation --no-cache-dir \
        "git+https://github.com/NVIDIA/TransformerEngine.git@$TRANSFORMER_ENGINE_REF"
else
    log "skipping Apex, flash-attn, and TransformerEngine builds because RUN_HEAVY_BUILDS=0"
fi

log "installing verl editable from $VERL_PATH"
uv pip install --python "$VENV_DIR/bin/python" --no-deps -e "$VERL_PATH"

if [ "$RUN_SMOKE_TEST" != "0" ]; then
    log "running import/GPU smoke test"
    "$VENV_DIR/bin/python" - <<'PY'
import importlib

modules = [
    "torch",
    "vllm",
    "transformers",
    "mbridge",
    "megatron",
    "verl",
]

optional_modules = [
    "apex",
    "flash_attn",
    "transformer_engine.pytorch",
]

loaded = {}
for name in modules:
    loaded[name] = importlib.import_module(name)

for name in optional_modules:
    try:
        loaded[name] = importlib.import_module(name)
    except Exception as exc:
        print(f"{name}: import skipped/failed ({exc})")

torch = loaded["torch"]
print("--- versions ---")
for name, module in loaded.items():
    version = getattr(module, "__version__", "unknown")
    print(f"{name}: {version}")

if torch.cuda.is_available():
    print(f"cuda: available devices={torch.cuda.device_count()} cap={torch.cuda.get_device_capability(0)}")
    if "flash_attn" in loaded:
        from flash_attn import flash_attn_func

        q = torch.randn(2, 8, 128, 64, dtype=torch.bfloat16, device="cuda")
        out = flash_attn_func(q, q, q)
        assert out.shape == q.shape
    if "transformer_engine.pytorch" in loaded:
        te = loaded["transformer_engine.pytorch"]
        lin = te.Linear(256, 256).cuda().bfloat16()
        x = torch.randn(4, 16, 256, device="cuda", dtype=torch.bfloat16)
        assert lin(x).shape == x.shape
    print("gpu smoke: OK")
else:
    print("cuda: unavailable; import smoke only")
PY
else
    log "skipping smoke test because RUN_SMOKE_TEST=0"
fi

cat <<NOTE

=======================================================================
Megatron setup complete.

venv:
  $VENV_DIR

activate:
  source "$VENV_DIR/bin/activate"
  unset PYTHONPATH

common runtime exports:
  export CUDA_HOME="$CUDA_HOME"
  export TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST"
  export LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
  export NCCL_SOCKET_IFNAME=lo
  export GLOO_SOCKET_IFNAME=lo

For the Gemma Megatron launcher, pass:
  MEGATRON_VENV="$VENV_DIR" CUDA_HOME_OVERRIDE="$CUDA_HOME" bash rl-distill-scripts/gemma3_4b_pt_megatron_20k.sh
=======================================================================

NOTE
