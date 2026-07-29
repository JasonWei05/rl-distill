#!/usr/bin/env bash
# NeMo-RL exact replication of DAPO-gemma4-e2b-pt-DeepScaleR-4of4strict-seed42 (8k variant),
# for a cross-framework comparison against the verl run (wandb bw9dcxso / recovery recbw9dcxso).
# Framework: third_party/nemo-rl @ 5f89b3ae, UNMODIFIED — all adaptation lives in
# rl-distill-scripts/nemo_rl_repro/ (dataset, strict reward env, config, wrapper).
# See rl-distill-scripts/nemo_rl_repro/PARITY_CHECKLIST.md for matched knobs + caveats.
#
# Flow: cuda-compat fallback (cu130 stack on a CUDA-12.8-driver fleet) -> uv env resolve ->
# torch/cuda fail-fast -> GO/NO-GO gate (1-step run; step-0 validation/accuracy must be in
# [0.045, 0.075], the verl baseline band) -> full training run.
set -uo pipefail
cd /workspace/rl-distill || exit 1

if [ -f .env ]; then set -a; source .env; set +a; fi

REPRO_DIR=/workspace/rl-distill/rl-distill-scripts/nemo_rl_repro
NEMO=/workspace/rl-distill/third_party/nemo-rl
CONFIG="${REPRO_DIR}/config/dapo_gemma4_e2b_pt_repro.yaml"
export NEMO_RL_ROOT="${NEMO}"

# --- data: same prep as the verl runs (parquets referenced by the repro config) ---
export DATA_DIR=/tmp/verl/data
mkdir -p "${DATA_DIR}"
# The pod shell's PATH drops the image's venv entry (login-shell reset), so bare
# python3 does not exist; use a baked venv python explicitly (verl venv on the plain
# image, nemo-rl driver venv on the nemorl image — both have huggingface_hub).
VERL_PY=/workspace/rl-distill/.venv/bin/python3
[ -x "${VERL_PY}" ] || VERL_PY="${NEMO}/.venv/bin/python3"
[ -x "${VERL_PY}" ] || VERL_PY=$(command -v python3 || command -v python)
DATA_DIR="${DATA_DIR}" PYTHON="${VERL_PY}" bash rl-distill-scripts/data/prepare_deepscaler_4of4strict_rl_data.sh
for f in deepscaler_4of4strict_rl_train.parquet deepscaler_4of4strict_rl_val200_x16.parquet; do
  [ -s "${DATA_DIR}/${f}" ] || { echo "FATAL: data prep failed: missing ${DATA_DIR}/${f}"; exit 1; }
done
echo "DATA_PREP_OK"

# --- CUDA 13 runtime on the p5 fleet's CUDA 12.8 driver: forward-compat libs ---
DRIVER_MAJOR=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
DRIVER_MAJOR=${DRIVER_MAJOR:-0}   # broken nvidia-smi -> take the compat path; CUDA_OK fail-fast catches a dead node
echo "driver major: ${DRIVER_MAJOR}"
if [ "${DRIVER_MAJOR}" -lt 580 ]; then
  echo "### installing cuda-compat-13 (forward compatibility for cu130 wheels)"
  # already baked into the nemorl image — reuse without touching apt
  COMPAT_DIR=$(ls -d /usr/local/cuda-13.*/compat 2>/dev/null | head -1)
  if [ -n "${COMPAT_DIR}" ]; then
    :
  elif { apt-get update -qq 2>/dev/null || true; } && apt-get install -y -qq cuda-compat-13-0 2>/dev/null; then
    COMPAT_DIR=$(ls -d /usr/local/cuda-13.*/compat 2>/dev/null | head -1)
  else
    # repo not configured: fetch the deb directly and extract (no install needed)
    mkdir -p /tmp/cuda-compat && cd /tmp/cuda-compat
    REPO=https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64
    DEB=$(curl -fsSL "${REPO}/" | grep -oE 'cuda-compat-13-[0-9]+_[0-9.\-]+_amd64\.deb' | sort -V | tail -1)
    curl -fsSLO "${REPO}/${DEB}" && dpkg -x "${DEB}" extracted
    COMPAT_DIR=$(ls -d /tmp/cuda-compat/extracted/usr/local/cuda-13.*/compat 2>/dev/null | head -1)
    cd /workspace/rl-distill || exit 1
  fi
  if [ -z "${COMPAT_DIR:-}" ]; then echo "FATAL: cuda-compat install failed"; exit 1; fi
  export LD_LIBRARY_PATH="${COMPAT_DIR}:${LD_LIBRARY_PATH:-}"
  echo "CUDA_COMPAT_OK ${COMPAT_DIR}"
fi

# --- CUDA 13.0 toolkit: the automodel extra SOURCE-BUILDS TransformerEngine/deep_ep/
# causal-conv1d/mamba-ssm, whose setups hard-require nvcc matching torch cu130 (the pod
# toolkit is 12.9 -> "detected CUDA version (12.8) mismatches ... (13.0)"). ~3GB apt. ---
# NOTE: test for nvcc, NOT the directory — the cuda-compat package above creates
# /usr/local/cuda-13.0/compat, which made a directory check falsely skip this install.
if [ ! -x /usr/local/cuda-13.0/bin/nvcc ]; then
  echo "### installing cuda-toolkit-13-0 (build toolchain for source-built extras)"
  curl -fsSLO https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb || echo "keyring install failed (may already be configured)"
  apt-get update 2>&1 | tail -2
  apt-get install -y cuda-toolkit-13-0 2>&1 | tail -3
fi
[ -x /usr/local/cuda-13.0/bin/nvcc ] || { echo "FATAL: cuda-13.0 toolkit install failed"; exit 1; }
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="/usr/local/cuda-13.0/bin:${PATH}"
export MAX_JOBS="${MAX_JOBS:-64}"   # TransformerEngine source build parallelism
# CUDA 13 relocated CCCL/libcu++ to include/cccl; host-compiler TUs including
# <cuda/std/...> (deep_ep) need it on the include path explicitly.
if [ -d /usr/local/cuda-13.0/include/cccl ]; then
  export CPATH="/usr/local/cuda-13.0/include/cccl:${CPATH:-}"
  export CPLUS_INCLUDE_PATH="/usr/local/cuda-13.0/include/cccl:${CPLUS_INCLUDE_PATH:-}"
fi
# deep_ep (pulled by BOTH the automodel and vllm extras via worker venv builds)
# compiles against InfiniBand verbs headers.
[ -f /usr/include/infiniband/verbs.h ] \
  || apt-get install -y libibverbs-dev librdmacm-dev 2>&1 | tail -1 || true
# TransformerEngine's build needs cudnn.h at compile time; the pip cudnn wheel lands
# mid-sync (race), so install the system dev package deterministically first
# (presence check first: already baked into the nemorl image).
if [ ! -f /usr/include/cudnn.h ] && ! ls /usr/include/x86_64-linux-gnu/cudnn.h >/dev/null 2>&1; then
  apt-get install -y cudnn9-cuda-13 libcudnn9-dev-cuda-13 2>&1 | tail -1 \
    || apt-get install -y cudnn-cuda-13 2>&1 | tail -1 || true
fi
if [ -f /usr/include/cudnn.h ] || ls /usr/include/x86_64-linux-gnu/cudnn.h >/dev/null 2>&1; then
  echo "CUDNN_DEV_OK (system headers present)"
else
  echo "WARN: system cudnn headers not found; relying on pip cudnn + CUDNN_HOME"
fi
echo "CUDA13_TOOLKIT_OK $(nvcc --version | tail -1)"

# --- remaining build deps, copied from their docker/Dockerfile (cmake via Kitware
# tarball; NVTE_* build envs; CUDNN_HOME at the venv's pip cudnn) ---
if ! command -v cmake >/dev/null 2>&1; then
  CMAKE_VERSION=4.0.3; ARCH=$(uname -m)
  curl --retry 3 --retry-delay 2 -fsSL -o /tmp/cmake.tgz \
    "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-${ARCH}.tar.gz" \
    && tar -xzf /tmp/cmake.tgz -C /tmp \
    && cp -r "/tmp/cmake-${CMAKE_VERSION}-linux-${ARCH}/bin/"* /usr/local/bin/ \
    && cp -r "/tmp/cmake-${CMAKE_VERSION}-linux-${ARCH}/share/"* /usr/local/share/
fi
command -v cmake >/dev/null 2>&1 || { echo "FATAL: cmake install failed"; exit 1; }
echo "CMAKE_OK $(cmake --version | head -1)"
export NVTE_BUILD_MAX_JOBS=8 NVTE_BUILD_THREADS_PER_JOB=2 NVTE_FRAMEWORK=pytorch
export NVTE_CUDA_ARCHS=90 TORCH_CUDA_ARCH_LIST="9.0"   # H100 only — skips sm100 compilation
# CUDNN_HOME is set AFTER the uv sync below (globbed from the venv's pip cudnn) —
# a hardcoded python3.13 path here broke silently on lockfile Python bumps.

# --- uv env (their locked cu130 stack; first resolve now includes TE/deep_ep source
# builds: expect ~30-50 min on the first attempt) ---
export PATH="${HOME}/.local/bin:/root/.local/bin:${PATH}"
command -v uv >/dev/null 2>&1 || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="${HOME}/.local/bin:${PATH}"; }
cd "${NEMO}" || exit 1
export UV_HTTP_TIMEOUT=300
# per-component worker venvs (their venvs.py) belong on local disk, not the repo dir.
# The nemorl image ships them prebuilt at /opt/ray_venvs (venvs.py reuses an existing
# venv as-is when <venv>/bin/python exists).
if [ -d /opt/ray_venvs ]; then
  export NEMO_RL_VENV_DIR=/opt/ray_venvs
else
  export NEMO_RL_VENV_DIR=/tmp/nemo-rl-worker-venvs
fi
echo "NEMO_RL_VENV_DIR=${NEMO_RL_VENV_DIR}"
# vllm 0.25's warmup spuriously probes the DeepGEMM FP8 path on this stack (bf16 model)
# and hard-fails; disable explicitly.
export VLLM_USE_DEEP_GEMM=0
# NOTE: do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True here — it propagates
# into the vLLM workers via ray runtime env and their memory pool asserts
# "Expandable segments are not compatible with memory pool" (pytorch#147851) at engine
# init. Activation checkpointing (config) is the actual training-OOM fix.
# HF cache on pod-local disk (concurrent worker reads of a shared-FS cache have
# produced truncated config.json reads locally)
export HF_HOME=/tmp/hf-home
# deep-ep (MoE expert-parallel comms) needs InfiniBand dev headers absent from the pod
# and is never imported by a dense-model run — resolve the lock but skip installing it.
# NOTE: --no-install-package is a `uv sync` flag (uv run rejects it); sync first, then
# run with --no-sync so the skipped package stays skipped.
echo "### syncing uv environment (with retries; first sync source-builds TE etc, ~30-50 min)"
env_ok=0
for attempt in 1 2 3; do
  uv sync --locked --extra automodel --no-install-package deep-ep && { env_ok=1; break; }
  echo "uv sync attempt ${attempt}/3 failed"
done
[ "${env_ok}" -eq 1 ] || { echo "FATAL: uv sync failed after 3 attempts"; exit 1; }
UVRUN=(uv run --no-sync)
"${UVRUN[@]}" python -c "print('UV_ENV_OK')" || { echo "FATAL: uv env unusable"; exit 1; }

# TransformerEngine (runtime + worker-venv builds) finds the pip cudnn via CUDNN_HOME;
# glob the just-synced driver venv rather than hardcoding the Python minor version.
CUDNN_DIR=$(ls -d "${NEMO}"/.venv/lib/python3.*/site-packages/nvidia/cudnn 2>/dev/null | head -1)
if [ -n "${CUDNN_DIR}" ]; then
  export CUDNN_HOME="${CUDNN_DIR}" CUDNN_PATH="${CUDNN_DIR}"
  echo "CUDNN_HOME_OK ${CUDNN_DIR}"
fi

# TransformerEngine's cmake (built inside the WORKER venvs by nemo-rl's venvs.py) needs
# nccl.h; wire the pip nvidia-nccl-cu13 from the just-synced driver venv.
NCCL_DIR=$(ls -d "${NEMO}"/.venv/lib/python3.*/site-packages/nvidia/nccl 2>/dev/null | head -1)
if [ -n "${NCCL_DIR}" ]; then
  export CPATH="${NCCL_DIR}/include:${CPATH:-}"
  export CPLUS_INCLUDE_PATH="${NCCL_DIR}/include:${CPLUS_INCLUDE_PATH:-}"
  export LIBRARY_PATH="${NCCL_DIR}/lib:${LIBRARY_PATH:-}"
  export NCCL_INCLUDE_DIR="${NCCL_DIR}/include" NCCL_LIB_DIR="${NCCL_DIR}/lib"
  echo "NCCL_HEADERS_OK ${NCCL_DIR}"
fi

echo "### torch/cuda fail-fast"
"${UVRUN[@]}" python -c "import torch; torch.cuda.init(); print('CUDA_OK torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0))" || { echo "FATAL: torch/cuda init failed"; exit 1; }
"${UVRUN[@]}" python -c "import math_verify; print('MATH_VERIFY_OK')" || { echo "FATAL: math_verify missing"; exit 1; }

# --- GO/NO-GO gate: 1 training step, assert step-0 val in the verl baseline band ---
GATE_LOG=/tmp/nemorl_gate.log
echo "### go/no-go gate (max_num_steps=1, val_at_start)"
"${UVRUN[@]}" python "${REPRO_DIR}/run_grpo_repro.py" --config "${CONFIG}" \
    cluster.gpus_per_node="${GPUS_PER_NODE:-8}" \
    grpo.max_num_steps=1 \
    checkpointing.enabled=false \
    logger.wandb.name='nemorl-dapo-gemma4-e2b-pt-ds4of4strict-s42-8k-GATE' \
    2>&1 | tee "${GATE_LOG}"
gate_run_rc=$?   # pipefail is on -> this is the python's rc, not tee's
if [ "${gate_run_rc}" -ne 0 ]; then
  echo "GATE_FAIL: gate run crashed rc=${gate_run_rc} (not a parity failure — check the traceback above)"
  exit "${gate_run_rc}"
fi
"${UVRUN[@]}" python - "$GATE_LOG" <<'PY'
import re
import sys

text = open(sys.argv[1], errors="replace").read()
# their console prints "• Accuracy: 0.0508" in the Validation Results block;
# wandb-style validation/accuracy appears only when the logger is enabled
vals = re.findall(r"validation/accuracy[\"':= ]+([0-9.]+)", text) or re.findall(
    r"Accuracy:\s+([0-9.]+)", text
)
if not vals:
    print("GATE_FAIL: no validation/accuracy found in gate log")
    sys.exit(1)
acc = float(vals[0])
if 0.045 <= acc <= 0.075:
    print(f"GATE_PASS validation/accuracy={acc}")
else:
    print(f"GATE_FAIL validation/accuracy={acc} outside [0.045, 0.075] — prompt/reward/sampling parity broken")
    sys.exit(1)
PY
gate_rc=$?
if [ "${gate_rc}" -ne 0 ]; then echo "RUN_DONE rc=${gate_rc}"; exit "${gate_rc}"; fi

# --- full run (MAX_STEPS env caps it, e.g. MAX_STEPS=10 for a short comparison run) ---
status=0
EXTRA_ARGS=()
if [ -n "${MAX_STEPS:-}" ]; then EXTRA_ARGS+=("grpo.max_num_steps=${MAX_STEPS}"); fi
"${UVRUN[@]}" python "${REPRO_DIR}/run_grpo_repro.py" --config "${CONFIG}" cluster.gpus_per_node="${GPUS_PER_NODE:-8}" "${EXTRA_ARGS[@]}" || status=$?

echo "===== final wandb sync (backfill any uploader failures) ====="
# NeMo-RL's WandbLogger writes under {logger.log_dir}/wandb, not cwd — point the
# backfill there or it silently finds nothing.
NEMORL_LOG_DIR=/tmp/verl/logs/nemorl-dapo-gemma4-e2b-pt-DeepScaleR-4of4strict-seed42-8k
"${UVRUN[@]}" wandb sync --sync-all "${NEMORL_LOG_DIR}/wandb" 2>&1 | tail -5 || true

echo "RUN_DONE rc=${status}"
exit "${status}"
