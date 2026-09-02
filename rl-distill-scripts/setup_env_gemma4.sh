#!/usr/bin/env bash
set -euo pipefail
# EXPERIMENTAL isolated env for Gemma 4 (gemma-4-E2B/E4B/12B...) on local H100 (sm_90).
# Builds .venv-gemma4 — does NOT touch the working FSDP2 .venv or .venv-megatron.
# Stack (gemma4-capable, resolved via uv; matches vLLM's official Gemma 4 recipe):
#   torch 2.11.0 cu129 + vllm 0.25.1 + transformers 5.14.1 + ray 2.56 + tensordict 0.13
# verl's own pins (vllm<=0.12) are advisory — we intentionally exceed them (installed --no-deps).
#
#   nohup bash rl-distill-scripts/setup_env_gemma4.sh > ~/verl/logs/setup_gemma4.log 2>&1 &
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${PROJECT_ROOT}"
set -a; source .env 2>/dev/null || true; set +a
VENV="${VENV:-${PROJECT_ROOT}/.venv-gemma4}"
export UV_HTTP_TIMEOUT=600            # EFS/network is slow under box load
# CUDA variant (gemma-4 does NOT require CUDA 13 — the official vLLM recipe's primary path is cu129):
#   cu130 (default) — for hosts with a CUDA-13 driver (e.g. this devbox, driver 580.x). Uses the default
#     PyPI vllm 0.25.1 wheel, which is CUDA-13-linked (needs libcudart.so.13 -> the nvidia cu13 wheel).
#   cu129 — for hosts with a CUDA-12.x driver (e.g. ScaleTrain p5 fleet, driver reports 12.8; cu129
#     binaries run there via CUDA 12.x minor-version compatibility). Uses the +cu129 VARIANT wheel from
#     the vllm GitHub release — the PyPI default wheel would fail with "libcudart.so.13: cannot open".
GEMMA4_CUDA_VARIANT="${GEMMA4_CUDA_VARIANT:-cu130}"
PT_INDEX="https://download.pytorch.org/whl/${GEMMA4_CUDA_VARIANT}"
VLLM_SPEC="vllm==0.25.1"
if [ "${GEMMA4_CUDA_VARIANT}" = "cu129" ]; then
  VLLM_SPEC="https://github.com/vllm-project/vllm/releases/download/v0.25.1/vllm-0.25.1+cu129-cp38-abi3-manylinux_2_28_$(uname -m).whl"
fi

echo "### [$(date +%H:%M:%S)] create venv ${VENV} (variant ${GEMMA4_CUDA_VARIANT})"
uv venv "${VENV}" --python 3.12
source "${VENV}/bin/activate"

echo "### [$(date +%H:%M:%S)] install cuda torch 2.11 (${GEMMA4_CUDA_VARIANT}) first"
uv pip install "torch==2.11.0" "torchvision==0.26.0" "torchaudio==2.11.0" --index-url "${PT_INDEX}"

echo "### [$(date +%H:%M:%S)] install gemma4-capable vllm/transformers + verl training deps"
uv pip install \
  "${VLLM_SPEC}" "transformers==5.14.1" "tensordict==0.13.0" \
  accelerate peft "datasets>=3" codetiming hydra-core omegaconf pyarrow pylatexenc torchdata \
  boto3 math-verify \
  ninja \
  "ray[default]" wandb \
  --extra-index-url "${PT_INDEX}" --index-strategy unsafe-best-match
# ninja: flashinfer JIT-compiles sampling kernels at engine init (even with TRITON_ATTN attention)
# and dies with FileNotFoundError('ninja') without it. We ALSO disable the flashinfer sampler at
# runtime (VLLM_USE_FLASHINFER_SAMPLER=0 in the run scripts) — JIT-compiling cu129 flashinfer with
# a mismatched host nvcc crashed workers natively on ScaleTrain p5 pods.

# vLLM 0.25.1's R3 capturer already understands Gemma 4's ``top_k_experts``
# while sizing buffers, but one logger argument still dereferences the
# Qwen-style ``num_experts_per_tok`` name and crashes the engine at startup.
# Apply a version-gated, idempotent patch and fail closed if the wheel changes.
python3 rl-distill-scripts/patch_vllm_gemma4_r3.py

echo "### [$(date +%H:%M:%S)] install verl editable (--no-deps: keep our resolved versions)"
uv pip install --no-deps -e .

echo "### [$(date +%H:%M:%S)] import + version check"
python3 - <<'PY'
import torch, vllm, transformers, ray, tensordict
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| avail", torch.cuda.is_available(), "| ndev", torch.cuda.device_count())
print("vllm", vllm.__version__, "| transformers", transformers.__version__, "| ray", ray.__version__)
PY

echo "### [$(date +%H:%M:%S)] does transformers 5.14 recognize all matrix configs?"
python3 - <<'PY'
from transformers import AutoConfig
for model, revision in (
    ("google/gemma-4-E2B", "d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f"),
    ("google/gemma-4-E4B", "411aa17b749aa952df1359d2dcea73917a544d9a"),
    ("google/gemma-4-12B", "023679ed352de9bb66cc873c9009ce3482585c08"),
):
    c = AutoConfig.from_pretrained(model, revision=revision)
    tc = getattr(c, "text_config", None)
    print(model, "model_type:", c.model_type, "| arch:", getattr(c, "architectures", None))
    print("  text_config model_type:", getattr(tc, "model_type", None) if tc else None)

from transformers.models.gemma4_unified.modeling_gemma4_unified import Gemma4UnifiedTextDecoderLayer
print("unified wrap class:", Gemma4UnifiedTextDecoderLayer.__name__)
PY

echo "### [$(date +%H:%M:%S)] does vllm register a gemma4 model class?"
python3 - <<'PY'
import importlib, glob, os, vllm
d = os.path.dirname(vllm.__file__)
print("vllm gemma4 model files:", sorted(os.path.basename(x) for x in glob.glob(d+"/model_executor/models/gemma4*")))
try:
    from vllm.model_executor.models.registry import ModelRegistry
    archs = ModelRegistry.get_supported_archs()
    print("gemma4 archs registered:", [a for a in archs if "gemma4" in a.lower() or "Gemma4" in a])
    assert "Gemma4ForConditionalGeneration" in archs
    assert "Gemma4UnifiedForConditionalGeneration" in archs
except Exception as e:
    raise RuntimeError(f"gemma4 registry check failed: {e}") from e
PY

echo "### [$(date +%H:%M:%S)] GEMMA4_ENV_CORE_OK"
