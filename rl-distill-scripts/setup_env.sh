#!/usr/bin/env bash
# One-shot environment setup for B200 DAPO training.
#
# Produces a `.venv/` with the known-working stack:
#   torch 2.9.1 + cu128   (B200 sm_100 support)
#   vllm  0.15.1          (torch 2.9 compatible)
#   flash-attn 2.8.3      (rebuilt against torch 2.9)
#   transformers 4.57.6   (vllm 0.15.1 requires this pin; 5.x breaks rope_scaling)
#   verl (this repo, editable, no-deps)
#
# Run once per fresh node before launching training.

set -xeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

# 1. uv + venv
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
fi
uv venv --python 3.12 --seed .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python3 -m pip install -U pip wheel setuptools

# 2. Core training stack. Install torch FIRST from the cu128 index so we get
#    the sm_100 build B200 needs. Everything else resolves against it.
uv pip install --index-url https://download.pytorch.org/whl/cu128 \
    "torch==2.9.1" "torchvision" "torchaudio"

uv pip install \
    "vllm==0.15.1" \
    "transformers==4.57.6" \
    "ray[default]" \
    "hydra-core" \
    "wandb" \
    "huggingface_hub" \
    "datasets" \
    "pandas" "pyarrow" \
    "math-verify" \
    "pre-commit"

# 3. flash-attn must be built against the installed torch (no prebuilt wheel
#    works for torch 2.9 + cu128 + sm_100 at time of writing).
uv pip install --no-build-isolation "flash-attn==2.8.3"

# 4. verl, editable, no deps (we already pinned the exact stack above).
uv pip install --no-deps -e .

# 5. pre-commit hooks (optional, for contributors)
pre-commit install || true

echo ""
echo "=== setup_env.sh done ==="
echo "Activate with: source ${PROJECT_ROOT}/.venv/bin/activate"
python3 -c "import torch, vllm, flash_attn, transformers; \
print('torch', torch.__version__, 'cuda', torch.version.cuda); \
print('vllm', vllm.__version__); \
print('flash_attn', flash_attn.__version__); \
print('transformers', transformers.__version__)"
