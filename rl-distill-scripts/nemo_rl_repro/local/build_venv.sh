#!/usr/bin/env bash
# Build the full verified local NeMo-RL environment: driver venv + both worker
# venvs + the gemma-4 KV-sharing patch. First run ~30-60 min (TransformerEngine
# and friends source-build); reruns fast-skip.
set -uo pipefail
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_HERE}/env.sh"
cd "${NEMO_RL_ROOT}" || exit 1
export UV_PROJECT_ENVIRONMENT="${NEMO_RL_DRIVER_VENV}"

echo "### driver venv: uv sync (first run source-builds TransformerEngine, ~30-50 min)"
env_ok=0
for attempt in 1 2 3; do
  uv sync --locked --extra automodel --no-install-package deep-ep && { env_ok=1; break; }
  echo "uv sync attempt ${attempt}/3 failed"
done
[ "${env_ok}" -eq 1 ] || { echo "FATAL: uv sync failed after 3 attempts"; exit 1; }
"${NEMO_RL_DRIVER_VENV}/bin/python" -c "print('UV_ENV_OK')" || { echo "FATAL: uv env unusable"; exit 1; }

# re-source so the CUDNN/NCCL globs (now present) get exported for worker-venv builds
source "${_HERE}/env.sh"

echo "### worker venvs (prefetch, same mechanism as the baked image)"
DT_VENV="${NEMO_RL_VENV_DIR}/nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2"
if [ ! -x "${DT_VENV}/bin/python" ]; then
  uv run --no-sync python nemo_rl/utils/prefetch_venvs.py VllmGenerationWorker DTensorPolicyWorkerV2 2>&1 | tail -3
fi
[ -x "${DT_VENV}/bin/python" ] || { echo "FATAL: worker venv prefetch failed"; exit 1; }
echo "WORKER_VENVS_OK"

echo "### gemma-4 KV-sharing patch (transformers 5.5.4 + sitecustomize)"
DRIVER_VENV="${NEMO_RL_DRIVER_VENV}" NEMO_RL_VENV_DIR="${NEMO_RL_VENV_DIR}" \
  bash "${_HERE}/patch_gemma4_kv_venvs.sh" || exit 1

echo "### fail-fast checks"
"${NEMO_RL_DRIVER_VENV}/bin/python" -c "import torch; torch.cuda.init(); print('CUDA_OK torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0))" || { echo "FATAL: torch/cuda init failed"; exit 1; }
"${NEMO_RL_DRIVER_VENV}/bin/python" -c "import math_verify; print('MATH_VERIFY_OK')" || { echo "FATAL: math_verify missing"; exit 1; }
echo "BUILD_VENV_DONE"
