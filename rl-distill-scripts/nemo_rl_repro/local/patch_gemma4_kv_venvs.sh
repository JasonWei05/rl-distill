#!/usr/bin/env bash
# Apply the verified gemma-4 KV-sharing fix stack to already-built venvs (idempotent).
# Used by both the local build (build_venv.sh) and the pod run file.
#
# What it does (PROGRESS_LOG 2026-07-30):
#   1. transformers==5.5.4 into the DRIVER venv and the DTensor POLICY WORKER venv
#      (within nemo-rl's <5.6.0 pin; the uv.lock's 5.5.0 silently corrupts gemma-4
#      E2B/E4B training whenever activation checkpointing is on). The vLLM worker
#      venv is deliberately untouched.
#   2. sitecustomize.py into the policy worker venv's site-packages (removes the
#      use_cache workaround + forces FSDP2 cast_forward_inputs=False on >= 5.5.2;
#      hard-fails act-ckpt launches on < 5.5.2).
#
# NOTE: a worker-venv rebuild (deleting NEMO_RL_VENV_DIR) reverts to the locked
# 5.5.0 — re-run this script afterwards. Env inputs (all have defaults):
#   DRIVER_VENV, NEMO_RL_VENV_DIR, REPRO_DIR, GEMMA4_TRANSFORMERS_VERSION
set -euo pipefail
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRO_DIR="${REPRO_DIR:-$(dirname "${_HERE}")}"
DRIVER_VENV="${DRIVER_VENV:-/tmp/nemo-rl-venv}"
NEMO_RL_VENV_DIR="${NEMO_RL_VENV_DIR:-/tmp/nemo-rl-worker-venvs}"
TF_VER="${GEMMA4_TRANSFORMERS_VERSION:-5.5.4}"
DT_VENV="${NEMO_RL_VENV_DIR}/nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2"

command -v uv >/dev/null || export PATH="$HOME/.local/bin:$PATH"

for venv in "${DRIVER_VENV}" "${DT_VENV}"; do
  [ -x "${venv}/bin/python" ] || { echo "FATAL: venv missing: ${venv}"; exit 1; }
  uv pip install -q --python "${venv}/bin/python" "transformers==${TF_VER}"
  "${venv}/bin/python" -c "import transformers; print('  ${venv##*/}: transformers', transformers.__version__)"
done

DT_SP=$(ls -d "${DT_VENV}"/lib/python3.*/site-packages | head -1)
cp "${REPRO_DIR}/sitecustomize.py" "${DT_SP}/sitecustomize.py"
"${DT_VENV}/bin/python" -c "import sitecustomize" >/dev/null
echo "GEMMA4_KV_PATCH_OK transformers=${TF_VER} sitecustomize->${DT_SP}"
