#!/usr/bin/env bash
# Launch the durable 8xH100 Gemma-4 E4B production supervisor.
set -euo pipefail

_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_HERE}/env.sh"
_REPO_ROOT="$(cd "${_HERE}/../../.." && pwd)"
cd "${_REPO_ROOT}"

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

exec "${NEMO_RL_DRIVER_VENV}/bin/python" \
  "${_HERE}/full_run_supervisor.py" \
  --max-steps 500 \
  --gpus 8 \
  "$@"
