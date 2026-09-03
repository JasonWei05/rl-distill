#!/usr/bin/env bash
# Async GPU-pool queue for all six best-checkpoint trace collections in ONE 8-GPU
# pod. Packs the 8 GPUs greedily: 2-GPU collections first (12b/26b), then 1-GPU
# collections (e4b), and starts the next queued collection the moment GPUs free up.
# No per-model synchronization barrier — every (model, band) is independent.
#
# Each collection is run_gemma4_bestckpt_trace_collection.sh, pinned to its GPU
# slice via TRACE_GPU_IDS. Reuses that script unchanged.

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

COLLECTION="rl-distill-scripts/scale_train/run_gemma4_bestckpt_trace_collection.sh"
POLL_SECONDS="${QUEUE_POLL_SECONDS:-15}"
LOG_ROOT="${QUEUE_LOG_ROOT:-/tmp/gemma4_bestckpt_trace_queue/logs}"
mkdir -p "${LOG_ROOT}"

# Queue order: all 2-GPU runs first, then all 1-GPU runs. "<spec>:<gpus>".
QUEUE=(
  "12b-easy:2"
  "12b-medium:2"
  "26b-easy:2"
  "e4b-easy:1"
  "e4b-medium:1"
  "e4b-hard:1"
)

if [[ -f .env ]]; then set -a; source .env; set +a; fi
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-west-2}"
export AWS_REGION="${AWS_REGION:-us-west-2}"

# Build the shared venv ONCE so concurrent collections never race on it.
export PATH="${HOME}/.local/bin:/root/.local/bin:${PATH}"
command -v uv >/dev/null 2>&1 || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="${HOME}/.local/bin:${PATH}"; }
export VENV="${VENV:-/tmp/.venv-gemma4}"
export GEMMA4_CUDA_VARIANT="${GEMMA4_CUDA_VARIANT:-cu129}"
if [[ ! -x ${VENV}/bin/python ]]; then
  echo "QUEUE building shared venv ${VENV}"
  VENV="${VENV}" GEMMA4_CUDA_VARIANT="${GEMMA4_CUDA_VARIANT}" bash rl-distill-scripts/setup_env_gemma4.sh
fi

# Physical GPU pool.
mapfile -t FREE_GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader | tr -d ' ')
TOTAL_GPUS=${#FREE_GPUS[@]}
if ((TOTAL_GPUS < 2)); then echo "FATAL: need >=2 GPUs, saw ${TOTAL_GPUS}" >&2; exit 2; fi
echo "QUEUE start gpus=${TOTAL_GPUS} queue=[${QUEUE[*]}]"

declare -A PID_SPEC=() PID_GPUS=() SPEC_STATUS=()
head=0

launch_spec() {
  local spec="$1" need="$2"
  local slice=("${FREE_GPUS[@]:0:need}")
  FREE_GPUS=("${FREE_GPUS[@]:need}")
  local gpu_csv; gpu_csv="$(IFS=,; echo "${slice[*]}")"
  local log="${LOG_ROOT}/${spec}.log"
  echo "QUEUE launch spec=${spec} gpus=${gpu_csv} log=${log} $(date -u +%FT%TZ)"
  # No parent CUDA_VISIBLE_DEVICES mask: each worker selects its absolute device.
  env -u CUDA_VISIBLE_DEVICES \
    TRACE_SPEC="${spec}" TRACE_GPU_IDS="${gpu_csv}" VENV="${VENV}" \
    bash "${COLLECTION}" >"${log}" 2>&1 &
  local pid=$!
  PID_SPEC[$pid]="${spec}"
  PID_GPUS[$pid]="${gpu_csv}"
}

reap_finished() {
  local pid spec status gpu progressed=1
  for pid in "${!PID_SPEC[@]}"; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      wait "${pid}"; status=$?
      spec="${PID_SPEC[$pid]}"
      SPEC_STATUS[$spec]="${status}"
      IFS=',' read -r -a gpu <<< "${PID_GPUS[$pid]}"
      FREE_GPUS+=("${gpu[@]}")
      unset 'PID_SPEC[$pid]' 'PID_GPUS[$pid]'
      if ((status == 0)); then
        echo "QUEUE done spec=${spec} status=0 freed_gpus=${PID_GPUS[$pid]:-} $(date -u +%FT%TZ)"
      else
        echo "QUEUE FAILED spec=${spec} status=${status} (log ${LOG_ROOT}/${spec}.log) $(date -u +%FT%TZ)" >&2
      fi
      progressed=0
    fi
  done
  return "${progressed}"
}

while true; do
  # Fill: launch the next queued collection whenever enough GPUs are free.
  while ((head < ${#QUEUE[@]})); do
    entry="${QUEUE[$head]}"; spec="${entry%%:*}"; need="${entry##*:}"
    ((need <= ${#FREE_GPUS[@]})) || break
    launch_spec "${spec}" "${need}"
    ((head++))
  done
  # Done when nothing is running and the queue is drained.
  if ((${#PID_SPEC[@]} == 0 && head >= ${#QUEUE[@]})); then break; fi
  # Wait for a collection to finish, then loop to refill freed GPUs.
  if ! reap_finished; then
    sleep "${POLL_SECONDS}"
  fi
done

fail=0
echo "QUEUE summary:"
for entry in "${QUEUE[@]}"; do
  spec="${entry%%:*}"
  st="${SPEC_STATUS[$spec]:-missing}"
  echo "  ${spec}: exit=${st}"
  [[ "${st}" == 0 ]] || fail=1
done
if ((fail != 0)); then
  echo "QUEUE_COMPLETE_WITH_FAILURES" >&2
  exit 1
fi
echo "QUEUE_COMPLETE_ALL_OK"
