#!/usr/bin/env bash
# Async GPU-pool queue for all twelve best-checkpoint trace collections in ONE 8-GPU
# pod. Packs the 8 GPUs greedily: 2-GPU collections first (12b/26b), then 1-GPU
# collections (e4b), and starts the next queued collection the moment GPUs free up.
# No per-model synchronization barrier — every (model, band) is independent.
#
# Each collection is run_gemma4_bestckpt_trace_collection.sh, pinned to its GPU
# slice via TRACE_GPU_IDS. Reuses that script unchanged.

set -uo pipefail

# Everything below runs inside main() so bash parses the WHOLE file before executing any of it.
# Bash otherwise reads a script incrementally by byte offset, so editing this file while an
# instance is running (hours-long collections) corrupts that instance: an edit that grew earlier
# lines made a running collection resume mid-line ("log_path: unbound variable") and fail after
# a completed train split.
main() {

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

COLLECTION="rl-distill-scripts/scale_train/run_gemma4_bestckpt_trace_collection.sh"
POLL_SECONDS="${QUEUE_POLL_SECONDS:-15}"
LOG_ROOT="${QUEUE_LOG_ROOT:-/tmp/gemma4_bestckpt_trace_queue/logs}"
mkdir -p "${LOG_ROOT}"

# Queue order: all 2-GPU runs first, then all 1-GPU runs. "<spec>:<gpus>".
# TRACE_QUEUE_SPECS (comma-separated "<spec>:<gpus>") overrides the list, so one node can
# run a subset. Two-node split used for the distillation study:
#   node 1: TRACE_QUEUE_SPECS=26b-easy:2,26b-medium:2,26b-hard:2,e2b-easy:1,e2b-medium:1,e2b-hard:1
#           (26b: 2 GPUs as TP2; e2b: 1 GPU) — the slowest teacher paired with the fastest
#   node 2: TRACE_QUEUE_SPECS=12b-easy:2,12b-medium:2,12b-hard:2,e4b-easy:2,e4b-medium:2,e4b-hard:2
#           (12b and e4b: 2 GPUs as DP2)
# This pairing balances total GPU-hours across the two nodes (gen + the distill runs that
# consume each node's own teachers) and gives each node a quick teacher so distillation can
# start while the slow teacher is still generating.
# Nodes write to the same S3 prefix and cooperate: completed shards are restored on startup
# and skipped, so a node can also pick up work another node (or this box) already did.
if [[ -n ${TRACE_QUEUE_SPECS:-} ]]; then
  IFS=',' read -r -a QUEUE <<< "${TRACE_QUEUE_SPECS}"
else
  QUEUE=(
    "12b-easy:2"
    "12b-medium:2"
    "12b-hard:2"
    "26b-easy:2"
    "26b-medium:2"
    "26b-hard:2"
    "e4b-easy:1"
    "e4b-medium:1"
    "e4b-hard:1"
    "e2b-easy:1"
    "e2b-medium:1"
    "e2b-hard:1"
  )
fi

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

# Physical GPU pool. TRACE_QUEUE_GPUS pins to explicit indices (for running on a
# shared box's free GPUs); otherwise use every visible GPU.
if [[ -n ${TRACE_QUEUE_GPUS:-} ]]; then
  IFS=',' read -r -a FREE_GPUS <<< "${TRACE_QUEUE_GPUS}"
else
  mapfile -t FREE_GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader | tr -d ' ')
fi
TOTAL_GPUS=${#FREE_GPUS[@]}
if ((TOTAL_GPUS < 2)); then echo "FATAL: need >=2 GPUs, saw ${TOTAL_GPUS}" >&2; exit 2; fi
echo "QUEUE start gpus=${TOTAL_GPUS} queue=[${QUEUE[*]}]"

declare -A PID_SPEC=() PID_GPUS=() SPEC_STATUS=() LAUNCHED=()
launched_count=0

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
  # Fill: launch ANY not-yet-launched queued collection that fits the free GPUs
  # (indices are tried in order, so 2-GPU runs are preferred, but a later 1-GPU
  # run fills a lone free GPU instead of leaving it idle behind a 2-GPU head).
  filled=1
  while ((filled == 1)); do
    filled=0
    for idx in "${!QUEUE[@]}"; do
      [[ -n "${LAUNCHED[$idx]:-}" ]] && continue
      entry="${QUEUE[$idx]}"; spec="${entry%%:*}"; need="${entry##*:}"
      if ((need <= ${#FREE_GPUS[@]})); then
        launch_spec "${spec}" "${need}"
        LAUNCHED[$idx]=1
        ((launched_count++))
        filled=1
      fi
    done
  done
  # Done when nothing is running and every collection has been launched.
  if ((${#PID_SPEC[@]} == 0 && launched_count >= ${#QUEUE[@]})); then break; fi
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
}

main "$@"
