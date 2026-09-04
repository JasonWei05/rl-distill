#!/usr/bin/env bash
# Async GPU-pool queue over distillation runs on one node. Each entry is
# "<teacher_spec>:<student>:<gpus>" and becomes one run_gemma4_distill_one.sh invocation
# pinned to a GPU slice. A run is launched only once its teacher's trace bundle is COMPLETE
# (locally or in S3), so this can start while trace generation is still finishing: runs whose
# teacher is done fill free GPUs immediately, the rest wait.
#
# Two-node split for the distillation study (DISTILL_QUEUE_RUNS overrides the default list):
#   node 1 (teachers 26b + e2b, 9 runs):
#     DISTILL_QUEUE_RUNS=26b-easy:e4b:2,26b-medium:e4b:2,26b-hard:e4b:2,26b-easy:e2b:1,26b-medium:e2b:1,26b-hard:e2b:1,e2b-easy:e2b:1,e2b-medium:e2b:1,e2b-hard:e2b:1
#   node 2 (teachers 12b + e4b, 12 runs):
#     DISTILL_QUEUE_RUNS=12b-easy:e4b:2,12b-medium:e4b:2,12b-hard:e4b:2,e4b-easy:e4b:2,e4b-medium:e4b:2,e4b-hard:e4b:2,12b-easy:e2b:1,12b-medium:e2b:1,12b-hard:e2b:1,e4b-easy:e2b:1,e4b-medium:e2b:1,e4b-hard:e2b:1
# (e4b-base students take 2 GPUs, e2b-base students 1 GPU; 2-GPU runs are listed first so they
# are preferred while the pool is full. Each node distills only from teachers it generated, so
# no cross-node dependency; node 2 carries more distill runs to offset node 1's slow 26b traces.)

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

RUNNER="rl-distill-scripts/scale_train/run_gemma4_distill_one.sh"
POLL_SECONDS="${DISTILL_QUEUE_POLL_SECONDS:-60}"
LOG_ROOT="${DISTILL_QUEUE_LOG_ROOT:-/tmp/gemma4_distill_queue/logs}"
mkdir -p "${LOG_ROOT}"
TRACE_S3_BASE="${TRACE_S3_BASE:-s3://scale-ml/genai/rl-distill/gemma4-bestckpt-traces-topk128-v2}"
TRACE_LOCAL_BASE="${TRACE_LOCAL_BASE:-/tmp/gemma4_bestckpt_traces_v2}"

export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-west-2}"
export AWS_REGION="${AWS_REGION:-us-west-2}"
if [[ -z "${AWS_PROFILE:-}" ]] && aws configure list-profiles 2>/dev/null | grep -qx ml-worker; then
  export AWS_PROFILE=ml-worker
fi

if [[ -n ${DISTILL_QUEUE_RUNS:-} ]]; then
  IFS=',' read -r -a QUEUE <<< "${DISTILL_QUEUE_RUNS}"
else
  # All 21 runs of the grid (e4b base <- {26b,12b,e4b}; e2b base <- {26b,12b,e4b,e2b}; x3 bands).
  QUEUE=(
    26b-easy:e4b:2 26b-medium:e4b:2 26b-hard:e4b:2
    12b-easy:e4b:2 12b-medium:e4b:2 12b-hard:e4b:2
    e4b-easy:e4b:2 e4b-medium:e4b:2 e4b-hard:e4b:2
    26b-easy:e2b:1 26b-medium:e2b:1 26b-hard:e2b:1
    12b-easy:e2b:1 12b-medium:e2b:1 12b-hard:e2b:1
    e4b-easy:e2b:1 e4b-medium:e2b:1 e4b-hard:e2b:1
    e2b-easy:e2b:1 e2b-medium:e2b:1 e2b-hard:e2b:1
  )
fi

if [[ -n ${DISTILL_QUEUE_GPUS:-} ]]; then
  IFS=',' read -r -a FREE_GPUS <<< "${DISTILL_QUEUE_GPUS}"
else
  mapfile -t FREE_GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader | tr -d ' ')
fi
echo "DISTILL_QUEUE start gpus=${#FREE_GPUS[@]} runs=[${QUEUE[*]}]"

teacher_ready() {  # the run's teacher bundle is complete locally or in S3
  local spec="$1"
  [[ -f ${TRACE_LOCAL_BASE}/${spec}/COMPLETE.json ]] && return 0
  aws s3 ls "${TRACE_S3_BASE}/${spec}/COMPLETE.json" >/dev/null 2>&1
}

declare -A PID_RUN=() PID_GPUS=() RUN_STATUS=() LAUNCHED=()
launched_count=0

launch_run() {
  local run="$1" need="$2"
  local slice=("${FREE_GPUS[@]:0:need}")
  FREE_GPUS=("${FREE_GPUS[@]:need}")
  local gpu_csv; gpu_csv="$(IFS=,; echo "${slice[*]}")"
  local spec="${run%%:*}" rest="${run#*:}" student; student="${rest%%:*}"
  local log="${LOG_ROOT}/${spec}-to-${student}.log"
  echo "DISTILL_QUEUE launch run=${run} gpus=${gpu_csv} log=${log} $(date -u +%FT%TZ)"
  env -u CUDA_VISIBLE_DEVICES TEACHER_SPEC="${spec}" STUDENT="${student}" DISTILL_GPU_IDS="${gpu_csv}" \
    bash "${RUNNER}" >"${log}" 2>&1 &
  local pid=$!
  PID_RUN[$pid]="${run}"
  PID_GPUS[$pid]="${gpu_csv}"
}

reap_finished() {
  local pid run status gpu progressed=1
  for pid in "${!PID_RUN[@]}"; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      wait "${pid}"; status=$?
      run="${PID_RUN[$pid]}"
      RUN_STATUS[$run]="${status}"
      IFS=',' read -r -a gpu <<< "${PID_GPUS[$pid]}"
      FREE_GPUS+=("${gpu[@]}")
      unset 'PID_RUN[$pid]' 'PID_GPUS[$pid]'
      if ((status == 0)); then
        echo "DISTILL_QUEUE done run=${run} status=0 $(date -u +%FT%TZ)"
      else
        echo "DISTILL_QUEUE FAILED run=${run} status=${status} (log ${LOG_ROOT}) $(date -u +%FT%TZ)" >&2
      fi
      progressed=0
    fi
  done
  return "${progressed}"
}

while true; do
  # Fill: any not-yet-launched run whose teacher is complete and whose GPU need fits.
  filled=1
  while ((filled == 1)); do
    filled=0
    for idx in "${!QUEUE[@]}"; do
      [[ -n "${LAUNCHED[$idx]:-}" ]] && continue
      entry="${QUEUE[$idx]}"; need="${entry##*:}"; spec="${entry%%:*}"
      if ((need <= ${#FREE_GPUS[@]})) && teacher_ready "${spec}"; then
        launch_run "${entry}" "${need}"
        LAUNCHED[$idx]=1
        ((launched_count++))
        filled=1
      fi
    done
  done
  if ((${#PID_RUN[@]} == 0 && launched_count >= ${#QUEUE[@]})); then break; fi
  if ! reap_finished; then
    sleep "${POLL_SECONDS}"
  fi
done

fail=0
echo "DISTILL_QUEUE summary:"
for entry in "${QUEUE[@]}"; do
  st="${RUN_STATUS[$entry]:-missing}"
  echo "  ${entry}: exit=${st}"
  [[ "${st}" == 0 ]] || fail=1
done
if ((fail != 0)); then
  echo "DISTILL_QUEUE_COMPLETE_WITH_FAILURES" >&2
  exit 1
fi
echo "DISTILL_QUEUE_COMPLETE_ALL_OK"
