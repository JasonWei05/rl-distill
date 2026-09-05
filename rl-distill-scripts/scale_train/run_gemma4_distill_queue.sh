#!/usr/bin/env bash
# Async GPU-pool queue over distillation runs on one node. Each entry is
# "<teacher_spec>:<student>:<gpus>[:VAR=value[;VAR=value...]]" and becomes one
# run_gemma4_distill_one.sh invocation pinned to a GPU slice; the optional fourth field sets
# per-run environment overrides for the runner (e.g. "e4b-easy:e4b:4:LR=1e-6" to redo one run
# with a different peak LR while the rest keep the queue-wide defaults). A run is launched only once its teacher's trace bundle is COMPLETE
# (locally or in S3), so this can start while trace generation is still finishing: runs whose
# teacher is done fill free GPUs immediately, the rest wait.
#
# Two-node split for the distillation study (DISTILL_QUEUE_RUNS overrides the default list):
#   node 1 (teachers 26b + e2b, 9 runs):
#     DISTILL_QUEUE_RUNS=26b-easy:e4b:4,26b-medium:e4b:4,26b-hard:e4b:4,26b-easy:e2b:2,26b-medium:e2b:2,26b-hard:e2b:2,e2b-easy:e2b:2,e2b-medium:e2b:2,e2b-hard:e2b:2
#   node 2 (teachers 12b + e4b, 12 runs):
#     DISTILL_QUEUE_RUNS=12b-easy:e4b:4,12b-medium:e4b:4,12b-hard:e4b:4,e4b-easy:e4b:4,e4b-medium:e4b:4,e4b-hard:e4b:4,12b-easy:e2b:2,12b-medium:e2b:2,12b-hard:e2b:2,e4b-easy:e2b:2,e4b-medium:e2b:2,e4b-hard:e2b:2
# (e4b-base students take 4 GPUs with KL chunk 4096; e2b-base students take 2 GPUs with KL chunk
# 2048 -- with a 4096 chunk the 2-GPU e2b layout ran out of memory on long traces. 4-GPU runs are
# listed first so they are preferred while the pool is full. Each node distills only from teachers
# it generated, so no cross-node dependency; node 2 carries more distill runs to offset node 1's
# slow 26b traces.)

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
    26b-easy:e4b:4 26b-medium:e4b:4 26b-hard:e4b:4
    12b-easy:e4b:4 12b-medium:e4b:4 12b-hard:e4b:4
    e4b-easy:e4b:4 e4b-medium:e4b:4 e4b-hard:e4b:4
    26b-easy:e2b:2 26b-medium:e2b:2 26b-hard:e2b:2
    12b-easy:e2b:2 12b-medium:e2b:2 12b-hard:e2b:2
    e4b-easy:e2b:2 e4b-medium:e2b:2 e4b-hard:e2b:2
    e2b-easy:e2b:2 e2b-medium:e2b:2 e2b-hard:e2b:2
  )
fi

# GPU pool. Static (default): DISTILL_QUEUE_GPUS or every visible GPU, owned exclusively by this queue.
# Dynamic (DISTILL_QUEUE_DYNAMIC_GPUS=true): share the node with another GPU consumer (e.g. the trace
# queue finishing its last collections) — before every fill pass a GPU is free only if it has no running
# compute process and is not assigned to one of this queue's own runs (which take a minute to allocate).
# A process that was just launched holds no GPU memory yet, so nvidia-smi alone is not enough: also pass
# the other consumer's queue log via DISTILL_QUEUE_RESERVED_GPUS_LOG (the trace queue's stdout) and every
# GPU named in a "QUEUE launch spec=<s> gpus=<a,b>" line without a later "QUEUE done spec=<s>" /
# "QUEUE FAILED spec=<s>" line is treated as reserved. Only use dynamic mode once the other consumer can
# no longer *start* new work, otherwise both may grab the same freed GPU inside the poll window.
DISTILL_QUEUE_DYNAMIC_GPUS="${DISTILL_QUEUE_DYNAMIC_GPUS:-false}"
DISTILL_QUEUE_RESERVED_GPUS_LOG="${DISTILL_QUEUE_RESERVED_GPUS_LOG:-}"
if [[ -n ${DISTILL_QUEUE_RESERVED_GPUS_LOG} && ! -f ${DISTILL_QUEUE_RESERVED_GPUS_LOG} ]]; then
  echo "FATAL: DISTILL_QUEUE_RESERVED_GPUS_LOG=${DISTILL_QUEUE_RESERVED_GPUS_LOG} does not exist" >&2
  exit 2
fi
case "${DISTILL_QUEUE_DYNAMIC_GPUS,,}" in
  true|false) ;;
  *) echo "FATAL: DISTILL_QUEUE_DYNAMIC_GPUS must be true or false" >&2; exit 2 ;;
esac
if [[ -n ${DISTILL_QUEUE_GPUS:-} ]]; then
  IFS=',' read -r -a POOL_GPUS <<< "${DISTILL_QUEUE_GPUS}"
else
  mapfile -t POOL_GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader | tr -d ' ')
fi
FREE_GPUS=("${POOL_GPUS[@]}")
echo "DISTILL_QUEUE start gpus=${#POOL_GPUS[@]} dynamic=${DISTILL_QUEUE_DYNAMIC_GPUS} reserved_log=${DISTILL_QUEUE_RESERVED_GPUS_LOG:-none} runs=[${QUEUE[*]}]"

# Dynamic mode: GPUs in POOL_GPUS with no compute process (per nvidia-smi) and not held by our runs.
refresh_free_gpus_dynamic() {
  local busy held gpu pid uuid_map
  declare -A busy_set=() held_set=()
  # nvidia-smi reports compute apps by GPU UUID; map UUID -> index.
  uuid_map="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | tr -d ' ')"
  while IFS=, read -r uuid; do
    [[ -n ${uuid} ]] || continue
    gpu="$(awk -F, -v u="${uuid}" '$2==u{print $1}' <<<"${uuid_map}")"
    [[ -n ${gpu} ]] && busy_set[$gpu]=1
  done < <(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | tr -d ' ')
  for pid in "${!PID_GPUS[@]}"; do
    IFS=',' read -r -a held <<< "${PID_GPUS[$pid]}"
    for gpu in "${held[@]}"; do held_set[$gpu]=1; done
  done
  # GPUs reserved by the other queue's ledger (launched, not yet done/failed), even if idle right now.
  if [[ -n ${DISTILL_QUEUE_RESERVED_GPUS_LOG} ]]; then
    local reserved
    reserved="$(awk '
      /QUEUE launch spec=/ { for (i=1;i<=NF;i++) { if ($i ~ /^spec=/) s=substr($i,6); if ($i ~ /^gpus=/) g=substr($i,6) } launched[s]=g }
      /QUEUE (done|FAILED) spec=/ { for (i=1;i<=NF;i++) if ($i ~ /^spec=/) s=substr($i,6); delete launched[s] }
      END { for (s in launched) print launched[s] }' "${DISTILL_QUEUE_RESERVED_GPUS_LOG}" | tr ',' '\n' | grep -E '^[0-9]+$' | sort -u | tr '\n' ' ')"
    for gpu in ${reserved}; do held_set[$gpu]=1; done
  fi
  FREE_GPUS=()
  for gpu in "${POOL_GPUS[@]}"; do
    [[ -n ${busy_set[$gpu]:-} || -n ${held_set[$gpu]:-} ]] && continue
    FREE_GPUS+=("${gpu}")
  done
}

teacher_ready() {  # the run's teacher bundle is complete locally or in S3
  local spec="$1"
  [[ -f ${TRACE_LOCAL_BASE}/${spec}/COMPLETE.json ]] && return 0
  aws s3 ls "${TRACE_S3_BASE}/${spec}/COMPLETE.json" >/dev/null 2>&1
}

declare -A PID_RUN=() PID_GPUS=() RUN_STATUS=() LAUNCHED=()
launched_count=0

run_field() { local run="$1" index="$2"; IFS=':' read -r -a _f <<< "${run}"; echo "${_f[$index]:-}"; }
run_gpus() { run_field "$1" 2; }

launch_run() {
  local run="$1" need="$2"
  local slice=("${FREE_GPUS[@]:0:need}")
  FREE_GPUS=("${FREE_GPUS[@]:need}")
  local gpu_csv; gpu_csv="$(IFS=,; echo "${slice[*]}")"
  local spec student overrides_raw
  spec="$(run_field "${run}" 0)"; student="$(run_field "${run}" 1)"; overrides_raw="$(run_field "${run}" 3)"
  local -a overrides=()
  if [[ -n ${overrides_raw} ]]; then
    IFS=';' read -r -a overrides <<< "${overrides_raw}"
    local kv override_re='^[A-Z][A-Z0-9_]*=[^;]*$'
    for kv in "${overrides[@]}"; do
      if [[ ! ${kv} =~ ${override_re} ]]; then
        echo "DISTILL_QUEUE FATAL bad override ${kv@Q} in run ${run}" >&2; exit 2
      fi
    done
  fi
  local tag=""; ((${#overrides[@]})) && tag="-$(IFS=-; echo "${overrides[*]//=/}" | tr -c 'A-Za-z0-9.\n-' '_')"
  local log="${LOG_ROOT}/${spec}-to-${student}${tag}.log"
  echo "DISTILL_QUEUE launch run=${run} gpus=${gpu_csv} overrides=[${overrides[*]:-}] log=${log} $(date -u +%FT%TZ)"
  env -u CUDA_VISIBLE_DEVICES "${overrides[@]}" TEACHER_SPEC="${spec}" STUDENT="${student}" DISTILL_GPU_IDS="${gpu_csv}" \
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
      if [[ ${DISTILL_QUEUE_DYNAMIC_GPUS,,} != true ]]; then FREE_GPUS+=("${gpu[@]}"); fi
      unset 'PID_RUN[$pid]' 'PID_GPUS[$pid]'
      if ((status == 0)); then
        echo "DISTILL_QUEUE done run=${run} status=0 freed_gpus=$(IFS=,; echo "${gpu[*]}") $(date -u +%FT%TZ)"
      else
        echo "DISTILL_QUEUE FAILED run=${run} status=${status} (log ${LOG_ROOT}) $(date -u +%FT%TZ)" >&2
      fi
      progressed=0
    fi
  done
  return "${progressed}"
}

while true; do
  if [[ ${DISTILL_QUEUE_DYNAMIC_GPUS,,} == true ]]; then refresh_free_gpus_dynamic; fi
  # Fill: any not-yet-launched run whose teacher is complete and whose GPU need fits.
  filled=1
  while ((filled == 1)); do
    filled=0
    for idx in "${!QUEUE[@]}"; do
      [[ -n "${LAUNCHED[$idx]:-}" ]] && continue
      entry="${QUEUE[$idx]}"; need="$(run_gpus "${entry}")"; spec="$(run_field "${entry}" 0)"
      if [[ ! ${need} =~ ^[1-9][0-9]*$ ]]; then echo "DISTILL_QUEUE FATAL bad gpu count in ${entry}" >&2; exit 2; fi
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
