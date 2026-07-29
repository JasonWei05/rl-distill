#!/usr/bin/env bash
# Retry orchestrator for the two LOCAL 1B-PT DeepScaleR strict-4/4 runs. The shared box is often
# oversubscribed (load > #CPUs), which starves Ray's local-cluster startup: the raylet aborts because
# the metrics agent (an EFS-venv Python import) can't write its port file within the raylet's fixed
# ~86s wait (no env override exists for that one). GPUs are free; the ONLY blocker is CPU/EFS load.
#
# This brings each seed up ONE AT A TIME (no concurrent-startup contention), and before each attempt it
# waits for the 1-min load to drop below LOAD_GATE — so it only tries when the agent can plausibly start
# in time, and doesn't pile onto a peak-load box. Stops the instant a seed reaches training and moves on.
#
#   nohup bash rl-distill-scripts/launch_1b_ds4of4strict_retry.sh > ~/verl/logs/1b_ds4of4strict_retry.out 2>&1 &
set -uo pipefail   # NOT -e: failures are handled inside the retry loop
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${PROJECT_ROOT}"
[ -f .venv/bin/activate ] && source .venv/bin/activate
set -a; source .env 2>/dev/null; set +a
DATA_DIR="${DATA_DIR:-${HOME}/verl/data}"
TRAIN="${DATA_DIR}/deepscaler_4of4strict_rl_train.parquet"
VAL="['${DATA_DIR}/deepscaler_4of4strict_rl_val200_x16.parquet']"
TPL="${PROJECT_ROOT}/rl-distill-scripts/data/gemma3_it_fewshot_math.jinja"
LOGDIR="${LOGDIR:-${HOME}/verl/logs/1b_ds4of4strict_sweep}"; mkdir -p "${LOGDIR}"
for f in "${TRAIN}" "${DATA_DIR}/deepscaler_4of4strict_rl_val200_x16.parquet"; do
  [ -f "${f}" ] || { echo "ERROR: missing strict-4/4 file ${f}"; exit 1; }
done

LOAD_GATE=${LOAD_GATE:-160}          # only attempt when 1-min load avg is below this
MAX_WALL_MIN=${MAX_WALL_MIN:-180}    # overall deadline
ATTEMPT_TIMEOUT=${ATTEMPT_TIMEOUT:-540}   # seconds to decide an attempt outcome
BACKOFF=${BACKOFF:-90}               # wait after a failed attempt
DEADLINE=$(( $(date +%s) + MAX_WALL_MIN*60 ))
now() { date +%H:%M:%S; }
load1() { cut -d' ' -f1 /proc/loadavg; }

wait_for_load() {   # block until load < LOAD_GATE or deadline; echo reason
  while [ "$(date +%s)" -lt "${DEADLINE}" ]; do
    awk -v l="$(load1)" -v g="${LOAD_GATE}" 'BEGIN{exit !(l<g)}' && { echo ok; return 0; }
    sleep 30
  done
  echo deadline; return 1
}

kill_seed() {   # $1=seed  (safe: grep -v grep drops both the grep proc and any wrapper cmdline)
  local pids; pids="$(ps -eo pid,cmd | grep "ray_1b_ds4of4s_seed${1}" | grep -v grep | awk '{print $1}')"
  [ -n "${pids}" ] && kill -9 ${pids} 2>/dev/null || true
}

launch_seed() {   # $1=seed $2=pair $3=portbase ; echoes driver pid
  local seed=$1 pair=$2 base=$3
  for off in 0 100 200; do
    local h; h="$(ss -ltnHp 2>/dev/null | grep -F ":$((base+off)) " | grep -oP 'pid=\K[0-9]+' | sort -u || true)"
    [ -n "${h}" ] && kill -9 ${h} 2>/dev/null || true
  done
  rm -rf "/tmp/ray_1b_ds4of4s_seed${seed}" 2>/dev/null || true; sleep 2
  local ng=$(( $(tr -cd ',' <<<"${pair}" | wc -c) + 1 ))
  CUDA_VISIBLE_DEVICES="${pair}" N_GPUS_PER_NODE="${ng}" VERL_VLLM_PORT_BASE="${base}" \
  RAY_TMP="/tmp/ray_1b_ds4of4s_seed${seed}" DATA_SEED="${seed}" SAVE_FREQ=25 \
  MAX_RESPONSE_LENGTH=8192 OVERLONG_BUFFER_LEN=2048 \
  LOG_VAL_GENERATIONS=100 LOG_TRAIN_GENERATIONS=100 \
  TRAIN_FILE="${TRAIN}" VAL_FILES="${VAL}" GEMMA3_CHAT_TEMPLATE_FILE="${TPL}" \
  EXP_NAME="DAPO-gemma3-1b-pt-DeepScaleR-4of4strict-seed${seed}" \
  CKPTS_DIR="${HOME}/verl/ckpts/DAPO/gemma3-1b-pt-deepscaler-4of4strict-seed${seed}" \
  HF_PUSH_REPO="JWei05/DAPO-Gemma3-1B-PT-DeepScaleR-4of4strict-seed${seed}" \
    nohup bash rl-distill-scripts/gemma3_1b_pt_fewshot_math_rl.sh > "${LOGDIR}/seed${seed}.log" 2>&1 &
  echo $!
}

bring_up() {   # $1=seed $2=pair $3=portbase ; returns 0 once past Ray bringup
  local seed=$1 pair=$2 base=$3 attempt=0
  local log="${LOGDIR}/seed${seed}.log"
  local FAIL='Timed out waiting for file|Check failed|timed out during startup|Traceback \(most recent|CUDA out of memory|Address already in use'
  while [ "$(date +%s)" -lt "${DEADLINE}" ]; do
    [ "$(wait_for_load)" = ok ] || break
    attempt=$((attempt+1)); : > "${log}"
    local pid; pid="$(launch_seed "${seed}" "${pair}" "${base}")"
    echo "[$(now)] seed${seed} attempt ${attempt}: pid ${pid}, load $(load1)"
    local t0; t0=$(date +%s); local outcome=""
    while [ $(( $(date +%s) - t0 )) -lt "${ATTEMPT_TIMEOUT}" ]; do
      if grep -qE "val-core|step:[0-9]" "${log}" 2>/dev/null; then outcome=ok; break; fi
      # >90 non-empty log lines with no failure marker and driver alive => past the flaky Ray bringup
      if ! grep -qE "${FAIL}" "${log}" 2>/dev/null && kill -0 "${pid}" 2>/dev/null \
         && [ "$(grep -c . "${log}" 2>/dev/null)" -gt 90 ]; then outcome=up; break; fi
      if grep -qE "${FAIL}" "${log}" 2>/dev/null; then outcome=fail; break; fi
      kill -0 "${pid}" 2>/dev/null || { outcome=died; break; }
      sleep 6
    done
    if [ "${outcome}" = ok ] || [ "${outcome}" = up ]; then
      echo "[$(now)] seed${seed} PAST RAY BRINGUP ✓ (attempt ${attempt}, outcome=${outcome})"; return 0
    fi
    echo "[$(now)] seed${seed} attempt ${attempt} failed (${outcome:-stuck}); cleaning up, backoff ${BACKOFF}s"
    kill_seed "${seed}"; sleep "${BACKOFF}"
  done
  echo "[$(now)] seed${seed} GAVE UP (deadline/backoff exhausted)"; return 1
}

echo "=== 1B strict-4/4 retry orchestrator: LOAD_GATE=${LOAD_GATE} deadline=$(date -d @${DEADLINE} +%H:%M) ==="
if bring_up 42 "2,3" 52000; then
  echo "=== seed 42 up — bringing up seed 43 ==="
  if bring_up 43 "4,5" 53000; then echo "=== BOTH SEEDS UP ✓✓ ==="; else echo "=== seed 42 up; seed 43 gave up ==="; fi
else
  echo "=== seed 42 never came up (box stayed above LOAD_GATE / deadline) ==="
fi
