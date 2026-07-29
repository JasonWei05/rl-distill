#!/usr/bin/env bash
set -uo pipefail
# Robust LOCAL launcher for ONE 1B strict-4/4 seed. Combines the two fixes for the raylet's fixed ~80s
# dashboard-agent port-file wait (which is hardcoded in ray's C++ WaitForFile, NOT tunable):
#   Way 1 — agent-only local-ray shim (codex fix): a sitecustomize (RAY_AGENT_SHIM) prepends a LOCAL
#           ray[default] target (LOCAL_RAY_SITE, ~200MB, matches the EFS ray version) onto sys.path ONLY
#           for the dashboard/metrics agent process, so its import loads from /tmp in seconds instead of
#           ~100s off the contended EFS venv -> it writes its port file inside the raylet's ~80s wait.
#           Driver/workers keep the EFS venv. Much lighter than copying the full 18GB venv.
#   Way 2 — retry ray.init until it clears the startup wall (in case a slow import still races).
# Deconflicted from the ScaleTrain seed via a -local suffix (own wandb name + HF repo). Cleans up each
# attempt by PROCESS GROUP so it can never leave a zombie retry loop behind.
#
#   VENV_LOCAL=/tmp/verl_venv_local SEED=42 PAIR=2,3 PORT=52000 \
#     nohup bash rl-distill-scripts/launch_1b_ds4of4strict_local_retry.sh > ~/verl/logs/s42_localretry.out 2>&1 &
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${PROJECT_ROOT}"
set -a; source .env 2>/dev/null || true; set +a
SEED="${SEED:-42}"; PAIR="${PAIR:-2,3}"; PORT="${PORT:-52000}"
RAY_AGENT_SHIM="${RAY_AGENT_SHIM:-/tmp/ray-agent-shim}"   # sitecustomize dir: agent-only local-ray
LOCAL_RAY_SITE="${LOCAL_RAY_SITE:-/tmp/ray-site-local}"   # local ray[default] target (match EFS ray ver)
DATA_DIR="${HOME}/verl/data"
TRAIN="${DATA_DIR}/deepscaler_4of4strict_rl_train.parquet"
VAL="['${DATA_DIR}/deepscaler_4of4strict_rl_val200_x16.parquet']"
TPL="${PROJECT_ROOT}/rl-distill-scripts/data/gemma3_it_fewshot_math.jinja"
LOGDIR="${HOME}/verl/logs/1b_ds4of4strict_sweep"; mkdir -p "${LOGDIR}"
LOG="${LOGDIR}/seed${SEED}_local.log"
RAYTMP="/tmp/ray_1b_ds4of4s_seed${SEED}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-12}"; ATTEMPT_TIMEOUT="${ATTEMPT_TIMEOUT:-360}"; BACKOFF="${BACKOFF:-30}"
FAIL='Timed out waiting for file|Check failed|timed out during startup|Traceback \(most recent|CUDA out of memory'
OK='Started a local Ray instance|val-core|step:[0-9]'
{ [ -f "${RAY_AGENT_SHIM}/sitecustomize.py" ] && [ -d "${LOCAL_RAY_SITE}/ray" ]; } || { echo "ERROR: agent-shim/local-ray missing (${RAY_AGENT_SHIM}/sitecustomize.py, ${LOCAL_RAY_SITE}/ray)"; exit 2; }

cleanup() {   # group-kill this seed's procs (safe: grep -v grep drops the grep + this script's cmdline)
  for p in $(ps -eo pid,cmd | grep -E "ray_1b_ds4of4s_seed${SEED}|4of4strict-seed${SEED}-local" | grep -v grep | awk '{print $1}'); do
    pgid=$(ps -o pgid= -p "$p" 2>/dev/null | tr -d ' '); [ -n "$pgid" ] && kill -9 -"$pgid" 2>/dev/null || true
  done
  for off in 0 100 200; do h="$(ss -ltnHp 2>/dev/null | grep -F ":$((PORT+off)) " | grep -oP 'pid=\K[0-9]+' | sort -u || true)"; [ -n "$h" ] && kill -9 $h 2>/dev/null || true; done
  rm -rf "${RAYTMP}" 2>/dev/null || true
}

ng=$(( $(tr -cd ',' <<<"${PAIR}" | wc -c) + 1 ))
a=0
while [ "$a" -lt "$MAX_ATTEMPTS" ]; do
  a=$((a+1)); cleanup; sleep 2; : > "${LOG}"
  echo "[$(date +%H:%M:%S)] attempt ${a}/${MAX_ATTEMPTS} seed ${SEED} (agent-shim local-ray, load $(cut -d' ' -f1 /proc/loadavg))"
  CUDA_VISIBLE_DEVICES="${PAIR}" N_GPUS_PER_NODE="${ng}" VERL_VLLM_PORT_BASE="${PORT}" \
  PYTHONPATH="${RAY_AGENT_SHIM}:${PYTHONPATH:-}" LOCAL_RAY_SITE="${LOCAL_RAY_SITE}" \
  RAY_TMP="${RAYTMP}" DATA_SEED="${SEED}" SAVE_FREQ=25 \
  MAX_RESPONSE_LENGTH=8192 OVERLONG_BUFFER_LEN=2048 LOG_VAL_GENERATIONS=100 LOG_TRAIN_GENERATIONS=100 \
  TRAIN_FILE="${TRAIN}" VAL_FILES="${VAL}" GEMMA3_CHAT_TEMPLATE_FILE="${TPL}" \
  EXP_NAME="DAPO-gemma3-1b-pt-DeepScaleR-4of4strict-seed${SEED}-local" \
  CKPTS_DIR="${HOME}/verl/ckpts/DAPO/gemma3-1b-pt-deepscaler-4of4strict-seed${SEED}-local" \
  HF_PUSH_REPO="JWei05/DAPO-Gemma3-1B-PT-DeepScaleR-4of4strict-seed${SEED}-local" \
    nohup bash rl-distill-scripts/gemma3_1b_pt_fewshot_math_rl.sh > "${LOG}" 2>&1 &
  drv=$!
  t0=$(date +%s); outcome=""
  while [ $(( $(date +%s) - t0 )) -lt "${ATTEMPT_TIMEOUT}" ]; do
    if grep -qE "${OK}" "${LOG}" 2>/dev/null; then outcome=ok; break; fi
    if grep -qE "${FAIL}" "${LOG}" 2>/dev/null; then outcome=fail; break; fi
    kill -0 "$drv" 2>/dev/null || { outcome=died; break; }
    sleep 6
  done
  if [ "$outcome" = ok ]; then echo "[$(date +%H:%M:%S)] SEED ${SEED} CLEARED ray.init ✓ (attempt ${a}) — training continues (pid ${drv})"; exit 0; fi
  echo "[$(date +%H:%M:%S)] attempt ${a} => ${outcome:-timeout}; cleanup + backoff ${BACKOFF}s"
  cleanup; sleep "${BACKOFF}"
done
echo "[$(date +%H:%M:%S)] GAVE UP after ${MAX_ATTEMPTS} attempts"; exit 1
