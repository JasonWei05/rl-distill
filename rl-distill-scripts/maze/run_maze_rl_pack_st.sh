#!/usr/bin/env bash
# ScaleTrain pod entry (GPUS GPUs, default 1): many small-model maze RL runs packed onto the pod's GPUs (maze/rl_tis_run.sh
# each), SLOTS at a time. Each run starts on the GPU with the least per-step cost already on it (256 prompts ~ 5x a small
# batch); when a GPU has no runs left while another still has >= 2 (and nothing is queued), the costliest run on the busiest
# GPU is stopped and resumed from its local checkpoint on the idle GPU (loses < 10 steps), so no GPU sits idle.
# A run spec is <level>.<small seed of the base>.<prompts per step>.<RL seed>, e.g. p1.s0.16.2; the base is the tuned small
# model s3://.../pretrain20k_sml/small_ds0_is<seed>/tune-<level>/best, or ${BASES_S3}/<level>_s<seed>/ when BASES_S3 is set
# (the shortest-path bases). Run state lives under ${OUT_S3}/runs/<suite>/:
# state/ (latest checkpoint dir, best_hf/ with best_step.json, best.json, latest_checkpointed_iteration.txt, wandb_id.txt),
# logs/, and COMPLETE.json once the run has finished (early stop or max steps). A sidecar mirrors state every SYNC_SECONDS;
# a relaunched pod skips finished runs and resumes the others from their last mirrored checkpoint (W&B id = suite, resumed
# in place). Writes ${OUT_S3}/pods/${TAG}.COMPLETE when every run in RUNS is complete.
#   env: RUNS ('+'-separated specs), SLOTS (concurrent runs), TAG (pod name), OUT_S3, MAX_STEPS (10000), PATIENCE (20),
#        SYNC_SECONDS (300, logs), CKPT_POLL_SECONDS (15, new checkpoints), RUN_RETRIES (2 relaunches per run inside the pod), REWARD_NUM_PROCESSES (2),
#        WANDB_PROJECT_NAME (tailrl-text-maze), WANDB_ID_SUFFIX (v1), MPS (0; 1 starts a CUDA MPS daemon for the packed runs),
#        SUITE_PREFIX (sml; run names <prefix>-<level>-b<prompts>-s<seed>), ACTOR_LR (1e-4, read by rl_tis_run.sh),
#        GPUS (1), FAST_DECODE (0, read by rl_tis_run.sh), REBALANCE_MIN_AGE (900 s a run must have run before it can move),
#        BASES_S3 (see above), RL_REWARD (goal | shortest, read by rl_tis_run.sh)
#        RUN_ALL_GPUS (0; 1: each run trains data-parallel on all GPUS of the pod, one run at a time: SLOTS is set to 1),
#        RL_ADV (grpo | reinforce_batch, read by rl_tis_run.sh; also names the run dir),
#        DATA_S3 (dir with main_parquet/ (RL prompts) and eval500/ (validation); default the Prim-maze data)
source "$(dirname "${BASH_SOURCE[0]}")/st_setup.sh"
RUNS="${RUNS:?}"; SLOTS="${SLOTS:?}"; TAG="${TAG:?}"; OUT_S3="${OUT_S3:?}"
SYNC_SECONDS="${SYNC_SECONDS:-300}"; RUN_RETRIES="${RUN_RETRIES:-2}"; SP="${SUITE_PREFIX:-sml}"; GPUS="${GPUS:-1}"
SMALL_S3=s3://scale-ml/genai/rl-distill/maze/pretrain20k_sml; DATA_S3="${DATA_S3:-s3://scale-ml/genai/rl-distill/maze/data}"
echo "MAZE_RL_START $(date -u +%FT%TZ) host=$(hostname) nproc=$(nproc) gpus=${GPUS} slots=${SLOTS} runs=${RUNS}"
D=/tmp/maze/data; RD=/tmp/maze/runs; LOGS=/tmp/maze/rl_logs
CK=${RD}/checkpoints/${WANDB_PROJECT_NAME:-tailrl-text-maze}   # train.sh: <TAILRL_MAZE_RUN_DIR>/checkpoints/<W&B project>/<experiment>
mkdir -p "${D}" "${CK}" "${LOGS}"
aws s3 sync --only-show-errors "${DATA_S3}/main_parquet" "${D}/main_parquet"; aws s3 sync --only-show-errors "${DATA_S3}/eval500" "${D}/eval500"
cat > "${T}/.env" <<EOF
TAILRL_MAZE_DATA_DIR=${D}
HF_CACHE_DIR=/tmp/maze/hf_sft_ckpts
TAILRL_MAZE_RUN_DIR=${RD}
WANDB_MODE=online
WANDB_ENTITY=rl-distill
WANDB_PROJECT=${WANDB_PROJECT_NAME:-tailrl-text-maze}
EOF
export TAILRL_ROOT="${T}" VAL_DIR="${D}/eval500" HOME_MAZE=/tmp/maze REWARD_NUM_PROCESSES="${REWARD_NUM_PROCESSES:-2}"
export WANDB_ENTITY=rl-distill WANDB_PROJECT="${WANDB_PROJECT_NAME:-tailrl-text-maze}" WANDB_MODE=online
if [ "${MPS:-0}" = 1 ]; then   # CUDA MPS: kernels from the packed runs share the GPU instead of time-slicing it
  export CUDA_MPS_PIPE_DIRECTORY=/tmp/maze/mps_pipe CUDA_MPS_LOG_DIRECTORY=/tmp/maze/mps_log
  mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"
  if nvidia-cuda-mps-control -d; then echo "MPS_ON"; else echo "MPS_FAILED"; unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY; fi
fi
( while sleep 60; do   # pod load: cgroup CPU throttling + GPU utilization
    # pod memory (cgroup v2 memory.current + memory.stat anon / file / shmem, else cgroup v1), in GiB: shows what fills the
    # 192 GiB pod limit before an OOM kill
    mem=$( { [ -f /sys/fs/cgroup/memory.current ] && awk -v c="$(cat /sys/fs/cgroup/memory.current)" '/^(anon|file|shmem) / {printf "%s=%.1f ", $1, $2/2^30} END {printf "total=%.1f", c/2^30}' /sys/fs/cgroup/memory.stat; } \
           || awk '/^(total_rss|total_cache|total_shmem) / {printf "%s=%.1f ", $1, $2/2^30}' /sys/fs/cgroup/memory/memory.stat 2>/dev/null)
    echo "PODSTAT $(date -u +%T) load=$(cut -d' ' -f1 /proc/loadavg) mem[GiB] ${mem} gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr -d ' ' | paste -sd';')"
  done ) & podstat=$!

# must match TailRL train.sh's EXPERIMENT_NAME (method tag + G = samples per prompt), else resumes download a checkpoint the
# trainer never reads (it then trains from scratch) and COMPLETE.json is never found (the run is reported failed)
case "${RL_ADV:-grpo}" in
  grpo) RUN_TAG=grpo; RUN_G=16 ;;
  reinforce_batch) RUN_TAG=reinforce_batch; RUN_G=1 ;;
  *) echo "unknown RL_ADV ${RL_ADV}" >&2; exit 1 ;;
esac
# RUN_ALL_GPUS=1: every run uses all GPUS of the pod (data parallel, one run at a time), else one GPU per run
if [ "${RUN_ALL_GPUS:-0}" = 1 ]; then
  RUN_NGPU="${GPUS}"; RUN_GPU_IDS="$(seq -s, 0 $((GPUS - 1)))"; SLOTS=1
else
  RUN_NGPU=1; RUN_GPU_IDS=0
fi
RUN_GPU_TAG="${RUN_GPU_IDS//,/_}"; export RUN_GPU_IDS
run_dir() { echo "${CK}/textmaze_$1_${RUN_TAG}_binary_ckpt0_bs$2_G${RUN_G}_${RUN_NGPU}gpu_${RUN_GPU_TAG}_seed$3"; }
push_state() {  # push_state <suite> <run dir>: latest checkpoint + best + bookkeeping (no older checkpoints)
  local s3="${OUT_S3}/runs/$1" d="$2" step
  [ -f "${LOGS}/$1.log" ] && aws s3 cp --only-show-errors "${LOGS}/$1.log" "${s3}/logs/$1.$(hostname).log"   # even before the first checkpoint
  [ -d "${d}" ] || return 0
  step="$(cat "${d}/latest_checkpointed_iteration.txt" 2>/dev/null || true)"
  # the checkpoint must be fully on S3 before latest_checkpointed_iteration.txt points at it (a failed / interrupted sync
  # used to leave "latest" pointing at a missing checkpoint, and the run could not resume)
  if [ -n "${step}" ] && [ -d "${d}/global_step_${step}" ]; then
    aws s3 sync --only-show-errors "${d}/global_step_${step}" "${s3}/state/global_step_${step}" \
      && aws s3 ls "${s3}/state/global_step_${step}/actor/model_world_size_${RUN_NGPU}_rank_$((RUN_NGPU - 1)).pt" >/dev/null 2>&1 \
      || return 1   # the last rank's shard: verl writes model_world_size_<N>_rank_<r>.pt
  fi
  [ -d "${d}/best_hf" ] && aws s3 sync --only-show-errors "${d}/best_hf" "${s3}/state/best_hf"
  # "latest" = the step just verified, not the trainer's file: it may already name a newer checkpoint that is not uploaded
  [ -n "${step}" ] && [ -d "${d}/global_step_${step}" ] \
    && { echo "${step}" | aws s3 cp --only-show-errors - "${s3}/state/latest_checkpointed_iteration.txt" || true; }
  for f in best.json wandb_id.txt COMPLETE.json; do
    [ -f "${d}/${f}" ] && aws s3 cp --only-show-errors "${d}/${f}" "${s3}/state/${f}"
  done
  return 0
}

declare -A PID SUITE_OF DIR_OF SPEC_OF TRIES GPU_OF COST_OF STARTED_AT PREF_GPU MIGRATE_TO
LOAD=(); NRUN=(); for ((g = 0; g < GPUS; g++)); do LOAD[g]=0; NRUN[g]=0; done
migrating=0   # a run stopped for rebalancing whose restart is pending (a plain counter: ${#assoc[@]} of an empty array trips set -u)
queue=(); IFS='+' read -r -a queue <<< "${RUNS}"
free_slot() { for ((i = 0; i < SLOTS; i++)); do [ -z "${PID[$i]:-}" ] && { echo "$i"; return; }; done; echo ""; }
cost_of() { case "$1" in 256) echo 54 ;; 64) echo 16 ;; 16) echo 12 ;; *) echo 11 ;; esac; }   # ~ single-run s/step x10
least_loaded_gpu() { local g best=0; for ((g = 1; g < GPUS; g++)); do
    if [ "${LOAD[g]}" -lt "${LOAD[best]}" ] || { [ "${LOAD[g]}" -eq "${LOAD[best]}" ] && [ "${NRUN[g]}" -lt "${NRUN[best]}" ]; }; then best=$g; fi
  done; echo "${best}"; }
kill_tree() { local c; for c in $(pgrep -P "$1" || true); do kill_tree "$c"; done; kill -TERM "$1" 2>/dev/null || true; }
kill_ray_band() {  # the run's Ray head + workers are daemonized (not its descendants): match its unique temp dir
  ps -eo pid=,args= | awk -v p="ray_textmaze_p$1_gpu${RUN_GPU_TAG}/" 'index($0, p) {print $1}' | xargs -r kill -9 2>/dev/null || true; }
start_run() {  # start_run <spec> <slot> [gpu]
  local spec="$1" slot="$2" lvl sb b seed suite d base src g
  IFS='.' read -r lvl sb b seed <<< "${spec}"
  suite="${SP}-${lvl}-b${b}-s${seed}"; d="$(run_dir "${suite}" "${b}" "${seed}")"
  if aws s3 ls "${OUT_S3}/runs/${suite}/COMPLETE.json" >/dev/null 2>&1; then echo "SKIP_COMPLETE ${suite}"; return 1; fi
  base=/tmp/maze/bases/${lvl}_${sb}
  src="${SMALL_S3}/small_ds0_is${sb#s}/tune-${lvl}/best"; [ -n "${BASES_S3:-}" ] && src="${BASES_S3}/${lvl}_${sb}"
  [ -f "${base}/ckpt-0/model.safetensors" ] || aws s3 sync --only-show-errors "${src}" "${base}/ckpt-0"
  if [ ! -f "${d}/latest_checkpointed_iteration.txt" ] && aws s3 ls "${OUT_S3}/runs/${suite}/state/latest_checkpointed_iteration.txt" >/dev/null 2>&1; then
    mkdir -p "${d}"; local step
    aws s3 cp --only-show-errors "${OUT_S3}/runs/${suite}/state/latest_checkpointed_iteration.txt" "${d}/latest_checkpointed_iteration.txt"
    step="$(cat "${d}/latest_checkpointed_iteration.txt")"
    aws s3 sync --only-show-errors "${OUT_S3}/runs/${suite}/state/global_step_${step}" "${d}/global_step_${step}"
    # a checkpoint from another world size (1-GPU vs RUN_ALL_GPUS run of the same suite) cannot be loaded: refuse it
    if [ ! -f "${d}/global_step_${step}/actor/model_world_size_${RUN_NGPU}_rank_0.pt" ]; then
      echo "STATE_WORLD_SIZE_MISMATCH ${suite}: S3 step ${step} was not written by ${RUN_NGPU} GPU(s); not resuming it"
      rm -rf "${d}"; return 1
    fi
    aws s3 sync --only-show-errors "${OUT_S3}/runs/${suite}/state/best_hf" "${d}/best_hf" || true
    for f in best.json wandb_id.txt; do aws s3 cp --only-show-errors "${OUT_S3}/runs/${suite}/state/${f}" "${d}/${f}" 2>/dev/null || true; done
    echo "RESUME ${suite} from step ${step}"
  fi
  g="${3:-$(least_loaded_gpu)}"; [ "${RUN_ALL_GPUS:-0}" = 1 ] && g=0
  local phys="${g}"; [ "${RUN_ALL_GPUS:-0}" = 1 ] && phys="${RUN_GPU_IDS}"
  ( export SUITE="${suite}" BATCH="${b}" SEED="${seed}" GPU="${phys}" SLOT="${slot}" BASE_DIR="${base}" WANDB_RUN_ID="${suite}-${WANDB_ID_SUFFIX:-v1}"
    exec bash "${M}/rl_tis_run.sh" ) >> "${LOGS}/${suite}.log" 2>&1 &
  PID[$slot]=$!; SUITE_OF[$slot]="${suite}"; DIR_OF[$slot]="${d}"; SPEC_OF[$slot]="${spec}"
  GPU_OF[$slot]="${g}"; COST_OF[$slot]="$(cost_of "${b}")"; STARTED_AT[$slot]="$(date +%s)"
  LOAD[g]=$(( LOAD[g] + COST_OF[$slot] )); NRUN[g]=$(( NRUN[g] + 1 ))
  echo "STARTED ${suite} slot=${slot} gpu=${g} pid=${PID[$slot]} gpu_loads=${LOAD[*]} $(date -u +%FT%TZ)"
}
suite_of_dir() { basename "$1" | sed -E "s/^textmaze_(${SP}-[^_]+)_.*/\1/"; }
# every new checkpoint (one per validation, every 10 steps) is mirrored within CKPT_POLL_SECONDS of being written, so a
# preempted pod loses at most the steps since the last checkpoint; logs are mirrored every SYNC_SECONDS
( last_logs=0
  while sleep "${CKPT_POLL_SECONDS:-15}"; do
    for d in "${CK}"/textmaze_"${SP}"-*; do
      [ -d "${d}" ] || continue
      step="$(cat "${d}/latest_checkpointed_iteration.txt" 2>/dev/null || true)"
      [ -n "${step}" ] && [ "${step}" != "$(cat "${d}/.pushed_step" 2>/dev/null)" ] || continue
      push_state "$(suite_of_dir "${d}")" "${d}" && echo "${step}" > "${d}/.pushed_step"
    done
    if [ $(( $(date +%s) - last_logs )) -ge "${SYNC_SECONDS}" ]; then
      last_logs=$(date +%s)
      for l in "${LOGS}"/"${SP}"-*.log; do [ -f "${l}" ] || continue; s="$(basename "${l}" .log)"; aws s3 cp --only-show-errors "${l}" "${OUT_S3}/runs/${s}/logs/${s}.$(hostname).log" || true; done
    fi
  done ) & sidecar=$!
# preemption: if SIGTERM reaches this script, mirror every run's latest checkpoint once more before the pod goes
final_push() {
  echo "SIGTERM $(date -u +%FT%TZ): final checkpoint push"
  local d pids=()
  for d in "${CK}"/textmaze_"${SP}"-*; do [ -d "${d}" ] && { push_state "$(suite_of_dir "${d}")" "${d}" & pids+=($!); }; done
  wait "${pids[@]}" 2>/dev/null; echo "SIGTERM_PUSHED ${#pids[@]} runs"; exit 143
}
trap final_push TERM

while :; do
  # reap finished runs
  for ((i = 0; i < SLOTS; i++)); do
    p="${PID[$i]:-}"; [ -n "${p}" ] || continue
    kill -0 "${p}" 2>/dev/null && continue
    rc=0; wait "${p}" || rc=$?; s="${SUITE_OF[$i]}"; d="${DIR_OF[$i]}"; g="${GPU_OF[$i]}"
    LOAD[g]=$(( LOAD[g] - COST_OF[$i] )); NRUN[g]=$(( NRUN[g] - 1 ))
    if [ -n "${MIGRATE_TO[$s]:-}" ] && [ ! -f "${d}/COMPLETE.json" ]; then   # stopped on purpose: resume on the idle GPU
      kill_ray_band "${i}"
      PREF_GPU[${SPEC_OF[$i]}]="${MIGRATE_TO[$s]}"; queue=("${SPEC_OF[$i]}" "${queue[@]}")
      echo "MIGRATE ${s} gpu ${g} -> ${MIGRATE_TO[$s]} (resumes from step $(cat "${d}/latest_checkpointed_iteration.txt" 2>/dev/null)) $(date -u +%FT%TZ)"
      unset "MIGRATE_TO[$s]"; migrating=0; PID[$i]=""; continue
    fi
    pushed=0   # a transient S3 error must not end the pod (set -e) with the other runs mid-training: retry, then move on
    for _ in 1 2 3; do push_state "${s}" "${d}" && { pushed=1; break; }; sleep 30; done
    [ "${pushed}" = 1 ] || echo "PUSH_FAILED ${s} $(date -u +%FT%TZ)"
    if [ -f "${d}/COMPLETE.json" ] && [ "${pushed}" = 1 ] \
        && aws s3 cp --only-show-errors "${d}/COMPLETE.json" "${OUT_S3}/runs/${s}/COMPLETE.json"; then
      echo "DONE ${s} rc=${rc} $(tr -d '\n ' < "${d}/COMPLETE.json") $(date -u +%FT%TZ)"
      rm -rf "${d}"/global_step_*   # mirrored; free local disk
    else
      TRIES[$s]=$(( ${TRIES[$s]:-0} + 1 ))
      echo "RUN_FAILED ${s} rc=${rc} try=${TRIES[$s]} $(tail -c 300 "${LOGS}/${s}.log" | tr '\n' ' ')"
      # the failed run's Ray head + workers are daemonized and outlive it; a retry in the same slot then dies at once
      # ("Connection lost", "resource_tracker died"), so kill them before the run is re-queued
      kill_ray_band "${i}"; sleep 5
      [ "${TRIES[$s]}" -le "${RUN_RETRIES}" ] && queue+=("${SPEC_OF[$i]}")
    fi
    PID[$i]=""
  done
  # fill free slots
  while [ "${#queue[@]}" -gt 0 ]; do
    slot="$(free_slot)"; [ -n "${slot}" ] || break
    spec="${queue[0]}"; queue=("${queue[@]:1}")
    start_run "${spec}" "${slot}" "${PREF_GPU[${spec}]:-}" || true
    unset "PREF_GPU[${spec}]"
    sleep 20 & wait $!   # stagger Ray head starts (interruptible, so SIGTERM is handled at once)
  done
  # keep every GPU busy: an idle GPU takes the costliest run (started > 15 min ago) from a GPU that still has >= 2
  if [ "${#queue[@]}" -eq 0 ] && [ "${GPUS}" -gt 1 ] && [ "${migrating}" -eq 0 ] && [ "${RUN_ALL_GPUS:-0}" != 1 ]; then
    idle=""; for ((g = 0; g < GPUS; g++)); do [ "${NRUN[g]}" -eq 0 ] && { idle=$g; break; }; done
    if [ -n "${idle}" ]; then
      victim=""; vcost=-1; now=$(date +%s)
      for ((i = 0; i < SLOTS; i++)); do
        [ -n "${PID[$i]:-}" ] || continue; g="${GPU_OF[$i]}"
        [ "${NRUN[g]}" -ge 2 ] && [ $(( now - STARTED_AT[$i] )) -gt "${REBALANCE_MIN_AGE:-900}" ] || continue
        score=$(( LOAD[g] * 100 + COST_OF[$i] )); [ "${score}" -gt "${vcost}" ] && { vcost=${score}; victim=$i; }
      done
      if [ -n "${victim}" ]; then
        MIGRATE_TO[${SUITE_OF[$victim]}]="${idle}"; migrating=1
        echo "REBALANCE gpu ${idle} idle: moving ${SUITE_OF[$victim]} off gpu ${GPU_OF[$victim]} (loads ${LOAD[*]}, runs ${NRUN[*]})"
        kill_tree "${PID[$victim]}"
      fi
    fi
  fi
  busy=0; for ((i = 0; i < SLOTS; i++)); do [ -n "${PID[$i]:-}" ] && busy=1; done
  [ "${busy}" = 0 ] && [ "${#queue[@]}" -eq 0 ] && break
  sleep 30 & wait $!
done
kill "${sidecar}" "${podstat}" 2>/dev/null || true
missing=0; IFS='+' read -r -a all <<< "${RUNS}"
for spec in "${all[@]}"; do IFS='.' read -r lvl sb b seed <<< "${spec}"
  aws s3 ls "${OUT_S3}/runs/${SP}-${lvl}-b${b}-s${seed}/COMPLETE.json" >/dev/null 2>&1 || { missing=$((missing + 1)); echo "NOT_COMPLETE ${SP}-${lvl}-b${b}-s${seed}"; }; done
if [ "${missing}" = 0 ]; then
  echo "{\"done\": \"$(date -u +%FT%TZ)\", \"runs\": \"${RUNS}\"}" | aws s3 cp - "${OUT_S3}/pods/${TAG}.COMPLETE"; echo "MAZE_RL_POD_DONE ${TAG}"
else
  echo "MAZE_RL_POD_INCOMPLETE ${TAG} missing=${missing}"; exit 1
fi
