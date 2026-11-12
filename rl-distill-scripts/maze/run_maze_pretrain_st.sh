#!/usr/bin/env bash
# ScaleTrain pod entry (GPUS GPUs): one model size of the shortest-path maze study, end to end (maze_pt_pipeline.py):
# pretrain SEEDS from scratch on the shortest-path-only SFT data, screen every SCREEN_STRIDE-th checkpoint as it lands,
# and tune + confirm a base at each of the 5 is_shortest pass@1 levels per seed -- all asynchronously on the pod's GPUs
# under one CUDA MPS daemon. Everything under /tmp/maze/pt is mirrored to ${OUT_S3}/${SIZE} every SYNC_SECONDS (a relaunch
# restores it: finished levels are kept; a run whose levels are not all done retrains under a new attempt name).
# Writes ${OUT_S3}/${SIZE}/COMPLETE when every level of every seed is done.
#   env: SIZE (3XS|2XS|XS|S|L), OUT_S3, SEEDS ("0 1 2"), GPUS (2), MAX_STEPS (10000), STOP_AT (0.20), SYNC_SECONDS (120),
#        WANDB_PROJECT_NAME (tailrl-text-maze-shortest), EXTRA_PIPELINE_ARGS,
#        DATA_S3 (pretraining train / test JSON; default the shortest-path-only data), TRAIN_JSON (train_optimal.json),
#        EVAL_S3 (dir with the dev-maze test.parquet for screening and tuning; default TailRL's committed eval1000),
#        MAX_LENGTH (512; trainer sequence cap = max_position_embeddings), MAZE_MAX_NEW_TOKENS (180; generation cap),
#        TUNE_FINE_MODE (fine | confirm, tune_pass1.py --fine-mode), CENTER_STEP (10; spacing of the averaged candidates),
#        LR (5e-4), SAVE_STEPS (5), NO_TUNE (set: train + screen only)
#   A pre-tokenized copy ${DATA_S3}/<TRAIN_JSON stem>.tok-L<MAX_LENGTH>/ (maze/tokenize_maze_sft.py) is synced with the data and
#   used by every trainer instead of tokenizing TRAIN_JSON (~35 min with 5 trainers on one pod)
source "$(dirname "${BASH_SOURCE[0]}")/st_setup.sh"
SIZE="${SIZE:?}"; OUT_S3="${OUT_S3:?}"; SEEDS="${SEEDS:-0 1 2}"; SEEDS="${SEEDS//[-:]/ }"   # 0-1-2-3-4 (env-var lists cannot hold spaces)
GPUS="${GPUS:-2}"; SYNC_SECONDS="${SYNC_SECONDS:-120}"; MAX_LENGTH="${MAX_LENGTH:-512}"
DATA_S3="${DATA_S3:-s3://scale-ml/genai/rl-distill/maze/data/sft_optimal}"; TRAIN_JSON="${TRAIN_JSON:-train_optimal.json}"
ROOT=/tmp/maze/pt; D=/tmp/maze/data/pt; mkdir -p "${ROOT}" "${D}"
echo "MAZE_PT_START $(date -u +%FT%TZ) host=$(hostname) size=${SIZE} seeds=${SEEDS} gpus=${GPUS}"
aws s3 sync --only-show-errors "${DATA_S3}" "${D}"
if [ -n "${EVAL_S3:-}" ]; then
  aws s3 sync --only-show-errors "${EVAL_S3}" /tmp/maze/data/eval; export MAZE_EVAL_PARQUET=/tmp/maze/data/eval/test.parquet
fi
aws s3 sync --only-show-errors "${OUT_S3}/${SIZE}" "${ROOT}" --exclude "*.tmp*" || true   # earlier attempt, if any
CACHE="${D}/${TRAIN_JSON%.json}.tok-L${MAX_LENGTH}"   # pre-tokenized TRAIN_JSON, synced with DATA_S3 when it exists
echo "MAZE_PT_DATA train=$(du -h "${D}/${TRAIN_JSON}" | cut -f1) cache=$(ls -d "${CACHE}" 2>/dev/null || echo none) eval=${MAZE_EVAL_PARQUET:-tailrl-eval1000} restored_runs=$(ls -d "${ROOT}/${SIZE}"_is* 2>/dev/null | wc -l)"
[ -d "${CACHE}" ] || echo "MAZE_PT_NO_CACHE ${CACHE}: every trainer tokenizes ${TRAIN_JSON} itself (~35 min for the 25x25 set; maze/tokenize_maze_sft.py builds the cache)"
export CUDA_MPS_PIPE_DIRECTORY=/tmp/maze/mps_pipe CUDA_MPS_LOG_DIRECTORY=/tmp/maze/mps_log
mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"
if nvidia-cuda-mps-control -d; then echo "MPS_ON"; else echo "MPS_FAILED"; unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY; fi
export WANDB_ENTITY=rl-distill WANDB_MODE=online HOME=/tmp/maze
# trainers, screeners and level tasks share each GPU: cap sequences per generate call (L: ~14 GB per process at 2048, 8192 peaks at 40-60 GB -> OOM)
export SCREEN_MAX_SEQS="${SCREEN_MAX_SEQS:-1024}"
# mirror: screens, selections, tune.json + each level's chosen base (best/), logs, every 100th checkpoint, and all the
# checkpoints a finished run's level tasks need (sync_ckpts); averaged / interpolated candidates are not mirrored
push() { aws s3 sync --only-show-errors "${ROOT}" "${OUT_S3}/${SIZE}" --exclude "*.tmp*" --exclude "*/wandb/*" \
  --exclude "*/ckpt-*" --include "*/ckpt-*00/*" --exclude "*/avg-*" --exclude "*/interp-*"; }
# a trained run whose levels are all queued: mirror its checkpoints up to SCREEN_UNTIL + 250 steps once (every level's
# candidate window), then mark it CKPTS_SYNCED so a relaunch restores and reuses it instead of retraining
sync_ckpts() { local r c s lim inc
  for r in "${ROOT}/${SIZE}"_is*; do
    # only runs trained on this pod: runs restored from S3 hold just every 100th checkpoint
    { [ -f "${r}/DONE" ] && [ -f "${r}/SCREEN_UNTIL" ] && [ ! -f "${r}/CKPTS_SYNCED" ] \
      && [ "$(cat "${r}/HOST" 2>/dev/null)" = "$(hostname)" ]; } || continue
    lim=$(( $(cat "${r}/SCREEN_UNTIL") + 250 )); inc=()
    for c in "${r}"/ckpt-*; do s="${c##*-}"; [[ "${s}" =~ ^[0-9]+$ ]] && [ "${s}" -le "${lim}" ] && inc+=(--include "ckpt-${s}/*"); done
    aws s3 sync --only-show-errors "${r}" "${OUT_S3}/${SIZE}/$(basename "${r}")" --exclude "*" "${inc[@]}" --exclude "*.tmp*" \
      && touch "${r}/CKPTS_SYNCED" && echo "CKPTS_SYNCED $(basename "${r}") ${#inc[@]} checkpoints up to ${lim}"
  done; }
( while sleep "${SYNC_SECONDS}"; do sync_ckpts || true; push || true; done ) & sidecar=$!
( while sleep 60; do echo "PODSTAT $(date -u +%T) load=$(cut -d' ' -f1 /proc/loadavg) gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr -d ' ' | paste -sd';')"; done ) & podstat=$!
rc=0
"${PY}" "${M}/maze_pt_pipeline.py" --size "${SIZE}" --seeds ${SEEDS} --gpus "${GPUS}" --root "${ROOT}" \
  --train-data "${D}/${TRAIN_JSON}" --train-cache "${CACHE}" \
  --val-data "${D}/test.json" --tailrl-root "${T}" --python "${PY}" \
  --attempt "a$(date -u +%m%d%H%M)" --max-steps "${MAX_STEPS:-10000}" --max-length "${MAX_LENGTH}" --fine-mode "${TUNE_FINE_MODE:-fine}" --center-step "${CENTER_STEP:-10}" --stop-at "${STOP_AT:-0.20}" \
  --lr "${LR:-5e-4}" --save-steps "${SAVE_STEPS:-5}" ${NO_TUNE:+--no-tune} \
  --wandb-project "${WANDB_PROJECT_NAME:-tailrl-text-maze-shortest}" ${EXTRA_PIPELINE_ARGS:-} || rc=$?
kill "${sidecar}" "${podstat}" 2>/dev/null || true
push
if [ "${rc}" = 0 ]; then
  echo "{\"done\": \"$(date -u +%FT%TZ)\", \"size\": \"${SIZE}\"}" | aws s3 cp - "${OUT_S3}/${SIZE}/COMPLETE"; echo "MAZE_PT_DONE ${SIZE}"
else
  echo "MAZE_PT_INCOMPLETE ${SIZE} rc=${rc}"; exit 1
fi
