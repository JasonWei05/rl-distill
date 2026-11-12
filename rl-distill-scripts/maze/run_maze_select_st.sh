#!/usr/bin/env bash
# ScaleTrain pod entry (1 GPU): select_matched_init.py for one pretraining run and one starting level, on a run mirrored in S3.
# Pulls ${RUN_S3}/<run> (checkpoints, earlier averaged candidates and selection results), builds averaged candidates over
# [LO, HI] (every CENTER_STEP steps, +-HALF_WINDOW), screens them on the 1000 dev mazes at K samples, re-screens the N_FINAL
# closest to TARGET at FINAL_K, and mirrors the new candidates + select_<LEVEL>_set.json back to S3 every few minutes and at
# the end (a relaunch resumes from what is there). Writes ${RUN_S3}/<run>/select_<LEVEL>_set.COMPLETE when done.
#   env: RUN (e.g. small_ds0_is0), LEVEL (p01|p03|p1|p3|p10), TARGET, LO, HI, K (64), FINAL_K (2048), N_FINAL (2),
#        CENTER_STEP (20), HALF_WINDOW (60), RUN_S3 (s3://scale-ml/genai/rl-distill/maze/pretrain20k_sml)
source "$(dirname "${BASH_SOURCE[0]}")/st_setup.sh"
RUN="${RUN:?}"; LEVEL="${LEVEL:?}"; TARGET="${TARGET:?}"; LO="${LO:?}"; HI="${HI:?}"
K="${K:-64}"; FINAL_K="${FINAL_K:-2048}"; N_FINAL="${N_FINAL:-2}"; CENTER_STEP="${CENTER_STEP:-20}"; HALF_WINDOW="${HALF_WINDOW:-60}"
RUN_S3="${RUN_S3:-s3://scale-ml/genai/rl-distill/maze/pretrain20k_sml}"
echo "MAZE_SELECT_START $(date -u +%FT%TZ) run=${RUN} level=${LEVEL} target=${TARGET} bracket=[${LO},${HI}] k=${K} final_k=${FINAL_K}"
D=/tmp/maze/runs/${RUN}; mkdir -p "${D}"
# checkpoints the bracket's averaging windows need, plus every averaged candidate and selection file already made
inc=(--exclude "*" --include "init_model/*" --include "select_*" --include "avg-*")
for s in $(seq $(( (LO - HALF_WINDOW) / 20 * 20 )) 20 $(( HI + HALF_WINDOW ))); do inc+=(--include "ckpt-${s}/*"); done
aws s3 sync --only-show-errors "${RUN_S3}/${RUN}" "${D}" "${inc[@]}"
echo "MAZE_SELECT_DATA $(ls -d ${D}/ckpt-* | wc -l) checkpoints, $(ls -d ${D}/avg-* 2>/dev/null | wc -l) averaged candidates"
push() { aws s3 sync --only-show-errors "${D}" "${RUN_S3}/${RUN}" --exclude "*" --include "avg-*" --include "select_${LEVEL}_set.json"; }
( while sleep 300; do push || true; done ) & sidecar=$!
(cd /tmp/maze && HOME=/tmp/maze "${PY}" "${M}/select_matched_init.py" --tailrl-root "${T}" --run "${D}" --target "${TARGET}" \
   --bracket "${LO}" "${HI}" --center-step "${CENTER_STEP}" --half-window "${HALF_WINDOW}" --k "${K}" --final-k "${FINAL_K}" \
   --n-final "${N_FINAL}" --n-prompts 1000 --out-name "select_${LEVEL}_set.json")
kill "${sidecar}" 2>/dev/null || true
push
echo "{\"done\": \"$(date -u +%FT%TZ)\", \"target\": ${TARGET}, \"bracket\": [${LO}, ${HI}], \"k\": ${K}, \"final_k\": ${FINAL_K}}" \
  | aws s3 cp - "${RUN_S3}/${RUN}/select_${LEVEL}_set.COMPLETE"
echo "MAZE_SELECT_DONE ${RUN} ${LEVEL}"
