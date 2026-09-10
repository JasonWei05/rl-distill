#!/usr/bin/env bash
# Prune stale rolling S3 checkpoints of the distillation runs from a box whose profile may delete objects.
# ScaleTrain pods run as eks-ml-worker2, which has no s3:DeleteObject, so the trainer's own pruning/retiring of the rolling
# slot is "deferred" (logged, harmless) and stale rolling steps (~170 GB per 12B step, ~420 GB per 26B step) pile up.
# Rule per run prefix: with rolling tracker N and permanent tracker P, delete rolling/global_step_M/ only if M < N or M <= P,
# and only if its _REMOTE_COMPLETE.json exists (never an upload in flight, never a step newer than both trackers). The rolling
# tracker is removed only when it is <= P and no newer rolling step exists. Permanent global_step_*/ dirs are never touched.
#   bash prune_rolling_checkpoints.sh            # one pass over CKPT_S3_ROOTS (distill + RL checkpoint roots)
#   PRUNE_INTERVAL_MIN=30 bash prune_rolling_checkpoints.sh loop
set -euo pipefail
export AWS_PROFILE="${AWS_PROFILE:-ml-worker}"
# One or more checkpoint roots (space-separated); each holds <run>/{global_step_*, rolling/, latest_checkpointed_iteration.txt}.
ROOTS="${CKPT_S3_ROOTS:-s3://scale-ml/genai/rl-distill/gemma4-e4b-base-distill-ckpts-v1 s3://scale-ml/genai/rl-distill/gemma4-12b-from-e4bbase-distill-rl-full-checkpoints s3://scale-ml/genai/rl-distill/gemma4-difficulty-s43-20260910-full-checkpoints}"
DRY_RUN="${DRY_RUN:-false}"
rm_prefix() { if [[ "${DRY_RUN}" == "true" ]]; then echo "  (dry-run) rm -r $1"; else aws s3 rm --recursive --only-show-errors "$1" && echo "  removed $1"; fi; }
pass() {
  for ROOT in ${ROOTS}; do
  for run in $(aws s3 ls "${ROOT}/" 2>/dev/null | awk '/PRE/ {print $2}' | grep -v '^_'); do
    prefix="${ROOT}/${run%/}"
    N="$(aws s3 cp --only-show-errors "${prefix}/rolling/latest_checkpointed_iteration.txt" - 2>/dev/null | tr -dc 0-9 || true)"
    P="$(aws s3 cp --only-show-errors "${prefix}/latest_checkpointed_iteration.txt" - 2>/dev/null | tr -dc 0-9 || true)"
    steps="$(aws s3 ls "${prefix}/rolling/" 2>/dev/null | awk '/PRE global_step_/ {print $2}' | sed -E 's#global_step_([0-9]+)/#\1#' | sort -n || true)"
    [[ -z "${steps}" ]] && continue
    echo "[$(date -u +%FT%TZ)] ${run%/}: rolling=${N:-none} permanent=${P:-none} rolling_steps=$(echo ${steps} | tr ' ' ,)"
    # SAFETY (2026-09-10 incident): a pod cannot delete, so after a permanent save the rolling tracker stays stale *below* the
    # permanent step while the trainer keeps uploading newer rolling steps. Never delete a step newer than both trackers, and
    # never delete a step without its completion marker (an upload in flight): the trainer verifies every uploaded object
    # right after the upload and a concurrent delete kills the run (RL) or drops the checkpoint (distill).
    for M in ${steps}; do
      stale=false
      if [[ -n "${N}" && "${M}" -lt "${N}" ]]; then stale=true; fi          # superseded by a newer rolling step
      if [[ -n "${P}" && "${M}" -le "${P}" ]]; then stale=true; fi          # superseded by a permanent checkpoint
      [[ "${stale}" == "true" ]] || continue
      if ! aws s3 ls "${prefix}/rolling/global_step_${M}/_REMOTE_COMPLETE.json" >/dev/null 2>&1; then
        echo "  keep rolling/global_step_${M}: no completion marker (upload in flight?)"; continue
      fi
      rm_prefix "${prefix}/rolling/global_step_${M}/"
    done
    # Drop the rolling tracker only when it points at a step we removed (<= permanent) and nothing newer remains.
    if [[ -n "${N}" && -n "${P}" && "${N}" -le "${P}" ]]; then
      newer=$(for M in ${steps}; do [[ "${M}" -gt "${P}" ]] && echo "${M}"; done)
      if [[ -z "${newer}" ]]; then
        echo "  rolling tracker (${N}) <= permanent (${P}) and no newer rolling step: retiring tracker"
        [[ "${DRY_RUN}" == "true" ]] || aws s3 rm --only-show-errors "${prefix}/rolling/latest_checkpointed_iteration.txt"
      fi
    fi
  done
  done
}
if [[ "${1:-once}" == "loop" ]]; then
  while true; do pass || echo "[$(date -u +%FT%TZ)] pass failed (retrying next interval)"; sleep "$(( ${PRUNE_INTERVAL_MIN:-30} * 60 ))"; done
else
  pass
fi
