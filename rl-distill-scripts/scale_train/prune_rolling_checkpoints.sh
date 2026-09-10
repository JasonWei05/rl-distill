#!/usr/bin/env bash
# Prune stale rolling S3 checkpoints of the distillation runs from a box whose profile may delete objects.
# ScaleTrain pods run as eks-ml-worker2, which has no s3:DeleteObject, so the trainer's own pruning/retiring of the rolling
# slot is "deferred" (logged, harmless) and stale rolling steps (~170 GB per 12B step, ~420 GB per 26B step) pile up.
# Rule per run prefix: read rolling/latest_checkpointed_iteration.txt = N; delete rolling/global_step_M/ for every M < N
# (never the current step, never a higher step that may be mid-upload); if the permanent tracker P >= N the rolling slot is
# superseded: delete its tracker, then all its steps. Permanent global_step_*/ dirs are never touched.
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
    if [[ -n "${N}" && -n "${P}" && "${P}" -ge "${N}" ]]; then
      echo "  rolling slot superseded by permanent step ${P}: retiring"
      [[ "${DRY_RUN}" == "true" ]] || aws s3 rm --only-show-errors "${prefix}/rolling/latest_checkpointed_iteration.txt"
      for M in ${steps}; do rm_prefix "${prefix}/rolling/global_step_${M}/"; done
      continue
    fi
    for M in ${steps}; do
      if [[ -n "${N}" && "${M}" -lt "${N}" ]]; then rm_prefix "${prefix}/rolling/global_step_${M}/"; fi
    done
  done
  done
}
if [[ "${1:-once}" == "loop" ]]; then
  while true; do pass || echo "[$(date -u +%FT%TZ)] pass failed (retrying next interval)"; sleep "$(( ${PRUNE_INTERVAL_MIN:-30} * 60 ))"; done
else
  pass
fi
