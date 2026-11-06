#!/usr/bin/env bash
# Launch or extend one RL run of the extra-levels runbook on federer through launch_single_rl.sh (Step 3 cadence).
#   [SIDE=L0] bash rl_launch.sh <pair> <seed> <steps> <gpu>        (SIDE defaults to S0)
# Two things the plain launcher does not handle here:
#   * the run dir name embeds the GPU id, so resuming on another (empty) GPU first aliases the existing run dir under the
#     new name (a symlink; resume_mode=auto then continues from its newest checkpoint, and the janitor skips the alias);
#   * the launcher truncates rl/<tag>_<SIDE>_rl.log, so a previous log is rotated to .part<K> first; assemble_rl_logs.sh
#     concatenates the parts before rl_curve_family.py extract.
#   * inference-only memory knobs (generation in 2000-sequence chunks, old-log-prob forward in 1024-sequence micro-batches,
#     expandable segments): ~12 GB per S run instead of 27 GB, so runs can share a card; PPO_MICRO_BATCH stays at its
#     default (the per-micro-batch token-mean makes it change the update);
#   * RAY_PORT_INDEX (0-7) must be unique among concurrent runs (it defaults to the GPU id, which two runs on one card share).
set -euo pipefail
source /work2/jiahaoz4/maze_extra/env.sh
PAIR=${1:?pair}; SEED=${2:?seed}; STEPS=${3:?steps}; GPU=${4:?gpu}; SIDE=${SIDE:-S0}
case "$SIDE" in S0) SL=s0 ;; L0) SL=l0 ;; *) echo "SIDE must be S0 or L0" >&2; exit 1 ;; esac
: "${RAY_PORT_INDEX:?set RAY_PORT_INDEX (0-7), unique among running RL jobs}"
# GPU rule (user): never start on a card another user has a process on
BUS=$(nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader | awk -F', ' -v g="$GPU" '$1 == g {print $2}')
OTHERS=$(nvidia-smi --query-compute-apps=gpu_bus_id,pid --format=csv,noheader | awk -F', ' -v b="$BUS" '$1 == b {print $2}' |
  while read -r p; do u=$(ps -o user= -p "$p" 2>/dev/null || true); [ -n "$u" ] && [ "$u" != "$(id -un)" ] && echo "$u:$p"; done || true)
if [ -n "$OTHERS" ]; then echo "GPU $GPU is in use by another user ($OTHERS); not launching" >&2; exit 1; fi
# CHECK_ONLY=1: only the check above (use it before stopping a run you mean to move to this card)
if [ "${CHECK_ONLY:-0}" = 1 ]; then echo "GPU $GPU: no other user's process"; exit 0; fi
export RAY_PORT_INDEX ROLLOUT_MICRO_BATCH_SIZE=${ROLLOUT_MICRO_BATCH_SIZE:-2000} LOG_PROB_MICRO_BATCH=${LOG_PROB_MICRO_BATCH:-1024} \
  PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True} \
  VERL_VAL_SKIP_BOOTSTRAP=${VERL_VAL_SKIP_BOOTSTRAP:-1}   # no best@n/worst@n bootstrap in validation (mean@16 unchanged)
TAG=$PAIR; [ "$SEED" != 0 ] && TAG="${PAIR}s${SEED}"
CK=$HOME/tailrl_maze_runs/checkpoints/tailrl-text-maze
NEW=$CK/textmaze_${TAG}-${SL}_grpo_binary_ckpt0_bs256_G16_1gpu_${GPU}_seed${SEED}
if [ ! -e "$NEW" ]; then
  OLD=$(find "$CK" -maxdepth 1 -type d -name "textmaze_${TAG}-${SL}_grpo_binary_ckpt0_bs256_G16_1gpu_*_seed${SEED}" 2>/dev/null | head -1)
  if [ -n "$OLD" ]; then ln -s "$(basename "$OLD")" "$NEW"; echo "aliased $(basename "$NEW") -> $(basename "$OLD")"; fi
fi
LOG=$HOME/tailrl_maze_runs/rl/${TAG}_${SIDE}_rl.log
if [ -s "$LOG" ]; then K=1; while [ -e "$LOG.part$K" ]; do K=$((K + 1)); done; mv "$LOG" "$LOG.part$K"; echo "rotated log -> $LOG.part$K"; fi
cd /work2/jiahaoz4/rl-distill
STEPS=$STEPS bash $M/launch_single_rl.sh "$PAIR" "$SIDE" "$SEED" "$GPU"
# record how this run was launched; watchdog.sh relaunches it from this file if it dies before STEPS
SUITE="${TAG}-${SL}"
printf "SIDE=%s\nPAIR=%s\nSEED=%s\nSTEPS=%s\nGPU=%s\nPORT=%s\nROLLOUT=%s\nLOGPROB=%s\nLAUNCHED=%s\n" "$SIDE" "$PAIR" "$SEED" "$STEPS" "$GPU" \
  "$RAY_PORT_INDEX" "$ROLLOUT_MICRO_BATCH_SIZE" "$LOG_PROB_MICRO_BATCH" "$(date -u +%FT%TZ)" > /work2/jiahaoz4/maze_extra/runs/$SUITE.env
rm -f /work2/jiahaoz4/maze_extra/runs/$SUITE.done
