#!/usr/bin/env bash
# Every 60s: back up the gemma4-8k ScaleTrain pod logs to EFS and refresh the wandb
# recovery run for E4B-8k, whose wandb uploader died mid-run on 2026-07-27 (metrics
# only reach the server via pod-log parsing; see recover_wandb_from_podlog.py and
# PROGRESS_LOG.md). Re-logging is idempotent: wandb drops steps it already has, so
# each cycle appends only new steps. Run inside tmux: tmux new -s wandb-recover ...
set -u
cd /mnt/efs/jasonwei/rl-distill
source .env
OUT=/mnt/efs/jasonwei/rl-distill/artifacts/gemma4_8k_logs
mkdir -p "$OUT"

while true; do
  ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  for m in e2b e4b; do
    pod=$(kubectl --context ir-ml-gpu-batch -n train get pods 2>/dev/null \
          | grep "gemma4-${m}-ds4of4s-8k" | grep Running | awk '{print $1}' | head -1)
    if [ -n "${pod}" ]; then
      kubectl --context ir-ml-gpu-batch -n train logs "${pod}" > "$OUT/${m}8k_pod.log.tmp" 2>/dev/null \
        && mv "$OUT/${m}8k_pod.log.tmp" "$OUT/${m}8k_pod.log"
    fi
  done
  if [ -s "$OUT/e4b8k_pod.log" ]; then
    WANDB_BASE_URL=https://api.wandb.ai WANDB_API_KEY="${WANDB_API_KEY}" \
      .venv-gemma4/bin/python rl-distill-scripts/recover_wandb_from_podlog.py \
      "$OUT/e4b8k_pod.log" \
      "DAPO-gemma4-e4b-pt-DeepScaleR-4of4strict-seed42-8k-recovered" rec1tyrqscv \
      2>&1 | grep -aE "parsed|RECOVERY_SYNCED|Error" || true
  fi
  # E2B's uploader died the same way at step 98 (2026-07-27 ~22:30Z)
  if [ -s "$OUT/e2b8k_pod.log" ]; then
    WANDB_BASE_URL=https://api.wandb.ai WANDB_API_KEY="${WANDB_API_KEY}" \
      .venv-gemma4/bin/python rl-distill-scripts/recover_wandb_from_podlog.py \
      "$OUT/e2b8k_pod.log" \
      "DAPO-gemma4-e2b-pt-DeepScaleR-4of4strict-seed42-8k-recovered" recbw9dcxso \
      2>&1 | grep -aE "parsed|RECOVERY_SYNCED|Error" || true
  fi
  echo "[$ts] cycle done"
  sleep 60
done
