#!/usr/bin/env bash
# One-shot: switch the in-flight q30bh-hard trace run (8 GPUs, TP 2 x 4 engines) to TP 1 x 8 engines without losing work:
# waits until shard batch MIN_SHARDS has landed on S3 (the 4 TP-2 workers finish shards in lockstep), stops that supervisor,
# cancels exactly its job, and relaunches with TRACE_PIN_ENGINE_IDENTITY=1 so the completed shards stay valid.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
MIN_SHARDS="${MIN_SHARDS:-20}"; sup=.scale_train_supervisors/q3-30bh-dist-4b-20261009-g8
S3=s3://scale-ml/genai/rl-distill/qwen3-30b-rl-traces-topk128-v1/q30bh-hard/train/
log() { echo "$(date -u +%FT%TZ) [cutover-tp1] $*"; }
until [ "$(AWS_PROFILE=ml-worker aws s3 ls $S3 | grep -c '\.parquet$')" -ge "${MIN_SHARDS}" ]; do sleep 30; done
sleep 60   # let the last shard's manifest upload finish
job=$(python3 -c "import json; print(json.load(open('${sup}/state.json'))['job_id'])")
log "$(AWS_PROFILE=ml-worker aws s3 ls $S3 | grep -c '\.parquet$') shards on S3; stopping supervisor and cancelling ${job}"
touch "${sup}/STOP"; tmux kill-session -t rl-q3-30bh-dist-4b 2>/dev/null || true
env -u AWS_PROFILE scale-train cancel job "${job}" 2>&1 | grep -ioE "cancel[a-z]*" | head -1 || true
sleep 90
log "relaunching: TP 1 x 8 engines, MAX_NUM_SEQS=48, GPU_MEMORY_UTILIZATION=0.92, pinned identity, 16 validation workers"
MAX_RELAUNCHES=200 FAILURE_BACKOFF_SECONDS=3600 MAX_FAILURE_BACKOFF_SECONDS=3600 TEACHER_SPEC=q30bh-hard GPUS=8 ALLOW_BORROWING=false SUP_TAG=-tp1 \
  EXTRA_ENV_VARS=TRACE_TP=1,MAX_NUM_SEQS=48,GPU_MEMORY_UTILIZATION=0.92,TRACE_PIN_ENGINE_IDENTITY=1,TRACE_VALIDATION_WORKERS=16 \
  bash rl-distill-scripts/scale_train/start_qwen3_30bm_to_4b_distill_supervisor.sh
log done
