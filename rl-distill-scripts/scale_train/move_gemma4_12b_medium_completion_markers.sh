#!/usr/bin/env bash
# The untrained-12B medium run finished by early stopping (step 130) and left durable completion markers; the run-file's
# preflight treats them as "already complete" and exits 0. Move them aside (reversibly) before resuming. Reversal:
# swap the two prefixes in the mv commands below.
set -euo pipefail
S3_BASE="${S3_BASE:-s3://scale-ml/genai/rl-distill/gemma4-difficulty-s42-20260819}"
STAMP="${STAMP:-pre-resume-20260915}"
export AWS_PROFILE="${AWS_PROFILE:-ml-worker}"
F="${S3_BASE}-full-checkpoints/12b-medium"; A="${S3_BASE}/gemma4-12b-medium"
aws s3 mv "$F/run_complete.json"            "$F/${STAMP}/run_complete.json"
aws s3 mv "$A/run_outcome.json"             "$A/${STAMP}/run_outcome.json"
aws s3 mv "$A/best_hf/_REMOTE_COMPLETE.json" "$A/${STAMP}/best_hf__REMOTE_COMPLETE.json"
echo "moved completion markers under ${STAMP}/; remaining:"; aws s3 ls "$F/" | grep -v global_step; aws s3 ls "$A/"
