#!/usr/bin/env bash
# §9.0j step 1: top-128 trace generation from the RESUMED untrained-12B medium RL teacher (best step 140, val 0.5271; §9.0h) on the
# full medium band: 3000 train prompts x 16 samples + 300 validation prompts x 1, two local GPUs as DP=2 (TP=1) vLLM workers.
# Teacher weights come from the 4-rank S3 prefix (TEACHER_SOURCE=s3; the collection adds the 12B base processor_config.json).
# Output: /tmp/gemma4_bestckpt_traces_v2/12br-medium (mirrored to s3://scale-ml/genai/rl-distill/gemma4-bestckpt-traces-topk128-v2/12br-medium).
# Re-running resumes (completed shards are skipped). Then: local_jobs/distill_12br_medium_to_e4b_gpu4.sh.
#
#   GPUS=0,1 bash rl-distill-scripts/local_jobs/trace_12br_medium_dp2.sh      # default: first 2 idle GPUs
set -euo pipefail
cd /mnt/efs/jasonwei/rl-distill
pick_free_gpus() { local want="$1" idx uuid used free=() busy; busy="$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null | sort -u || true)"
  while IFS=', ' read -r idx uuid used; do [ "${used}" -lt 512 ] && ! grep -qx "${uuid}" <<<"${busy}" && free+=("${idx}"); done < <(nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits)
  local IFS=,; echo "${free[*]:0:${want}}"; }
GPUS="${GPUS:-$(pick_free_gpus 2)}"
[ "$(awk -F, '{print NF}' <<<"${GPUS}")" -eq 2 ] || { echo "FATAL: need 2 GPUs, got '${GPUS}'" >&2; exit 2; }
echo "TRACE_12BR_MEDIUM gpus=${GPUS} (DP=2, TP=1)"
export TRACE_SPEC=12br-medium TEACHER_SOURCE=s3 TRACE_GPU_IDS="${GPUS}" TENSOR_PARALLEL_SIZE=1
export TRAIN_SAMPLES_PER_QUESTION=16 VALIDATION_SAMPLES_PER_QUESTION=1
export VENV="${VENV:-/tmp/.venv-gemma4}" HF_HOME="${HF_HOME:-/tmp/hf_cache}" AWS_PROFILE="${AWS_PROFILE:-ml-worker}"
exec bash rl-distill-scripts/scale_train/run_gemma4_bestckpt_trace_collection.sh
