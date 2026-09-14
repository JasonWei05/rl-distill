#!/usr/bin/env bash
# Forward KL of the pre-on-policy student (off-policy step_001000 export) on the SAME 512 teacher samples used for the step-50 forward KL.
set -euo pipefail
cd /mnt/efs/jasonwei/rl-distill; set -a; source .env; set +a
export PATH="/mnt/efs/jasonwei/rl-distill/.venv-gemma4/bin:/usr/local/cuda/bin:${PATH:-/usr/bin:/bin}" CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export LD_LIBRARY_PATH="/mnt/efs/jasonwei/rl-distill/.venv-gemma4/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="$HOME/.cache/huggingface"
PY=.venv-gemma4/bin/python
STK0=/opt/dlami/nvme/tmp/jasonwei_hf_stage/offpolicy_12b_step1000; mkdir -p "$STK0"
AWS_PROFILE=ml-worker aws s3 sync s3://scale-ml/genai/rl-distill/gemma4-e4b-base-distill-final-exports/Distill-gemma4-e4b-base-medium-to-12b-base/step_001000/ "$STK0/" --only-show-errors
cp -n /opt/dlami/nvme/tmp/jasonwei_hf_stage/onpolicy_stk_step50/processor_config.json "$STK0/"   # same base (12B @023679ed) metadata
E4B="$HOME/.cache/huggingface/hub/models--google--gemma-4-E4B/snapshots/411aa17b749aa952df1359d2dcea73917a544d9a"
DATA="$HOME/.cache/huggingface/hub/datasets--JWei05--DeepScaleR-Easy-Medium-Hard-Gemma-26B-PT-10k/snapshots/a0ba3c3dc07c7bc27e901670ceb1a0b0ceeaa8db/medium"
OUT=/tmp/gemma4_stk50_kl/forward_step0; mkdir -p "$OUT"; ln -sfn /tmp/gemma4_stk50_kl/forward/traces "$OUT/traces"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
echo "$(date -u +%FT%TZ) FORWARD score step-0 student on the shared teacher samples"
$PY rl-distill-scripts/reverse_kl_topk.py score --student "$E4B" --teacher "$STK0" --train_parquet "$DATA/train.parquet" --val_parquet "$DATA/validation.parquet" --questions_per_split 128 --samples_per_question 4 --topk 128 --splits validation --seed 0 --trace_dir "$OUT/traces" --out "$OUT/metrics.json"
echo "$(date -u +%FT%TZ) STEP0_FORWARD_DONE"
