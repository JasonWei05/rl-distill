#!/usr/bin/env bash
# Forward + reverse KL between the on-policy (student-top-128) 12B step-50 checkpoint and the E4B base teacher,
# 128 medium validation questions x 4 samples. Reverse = student samples scored by the teacher (reverse_kl_topk.py as
# designed); forward = the same script with the roles swapped (teacher samples scored by the student).
# Shared-box aware: each phase picks the GPU with the most free memory (among CANDIDATE_GPUS) and sizes the vLLM
# memory fraction from what is free at that moment; completed generation phases are skipped (traces reused).
set -euo pipefail
cd /mnt/efs/jasonwei/rl-distill; set -a; source .env; set +a
export PATH="/mnt/efs/jasonwei/rl-distill/.venv-gemma4/bin:/usr/local/cuda/bin:${PATH:-/usr/bin:/bin}" CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export HF_HOME="$HOME/.cache/huggingface" VLLM_CACHE_ROOT=/tmp/vllm_cache_kl TRITON_CACHE_DIR=/tmp/triton_kl
PY=.venv-gemma4/bin/python
STK="${STK:-/opt/dlami/nvme/tmp/jasonwei_hf_stage/onpolicy_stk_step50}"
E4B="$HOME/.cache/huggingface/hub/models--google--gemma-4-E4B/snapshots/411aa17b749aa952df1359d2dcea73917a544d9a"
DATA="$HOME/.cache/huggingface/hub/datasets--JWei05--DeepScaleR-Easy-Medium-Hard-Gemma-26B-PT-10k/snapshots/a0ba3c3dc07c7bc27e901670ceb1a0b0ceeaa8db/medium"
OUT="${OUT:-/tmp/gemma4_stk50_kl}"; mkdir -p "$OUT/reverse" "$OUT/forward"
CANDIDATE_GPUS="${CANDIDATE_GPUS:-0 1 2 3 4 5 6 7}"
N_RESP=$((128*4))
pick_gpu() {  # $1 = MiB needed -> sets GPU and UTIL (fraction of the 81559 MiB card, capped 0.85)
  local need=$1 best= bestfree=0
  for g in $CANDIDATE_GPUS; do local free; free=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | awk -F', ' -v g=$g '$1==g{print 81559-$2}'); [ "${free:-0}" -gt "$bestfree" ] && { bestfree=$free; best=$g; }; done
  if [ "$bestfree" -lt "$need" ]; then echo "$(date -u +%FT%TZ) WAIT: need ${need} MiB, best GPU $best has ${bestfree} MiB free; sleeping 120 s"; sleep 120; pick_gpu "$need"; return; fi
  GPU=$best; UTIL=$(python3 -c "print(round(min(0.85, ($bestfree-3000)/81559),2))"); export CUDA_VISIBLE_DEVICES=$GPU
  echo "$(date -u +%FT%TZ) using GPU $GPU (free ${bestfree} MiB) util $UTIL"
}
COMMON=(--train_parquet "$DATA/train.parquet" --val_parquet "$DATA/validation.parquet" --questions_per_split 128 --samples_per_question 4 --topk 128 --splits validation --seed 0 --gen_batch 128 --max_tokens "${MAX_TOKENS:-8192}" --max_model_len "${MAX_MODEL_LEN:-12288}")
complete() { [ -f "$1" ] && [ "$(wc -l < "$1")" -eq "$N_RESP" ]; }
# --- reverse KL: student samples, teacher scores ---
if complete "$OUT/reverse/traces/validation.jsonl"; then echo "$(date -u +%FT%TZ) REVERSE generate: traces complete, skipping"; else
  pick_gpu 34000; echo "$(date -u +%FT%TZ) REVERSE generate (student samples)"
  $PY rl-distill-scripts/reverse_kl_topk.py generate --student "$STK" "${COMMON[@]}" --gpu_memory_utilization "$UTIL" --trace_dir "$OUT/reverse/traces"; fi
if [ -f "$OUT/reverse/metrics.json" ]; then echo "REVERSE score: metrics present, skipping"; else
  pick_gpu 26000; echo "$(date -u +%FT%TZ) REVERSE score (teacher scores student samples)"
  $PY rl-distill-scripts/reverse_kl_topk.py score --student "$STK" --teacher "$E4B" "${COMMON[@]}" --trace_dir "$OUT/reverse/traces" --out "$OUT/reverse/metrics.json"; fi
# --- forward KL: teacher samples, student scores (roles swapped) ---
if complete "$OUT/forward/traces/validation.jsonl"; then echo "$(date -u +%FT%TZ) FORWARD generate: traces complete, skipping"; else
  pick_gpu 26000; echo "$(date -u +%FT%TZ) FORWARD generate (teacher samples)"
  $PY rl-distill-scripts/reverse_kl_topk.py generate --student "$E4B" "${COMMON[@]}" --gpu_memory_utilization "$UTIL" --trace_dir "$OUT/forward/traces"; fi
if [ -f "$OUT/forward/metrics.json" ]; then echo "FORWARD score: metrics present, skipping"; else
  pick_gpu 34000; echo "$(date -u +%FT%TZ) FORWARD score (student scores teacher samples)"
  $PY rl-distill-scripts/reverse_kl_topk.py score --student "$E4B" --teacher "$STK" "${COMMON[@]}" --trace_dir "$OUT/forward/traces" --out "$OUT/forward/metrics.json"; fi
echo "$(date -u +%FT%TZ) KL_ALL_DONE"
