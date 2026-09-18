#!/usr/bin/env bash
# Sweep MICRO_BATCH_SIZE_PER_GPU x MAX_PADDED_TOKENS_PER_MICROBATCH for the E4B RL update phase on NGPUS (default 3) free local GPUs (torchrun, FSDP2-sharded like the pod)
# (gemma4_e4b_microbatch_sweep.py, one process per config so an OOM cannot poison the next). Realistic response-length
# batch first, then the 8k worst case for every layout that survived. Results: $OUT/results.jsonl (+ per-config logs).
#   CONFIGS="4:8192 8:16384" bash rl-distill-scripts/local_jobs/e4b_microbatch_sweep.sh
set -uo pipefail
cd /mnt/efs/jasonwei/rl-distill
export PATH="/tmp/.venv-gemma4/bin:/usr/local/cuda/bin:$PATH" HF_HUB_OFFLINE=1 TORCH_CUDNN_SDPA_ENABLED=0 VERL_FSDP2_LOCAL_LOAD=1
PY=/tmp/.venv-gemma4/bin/python
OUT="${OUT:-/tmp/gemma4_e4b_mb_sweep}"; mkdir -p "$OUT"
CONFIGS="${CONFIGS:-4:8192 8:16384 12:24576 16:32768 24:49152 32:65536}"
NGPUS="${NGPUS:-3}"
pick_gpu() { local g; g="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2<1500{print $1}' | grep -vE "^(${AVOID_GPUS:-99})$" | head -n "$NGPUS" | paste -sd,)"; [ "$(awk -F, '{print NF}' <<<"$g")" -eq "$NGPUS" ] && echo "$g"; }
run_one() { local mbs=$1 cap=$2 worst=$3; local tag="mbs${mbs}_cap${cap}"; [ "$worst" = 1 ] && tag="${tag}_worst"
  local gpu; until gpu="$(pick_gpu)"; [ -n "$gpu" ]; do echo "$(date -u +%FT%TZ) GPU_WAIT $tag"; sleep 120; done
  echo "$(date -u +%FT%TZ) RUN $tag gpu=$gpu"
  CUDA_VISIBLE_DEVICES=$gpu MICRO_BATCH_SIZE_PER_GPU=$mbs MAX_PADDED_TOKENS_PER_MICROBATCH=$cap WORST_CASE=$worst TAG=$tag \
    timeout 2400 $PY -m torch.distributed.run --standalone --nproc_per_node="$NGPUS" rl-distill-scripts/gemma4_e4b_microbatch_sweep.py > "$OUT/$tag.log" 2>&1
  local rc=$?; local line; line="$(grep -a "^SWEEP_RESULT" "$OUT/$tag.log" | tail -1)"
  if [ -n "$line" ]; then echo "${line#SWEEP_RESULT }" >> "$OUT/results.jsonl"; echo "$(date -u +%FT%TZ) DONE $tag rc=$rc $(echo "${line#SWEEP_RESULT }" | $PY -c 'import json,sys; d=json.load(sys.stdin); print({k:(v if k!="phases" else {p:(q.get("peak_gib"),q.get("seconds"),q.get("oom")) for p,q in v.items()}) for k,v in d.items() if k in ("num_micro_batches","baseline_gib_after_load","phases")})')"
  else echo "$(date -u +%FT%TZ) FAILED $tag rc=$rc $(grep -aE "OutOfMemory|Error|error" "$OUT/$tag.log" | tail -1 | cut -c1-160)"; echo "{\"tag\":\"$tag\",\"mbs\":$mbs,\"cap\":$cap,\"worst_case\":$([ "$worst" = 1 ] && echo true || echo false),\"crashed\":true,\"rc\":$rc}" >> "$OUT/results.jsonl"; fi
  return $rc; }
ok=()
for c in $CONFIGS; do mbs=${c%%:*}; cap=${c##*:}; run_one "$mbs" "$cap" 0 && grep -q "\"tag\": \"mbs${mbs}_cap${cap}\"" "$OUT/results.jsonl" && ! grep "\"tag\": \"mbs${mbs}_cap${cap}\"" "$OUT/results.jsonl" | tail -1 | grep -q '"oom": true' && ok+=("$c"); done
echo "$(date -u +%FT%TZ) REALISTIC_DONE survivors: ${ok[*]:-none}"
for c in "${ok[@]}"; do mbs=${c%%:*}; cap=${c##*:}; run_one "$mbs" "$cap" 1; done
echo "$(date -u +%FT%TZ) SWEEP_ALL_DONE"
