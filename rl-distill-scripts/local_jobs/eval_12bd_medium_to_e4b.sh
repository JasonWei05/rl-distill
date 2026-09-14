#!/usr/bin/env bash
# §9.0g step 3: evaluate the 12bd-medium -> E4B-base distillation checkpoints (S3 hf_exports/global_step_N) against the
# references, on local GPUs (default 0 and 2, one eval per GPU in parallel):
#   (a) pass@k x32 on the medium validation set (300 q; protocol gemma4_rl_distill_math_eval_v2_x32, same as §9.1) for every
#       student export + the 12bd teacher itself + the §4 control student (untrained-12B RL medium teacher -> E4B, step 500);
#       the E4B base x32 trace already exists under /tmp/gemma4_e4b_val32/id_medium/traces/.
#   (b) offline reverse KL (student samples, teacher scores) and forward KL (teacher samples, student scores) vs the 12bd
#       teacher on 128 validation questions x 4 samples (reverse_kl_topk.py; §9.0c/§9.0e protocol) for the final student.
#   (c) the pass@k figure.
#
#   STEPS="50 100 ... 1000" GPUS="0 2" bash rl-distill-scripts/local_jobs/eval_12bd_medium_to_e4b.sh
#   (default STEPS = every hf_exports/global_step_N with a completion marker on S3)
set -euo pipefail
cd /mnt/efs/jasonwei/rl-distill; set -a; source .env; set +a
export PATH="/mnt/efs/jasonwei/rl-distill/.venv-gemma4/bin:/usr/local/cuda/bin:${PATH:-/usr/bin:/bin}" CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export LD_LIBRARY_PATH="/mnt/efs/jasonwei/rl-distill/.venv-gemma4/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="$HOME/.cache/huggingface" VLLM_CACHE_ROOT=/tmp/vllm_cache_12bd_eval TRITON_CACHE_DIR=/tmp/triton_12bd_eval
export VERL_MATH_VERIFY_STRICT_BOXED=1 VERL_MATH_VERIFY_TIMEOUT=30 VERL_MATH_SYMPY_TIMEOUT=5.0 AWS_PROFILE=ml-worker
PY=.venv-gemma4/bin/python; S=rl-distill-scripts
CKPT_S3="${CKPT_S3:-s3://scale-ml/genai/rl-distill/gemma4-12bd-distill-ckpts-v1/12bd-medium-to-e4b-bs128-s1000-lr2e-6}"
OUT="${OUT:-/tmp/gemma4_12bd_evals}"; MODELS="$OUT/models"; mkdir -p "$MODELS" "$OUT/passk" "$OUT/kl"
VAL=/tmp/gemma4_e4b_val32
TEACHER="${TEACHER:-/tmp/gemma4_trace_models/12bd-medium}"          # the step-190 export the traces came from
GPUS="${GPUS:-0 2}"
DATA="$HOME/.cache/huggingface/hub/datasets--JWei05--DeepScaleR-Easy-Medium-Hard-Gemma-26B-PT-10k/snapshots/a0ba3c3dc07c7bc27e901670ceb1a0b0ceeaa8db/medium"
SAMPLING=(--temperature 1.0 --top_k -1 --top_p 1.0 --max_tokens 8192 --max_prompt_tokens 4096 --max_model_len 12288 --predictive_topk_width 0 --request_batch_size 2048 --questions_per_batch 64 --subset_strategy monte_carlo --monte_carlo_resamples 4096 --ks 1 2 4 8 16 32)
log() { echo "$(date -u +%FT%TZ) $*"; }
identity() { $PY - "$1" <<'PY'
import sys; sys.path.insert(0, "rl-distill-scripts/data")
from gemma4_model_identity import inspect_local_hf_model
print(inspect_local_hf_model(sys.argv[1]).model_identity_sha256)
PY
}
# --- 1. models --------------------------------------------------------------------------------
if [[ -z ${STEPS:-} ]]; then
  STEPS="$(aws s3 ls "$CKPT_S3/hf_exports/" | grep -oE "global_step_[0-9]+" | sed 's/global_step_//' | sort -n | tr '\n' ' ')"
fi
log "steps: $STEPS"
declare -A MODEL TAG
for st in $STEPS; do
  d="$MODELS/step_$(printf %06d "$st")"
  if [[ ! -f $d/config.json ]]; then
    aws s3 ls "$CKPT_S3/hf_exports/global_step_$st/_REMOTE_COMPLETE.json" >/dev/null || { log "step $st has no completion marker on S3; skipping"; continue; }
    log "download step $st"; aws s3 sync --only-show-errors "$CKPT_S3/hf_exports/global_step_$st/huggingface/" "$d/"
  fi
  MODEL["student_$st"]="$d"; TAG["student_$st"]="distill_12bd_medium_to_e4b_base__step_$(printf %06d "$st")"
done
# the §4 control: untrained-12B RL medium teacher -> E4B base (step 500), pinned at download time
CTRL="$MODELS/control_12b_medium_to_e4b_step500"
if [[ ! -f $CTRL/config.json ]]; then
  log "download control JWei05/gemma4-distill-v2-12b-medium-to-e4b-base@b015fe88/step_000500 (eval-registry pin, tag distill_12b_medium_to_e4b)"
  $PY - "$CTRL" <<'PY'
import shutil, sys
from pathlib import Path
from huggingface_hub import snapshot_download
dst = Path(sys.argv[1]); snap = Path(snapshot_download("JWei05/gemma4-distill-v2-12b-medium-to-e4b-base", revision="b015fe8827dbe11a58225f961340b615dbf1fdc8", allow_patterns=["step_000500/*"]))
shutil.copytree(snap / "step_000500", dst, dirs_exist_ok=True); (dst / "PINNED_COMMIT").write_text(snap.name + "\n"); print("control commit", snap.name)
PY
fi
MODEL["control"]="$CTRL"; TAG["control"]="distill_gemma4_12b_medium_to_e4b_base__step_000500__x32"
MODEL["teacher"]="$TEACHER"; TAG["teacher"]="teacher_12bd_medium__step_000190__x32"
# --- 2. pass@k x32, one eval per GPU in parallel ----------------------------------------------
run_passk() { local key=$1 gpu=$2; local model=${MODEL[$key]} tag=${TAG[$key]} out="$OUT/passk/${TAG[$key]}"
  [[ -f $out/metrics.json ]] && { log "passk $tag: done, skipping"; return 0; }
  mkdir -p "$out/traces"; local sha; sha=$(identity "$model")
  log "passk $tag on GPU $gpu (identity $sha)"
  CUDA_VISIBLE_DEVICES=$gpu $PY $S/eval_math_passk.py --model "$model" --expected_model_identity_sha256 "$sha" --tag "$tag" \
    --chat_template $S/data/gemma3_it_fewshot_math.jinja --datasets $VAL/data/id_medium.parquet --dataset_manifest $VAL/data/math_eval_manifest.json \
    --out "$out/metrics.json" --trace_dir "$out/traces" --tensor_parallel_size 1 --gpu_memory_utilization 0.85 "${SAMPLING[@]}" >"$out/eval.log" 2>&1 \
    && log "passk $tag: DONE" || { log "passk $tag: FAILED (see $out/eval.log)"; return 1; }
}
keys=(teacher control); for st in $STEPS; do [[ -n ${MODEL[student_$st]:-} ]] && keys+=("student_$st"); done
gpu_arr=($GPUS); i=0; pids=()
for key in "${keys[@]}"; do
  gpu=${gpu_arr[$((i % ${#gpu_arr[@]}))]}; run_passk "$key" "$gpu" & pids+=($!); i=$((i+1))
  if (( i % ${#gpu_arr[@]} == 0 )); then for p in "${pids[@]}"; do wait "$p" || true; done; pids=(); fi
done
for p in "${pids[@]:-}"; do [[ -n $p ]] && wait "$p" || true; done
# --- 3. KL vs the 12bd teacher for the final student (128 val q x 4) ---------------------------
last=$(echo $STEPS | tr ' ' '\n' | sort -n | tail -1); STUDENT=${MODEL[student_$last]:-}
if [[ -n $STUDENT ]]; then
  K="$OUT/kl/step_$last"; mkdir -p "$K/reverse" "$K/forward"; N=$((128*4))
  COMMON=(--train_parquet "$DATA/train.parquet" --val_parquet "$DATA/validation.parquet" --questions_per_split 128 --samples_per_question 4 --topk 128 --splits validation --seed 0 --gen_batch 128 --max_tokens 8192 --max_model_len 12288)
  complete() { [[ -f $1 && $(wc -l < "$1") -eq $N ]]; }
  g0=${gpu_arr[0]}
  complete "$K/reverse/traces/validation.jsonl" || CUDA_VISIBLE_DEVICES=$g0 $PY $S/reverse_kl_topk.py generate --student "$STUDENT" "${COMMON[@]}" --gpu_memory_utilization 0.85 --trace_dir "$K/reverse/traces"
  [[ -f $K/reverse/metrics.json ]] || CUDA_VISIBLE_DEVICES=$g0 $PY $S/reverse_kl_topk.py score --student "$STUDENT" --teacher "$TEACHER" "${COMMON[@]}" --trace_dir "$K/reverse/traces" --out "$K/reverse/metrics.json"
  complete "$K/forward/traces/validation.jsonl" || CUDA_VISIBLE_DEVICES=$g0 $PY $S/reverse_kl_topk.py generate --student "$TEACHER" "${COMMON[@]}" --gpu_memory_utilization 0.85 --trace_dir "$K/forward/traces"
  [[ -f $K/forward/metrics.json ]] || CUDA_VISIBLE_DEVICES=$g0 $PY $S/reverse_kl_topk.py score --student "$TEACHER" --teacher "$STUDENT" "${COMMON[@]}" --trace_dir "$K/forward/traces" --out "$K/forward/metrics.json"
  log "KL done: $K/{reverse,forward}/metrics.json"
fi
# --- 4. figure --------------------------------------------------------------------------------
args=(--trace "E4B base x32=$VAL/id_medium/traces/base_e4b__id_medium.jsonl")
[[ -f $OUT/passk/${TAG[teacher]}/metrics.json ]] && args+=(--trace "12B distilled+RL teacher (step 190) x32=$OUT/passk/${TAG[teacher]}/traces/${TAG[teacher]}__id_medium.jsonl")
[[ -f $OUT/passk/${TAG[control]}/metrics.json ]] && args+=(--trace "E4B <- untrained-12B RL teacher (§4 control, step 500) x32=$OUT/passk/${TAG[control]}/traces/${TAG[control]}__id_medium.jsonl")
for st in $STEPS; do t=${TAG[student_$st]:-}; [[ -n $t && -f $OUT/passk/$t/metrics.json ]] && args+=(--trace "E4B <- 12B distilled+RL teacher, step $st=$OUT/passk/$t/traces/${t}__id_medium.jsonl"); done
.venv/bin/python $S/plot_passk_from_traces.py --out $S/figures/passk_e4b_from_12bd_medium.png \
  --title "pass@k on the medium validation set (300 q, 32 samples): E4B base distilled from the RL'd distilled 12B (§9.0g) vs references" "${args[@]}"
log "EVAL_ALL_DONE"
