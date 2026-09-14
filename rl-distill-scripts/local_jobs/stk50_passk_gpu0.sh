#!/usr/bin/env bash
# pass@k x32 on the medium validation set (300 q) for the on-policy (student-top-128) 12B step-50 checkpoint, then the
# untrained E2B base with the same protocol, then the comparison figure. One GPU.
set -euo pipefail
cd /mnt/efs/jasonwei/rl-distill; set -a; source .env; set +a
# vLLM compile/CUDA-graph capture shells out to ninja + nvcc: put the venv and CUDA toolkit on PATH (a bare nohup env lacks them).
export PATH="/mnt/efs/jasonwei/rl-distill/.venv-gemma4/bin:/usr/local/cuda/bin:${PATH:-/usr/bin:/bin}" CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}" HF_HOME="$HOME/.cache/huggingface" VLLM_CACHE_ROOT=/tmp/vllm_cache_pk TRITON_CACHE_DIR=/tmp/triton_pk
export VERL_MATH_VERIFY_STRICT_BOXED=1 VERL_MATH_VERIFY_TIMEOUT=30 VERL_MATH_SYMPY_TIMEOUT=5.0
PY=.venv-gemma4/bin/python; S=rl-distill-scripts
STK="${STK:-/opt/dlami/nvme/tmp/jasonwei_hf_stage/onpolicy_stk_step50}"
E2B="$HOME/.cache/huggingface/hub/models--google--gemma-4-E2B/snapshots/d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f"
VAL=/tmp/gemma4_e4b_val32
SAMPLING=(--temperature 1.0 --top_k -1 --top_p 1.0 --max_tokens 8192 --max_prompt_tokens 4096 --max_model_len 12288 --predictive_topk_width 0 --request_batch_size 2048 --questions_per_batch 64 --subset_strategy monte_carlo --monte_carlo_resamples 4096 --ks 1 2 4 8 16 32)
run_eval() { local model=$1 tag=$2 sha=$3; local out=$VAL/students/$tag; mkdir -p "$out/traces"
  echo "$(date -u +%FT%TZ) EVAL $tag"
  $PY $S/eval_math_passk.py --model "$model" --expected_model_identity_sha256 "$sha" --tag "$tag" --chat_template $S/data/gemma3_it_fewshot_math.jinja --datasets $VAL/data/id_medium.parquet --dataset_manifest $VAL/data/math_eval_manifest.json --out "$out/metrics.json" --trace_dir "$out/traces" --tensor_parallel_size 1 --gpu_memory_utilization "${UTIL:-0.85}" "${SAMPLING[@]}"; }
# identity hashes from data/gemma4_model_identity.inspect_local_hf_model (2026-09-13)
run_eval "$STK" onpolicy_studenttop128_12b__step_000050 725ff2a3367cae4886319d91adba11236003c64dfdfc4faa53bc882c841d84f2
[ -f $VAL/students/base_e2b__x32_medium/metrics.json ] || run_eval "$E2B" base_e2b__x32_medium bde9e800223cdd62228ce39e0305398f6ada05b98adaf438b0b3d3d3c3015561
echo "$(date -u +%FT%TZ) PLOT"
$PY $S/plot_passk_from_traces.py --out $S/figures/passk_12b_onpolicy_stk50_vs_teacher.png \
  --title "pass@k on the medium validation set (300 q, 32 samples): on-policy (student-top-128) 12B step 50 vs the E4B base teacher, the off-policy 12B (step 1000), the 12B base and the E2B base" \
  --trace "E4B base (teacher) x32=$VAL/id_medium/traces/base_e4b__id_medium.jsonl" \
  --trace "E2B base x32=$VAL/students/base_e2b__x32_medium/traces/base_e2b__x32_medium__id_medium.jsonl" \
  --trace "12B base x32=$VAL/students/base_12b__x32_medium/traces/base_12b__x32_medium__id_medium.jsonl" \
  --trace "12B off-policy distilled step 1000=$VAL/students/distill_gemma4_e4b_base_medium_to_12b_base__step_001000/traces/distill_gemma4_e4b_base_medium_to_12b_base__step_001000__id_medium.jsonl" \
  --trace "12B on-policy (student top-128) step 50=$VAL/students/onpolicy_studenttop128_12b__step_000050/traces/onpolicy_studenttop128_12b__step_000050__id_medium.jsonl"
echo "$(date -u +%FT%TZ) PASSK_ALL_DONE"
