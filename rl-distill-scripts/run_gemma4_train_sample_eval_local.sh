#!/usr/bin/env bash
# Local (this box) eval of the Gemma 4 medium-trained E4B RL / distilled models on a 300-question random sample of the medium TRAIN
# split (data/prepare_gemma4_train_sample_eval_data.py), 16 samples/question, with the registered Gemma 4 math protocol (12-shot
# template, temp 1, top-p 1, 8192 tokens, strict boxed grading, monte-carlo pass@k) exactly as recorded in the study's metrics.json
# configs (2 vLLM slots per GPU at util 0.4 + fixed 16 GiB KV, no logprobs). Results: ${OUT}/<tag>/metrics.json (+ mirrored to S3).
#   GPU_SLOTS="0 0 1 1 5 6" bash run_gemma4_train_sample_eval_local.sh      # one slot per listed GPU entry, models assigned in order
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT=/tmp/gemma4_train_sample_eval; DATA=${ROOT}/data; OUT=${ROOT}/results; MODELS_DIR=${ROOT}/models; mkdir -p "${OUT}" "${MODELS_DIR}"
S3_OUT="${S3_OUT:-s3://scale-ml/genai/rl-distill/gemma4-train-sample-evals-v1/train_medium}"
PY=.venv-gemma4/bin/python
export AWS_PROFILE=ml-worker VERL_MATH_VERIFY_STRICT_BOXED=1 VERL_MATH_VERIFY_TIMEOUT=30 VERL_MATH_SYMPY_TIMEOUT=5.0 HF_HOME=${ROOT}/hf
set -a; source .env; set +a
read -r -a SLOTS <<<"${GPU_SLOTS:-0 0 1 1 5 6}"
# tag | S3 HF export (registry sources in config/gemma4_distill_study_eval_sources.json)
B=s3://scale-ml/genai/rl-distill
MODELS=(
  "rl_e4b_medium_s43|${B}/gemma4-difficulty-s43-20260910/gemma4-e4b-medium/best_hf"
  "rl_e4b_medium_bs256_s46|${B}/gemma4-e4b-bs256-s46-20260919/gemma4-e4b-medium/best_hf"
  "distill_12br_lr2e6_medium_to_e4b_step1000|${B}/gemma4-12br-distill-ckpts-v1/12br-medium-to-e4b-bs128-s1000-lr2e-6/global_step_1000/huggingface"
  "distill_12bd_medium_to_e4b_step1000|${B}/gemma4-12bd-distill-ckpts-v1/12bd-medium-to-e4b-bs128-s1000-lr2e-6/global_step_1000/huggingface"
  "distill_26bw_lr2e6_medium_to_e4b_step1000|${B}/gemma4-26bw-distill-ckpts-v1/26bw-medium-to-e4b-bs128-s1000-lr2e-6-n8/global_step_1000/huggingface"   # the -n8 (8-GPU) run = the reported 26bw student; the registry URI without -n8 is empty
  "distill_26bd_lr2e6_medium_to_e4b_step1000|${B}/gemma4-26bd-distill-ckpts-v1/26bd-medium-to-e4b-bs128-s1000-lr2e-6/global_step_1000/huggingface"
)
# ONLY="tag1 tag2": run just those models (e.g. a retry), GPU_SLOTS then lists one slot per selected model
if [ -n "${ONLY:-}" ]; then sel=(); for e in "${MODELS[@]}"; do [[ " ${ONLY} " == *" ${e%%|*} "* ]] && sel+=("${e}"); done; MODELS=("${sel[@]}"); fi
(( ${#SLOTS[@]} >= ${#MODELS[@]} )) || { echo "FATAL: need >= ${#MODELS[@]} GPU slots" >&2; exit 2; }
BASE_DIR="$(${PY} -c "from huggingface_hub import snapshot_download; print(snapshot_download('google/gemma-4-E4B', revision='411aa17b749aa952df1359d2dcea73917a544d9a', allow_patterns=['processor_config.json','preprocessor_config.json']))")"
pids=()
for i in "${!MODELS[@]}"; do
  IFS='|' read -r tag src <<<"${MODELS[$i]}"; gpu=${SLOTS[$i]}; m=${MODELS_DIR}/${tag}; o=${OUT}/${tag}; mkdir -p "${o}"
  (
    set -e
    aws s3 sync --only-show-errors "${src}/" "${m}/"
    # Gemma 4 must load as its native unified arch (never an arch override); provide the base processor config if the export lacks it
    [ -f "${m}/processor_config.json" ] || cp "${BASE_DIR}/processor_config.json" "${m}/"
    sleep $((i * 30))   # stagger vLLM start-ups (device-wide profiling miscounts concurrent engines)
    echo "EVAL_START $(date -u +%FT%TZ) ${tag} gpu=${gpu}"
    CUDA_VISIBLE_DEVICES=${gpu} VLLM_CACHE_ROOT=/tmp/vllm_cache_ts${i} TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_cache_ts${i} TRITON_CACHE_DIR=/tmp/triton_cache_ts${i} \
      ${PY} rl-distill-scripts/eval_math_passk.py --model "${m}" --allow_unpinned_local_model --tag "${tag}" \
      --chat_template rl-distill-scripts/data/gemma3_it_fewshot_math.jinja --datasets "${DATA}/train_medium.parquet" --dataset_manifest "${DATA}/math_eval_manifest.json" \
      --out "${o}/metrics.json" --trace_dir "${o}/traces" --samples_per_question 16 --seed 0 --ks 1 2 4 8 16 --subset_strategy monte_carlo \
      --monte_carlo_resamples 4096 --metric_seed 0 --max_tokens 8192 --max_prompt_tokens 4096 --max_model_len 12288 --temperature 1.0 --top_p 1.0 --top_k -1 \
      --predictive_topk_width 0 --questions_per_batch 64 --request_batch_size 1024 --tensor_parallel_size 1 --dtype bfloat16 \
      --gpu_memory_utilization 0.4 --kv_cache_memory_gib 16 --resume_traces > "${o}/log.txt" 2>&1
    aws s3 cp --only-show-errors "${o}/metrics.json" "${S3_OUT}/${tag}/metrics.json"; aws s3 cp --only-show-errors "${o}/log.txt" "${S3_OUT}/${tag}/log.txt"
    "${PY}" -c "import json,sys; r=json.load(open(sys.argv[1]))['results']['train_medium']['by_k']['16']; print(f'EVAL_DONE {sys.argv[2]} mean@16 {100*r[\"mean_at_k\"]:.1f} pass@16 {100*r[\"pass_at_k\"]:.1f}')" "${o}/metrics.json" "${tag}"
  ) || echo "EVAL_FAILED ${tag} (see ${o}/log.txt)" &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "${p}"; done
echo "TRAIN_SAMPLE_EVAL_DONE $(date -u +%FT%TZ) out=${OUT}"
