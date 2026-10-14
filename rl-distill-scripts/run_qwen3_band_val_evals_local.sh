#!/usr/bin/env bash
# Local (this box) n=16 eval of the BAND's RL best checkpoints + the Qwen3 bases on the BAND validation split, with the
# exact protocol of the distill stage-3 evals (run_qwen3_30bm_to_4b_distill_st.sh: 0-shot native ChatML + boxed instruction,
# temp 1, 8192 tokens, strict boxed grading, unbiased pass@k), so RL / base pass@16 is comparable with the distilled students.
# Results: ${S3_OUT}/<tag>/summary.json (+ samples, grades, log).
#   BAND=medium GPUS="1 5" setsid bash rl-distill-scripts/run_qwen3_band_val_evals_local.sh > <log> 2>&1 &   # BAND: medium|hard|very_hard
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PROJECT_ROOT="$(pwd)"
BAND="${BAND:-very_hard}"
ROOT=/tmp/qwen3_${BAND}_val_evals; MODELS_DIR=${ROOT}/models; mkdir -p "${MODELS_DIR}"
S3_OUT="${S3_OUT:-s3://scale-ml/genai/rl-distill/qwen3-${BAND}-val-evals-v1}"
PY=${PROJECT_ROOT}/.venv-gemma4/bin/python
# flashinfer's sampler JIT shells out to ninja / nvcc: the venv bin and CUDA must be on PATH (as in the pod run-files)
export CUDA_HOME=$(readlink -f /usr/local/cuda); export PATH="${PROJECT_ROOT}/.venv-gemma4/bin:${CUDA_HOME}/bin:${PATH}"
export AWS_PROFILE=ml-worker VERL_MATH_VERIFY_STRICT_BOXED=1 HF_HOME=${ROOT}/hf
set -a; source .env; set +a
read -r -a GPU_LIST <<<"${GPUS:-1 5}"
B=s3://scale-ml/genai/rl-distill
# best RL steps (run_outcome.json best_step) per band
case ${BAND} in
  medium) SHORT=med; S30=320; S4=510; S1P7=520 ;;
  hard) SHORT=hard; S30=410; S4=560; S1P7=550 ;;
  very_hard) SHORT=vh; S30=460; S4=620; S1P7=810 ;;
  *) echo "FATAL: unknown BAND=${BAND}" >&2; exit 2 ;;
esac
# tag | source (s3://... HF export or hf:repo@revision); listed per GPU queue below
MODELS=(
  "rl_30b_${SHORT}_s${S30}|s3:${B}/qwen3-30b-a3b-base-rl/qwen3-30b-a3b-${BAND}-es5/best_hf"
  "rl_4b_${SHORT}_s${S4}|s3:${B}/qwen3-4b-base-rl/qwen3-4b-${BAND}-es5/best_hf"
  "rl_1p7b_${SHORT}_s${S1P7}|s3:${B}/qwen3-1.7b-base-rl/qwen3-1.7b-${BAND}-es5/best_hf"
  "base_30b|hf:Qwen/Qwen3-30B-A3B-Base@1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9"
  "base_4b|hf:Qwen/Qwen3-4B-Base@906bfd4b4dc7f14ee4320094d8b41684abff8539"
  "base_1p7b|hf:Qwen/Qwen3-1.7B-Base@ea980cb0a6c2ae4b936e82123acc929f1cec04c1"
)
# MODEL_SPECS="tag|s3:uri tag|s3:uri ...": evaluate exactly these instead (e.g. other RL checkpoints' actor/huggingface exports)
if [ -n "${MODEL_SPECS:-}" ]; then read -r -a MODELS <<<"${MODEL_SPECS}"; fi
if [ -n "${ONLY:-}" ]; then sel=(); for e in "${MODELS[@]}"; do [[ " ${ONLY} " == *" ${e%%|*} "* ]] && sel+=("${e}"); done; MODELS=("${sel[@]}"); fi
VAL_PARQUET=${ROOT}/${BAND}_validation.parquet
"${PY}" - "${VAL_PARQUET}" "${BAND}" <<'PY'
import shutil, sys
from huggingface_hub import hf_hub_download
# same revisions as the distill stage-3 evals: very_hard was added in a later dataset commit (other band files unchanged there)
revision = {"very_hard": "9bec69ca0c432e6286cafdf8b26b4c6aa737084e"}.get(sys.argv[2], "be4dcd8b1fad7d5eb7bf721f0210baf78277aaa9")
p = hf_hub_download("JWei05/DeepScaleR-Easy-Medium-Hard-Qwen3-30B-A3B-Base-10k", f"{sys.argv[2]}/validation.parquet", repo_type="dataset",
                    revision=revision)
shutil.copy(p, sys.argv[1])
PY
DEFAULT_INSTRUCTION='Please reason step by step, and put your final answer within \boxed{}.'
NL=$'\n'; STOP_STRINGS="<|im_end|>,${NL}Human:,${NL}user${NL}"; STOP_TOKEN_IDS=151645,151643,151644
n_gpus=${#GPU_LIST[@]}; pids=()
for g in $(seq 0 $((n_gpus - 1))); do
  (
    set -e
    for ((i = g; i < ${#MODELS[@]}; i += n_gpus)); do
      IFS='|' read -r tag src <<<"${MODELS[$i]}"; out=${ROOT}/${tag}; m=${MODELS_DIR}/${tag}; mkdir -p "${out}"
      if aws s3 ls "${S3_OUT}/${tag}/summary.json" >/dev/null 2>&1; then echo "EVAL_ON_S3 ${tag}"; continue; fi
      if [[ ${src} == s3:* ]]; then aws s3 sync --only-show-errors "${src#s3:}/" "${m}/"
      else r=${src#hf:}; m=$("${PY}" -c "import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1], revision=sys.argv[2]))" "${r%@*}" "${r#*@}"); fi
      echo "EVAL_START $(date -u +%FT%TZ) ${tag} gpu=${GPU_LIST[$g]}"
      CUDA_VISIBLE_DEVICES=${GPU_LIST[$g]} VLLM_CACHE_ROOT=/tmp/vllm_cache_${BAND}_gpu${GPU_LIST[$g]} "${PY}" rl-distill-scripts/data/generate_deepscaler_4bpt_samples.py \
        --model "${m}" --input "${VAL_PARQUET}" --chat-template rl-distill-scripts/data/qwen3_base_native_chat.jinja \
        --output "${out}/samples.parquet" --shard-id 0 --num-shards 1 --n 16 \
        --max-tokens 8192 --max-model-len 12288 --max-prompt-tokens 4096 \
        --temperature 1.0 --top-p 1.0 --top-k -1 --stop-strings "${STOP_STRINGS}" --stop-token-ids "${STOP_TOKEN_IDS}" \
        --append-boxed-instruction --boxed-instruction "${DEFAULT_INSTRUCTION}" --attention-backend TRITON_ATTN \
        --chunk-size 300 --max-num-seqs 512 --gpu-memory-utilization 0.9 > "${out}/log.txt" 2>&1 \
        || { echo "EVAL_FAILED ${tag} (see ${out}/log.txt)"; continue; }
      PYTHONPATH="${PROJECT_ROOT}" "${PY}" rl-distill-scripts/data/grade_band_eval_samples.py "${out}/samples.parquet"
      "${PY}" - "${out}/grades.json" "${out}/summary.json" "${tag}" <<'PY'
import json, sys
from math import comb
rows = json.load(open(sys.argv[1])); n = len(rows[0]["acc"])
c = [sum(r["acc"]) for r in rows]; q = len(rows)
pass_k = {k: 100 * sum(1 - comb(n - ci, k) / comb(n, k) for ci in c) / q for k in (1, 2, 4, 8, 16) if k <= n}
summary = {"tag": sys.argv[3], "questions": q, "n": n, "mean_at_n": 100 * sum(c) / (q * n), "pass_at_k": pass_k,
           "boxed_frac": sum(r["boxed"] for r in rows) / (q * n), "trunc_frac": sum(r["trunc"] for r in rows) / (q * n),
           "mean_response_len": sum(r["mean_len"] for r in rows) / q}
json.dump(summary, open(sys.argv[2], "w"), indent=2)
print("EVAL_SUMMARY", json.dumps(summary))
PY
      for f in log.txt samples.parquet grades.json summary.json; do aws s3 cp --only-show-errors "${out}/${f}" "${S3_OUT}/${tag}/${f}"; done   # summary last: marks done
      echo "EVAL_DONE $(date -u +%FT%TZ) ${tag}"
    done
  ) &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "${p}"; done
echo "BAND_VAL_EVALS_DONE ${BAND} $(date -u +%FT%TZ)"
