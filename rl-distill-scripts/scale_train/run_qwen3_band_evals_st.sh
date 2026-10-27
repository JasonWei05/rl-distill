#!/usr/bin/env bash
# ScaleTrain run-file: sample N=16 responses from each student model on each teacher-difficulty band set built by
# rl-distill-scripts/data/build_qwen3_teacher_band_sets.py (300 questions each), with the same protocol as
# run_deepscaler_samples_st.sh (0-shot native ChatML + Qwen-Math boxed instruction, temp 1, 8192 response tokens).
# One generator per GPU; GPU g runs tasks g, g+#GPUs, ... Each finished task is uploaded to
# OUT_S3/<model>/<set>/samples.parquet immediately, and a relaunch skips tasks already on S3 (preemption-safe).
#   bash launch_st_with_code.sh --gpus-per-instance 4 --priority high --allow-borrowing \
#     --run-file run_qwen3_band_evals_st.sh --job-name q3-band-evals --env-vars "OUT_S3=s3://.../band_evals"
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "${PROJECT_ROOT}"
export PATH="${PROJECT_ROOT}/.venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH:-}"
echo "BAND_EVALS_START $(date -u +%FT%TZ) host=$(hostname) out=${OUT_S3:?}"
if [ -n "${CODE_S3_URI:-}" ]; then
  AWS_BIN="$(command -v aws || echo "${PROJECT_ROOT}/.venv/bin/aws")"
  "${AWS_BIN}" s3 cp --only-show-errors "${CODE_S3_URI}" /tmp/rl-distill-code.tar.gz
  tar -xzf /tmp/rl-distill-code.tar.gz --unlink-first --recursive-unlink -C "${PROJECT_ROOT}"
  echo "CODE_REFRESHED $(sha256sum /tmp/rl-distill-code.tar.gz | cut -c1-16)"
fi
if [ -f .env ]; then set -a; source .env; set +a; fi
: "${HF_TOKEN:?HF_TOKEN missing (forward it with --dotenv-keys)}"
PY="${PROJECT_ROOT}/.venv/bin/python"
"${PY}" -c "import vllm; print('VLLM', vllm.__version__)"
command -v aws >/dev/null || { echo "FATAL: aws CLI missing" >&2; exit 2; }

# name=repo@revision, space separated
MODELS="${MODELS:-Qwen3-1.7B-Base=Qwen/Qwen3-1.7B-Base@ea980cb0a6c2ae4b936e82123acc929f1cec04c1 Qwen3-4B-Base=Qwen/Qwen3-4B-Base@906bfd4b4dc7f14ee4320094d8b41684abff8539}"
SETS="${SETS:-teacher8b_easy teacher8b_medium teacher8b_hard teacher30b_easy teacher30b_medium teacher30b_hard}"
SETS_S3="${SETS_S3:-s3://scale-ml/genai/rl-distill/qwen3-deepscaler-samples/band_sets}"
N="${N:-16}"; MAX_TOKENS="${MAX_TOKENS:-8192}"; MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-4096}"
# Single-quoted default: inside ${VAR:-...} bash ends the default at the first "}" and mangles "\boxed{}".
DEFAULT_INSTRUCTION='Please reason step by step, and put your final answer within \boxed{}.'
PROMPT_INSTRUCTION="${PROMPT_INSTRUCTION:-${DEFAULT_INSTRUCTION}}"
echo "PROMPT_INSTRUCTION=${PROMPT_INSTRUCTION}"
TEMPLATE=rl-distill-scripts/data/qwen3_base_native_chat.jinja
NL=$'\n'; STOP_STRINGS="<|im_end|>,${NL}Human:,${NL}user${NL}"; STOP_TOKEN_IDS=151645,151643,151644
WORK="${WORK:-/tmp/band_evals}"
n_gpus="${N_GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)}"
export HF_HOME=/tmp/hf_cache
mkdir -p "${WORK}/sets"
aws s3 sync --only-show-errors "${SETS_S3}" "${WORK}/sets"

tasks=()
for spec in ${MODELS}; do
  name="${spec%%=*}"; repo_rev="${spec#*=}"
  "${PY}" - <<PY
from huggingface_hub import snapshot_download
snapshot_download("${repo_rev%@*}", revision="${repo_rev#*@}", local_dir="${WORK}/models/${name}")
PY
  for s in ${SETS}; do
    if aws s3 ls "${OUT_S3}/${name}/${s}/samples.parquet" >/dev/null 2>&1; then echo "TASK_ON_S3 ${name} ${s}"; else tasks+=("${name} ${s}"); fi
  done
done
echo "TASKS_TODO ${#tasks[@]} gpus=${n_gpus}"

pids=()
for g in $(seq 0 $((n_gpus - 1))); do
  (
    set -e
    for ((i = g; i < ${#tasks[@]}; i += n_gpus)); do
      read -r m s <<<"${tasks[$i]}"; out="${WORK}/out/${m}/${s}"; mkdir -p "${out}"
      CUDA_VISIBLE_DEVICES="${g}" "${PY}" rl-distill-scripts/data/generate_deepscaler_4bpt_samples.py \
        --model "${WORK}/models/${m}" --input "${WORK}/sets/${s}.parquet" --chat-template "${TEMPLATE}" \
        --output "${out}/samples.parquet" --shard-id 0 --num-shards 1 --n "${N}" \
        --max-tokens "${MAX_TOKENS}" --max-model-len $((MAX_TOKENS + MAX_PROMPT_TOKENS)) --max-prompt-tokens "${MAX_PROMPT_TOKENS}" \
        --temperature 1.0 --top-p 1.0 --top-k -1 --stop-strings "${STOP_STRINGS}" --stop-token-ids "${STOP_TOKEN_IDS}" \
        --append-boxed-instruction --boxed-instruction "${PROMPT_INSTRUCTION}" --attention-backend TRITON_ATTN \
        --chunk-size 300 --max-num-seqs 512 --gpu-memory-utilization 0.9 > "${out}/log.txt" 2>&1 \
        || { echo "FATAL: ${m} ${s} failed" >&2; tail -n 30 "${out}/log.txt" >&2; aws s3 cp --only-show-errors "${out}/log.txt" "${OUT_S3}/${m}/${s}/log.failed.txt"; exit 1; }
      aws s3 cp --only-show-errors "${out}/log.txt" "${OUT_S3}/${m}/${s}/log.txt"
      aws s3 cp --only-show-errors "${out}/samples.parquet.manifest.json" "${OUT_S3}/${m}/${s}/samples.parquet.manifest.json"
      aws s3 cp --only-show-errors "${out}/samples.parquet" "${OUT_S3}/${m}/${s}/samples.parquet"   # last: marks the task done
      echo "TASK_DONE $(date -u +%FT%TZ) ${m} ${s}"
    done
  ) &
  pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "${p}" || rc=1; done
if [ "${rc}" != 0 ]; then echo "FATAL: a band eval task failed" >&2; exit 1; fi
echo "{\"n\": ${N}, \"max_tokens\": ${MAX_TOKENS}, \"models\": \"${MODELS}\", \"sets\": \"${SETS}\"}" > "${WORK}/RUN_COMPLETE.json"
aws s3 cp --only-show-errors "${WORK}/RUN_COMPLETE.json" "${OUT_S3}/RUN_COMPLETE.json"
echo "BAND_EVALS_DONE $(date -u +%FT%TZ)"
