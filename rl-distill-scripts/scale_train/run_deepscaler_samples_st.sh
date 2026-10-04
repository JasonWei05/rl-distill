#!/usr/bin/env bash
# ScaleTrain run-file: sample N responses per question for the full deduplicated DeepScaleR set (38,796 questions)
# with one vLLM generator per GPU (DP = #GPUs, question shards by source_index % DP), preemption-safe via S3:
#   - on (re)start, every finished chunk under OUT_S3 is restored, so generate_deepscaler_4bpt_samples.py skips it
#     (CHUNK_REUSED; its manifest check refuses to mix settings);
#   - a background loop mirrors finished chunk parquets (never *.tmp) to OUT_S3 every SYNC_SECONDS;
#   - when every shard finishes, the merged shard outputs + all.parquet + RUN_COMPLETE.json are uploaded.
# Env: MODEL_REPO, MODEL_REVISION (pinned), OUT_S3 (required); PROMPT_INSTRUCTION, N, MAX_TOKENS, CHUNK_SIZE optional.
#   bash launch_st_with_code.sh --gpus-per-instance 4 --priority high --allow-borrowing \
#     --run-file run_deepscaler_samples_st.sh --job-name q3-8b-ds-n8 \
#     --env-vars "MODEL_REPO=Qwen/Qwen3-8B-Base,MODEL_REVISION=<sha>,OUT_S3=s3://.../qwen3-8b-base"
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "${PROJECT_ROOT}"
export PATH="${PROJECT_ROOT}/.venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH:-}"
echo "DS_SAMPLES_START $(date -u +%FT%TZ) host=$(hostname) model=${MODEL_REPO:?}@${MODEL_REVISION:?} out=${OUT_S3:?}"
if [ -n "${CODE_S3_URI:-}" ]; then
  AWS_BIN="$(command -v aws || echo "${PROJECT_ROOT}/.venv/bin/aws")"
  "${AWS_BIN}" s3 cp --only-show-errors "${CODE_S3_URI}" /tmp/rl-distill-code.tar.gz
  tar -xzf /tmp/rl-distill-code.tar.gz --unlink-first --recursive-unlink -C "${PROJECT_ROOT}"
  echo "CODE_REFRESHED $(sha256sum /tmp/rl-distill-code.tar.gz | cut -c1-16)"
fi
if [ -f .env ]; then set -a; source .env; set +a; fi
: "${HF_TOKEN:?HF_TOKEN missing (forward it with --dotenv-keys)}"
PY="${PROJECT_ROOT}/.venv/bin/python"
if ! "${PY}" -c "import vllm, pandas" 2>/dev/null; then
  export PATH="${HOME}/.local/bin:/root/.local/bin:${PATH}"
  command -v uv >/dev/null 2>&1 || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="${HOME}/.local/bin:${PATH}"; }
  export VENV="${VENV:-/tmp/.venv-gemma4}" GEMMA4_CUDA_VARIANT="${GEMMA4_CUDA_VARIANT:-cu129}"
  [ -x "${VENV}/bin/python" ] || VENV="${VENV}" bash rl-distill-scripts/setup_env_gemma4.sh
  PY="${VENV}/bin/python"
fi
"${PY}" -c "import vllm; print('VLLM', vllm.__version__)"
command -v aws >/dev/null || { echo "FATAL: aws CLI missing" >&2; exit 2; }

WORK="${WORK:-/tmp/ds_samples}"   # fixed paths: the per-shard manifest records them and must match across relaunches
N="${N:-8}"; MAX_TOKENS="${MAX_TOKENS:-8192}"; CHUNK_SIZE="${CHUNK_SIZE:-256}"; SYNC_SECONDS="${SYNC_SECONDS:-120}"
# Single-quoted default: inside ${VAR:-...} bash ends the default at the first "}" and mangles "\boxed{}".
DEFAULT_INSTRUCTION='Please reason step by step, and put your final answer within \boxed{}.'
PROMPT_INSTRUCTION="${PROMPT_INSTRUCTION:-${DEFAULT_INSTRUCTION}}"
echo "PROMPT_INSTRUCTION=${PROMPT_INSTRUCTION}"
INPUT_S3="${INPUT_S3:-s3://scale-ml/genai/rl-distill/data/deepscaler_it_gen_dedup_v2.parquet}"
TEMPLATE="${TEMPLATE:-rl-distill-scripts/data/qwen3_base_native_chat.jinja}"
NL=$'\n'
STOP_STRINGS="${STOP_STRINGS:-<|im_end|>,${NL}Human:,${NL}user${NL}}"   # base models continue as plain-text chat turns
STOP_TOKEN_IDS="${STOP_TOKEN_IDS:-151645,151643,151644}"
# Pin the backend: the image .venv's flash_attn.cute import crashes vLLM's backend auto-selection
# (AttributeError: cutlass.cute.core has no ThrMma); an explicit backend skips that probe.
ATTENTION_BACKEND="${ATTENTION_BACKEND:-TRITON_ATTN}"
n_gpus="${N_GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)}"
MAX_QUESTIONS="${MAX_QUESTIONS:-0}"   # >0 only for a local smoke test of this run-file
mkdir -p "${WORK}/out"
export HF_HOME=/tmp/hf_cache
"${PY}" - <<PY
from huggingface_hub import snapshot_download
snapshot_download("${MODEL_REPO}", revision="${MODEL_REVISION}", local_dir="${WORK}/model")
PY
aws s3 cp --only-show-errors "${INPUT_S3}" "${WORK}/input.parquet"
aws s3 sync --only-show-errors --exclude "*.tmp" "${OUT_S3}/out" "${WORK}/out" || true
echo "RESTORED parts=$(find "${WORK}/out" -name 'part_*.parquet' | wc -l) gpus=${n_gpus}"

( while true; do sleep "${SYNC_SECONDS}"; aws s3 sync --only-show-errors --exclude "*.tmp" "${WORK}/out" "${OUT_S3}/out" || true; done ) &
SYNC_PID=$!
pids=()
for g in $(seq 0 $((n_gpus - 1))); do
  CUDA_VISIBLE_DEVICES="${g}" "${PY}" rl-distill-scripts/data/generate_deepscaler_4bpt_samples.py \
    --model "${WORK}/model" --input "${WORK}/input.parquet" --chat-template "${TEMPLATE}" \
    --output "${WORK}/out/shard${g}.parquet" --shard-id "${g}" --num-shards "${n_gpus}" --n "${N}" \
    --max-tokens "${MAX_TOKENS}" --max-model-len $((MAX_TOKENS + 2048)) --max-prompt-tokens 2048 \
    --temperature 1.0 --top-p 1.0 --top-k -1 --stop-strings "${STOP_STRINGS}" --stop-token-ids "${STOP_TOKEN_IDS}" \
    --append-boxed-instruction --boxed-instruction "${PROMPT_INSTRUCTION}" \
    --attention-backend "${ATTENTION_BACKEND}" \
    --chunk-size "${CHUNK_SIZE}" --max-questions "${MAX_QUESTIONS}" --max-num-seqs 512 --gpu-memory-utilization 0.9 \
    > "${WORK}/out/shard${g}.log" 2>&1 &
  pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "${p}" || rc=1; done
kill "${SYNC_PID}" 2>/dev/null || true
aws s3 sync --only-show-errors --exclude "*.tmp" "${WORK}/out" "${OUT_S3}/out"
if [ "${rc}" != 0 ]; then
  echo "FATAL: a generator shard failed" >&2; tail -n 30 "${WORK}"/out/shard*.log >&2; exit 1
fi
"${PY}" - <<PY
import glob, json, pandas as pd
parts = sorted(glob.glob("${WORK}/out/shard*.parquet"))
df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True).sort_values("source_index").reset_index(drop=True)
assert len(df) == df["source_index"].nunique() and (${MAX_QUESTIONS} > 0 or len(df) == 38796), len(df)
df.to_parquet("${WORK}/all.parquet", index=False)
json.dump({"rows": len(df), "n": ${N}, "max_tokens": ${MAX_TOKENS}, "model": "${MODEL_REPO}@${MODEL_REVISION}",
           "mean_response_tokens": float(df["mean_len"].mean())}, open("${WORK}/RUN_COMPLETE.json", "w"), indent=2)
print("MERGED", len(df))
PY
aws s3 cp --only-show-errors "${WORK}/all.parquet" "${OUT_S3}/all.parquet"
aws s3 cp --only-show-errors "${WORK}/RUN_COMPLETE.json" "${OUT_S3}/RUN_COMPLETE.json"
echo "DS_SAMPLES_DONE $(date -u +%FT%TZ)"
