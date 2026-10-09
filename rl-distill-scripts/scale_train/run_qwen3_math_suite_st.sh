#!/usr/bin/env bash
# ScaleTrain pod entry: the Qwen math suite for 4B-class checkpoints, one vLLM engine (TP 1) per GPU pulling (model, set) tasks
# from a shared queue. Sets: the Qwen3-30B-defined DeepScaleR easy / medium / hard validation splits (300 q each, the RL val
# sets), GSM8K test (1,319), MATH500 (500), AIME 2025 and AIME 2026 (30 each), at the pinned revisions of the Gemma 4 suite
# (data/prepare_gemma4_three_model_eval_data.py). Samples per question: 16, AIME 32 (= the x32 duplication).
# Protocol = the Qwen RL prompt contract: 0-shot native ChatML + boxed instruction, temp 1, top-p 1, 8192 response tokens,
# strict last-box math_verify (= the RL reward) -> mean@n, unbiased pass@k / pass^k for k = 1, 2, 4, ... n.
# Each task writes log.txt / samples.parquet / grades.json / summary.json (last = done) under ${EVAL_S3}/<model>/<set>/;
# a relaunch skips finished tasks.
#   launched by launch_qwen3_math_suite_st.sh / start_qwen3_math_suite_supervisor.sh
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "${PROJECT_ROOT}"
export PATH="${PROJECT_ROOT}/.venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH:-}"
echo "MATH_SUITE_START $(date -u +%FT%TZ) host=$(hostname)"
if [ -n "${CODE_S3_URI:-}" ]; then
  AWS_BIN="$(command -v aws || echo "${PROJECT_ROOT}/.venv/bin/aws")"
  "${AWS_BIN}" s3 cp --only-show-errors "${CODE_S3_URI}" /tmp/rl-distill-code.tar.gz
  tar -xzf /tmp/rl-distill-code.tar.gz --unlink-first --recursive-unlink -C "${PROJECT_ROOT}"
  echo "CODE_REFRESHED $(sha256sum /tmp/rl-distill-code.tar.gz | cut -c1-16)"
fi
if [ -f .env ]; then set -a; source .env; set +a; fi
: "${HF_TOKEN:?HF_TOKEN missing (forward it with --dotenv-keys)}"
export PATH="${HOME}/.local/bin:/root/.local/bin:${PATH}"
command -v uv >/dev/null 2>&1 || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="${HOME}/.local/bin:${PATH}"; }
export VENV="${VENV:-/tmp/.venv-gemma4}" GEMMA4_CUDA_VARIANT="${GEMMA4_CUDA_VARIANT:-cu129}"   # p5 fleet: CUDA 12.8 driver
[ -x "${VENV}/bin/python" ] || VENV="${VENV}" bash rl-distill-scripts/setup_env_gemma4.sh
export PATH="${VENV}/bin:${PATH}"; source "${VENV}/bin/activate"
if [ -x /usr/local/cuda/bin/nvcc ]; then export CUDA_HOME="${CUDA_HOME:-$(readlink -f /usr/local/cuda)}"; export PATH="${CUDA_HOME}/bin:${PATH}"; fi
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
python -c "import torch, vllm, verl, datasets" || { echo "FATAL: venv import check failed" >&2; exit 2; }
command -v aws >/dev/null || { echo "FATAL: aws CLI missing (models and evals live on S3)" >&2; exit 2; }
export HF_HOME="${HF_HOME:-/tmp/hf_cache}" HF_HUB_CACHE="${HF_HUB_CACHE:-/tmp/hf_cache/hub}"
# grader pinned to the RL reward defaults (strict last well-formed box)
export VERL_MATH_VERIFY_STRICT_BOXED=1 VERL_MATH_VERIFY_TIMEOUT=30 VERL_MATH_SYMPY_TIMEOUT=5.0
n_gpus="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
unset CUDA_VISIBLE_DEVICES

EVAL_S3="${EVAL_S3:-s3://scale-ml/genai/rl-distill/qwen3-math-suite-v1}"
RL=s3://scale-ml/genai/rl-distill/qwen3-4b-base-rl
DIST=s3://scale-ml/genai/rl-distill/qwen3-30b-rl-distill-ckpts-v1
# name=source (hf:<repo>@<rev> | s3:<HF export prefix>)
MODELS="${MODELS:-q4b-rl-medium-best=s3:${RL}/qwen3-4b-medium-es5/best_hf q4b-rl-hard-best=s3:${RL}/qwen3-4b-hard-es5/best_hf q30bm-to-q4b-s1000=s3:${DIST}/q30bm-medium-to-q4b-bs128-s1000-lr2e-6/global_step_1000/huggingface q30bh-to-q4b-s1000=s3:${DIST}/q30bh-hard-to-q4b-bs128-s1000-lr2e-6/global_step_1000/huggingface q4b-base=hf:Qwen/Qwen3-4B-Base@906bfd4b4dc7f14ee4320094d8b41684abff8539}"
# queue order: longest sets first so the tail is short
SETS="${SETS:-hard aime2025 aime2026 medium math500 easy gsm8k}"
n_for() { case "$1" in aime*) echo "${AIME_N:-32}" ;; *) echo "${SET_N:-16}" ;; esac; }
WORK=/tmp/qwen3_math_suite; mkdir -p "${WORK}/data" "${WORK}/models"
tasks=()
for s in ${SETS}; do for m in ${MODELS}; do
  name="${m%%=*}"
  if aws s3 ls "${EVAL_S3}/${name}/${s}/summary.json" >/dev/null 2>&1; then echo "EVAL_ON_S3 ${name}/${s}"; else tasks+=("${name} ${s} ${m#*=}"); fi
done; done
echo "MATH_SUITE todo ${#tasks[@]} slots=${n_gpus}xTP1 ($(date -u +%FT%TZ)) out=${EVAL_S3}"
(( ${#tasks[@]} > 0 )) || { echo "MATH_SUITE_DONE $(date -u +%FT%TZ) (all on S3)"; exit 0; }

"${VENV}/bin/python" - "${WORK}/data" <<'PY'
import shutil, sys
from pathlib import Path
import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download
out = Path(sys.argv[1])
for band in ("easy", "medium", "hard"):   # the RL validation sets (raw problems; the generator appends the boxed instruction)
    p = hf_hub_download("JWei05/DeepScaleR-Easy-Medium-Hard-Qwen3-30B-A3B-Base-10k", f"{band}/validation.parquet", repo_type="dataset",
                        revision="be4dcd8b1fad7d5eb7bf721f0210baf78277aaa9")
    shutil.copy(p, out / f"{band}.parquet")
# OOD sets: same sources / revisions / answer extraction as data/prepare_gemma4_three_model_eval_data.py
SOURCES = {
    "math500": ("HuggingFaceH4/MATH-500", None, "test", "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be", 500),
    "gsm8k": ("openai/gsm8k", "main", "test", "740312add88f781978c0658806c59bc2815b9866", 1319),
    "aime2025": ("MathArena/aime_2025", None, "train", "c94da77eb22bbd6439e62a323bec18493a421302", 30),
    "aime2026": ("MathArena/aime_2026", None, "train", "d2de22f3c656b4f56cf8981212186377d1e23bc3", 30),
}
for name, (repo, config, split, rev, expected) in SOURCES.items():
    rows = list(load_dataset(repo, config, split=split, revision=rev))
    assert len(rows) == expected, (name, len(rows))
    if name == "gsm8k":
        df = pd.DataFrame({"uid": [f"gsm8k-{i}" for i in range(len(rows))], "problem": [r["question"] for r in rows],
                           "answer": [str(r["answer"]).split("####")[-1].strip() for r in rows]})
    else:
        ids = [r.get("unique_id", r.get("problem_idx", i + 1)) for i, r in enumerate(rows)]
        df = pd.DataFrame({"uid": [f"{name}-{x}" for x in ids], "problem": [r["problem"] for r in rows], "answer": [str(r["answer"]).strip() for r in rows]})
    assert (df["answer"].str.len() > 0).all(), name
    df.to_parquet(out / f"{name}.parquet", index=False)
    print("DATA", name, len(df))
PY

# fetch_model <source> -> prints the local model dir (serialized per model with flock: several slots can need the same model)
fetch_model() {
  local src="$1" key dir
  key="$(echo -n "${src}" | sha256sum | cut -c1-16)"; dir="${WORK}/models/${key}"
  (
    flock 9
    if [ ! -f "${dir}/.complete" ]; then
      case "${src}" in
        hf:*) local spec="${src#hf:}"
              "${VENV}/bin/python" -c "import sys; from huggingface_hub import snapshot_download; snapshot_download(sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3])" \
                "${spec%@*}" "${spec#*@}" "${dir}" >&2 ;;
        s3:*) aws s3 sync --only-show-errors "${src#s3:}/" "${dir}/" >&2 ;;
      esac
      touch "${dir}/.complete"
    fi
  ) 9>"${dir}.lock"
  echo "${dir}"
}

DEFAULT_INSTRUCTION='Please reason step by step, and put your final answer within \boxed{}.'
NL=$'\n'; STOP_STRINGS="<|im_end|>,${NL}Human:,${NL}user${NL}"; STOP_TOKEN_IDS=151645,151643,151644
echo 0 > "${WORK}/next"
claim_task() { ( flock 8; i="$(cat "${WORK}/next")"; (( i < ${#tasks[@]} )) && { echo "${i}"; echo $((i + 1)) > "${WORK}/next"; } || true ) 8>"${WORK}/next.lock"; }

run_task() {   # gpu name set src
  local gpu="$1" name="$2" set="$3" src="$4" out model attempt n
  out="${WORK}/out/${name}/${set}"; mkdir -p "${out}"; n="$(n_for "${set}")"
  model="$(fetch_model "${src}")"
  echo "EVAL_START $(date -u +%FT%TZ) ${name}/${set} gpu=${gpu} n=${n}"
  for attempt in 1 2; do   # one retry: concurrent vLLM engine start-up can lose a compile-cache race
    CUDA_VISIBLE_DEVICES="${gpu}" VLLM_CACHE_ROOT="/tmp/vllm_cache_gpu${gpu}" TORCHINDUCTOR_CACHE_DIR="/tmp/inductor_cache_gpu${gpu}" \
      TRITON_CACHE_DIR="/tmp/triton_cache_gpu${gpu}" "${VENV}/bin/python" rl-distill-scripts/data/generate_deepscaler_4bpt_samples.py \
      --model "${model}" --input "${WORK}/data/${set}.parquet" --chat-template rl-distill-scripts/data/qwen3_base_native_chat.jinja \
      --output "${out}/samples.parquet" --shard-id 0 --num-shards 1 --n "${n}" \
      --max-tokens 8192 --max-model-len 12288 --max-prompt-tokens 4096 \
      --temperature 1.0 --top-p 1.0 --top-k -1 --stop-strings "${STOP_STRINGS}" --stop-token-ids "${STOP_TOKEN_IDS}" \
      --append-boxed-instruction --boxed-instruction "${DEFAULT_INSTRUCTION}" --attention-backend TRITON_ATTN \
      --chunk-size 300 --max-num-seqs 512 --gpu-memory-utilization 0.9 > "${out}/log.txt" 2>&1 && break
    echo "EVAL_ATTEMPT_FAILED ${name}/${set} attempt=${attempt}: $(grep -aE 'Error|error' "${out}/log.txt" | tail -1 | cut -c1-200)" >&2
    if (( attempt == 2 )); then
      tail -n 30 "${out}/log.txt" >&2; aws s3 cp --only-show-errors "${out}/log.txt" "${EVAL_S3}/${name}/${set}/log.failed.txt"; return 1
    fi
    sleep 30
  done
  PYTHONPATH="${PROJECT_ROOT}" "${VENV}/bin/python" rl-distill-scripts/data/grade_band_eval_samples.py "${out}/samples.parquet"
  "${VENV}/bin/python" - "${out}/grades.json" "${out}/summary.json" "${name}" "${set}" "${src}" <<'PY'
import json, sys
from math import comb
rows = json.load(open(sys.argv[1])); n = len(rows[0]["acc"])
c = [sum(r["acc"]) for r in rows]; q = len(rows)
ks = [k for k in (1, 2, 4, 8, 16, 32) if k <= n]
summary = {"model": sys.argv[3], "set": sys.argv[4], "source": sys.argv[5], "questions": q, "n": n, "mean_at_n": 100 * sum(c) / (q * n),
           "pass_at_k": {k: 100 * sum(1 - comb(n - ci, k) / comb(n, k) for ci in c) / q for k in ks},
           "pass_hat_k": {k: 100 * sum(comb(ci, k) / comb(n, k) for ci in c) / q for k in ks},
           "boxed_frac": sum(r["boxed"] for r in rows) / (q * n), "trunc_frac": sum(r["trunc"] for r in rows) / (q * n),
           "mean_response_len": sum(r["mean_len"] for r in rows) / q}
json.dump(summary, open(sys.argv[2], "w"), indent=2)
print("EVAL_SUMMARY", json.dumps(summary))
PY
  for f in log.txt samples.parquet grades.json summary.json; do aws s3 cp --only-show-errors "${out}/${f}" "${EVAL_S3}/${name}/${set}/${f}"; done   # summary last: marks done
  echo "EVAL_DONE $(date -u +%FT%TZ) ${name}/${set}"
}

pids=()
for ((g = 0; g < n_gpus; g++)); do
  (
    set -e
    sleep $((g * ${EVAL_SLOT_STAGGER_S:-20}))
    while i="$(claim_task)"; [ -n "${i}" ]; do
      read -r name set src <<<"${tasks[$i]}"
      run_task "${g}" "${name}" "${set}" "${src}"
    done
  ) &
  pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait "${p}" || rc=1; done
(( rc == 0 )) || { echo "FATAL: an eval task failed" >&2; exit 1; }
echo "MATH_SUITE_DONE $(date -u +%FT%TZ) evals=${EVAL_S3}"
