#!/usr/bin/env bash
# ScaleTrain pod entry (8 GPUs): pass@k (k <= 32) evals on the Qwen3-30B-defined DeepScaleR validation splits (300 q per band, the RL
# val sets) for the Qwen3-4B-Base -> Qwen3-30B-A3B-Base distillation comparison:
#   q30b-base        Qwen3-30B-A3B-Base                 medium + hard
#   q4b-base         Qwen3-4B-Base (the distill teacher) medium + hard
#   q4b30b-<b>-s<N>  30B student distilled from 4B-base <b>-band traces, step N in 250/500/750/1000, on its own band
# Protocol = the Qwen band-eval protocol at n=32: Qwen 0-shot ChatML + boxed instruction, temp 1, top-p 1, 8192 tokens, strict boxed
# math_verify (= the RL reward) -> mean@32, unbiased pass@k / pass^k for k in 1..32.
# Tasks are pulled from a shared queue by n_gpus/2 TP-2 vLLM engines (step-1000 students and the bases first); each task writes
# log.txt / samples.parquet / grades.json / summary.json (last = done) under ${EVAL_S3}/<model>/<band>/, so a relaunch skips them.
#   launched by launch_qwen3_passk32_evals_st.sh / start_qwen3_passk32_evals_supervisor.sh
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "${PROJECT_ROOT}"
export PATH="${PROJECT_ROOT}/.venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH:-}"
echo "PASSK32_START $(date -u +%FT%TZ) host=$(hostname)"
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
python -c "import torch, vllm, verl" || { echo "FATAL: venv import check failed" >&2; exit 2; }
command -v aws >/dev/null || { echo "FATAL: aws CLI missing (models and evals live on S3)" >&2; exit 2; }
export HF_HOME="${HF_HOME:-/tmp/hf_cache}" HF_HUB_CACHE="${HF_HUB_CACHE:-/tmp/hf_cache/hub}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-${NETWORK_INTERFACE_NAME:-eth0}}" NCCL_SOCKET_FAMILY="${NCCL_SOCKET_FAMILY:-AF_INET6}" GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${NETWORK_INTERFACE_NAME:-eth0}}"
n_gpus="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
TP="${EVAL_TP:-2}"; n_slots=$((n_gpus / TP))
(( n_slots >= 1 )) || { echo "FATAL: needs >= ${TP} GPUs, got ${n_gpus}" >&2; exit 2; }
unset CUDA_VISIBLE_DEVICES

EVAL_S3="${EVAL_S3:-s3://scale-ml/genai/rl-distill/qwen3-passk32-evals-v1}"; N="${EVAL_N:-32}"
DISTILL_S3=s3://scale-ml/genai/rl-distill/qwen3-4b-base-distill-ckpts-v1
Q30B=hf:Qwen/Qwen3-30B-A3B-Base@1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9
Q4B=hf:Qwen/Qwen3-4B-Base@906bfd4b4dc7f14ee4320094d8b41684abff8539
student() { echo "s3:${DISTILL_S3}/q4b-base-$1-to-q30b-bs128-s1000-lr2e-6/global_step_$2/huggingface"; }
# name band source -- queue order = priority (longest hard tasks first within each tier)
ALL_TASKS=(
  "q30b-base hard ${Q30B}"           "q4b30b-hard-s1000 hard $(student hard 1000)"
  "q30b-base medium ${Q30B}"         "q4b30b-medium-s1000 medium $(student medium 1000)"
  "q4b-base hard ${Q4B}"             "q4b-base medium ${Q4B}"
  "q4b30b-hard-s250 hard $(student hard 250)"     "q4b30b-hard-s500 hard $(student hard 500)"     "q4b30b-hard-s750 hard $(student hard 750)"
  "q4b30b-medium-s250 medium $(student medium 250)" "q4b30b-medium-s500 medium $(student medium 500)" "q4b30b-medium-s750 medium $(student medium 750)"
)
WORK=/tmp/passk32_evals; mkdir -p "${WORK}/models"
tasks=()
for t in "${ALL_TASKS[@]}"; do
  read -r name band src <<<"${t}"
  if aws s3 ls "${EVAL_S3}/${name}/${band}/summary.json" >/dev/null 2>&1; then echo "EVAL_ON_S3 ${name}/${band}"; else tasks+=("${t}"); fi
done
echo "PASSK32 evals todo ${#tasks[@]} / ${#ALL_TASKS[@]} slots=${n_slots}xTP${TP} n=${N} ($(date -u +%FT%TZ)) out=${EVAL_S3}"
(( ${#tasks[@]} > 0 )) || { echo "PASSK32_DONE $(date -u +%FT%TZ) (all on S3)"; exit 0; }

for band in medium hard; do
  "${VENV}/bin/python" - "${WORK}/${band}_validation.parquet" "${band}" <<'PY'
import shutil, sys
from huggingface_hub import hf_hub_download
# the Qwen3-30B-defined band validation split = the RL validation set (raw problems; the generator appends the boxed instruction)
p = hf_hub_download("JWei05/DeepScaleR-Easy-Medium-Hard-Qwen3-30B-A3B-Base-10k", f"{sys.argv[2]}/validation.parquet", repo_type="dataset",
                    revision="be4dcd8b1fad7d5eb7bf721f0210baf78277aaa9")
shutil.copy(p, sys.argv[1])
PY
done

# fetch_model <source> -> prints the local model dir (serialized per model with flock: two slots can need the same base model)
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
# claim_task -> prints the next unclaimed task index (or nothing when the queue is empty)
claim_task() { ( flock 8; i="$(cat "${WORK}/next")"; (( i < ${#tasks[@]} )) && { echo "${i}"; echo $((i + 1)) > "${WORK}/next"; } || true ) 8>"${WORK}/next.lock"; }

run_task() {   # slot gpus name band src
  local slot="$1" gpus="$2" name="$3" band="$4" src="$5" out model attempt
  out="${WORK}/out/${name}/${band}"; mkdir -p "${out}"
  model="$(fetch_model "${src}")"
  echo "EVAL_START $(date -u +%FT%TZ) ${name}/${band} slot=${slot} gpus=${gpus} model=${src}"
  for attempt in 1 2; do   # one retry: concurrent vLLM engine start-up can lose a compile-cache race
    CUDA_VISIBLE_DEVICES="${gpus}" VLLM_CACHE_ROOT="/tmp/vllm_cache_slot${slot}" TORCHINDUCTOR_CACHE_DIR="/tmp/inductor_cache_slot${slot}" \
      TRITON_CACHE_DIR="/tmp/triton_cache_slot${slot}" "${VENV}/bin/python" rl-distill-scripts/data/generate_deepscaler_4bpt_samples.py \
      --model "${model}" --input "${WORK}/${band}_validation.parquet" --chat-template rl-distill-scripts/data/qwen3_base_native_chat.jinja \
      --output "${out}/samples.parquet" --shard-id 0 --num-shards 1 --n "${N}" \
      --max-tokens 8192 --max-model-len 12288 --max-prompt-tokens 4096 \
      --temperature 1.0 --top-p 1.0 --top-k -1 --stop-strings "${STOP_STRINGS}" --stop-token-ids "${STOP_TOKEN_IDS}" \
      --append-boxed-instruction --boxed-instruction "${DEFAULT_INSTRUCTION}" --attention-backend TRITON_ATTN \
      --tensor-parallel-size "${TP}" --chunk-size 300 --max-num-seqs 512 --gpu-memory-utilization 0.9 > "${out}/log.txt" 2>&1 && break
    echo "EVAL_ATTEMPT_FAILED ${name}/${band} attempt=${attempt}: $(grep -aE 'Error|error' "${out}/log.txt" | tail -1 | cut -c1-200)" >&2
    if (( attempt == 2 )); then
      tail -n 30 "${out}/log.txt" >&2; aws s3 cp --only-show-errors "${out}/log.txt" "${EVAL_S3}/${name}/${band}/log.failed.txt"; return 1
    fi
    sleep 30
  done
  PYTHONPATH="${PROJECT_ROOT}" "${VENV}/bin/python" rl-distill-scripts/data/grade_band_eval_samples.py "${out}/samples.parquet"
  "${VENV}/bin/python" - "${out}/grades.json" "${out}/summary.json" "${name}" "${band}" "${src}" <<'PY'
import json, sys
from math import comb
rows = json.load(open(sys.argv[1])); n = len(rows[0]["acc"])
c = [sum(r["acc"]) for r in rows]; q = len(rows)
ks = [k for k in (1, 2, 4, 8, 16, 32) if k <= n]
summary = {"model": sys.argv[3], "band": sys.argv[4], "source": sys.argv[5], "questions": q, "n": n, "mean_at_n": 100 * sum(c) / (q * n),
           "pass_at_k": {k: 100 * sum(1 - comb(n - ci, k) / comb(n, k) for ci in c) / q for k in ks},
           "pass_hat_k": {k: 100 * sum(comb(ci, k) / comb(n, k) for ci in c) / q for k in ks},
           "boxed_frac": sum(r["boxed"] for r in rows) / (q * n), "trunc_frac": sum(r["trunc"] for r in rows) / (q * n),
           "mean_response_len": sum(r["mean_len"] for r in rows) / q}
json.dump(summary, open(sys.argv[2], "w"), indent=2)
print("EVAL_SUMMARY", json.dumps(summary))
PY
  for f in log.txt samples.parquet grades.json summary.json; do aws s3 cp --only-show-errors "${out}/${f}" "${EVAL_S3}/${name}/${band}/${f}"; done   # summary last: marks done
  echo "EVAL_DONE $(date -u +%FT%TZ) ${name}/${band}"
}

pids=()
for ((slot = 0; slot < n_slots; slot++)); do
  gpus="$(seq -s, $((slot * TP)) $((slot * TP + TP - 1)))"
  (
    set -e
    sleep $((slot * ${EVAL_SLOT_STAGGER_S:-20}))
    while i="$(claim_task)"; [ -n "${i}" ]; do
      read -r name band src <<<"${tasks[$i]}"
      run_task "${slot}" "${gpus}" "${name}" "${band}" "${src}"
    done
  ) &
  pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait "${p}" || rc=1; done
(( rc == 0 )) || { echo "FATAL: an eval task failed" >&2; exit 1; }
echo "PASSK32_DONE $(date -u +%FT%TZ) evals=${EVAL_S3}"
