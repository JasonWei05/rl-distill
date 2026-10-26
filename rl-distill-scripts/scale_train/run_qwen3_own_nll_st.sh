#!/usr/bin/env bash
# ScaleTrain pod entry: own-sample NLL per token (entropy estimate) of the base -> base FORWARD-KL distilled students and their
# bases, scored on each model's own pass@32 eval samples (s3://.../qwen3-passk32-evals-v1/<name>/<band>/samples.parquet):
#   q4b-base, q30b-base (medium + hard) and q4b30b-<band>-s{250,500,750,1000} (4B base -> 30B base off-policy distillation).
# Tasks are pulled from a shared queue by n_gpus/TP vLLM engines; each writes ${OUT_S3}/<name>/<band>/nll.json (skip if present).
#   launched by launch_qwen3_own_nll_st.sh
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "${PROJECT_ROOT}"
export PATH="${PROJECT_ROOT}/.venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH:-}"
echo "OWN_NLL_START $(date -u +%FT%TZ) host=$(hostname)"
if [ -n "${CODE_S3_URI:-}" ]; then
  AWS_BIN="$(command -v aws || echo "${PROJECT_ROOT}/.venv/bin/aws")"
  "${AWS_BIN}" s3 cp --only-show-errors "${CODE_S3_URI}" /tmp/rl-distill-code.tar.gz
  tar -xzf /tmp/rl-distill-code.tar.gz --unlink-first --recursive-unlink -C "${PROJECT_ROOT}"
fi
if [ -f .env ]; then set -a; source .env; set +a; fi
: "${HF_TOKEN:?HF_TOKEN missing (forward it with --dotenv-keys)}"
export PATH="${HOME}/.local/bin:/root/.local/bin:${PATH}"
command -v uv >/dev/null 2>&1 || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="${HOME}/.local/bin:${PATH}"; }
export VENV="${VENV:-/tmp/.venv-gemma4}" GEMMA4_CUDA_VARIANT="${GEMMA4_CUDA_VARIANT:-cu129}"
[ -x "${VENV}/bin/python" ] || VENV="${VENV}" bash rl-distill-scripts/setup_env_gemma4.sh
export PATH="${VENV}/bin:${PATH}"; source "${VENV}/bin/activate"
if [ -x /usr/local/cuda/bin/nvcc ]; then export CUDA_HOME="${CUDA_HOME:-$(readlink -f /usr/local/cuda)}"; export PATH="${CUDA_HOME}/bin:${PATH}"; fi
python -c "import torch, vllm" || { echo "FATAL: venv import check failed" >&2; exit 2; }
export HF_HOME="${HF_HOME:-/tmp/hf_cache}" HF_HUB_CACHE="${HF_HUB_CACHE:-/tmp/hf_cache/hub}"
n_gpus="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
TP="${EVAL_TP:-2}"; n_slots=$((n_gpus / TP)); (( n_slots >= 1 )) || { echo "FATAL: needs >= ${TP} GPUs" >&2; exit 2; }
unset CUDA_VISIBLE_DEVICES

EVAL_S3=s3://scale-ml/genai/rl-distill/qwen3-passk32-evals-v1
OUT_S3="${OUT_S3:-s3://scale-ml/genai/rl-distill/qwen3-own-nll-v1}"; PER_Q="${PER_QUESTION:-8}"
DISTILL_S3=s3://scale-ml/genai/rl-distill/qwen3-4b-base-distill-ckpts-v1
Q30B=hf:Qwen/Qwen3-30B-A3B-Base@1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9
Q4B=hf:Qwen/Qwen3-4B-Base@906bfd4b4dc7f14ee4320094d8b41684abff8539
student() { echo "s3:${DISTILL_S3}/q4b-base-$1-to-q30b-bs128-s1000-lr2e-6/global_step_$2/huggingface"; }
ALL_TASKS=("q4b-base medium ${Q4B}" "q4b-base hard ${Q4B}" "q30b-base medium ${Q30B}" "q30b-base hard ${Q30B}")
for b in medium hard; do for s in 250 500 750 1000; do ALL_TASKS+=("q4b30b-${b}-s${s} ${b} $(student "${b}" "${s}")"); done; done
WORK=/tmp/own_nll; mkdir -p "${WORK}/models"
tasks=()
for t in "${ALL_TASKS[@]}"; do
  read -r name band src <<<"${t}"
  if aws s3 ls "${OUT_S3}/${name}/${band}/nll.json" >/dev/null 2>&1; then echo "ON_S3 ${name}/${band}"; else tasks+=("${t}"); fi
done
echo "OWN_NLL todo ${#tasks[@]} / ${#ALL_TASKS[@]} slots=${n_slots}xTP${TP} per_question=${PER_Q}"
(( ${#tasks[@]} > 0 )) || { echo "OWN_NLL_DONE $(date -u +%FT%TZ) (all on S3)"; exit 0; }

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
echo 0 > "${WORK}/next"
claim_task() { ( flock 8; i="$(cat "${WORK}/next")"; (( i < ${#tasks[@]} )) && { echo "${i}"; echo $((i + 1)) > "${WORK}/next"; } || true ) 8>"${WORK}/next.lock"; }
run_task() {   # slot gpus name band src
  local slot="$1" gpus="$2" name="$3" band="$4" src="$5" out model
  out="${WORK}/out/${name}/${band}"; mkdir -p "${out}"
  aws s3 cp --only-show-errors "${EVAL_S3}/${name}/${band}/samples.parquet" "${out}/samples.parquet"
  model="$(fetch_model "${src}")"
  echo "TASK_START $(date -u +%FT%TZ) ${name}/${band} slot=${slot} gpus=${gpus}"
  CUDA_VISIBLE_DEVICES="${gpus}" VLLM_CACHE_ROOT="/tmp/vllm_cache_slot${slot}" TORCHINDUCTOR_CACHE_DIR="/tmp/inductor_cache_slot${slot}" \
    TRITON_CACHE_DIR="/tmp/triton_cache_slot${slot}" "${VENV}/bin/python" rl-distill-scripts/data/score_own_sample_nll.py \
    --model "${model}" --samples "${out}/samples.parquet" --chat-template rl-distill-scripts/data/qwen3_base_native_chat.jinja \
    --per-question "${PER_Q}" --tensor-parallel-size "${TP}" --output "${out}/nll.json" > "${out}/log.txt" 2>&1 \
    || { tail -n 30 "${out}/log.txt" >&2; aws s3 cp --only-show-errors "${out}/log.txt" "${OUT_S3}/${name}/${band}/log.failed.txt"; return 1; }
  aws s3 cp --only-show-errors "${out}/log.txt" "${OUT_S3}/${name}/${band}/log.txt"
  aws s3 cp --only-show-errors "${out}/nll.json" "${OUT_S3}/${name}/${band}/nll.json"
  echo "TASK_DONE $(date -u +%FT%TZ) ${name}/${band} $(cat "${out}/nll.json" | tr -d '\n ' | cut -c1-200)"
}
slot_loop() {   # slot gpus
  local i
  while i="$(claim_task)"; [ -n "${i}" ]; do
    read -r name band src <<<"${tasks[${i}]}"
    run_task "$1" "$2" "${name}" "${band}" "${src}" || echo "TASK_FAILED ${name}/${band}" >&2
  done
}
pids=()
for ((s = 0; s < n_slots; s++)); do
  gpus="$(seq -s, $((s * TP)) $((s * TP + TP - 1)))"
  slot_loop "${s}" "${gpus}" & pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait "${p}" || rc=1; done
missing=0; for t in "${ALL_TASKS[@]}"; do read -r name band src <<<"${t}"; aws s3 ls "${OUT_S3}/${name}/${band}/nll.json" >/dev/null 2>&1 || missing=$((missing + 1)); done
echo "OWN_NLL_END $(date -u +%FT%TZ) missing=${missing}"
(( missing == 0 )) || exit 1
