#!/usr/bin/env bash
# ScaleTrain pod entry (1 GPU): reverse KL of the distilled 12B (and, for reference, the untrained 12B base) vs the E4B base
# teacher on 128 medium-train + 128 medium-validation questions, student top-128 tokens. See reverse_kl_topk.py.
#   --env-vars "STUDENT_SIZE=12b|26b,MODELS=distilled,base"   (defaults: 12b, both)   STUDENT_REPO/REVISION/SUBFOLDER override the pin.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "${PROJECT_ROOT}"
export PATH="${PROJECT_ROOT}/.venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH:-}"
echo "ST_RKL_START $(date -u +%FT%TZ) host=$(hostname)"
if [ -n "${CODE_S3_URI:-}" ]; then
  AWS_BIN="$(command -v aws || echo "${PROJECT_ROOT}/.venv/bin/aws")"
  "${AWS_BIN}" s3 cp --only-show-errors "${CODE_S3_URI}" /tmp/rl-distill-code.tar.gz
  tar -xzf /tmp/rl-distill-code.tar.gz --unlink-first --recursive-unlink -C "${PROJECT_ROOT}"; echo "CODE_REFRESHED"
fi
if [ -f .env ]; then set -a; source .env; set +a; fi
: "${HF_TOKEN:?HF_TOKEN missing}"
export PATH="${HOME}/.local/bin:/root/.local/bin:${PATH}"
command -v uv >/dev/null 2>&1 || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="${HOME}/.local/bin:${PATH}"; }
export VENV="${VENV:-/tmp/.venv-gemma4}" GEMMA4_CUDA_VARIANT="${GEMMA4_CUDA_VARIANT:-cu129}"
[ -x "${VENV}/bin/python" ] || VENV="${VENV}" bash rl-distill-scripts/setup_env_gemma4.sh
export PATH="${VENV}/bin:${PATH}"; source "${VENV}/bin/activate"
python -c "import torch, vllm, verl, transformers" || { echo "FATAL: venv import check failed" >&2; exit 2; }
export HF_HOME="${HF_HOME:-/tmp/hf_cache}" HF_HUB_CACHE="${HF_HUB_CACHE:-/tmp/hf_cache/hub}" TRITON_CACHE_DIR=/tmp/triton_cache
MODELS="${MODELS:-distilled,base}"
STUDENT_SIZE="${STUDENT_SIZE:-12b}"      # 12b | 26b : which distilled student (and matching untrained base) to measure
case "${STUDENT_SIZE}" in
  12b) BASE_REPO=google/gemma-4-12B;     BASE_REV=023679ed352de9bb66cc873c9009ce3482585c08; DEFAULT_STUDENT_REPO=JWei05/Distill-gemma4-e4b-base-medium-to-12b-base; DEFAULT_STUDENT_REV=92368d1f020e685436113d93dc1f75c64033570f ;;
  26b) BASE_REPO=google/gemma-4-26B-A4B; BASE_REV=24548b62aa021d562695c04aaf7758a1ea47990b; DEFAULT_STUDENT_REPO=JWei05/Distill-gemma4-e4b-base-medium-to-26b-base; DEFAULT_STUDENT_REV=dbf634856e38998d25740f4f4ac7da48416929f1 ;;
  *) echo "FATAL: STUDENT_SIZE must be 12b or 26b" >&2; exit 2 ;;
esac
STUDENT_REPO="${STUDENT_REPO:-${DEFAULT_STUDENT_REPO}}"; STUDENT_REVISION="${STUDENT_REVISION:-${DEFAULT_STUDENT_REV}}"; STUDENT_SUBFOLDER="${STUDENT_SUBFOLDER:-step_001000}"
TEACHER_REV=411aa17b749aa952df1359d2dcea73917a544d9a
DATA_REPO=JWei05/DeepScaleR-Easy-Medium-Hard-Gemma-26B-PT-10k; DATA_REV=a0ba3c3dc07c7bc27e901670ceb1a0b0ceeaa8db
QUESTIONS="${QUESTIONS_PER_SPLIT:-128}"; SAMPLES="${SAMPLES_PER_QUESTION:-4}"; TOPK="${TOPK:-128}"
S3_ROOT="${RKL_S3_ROOT:-s3://scale-ml/genai/rl-distill/gemma4-e4b-base-reverse-kl-v1}"
WORK=/tmp/gemma4_rkl; mkdir -p "${WORK}/data" "${WORK}/models"
echo "### data + models"
python - <<PY
from huggingface_hub import hf_hub_download, snapshot_download
for split in ("train", "validation"):
    print(hf_hub_download(repo_id="${DATA_REPO}", filename=f"medium/{split}.parquet", revision="${DATA_REV}", repo_type="dataset", local_dir="${WORK}/data"))
print("teacher", snapshot_download("google/gemma-4-E4B", revision="${TEACHER_REV}"))
PY
TEACHER="$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download('google/gemma-4-E4B', revision='${TEACHER_REV}'))")"
declare -A STUDENT_PATHS
for m in ${MODELS//,/ }; do
  case "$m" in
    distilled)
      STUDENT_PATHS[distilled]="${WORK}/models/distilled_${STUDENT_SIZE}_${STUDENT_SUBFOLDER}"
      python rl-distill-scripts/data/download_hf_subfolder.py --repo-id "${STUDENT_REPO}" --revision "${STUDENT_REVISION}" --subfolder "${STUDENT_SUBFOLDER}" \
        --output-dir "${STUDENT_PATHS[distilled]}" --overwrite --metadata-repo "${BASE_REPO}" --metadata-revision "${BASE_REV}" ;;
    base)
      STUDENT_PATHS[base]="$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download('${BASE_REPO}', revision='${BASE_REV}'))")" ;;
    *) echo "FATAL: unknown model tag ${m}" >&2; exit 2 ;;
  esac
done
for m in ${MODELS//,/ }; do
  TAG="${STUDENT_SIZE}_${m}__vs_e4b_base__medium_q${QUESTIONS}_s${SAMPLES}_top${TOPK}"; OUT="${WORK}/results/${TAG}"; mkdir -p "${OUT}"
  echo "### ${TAG}: student=${STUDENT_PATHS[$m]}"
  python rl-distill-scripts/reverse_kl_topk.py generate --student "${STUDENT_PATHS[$m]}" --train_parquet "${WORK}/data/medium/train.parquet" \
    --val_parquet "${WORK}/data/medium/validation.parquet" --questions_per_split "${QUESTIONS}" --samples_per_question "${SAMPLES}" --topk "${TOPK}" --trace_dir "${OUT}/traces"
  python rl-distill-scripts/reverse_kl_topk.py score --student "${STUDENT_PATHS[$m]}" --teacher "${TEACHER}" --train_parquet "${WORK}/data/medium/train.parquet" \
    --val_parquet "${WORK}/data/medium/validation.parquet" --questions_per_split "${QUESTIONS}" --samples_per_question "${SAMPLES}" --topk "${TOPK}" \
    --trace_dir "${OUT}/traces" --out "${OUT}/metrics.json"
  aws s3 sync "${OUT}" "${S3_ROOT}/${TAG}/" --only-show-errors && echo "UPLOADED ${S3_ROOT}/${TAG}/"
done
echo "ST_RKL_DONE $(date -u +%FT%TZ)"
