#!/usr/bin/env bash
# ScaleTrain pod entry (2 GPUs, no borrowing) for the per-checkpoint pass@k evals of the pre-training-control students
# (§9 of DISTILLATION_EXPERIMENTS.md). Polls the student's Hub repo, evaluates every new step_NNNNNN export on the band's
# 300-question validation set with the x32 protocol and uploads each finished step to S3; the plots are drawn locally
# from S3 (eval_student_checkpoints_passk.py --plot-from-s3).
#
#   --env-vars "STUDENT=12b"   -> dp 2 (one vLLM per GPU on question shards)
#   --env-vars "STUDENT=26b"   -> tp 2 (one vLLM tensor-parallel over both GPUs)
#   BANDS=medium (default) | medium,hard ; FINAL_STEP=1000 ; MAX_IDLE_HOURS=12
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"
# The pod's login shell drops the image PATH: put the baked FSDP2 venv (aws CLI) and the system dirs back first.
export PATH="${PROJECT_ROOT}/.venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH:-}"
echo "ST_PASSK_START $(date -u +%FT%TZ) host=$(hostname) commit=$(git rev-parse --short HEAD 2>/dev/null || echo baked)"

if [ -n "${CODE_S3_URI:-}" ]; then   # pre-built image: unpack the git-archive tarball of the commit to run
  echo "### refreshing repo code from ${CODE_S3_URI}"
  AWS_BIN="$(command -v aws || echo "${PROJECT_ROOT}/.venv/bin/aws")"
  "${AWS_BIN}" s3 cp --only-show-errors "${CODE_S3_URI}" /tmp/rl-distill-code.tar.gz
  tar -xzf /tmp/rl-distill-code.tar.gz --unlink-first --recursive-unlink -C "${PROJECT_ROOT}"
  echo "CODE_REFRESHED $(sha256sum /tmp/rl-distill-code.tar.gz | cut -c1-16) files=$(tar -tzf /tmp/rl-distill-code.tar.gz | wc -l)"
fi

STUDENT="${STUDENT:?12b or 26b}"
case "${STUDENT}" in
  12b) PARALLELISM="${PARALLELISM:-dp}" ;;
  26b) PARALLELISM="${PARALLELISM:-tp}" ;;
  *) echo "FATAL: STUDENT must be 12b or 26b" >&2; exit 2 ;;
esac
BANDS="${BANDS:-medium}"
FINAL_STEP="${FINAL_STEP:-1000}"
MAX_IDLE_HOURS="${MAX_IDLE_HOURS:-12}"
POLL_MINUTES="${POLL_MINUTES:-10}"
PASSK_S3_ROOT="${PASSK_S3_ROOT:-s3://scale-ml/genai/rl-distill/gemma4-e4b-base-student-passk-v1}"
DATA_ROOT="${DATA_ROOT:-/tmp/gemma4_e4b_val32/data}"
OUT_ROOT="${OUT_ROOT:-/tmp/gemma4_e4b_val32/students}"

if [ -f .env ]; then set -a; source .env; set +a; fi
: "${HF_TOKEN:?HF_TOKEN missing (forward it with --dotenv-keys)}"

# --- gemma-4 venv (vLLM 0.25 + verl grader) on local disk, same recipe as the training run-files -------------------
export PATH="${HOME}/.local/bin:/root/.local/bin:${PATH}"
command -v uv >/dev/null 2>&1 || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="${HOME}/.local/bin:${PATH}"; }
export VENV="${VENV:-/tmp/.venv-gemma4}"
export GEMMA4_CUDA_VARIANT="${GEMMA4_CUDA_VARIANT:-cu129}"
if [ ! -x "${VENV}/bin/python" ]; then
  echo "### building ${VENV} via setup_env_gemma4.sh (${GEMMA4_CUDA_VARIANT})"
  VENV="${VENV}" bash rl-distill-scripts/setup_env_gemma4.sh
fi
export PATH="${VENV}/bin:${PATH}"; source "${VENV}/bin/activate"
python -c "import torch, vllm, verl" || { echo "FATAL: venv import check failed" >&2; exit 2; }
command -v aws >/dev/null || { echo "FATAL: aws CLI missing (results go to S3)" >&2; exit 2; }
export HF_HOME="${HF_HOME:-/tmp/hf_cache}"; export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache}"

n_gpus="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
(( n_gpus >= 2 )) || { echo "FATAL: need 2 GPUs, got ${n_gpus}" >&2; exit 2; }
GPUS="$(seq -s, 0 $((n_gpus - 1)))"
echo "ST_PASSK config student=${STUDENT} parallelism=${PARALLELISM} gpus=${GPUS} bands=${BANDS} final_step=${FINAL_STEP} s3=${PASSK_S3_ROOT}"
echo "ST_PASSK disk: $(df -h /tmp | tail -1)"

# x32 validation protocol data (id_medium / id_hard at 32 samples per question), same command as the E4B-base reference.
python rl-distill-scripts/data/prepare_gemma4_rl_distill_eval_data.py --output-dir "${DATA_ROOT}" --overwrite \
  --samples-override "id_medium=32,id_hard=32" --protocol gemma4_rl_distill_math_eval_v2_x32
test -s "${DATA_ROOT}/math_eval_manifest.json"

REPO_ARGS=()
IFS=',' read -r -a band_list <<< "${BANDS}"
for band in "${band_list[@]}"; do REPO_ARGS+=(--repo "JWei05/Distill-gemma4-e4b-base-${band}-to-${STUDENT}-base"); done
# Every eval is exactly the RL reward grader (strict last-\boxed{}, 30 s verify, 5 s SymPy); the controller pins these too.
python rl-distill-scripts/eval_student_checkpoints_passk.py "${REPO_ARGS[@]}" \
  --gpus "${GPUS}" --parallelism "${PARALLELISM}" --poll-minutes "${POLL_MINUTES}" \
  --manifest "${DATA_ROOT}/math_eval_manifest.json" --out-root "${OUT_ROOT}" \
  --s3-root "${PASSK_S3_ROOT}" --no-plot --final-step "${FINAL_STEP}" --max-idle-hours "${MAX_IDLE_HOURS}"
status=$?
echo "ST_PASSK_DONE student=${STUDENT} exit=${status} $(date -u +%FT%TZ)"
exit "${status}"
