#!/usr/bin/env bash
# One off-policy top-128 forward-KL distillation run: TEACHER_SPEC's v2 traces -> STUDENT base.
#
#   TEACHER_SPEC=12b-easy STUDENT=e4b DISTILL_GPU_IDS=0,1 bash run_gemma4_distill_one.sh
#
# Pipeline (all steps idempotent / cached under /tmp):
#   1. locate the teacher's COMPLETE trace bundle (local trace root, else synced from S3)
#   2. build the derived training view (build_gemma4_distill_training_view.py)
#   3. snapshot the student base model; compute the teacher/student identity SHAs the
#      audited launcher pins
#   4. exec gemma4_topk_distill_fsdp2.sh with the study's recipe: global batch 64, 1000 steps
#      (epochs capped at 100 so steps bind), lr 2e-6, 100 warmup, linear decay to 2e-7
#
# Students: e4b -> google/gemma-4-E4B (2 GPUs), e2b -> google/gemma-4-E2B (1 GPU). A teacher's
# trace set is reused for both students; the recorded direction label is passed through and
# the student is verified by identity SHA (see preflight_gemma4_distill_training_view.py).

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

TEACHER_SPEC="${TEACHER_SPEC:?e.g. 12b-easy (any trace spec of run_gemma4_bestckpt_trace_collection.sh)}"
STUDENT="${STUDENT:?e4b or e2b}"
DISTILL_GPU_IDS="${DISTILL_GPU_IDS:?comma-separated physical GPU indices for this run}"
IFS=',' read -r -a GPU_IDS <<< "${DISTILL_GPU_IDS}"

case "${STUDENT}" in
  e4b) STUDENT_REPO=google/gemma-4-E4B ;;
  e2b) STUDENT_REPO=google/gemma-4-E2B ;;
  *) echo "FATAL: STUDENT must be e4b or e2b, got ${STUDENT}" >&2; exit 2 ;;
esac
# Pin to an immutable revision for reproducible identities (default resolves `main` and logs it).
STUDENT_REVISION="${STUDENT_REVISION:-main}"

TRACE_S3_BASE="${TRACE_S3_BASE:-s3://scale-ml/genai/rl-distill/gemma4-bestckpt-traces-topk128-v2}"
TRACE_LOCAL_ROOT="${TRACE_LOCAL_ROOT:-/tmp/gemma4_bestckpt_traces_v2/${TEACHER_SPEC}}"
VIEW_S3_BASE="${VIEW_S3_BASE:-s3://scale-ml/genai/rl-distill/gemma4-distill-views-v2}"
SOURCE_ROOT="${SOURCE_ROOT:-/tmp/gemma4_distill_sources/${TEACHER_SPEC}}"
VIEW_ROOT="${VIEW_ROOT:-/tmp/gemma4_distill_views/${TEACHER_SPEC}}"
STUDENTS_ROOT="${STUDENTS_ROOT:-/tmp/gemma4_distill_students}"

# Band trace bundles: 3000 train questions x 8 samples + 300 validation questions x 1 sample.
# Default view: train on ALL 3000 x 8 = 24,000 rows; validate every TEST_FREQ steps on 128 of the
# teacher's own validation-split generations (VALIDATION_SOURCE=validation). VALIDATION_SOURCE=train
# restores the old behavior of carving validation questions out of the train roster.
SOURCE_QUESTIONS="${SOURCE_QUESTIONS:-3000}"
VALIDATION_SOURCE="${VALIDATION_SOURCE:-validation}"
TRAIN_QUESTIONS="${TRAIN_QUESTIONS:-3000}"
VALIDATION_QUESTIONS="${VALIDATION_QUESTIONS:-128}"
TRAIN_SAMPLES_PER_QUESTION="${TRAIN_SAMPLES_PER_QUESTION:-8}"
VALIDATION_SAMPLE_INDEX="${VALIDATION_SAMPLE_INDEX:-0}"
VIEW_SEED="${VIEW_SEED:-42}"

export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-west-2}"
export AWS_REGION="${AWS_REGION:-us-west-2}"
if [[ -z "${AWS_PROFILE:-}" ]] && aws configure list-profiles 2>/dev/null | grep -qx ml-worker; then
  export AWS_PROFILE=ml-worker
fi
if [[ -f .env ]]; then set -a; source .env; set +a; fi

export VENV="${VENV:-/tmp/.venv-gemma4}"
if [[ ! -x ${VENV}/bin/python ]]; then
  echo "FATAL: VENV=${VENV} has no python; build it with rl-distill-scripts/setup_env_gemma4.sh" >&2
  exit 2
fi
PY="${VENV}/bin/python"
export HF_HOME="${HF_HOME:-/tmp/hf_cache}"

echo "DISTILL_ONE teacher=${TEACHER_SPEC} student=${STUDENT} gpus=${DISTILL_GPU_IDS} nproc=${#GPU_IDS[@]}"

# --- 1. teacher trace bundle -----------------------------------------------------------------
if [[ -f ${TRACE_LOCAL_ROOT}/dataset_index.json && -f ${TRACE_LOCAL_ROOT}/COMPLETE.json ]]; then
  SOURCE_ROOT="${TRACE_LOCAL_ROOT}"
  echo "DISTILL_ONE bundle=local ${SOURCE_ROOT}"
else
  if ! aws s3 ls "${TRACE_S3_BASE}/${TEACHER_SPEC}/COMPLETE.json" >/dev/null 2>&1; then
    echo "FATAL: ${TEACHER_SPEC} trace bundle is not complete (no COMPLETE.json at ${TRACE_S3_BASE}/${TEACHER_SPEC})" >&2
    exit 3
  fi
  mkdir -p "${SOURCE_ROOT}"
  echo "DISTILL_ONE bundle=s3 ${TRACE_S3_BASE}/${TEACHER_SPEC} -> ${SOURCE_ROOT}"
  aws s3 sync --only-show-errors "${TRACE_S3_BASE}/${TEACHER_SPEC}" "${SOURCE_ROOT}"
  if ! compgen -G "${SOURCE_ROOT}/source/*.parquet" >/dev/null; then
    echo "FATAL: the view builder needs the bundle's source/ roster parquet; it is not in S3 for ${TEACHER_SPEC}." >&2
    echo "       Run this on the node that generated the traces (local root ${TRACE_LOCAL_ROOT}) or copy source/ over." >&2
    exit 3
  fi
fi

# --- 2. derived training view ----------------------------------------------------------------
if [[ -f ${VIEW_ROOT}/dataset_index.json ]]; then
  echo "DISTILL_ONE view=cached ${VIEW_ROOT}"
else
  echo "DISTILL_ONE view=build ${VIEW_ROOT}"
  # --source-s3-uri is provenance metadata recorded in the view (no S3 access needed); the
  # view is only uploaded when DISTILL_S3_ENABLE=true (default) — set false on nodes without
  # scale-ml S3 write access to run fully locally.
  view_s3_args=()
  if [[ ${DISTILL_S3_ENABLE:-true} == true ]]; then view_s3_args=(--output-s3-uri "${VIEW_S3_BASE}/${TEACHER_SPEC}"); fi
  "${PY}" rl-distill-scripts/data/build_gemma4_distill_training_view.py \
    --source-root "${SOURCE_ROOT}" \
    --output-root "${VIEW_ROOT}" \
    --source-s3-uri "${TRACE_S3_BASE}/${TEACHER_SPEC}" \
    "${view_s3_args[@]}" \
    --seed "${VIEW_SEED}" \
    --train-questions "${TRAIN_QUESTIONS}" \
    --validation-questions "${VALIDATION_QUESTIONS}" \
    --train-samples-per-question "${TRAIN_SAMPLES_PER_QUESTION}" \
    --validation-sample-index "${VALIDATION_SAMPLE_INDEX}" \
    --validation-source "${VALIDATION_SOURCE}" \
    --expected-source-questions "${SOURCE_QUESTIONS}" \
    --expected-source-samples-per-question "${TRAIN_SAMPLES_PER_QUESTION}"
fi

# --- 3. student snapshot + pinned identities -------------------------------------------------
mkdir -p "${STUDENTS_ROOT}"
IDENTITY_JSON="$(cd rl-distill-scripts/data && "${PY}" - "${STUDENT_REPO}" "${STUDENT_REVISION}" "${STUDENTS_ROOT}" "${VIEW_ROOT}/dataset_index.json" <<'PY'
import json, sys
from pathlib import Path

from huggingface_hub import snapshot_download

import preflight_gemma4_topk_distill as source_preflight
from gemma4_distill_trace_schema import hash_json

repo, revision, students_root, view_index_path = sys.argv[1:5]
snap = Path(snapshot_download(repo, revision=revision))  # HF cache path ends in snapshots/<commit>
commit = snap.name
model_dir = Path(students_root) / f"{repo.split('/')[-1]}-{commit}"
if not model_dir.exists():
    import shutil
    shutil.copytree(snap, model_dir, symlinks=False)

view = json.loads(Path(view_index_path).read_text())
teacher = view.get("teacher")
if not isinstance(teacher, dict):
    teacher = (view.get("generation") or {}).get("teacher")
if not isinstance(teacher, dict):
    raise SystemExit("view dataset_index has no teacher identity block")
# Emit shell assignments (no jq dependency on the node).
print(f"MODEL_PATH={model_dir}")
print(f"STUDENT_COMMIT={commit}")
print(f"EXPECTED_TEACHER_IDENTITY_SHA256={hash_json(teacher)}")
print(f"EXPECTED_STUDENT_IDENTITY_SHA256={source_preflight._student_identity_sha256(str(model_dir), None)}")
print(f"DISTILL_DIRECTION={view['direction']}")
PY
)"
eval "${IDENTITY_JSON}"
echo "DISTILL_ONE student_snapshot=${MODEL_PATH} (${STUDENT_REPO}@${STUDENT_COMMIT})"
echo "DISTILL_ONE direction=${DISTILL_DIRECTION} teacher_identity=${EXPECTED_TEACHER_IDENTITY_SHA256} student_identity=${EXPECTED_STUDENT_IDENTITY_SHA256}"

# --- 4. launch --------------------------------------------------------------------------------
export MODEL_PATH DISTILL_DIRECTION EXPECTED_TEACHER_IDENTITY_SHA256 EXPECTED_STUDENT_IDENTITY_SHA256
export DATASET_INDEX="${VIEW_ROOT}/dataset_index.json"
export EXPECTED_TRAIN_QUESTIONS="${TRAIN_QUESTIONS}"
export EXPECTED_VALIDATION_QUESTIONS="${VALIDATION_QUESTIONS}"
export EXPECTED_TRAIN_SAMPLES_PER_QUESTION="${TRAIN_SAMPLES_PER_QUESTION}"
export EXPECTED_VALIDATION_SAMPLES_PER_QUESTION=1
export NPROC_PER_NODE="${#GPU_IDS[@]}"
export CUDA_VISIBLE_DEVICES="${DISTILL_GPU_IDS}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
export MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-1}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-1000}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-100}"
export LR="${LR:-2e-6}"
export LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-100}"
export LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-linear}"
export MIN_LR_RATIO="${MIN_LR_RATIO:-0.1}"   # 2e-6 * 0.1 = 2e-7 final LR
export PROJECT_NAME="${PROJECT_NAME:-gemma4-bestckpt-distill-v2}"
export EXP_NAME="${EXP_NAME:-${TEACHER_SPEC}-to-${STUDENT}-base-bs${TRAIN_BATCH_SIZE}-s${TOTAL_TRAINING_STEPS}}"
# Logging: console + wandb (project PROJECT_NAME); validation KL on the held-out rows every TEST_FREQ steps.
export TRAIN_LOGGER="${TRAIN_LOGGER:-[\"console\",\"wandb\"]}"
export TEST_FREQ="${TEST_FREQ:-10}"
# Checkpoints: no periodic local saves (SAVE_FREQ=0 -> the trainer saves only at the final step), the
# final save contains just the HF export (no FSDP/Adam shards), it is pushed to the Hub as
# <HF_PUSH_REPO>/step_<N>/ and the local copy is deleted after a successful upload.
export SAVE_FREQ="${SAVE_FREQ:-0}"
export CHECKPOINT_SAVE_CONTENTS="${CHECKPOINT_SAVE_CONTENTS:-[\"hf_model\"]}"
export HF_PUSH_ENABLE="${HF_PUSH_ENABLE:-true}"
export HF_PUSH_REPO="${HF_PUSH_REPO:-JWei05/Distill-gemma4-${TEACHER_SPEC}-to-${STUDENT}-base}"
export HF_PUSH_PRIVATE="${HF_PUSH_PRIVATE:-false}"
export HF_PUSH_DELETE_LOCAL="${HF_PUSH_DELETE_LOCAL:-true}"
export PYTHON_BIN="${PY}"
echo "DISTILL_ONE launch exp=${EXP_NAME} bs=${TRAIN_BATCH_SIZE} steps=${TOTAL_TRAINING_STEPS} epochs_cap=${TOTAL_EPOCHS} lr=${LR} warmup=${LR_WARMUP_STEPS} ${LR_SCHEDULER_TYPE}->min_ratio ${MIN_LR_RATIO}"
exec bash rl-distill-scripts/gemma4_topk_distill_fsdp2.sh
