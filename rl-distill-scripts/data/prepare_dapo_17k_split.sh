#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [ -f "${PROJECT_ROOT}/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "${PROJECT_ROOT}/.env"
    set +a
fi

DATA_DIR="${DATA_DIR:-${HOME}/verl/data}"
OVERWRITE="${OVERWRITE:-0}"
SEED="${DAPO_SPLIT_SEED:-42}"
TEST_SIZE="${DAPO_TEST_SIZE:-100}"
SOURCE_MODE="${SOURCE_MODE:-split_repo}"
SOURCE_FILE="${SOURCE_FILE:-${DATA_DIR}/dapo-math-17k.parquet}"
TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/dapo_17k_train.parquet}"
TEST_FILE="${TEST_FILE:-${DATA_DIR}/dapo_17k_test.parquet}"
SOURCE_TRAIN_FILE="${SOURCE_TRAIN_FILE:-${DATA_DIR}/dapo_17k_source_train.parquet}"
SOURCE_VALIDATION_FILE="${SOURCE_VALIDATION_FILE:-${DATA_DIR}/dapo_17k_source_validation.parquet}"
EVAL_REPEAT="${DAPO_EVAL_REPEAT:-16}"
HELDOUT_JSON="${HELDOUT_JSON:-${DATA_DIR}/dapo_17k_test_seed${SEED}_n${TEST_SIZE}.heldout.json}"
HELDOUT_TXT="${HELDOUT_TXT:-${DATA_DIR}/dapo_17k_test_seed${SEED}_n${TEST_SIZE}.heldout.txt}"

mkdir -p "${DATA_DIR}"

if [[ "${SOURCE_MODE}" == "split_repo" ]]; then
    if [[ ! -f "${SOURCE_TRAIN_FILE}" || "${OVERWRITE}" -eq 1 ]]; then
        echo "Downloading JWei05/DAPO-17.4k source train split..."
        wget -q --show-progress -O "${SOURCE_TRAIN_FILE}" \
            "https://huggingface.co/datasets/JWei05/DAPO-17.4k/resolve/main/data/train.parquet?download=true"
    else
        echo "Source train split found: ${SOURCE_TRAIN_FILE}"
    fi

    if [[ ! -f "${SOURCE_VALIDATION_FILE}" || "${OVERWRITE}" -eq 1 ]]; then
        echo "Downloading JWei05/DAPO-17.4k source validation split..."
        wget -q --show-progress -O "${SOURCE_VALIDATION_FILE}" \
            "https://huggingface.co/datasets/JWei05/DAPO-17.4k/resolve/main/data/validation.parquet?download=true"
    else
        echo "Source validation split found: ${SOURCE_VALIDATION_FILE}"
    fi

    if [[ ! -f "${TRAIN_FILE}" || ! -f "${TEST_FILE}" || "${OVERWRITE}" -eq 1 ]]; then
        python3 - <<PY
import json
from pathlib import Path
import pandas as pd

train_path = Path("${TRAIN_FILE}")
test_path = Path("${TEST_FILE}")
source_train_path = Path("${SOURCE_TRAIN_FILE}")
source_validation_path = Path("${SOURCE_VALIDATION_FILE}")
heldout_json = Path("${HELDOUT_JSON}")
heldout_txt = Path("${HELDOUT_TXT}")
seed = int("${SEED}")
test_size = int("${TEST_SIZE}")
eval_repeat = int("${EVAL_REPEAT}")

source_train = pd.read_parquet(source_train_path)
source_validation = pd.read_parquet(source_validation_path)
source = pd.concat([source_train, source_validation], ignore_index=True)
if test_size >= len(source):
    raise SystemExit(f"test_size={test_size} must be smaller than source size {len(source)}")
if eval_repeat <= 0:
    raise SystemExit(f"eval_repeat must be positive, got {eval_repeat}")

def set_split(x, split):
    out = dict(x) if isinstance(x, dict) else {}
    out["split"] = split
    return out

def get_idx(x, fallback):
    if isinstance(x, dict) and "index" in x:
        return str(x["index"])
    return str(fallback)

test_base = source.sample(n=test_size, random_state=seed).copy()
train = source.drop(index=test_base.index).copy()

train["extra_info"] = train["extra_info"].apply(lambda x: set_split(x, "train"))
test_base["extra_info"] = test_base["extra_info"].apply(lambda x: set_split(x, "test"))

test_base = test_base.reset_index(drop=False).rename(columns={"index": "source_row_idx"})
test_base["uid"] = [
    f"dapo17k-eval-{get_idx(extra_info, source_row_idx)}"
    for extra_info, source_row_idx in zip(test_base["extra_info"], test_base["source_row_idx"], strict=True)
]
test = pd.concat([test_base.copy() for _ in range(eval_repeat)], ignore_index=True)
test["eval_repeat_idx"] = [i % eval_repeat for i in range(len(test))]
test = test.sample(frac=1.0, random_state=seed + 1).reset_index(drop=True)
train = train.sample(frac=1.0, random_state=seed).reset_index(drop=True)

train.to_parquet(train_path, index=False)
test.to_parquet(test_path, index=False)

train_idx = {get_idx(x, i) for i, x in enumerate(train["extra_info"].tolist())}
test_idx = [get_idx(x, i) for i, x in enumerate(test_base["extra_info"].tolist())]
overlap = train_idx.intersection(test_idx)
if overlap:
    raise SystemExit(f"train/test overlap by extra_info.index: {len(overlap)}")

heldout = {
    "source": [
        "JWei05/DAPO-17.4k/data/train.parquet",
        "JWei05/DAPO-17.4k/data/validation.parquet",
    ],
    "seed": seed,
    "test_size_unique": len(test_base),
    "eval_repeat": eval_repeat,
    "test_rows": len(test),
    "extra_info_index": test_idx,
    "uid": test_base["uid"].tolist(),
    "source_row_idx": [int(i) for i in test_base["source_row_idx"].tolist()],
}
heldout_json.write_text(json.dumps(heldout, indent=2) + "\\n")
heldout_txt.write_text("\\n".join(test_idx) + "\\n")
uid_counts = test["uid"].value_counts()
if len(uid_counts) != test_size or uid_counts.min() != eval_repeat or uid_counts.max() != eval_repeat:
    raise SystemExit(f"bad uid repeat distribution: {uid_counts.describe()}")
print(f"train rows: {len(train)} -> {train_path}")
print(f"test unique prompts: {len(test_base)}")
print(f"test rows:  {len(test)} ({eval_repeat}x each) -> {test_path}")
print(f"heldout:    {heldout_json}")
print(f"heldout txt:{heldout_txt}")
PY
    else
        echo "Train split found: ${TRAIN_FILE}"
        echo "Test split found:  ${TEST_FILE}"
    fi
elif [[ "${SOURCE_MODE}" == "source" ]]; then
    if [[ ! -f "${SOURCE_FILE}" || "${OVERWRITE}" -eq 1 ]]; then
        echo "Downloading DAPO-Math-17k source parquet..."
        wget -q --show-progress -O "${SOURCE_FILE}" \
            "https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k/resolve/main/data/dapo-math-17k.parquet?download=true"
    else
        echo "Source file found: ${SOURCE_FILE}"
    fi

    python3 "${SCRIPT_DIR}/split_dapo_17k.py" \
        --input "${SOURCE_FILE}" \
        --output-train "${TRAIN_FILE}" \
        --output-test "${TEST_FILE}" \
        --heldout-json "${HELDOUT_JSON}" \
        --heldout-txt "${HELDOUT_TXT}" \
        --test-size "${TEST_SIZE}" \
        --seed "${SEED}"
else
    echo "Unknown SOURCE_MODE=${SOURCE_MODE}; expected split_repo or source" >&2
    exit 2
fi

echo "DAPO train: ${TRAIN_FILE}"
echo "DAPO test:  ${TEST_FILE}"
