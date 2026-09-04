#!/usr/bin/env python3
"""Prepare the pinned easy/medium Gemma 4 RL datasets and >=8k validation sets.

The two training splits are downloaded from their immutable Hugging Face
revisions.  Their matching 500-question validation splits, GSM8K test, and
MATH-500 are repeated to at least ``--min-validation-rows`` rows while keeping
a stable per-question ``uid``.  Stable UIDs let verl report mean@k/pass@k/maj@k
over repeated generations instead of treating every duplicate as a new prompt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd

BOXED = "Please output the final answer within \\boxed{}."

EASY_REPO = "JWei05/DeepScaleR-Easy-10k"
EASY_REVISION = "0c3e81d98fad8783f6ab93cf3732ce58f159b555"
MEDIUM_REPO = "JWei05/DeepScaleR-Medium-20k"
MEDIUM_REVISION = "c3db94f80a3abe079fdf457fe01555544b8bc2dd"
GSM8K_REPO = "openai/gsm8k"
GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"
MATH500_REPO = "HuggingFaceH4/MATH-500"
MATH500_REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"

TRAIN_FILES = {
    "easy": "deepscaler_easy_10k_train.parquet",
    "medium": "deepscaler_medium_20k_train.parquet",
}
BASE_VAL_FILES = {
    "easy": "deepscaler_easy_10k_val500.parquet",
    "medium": "deepscaler_medium_20k_val500.parquet",
}
REPEATED_VAL_FILES = {
    "easy": "deepscaler_easy_10k_val500_x16.parquet",
    "medium": "deepscaler_medium_20k_val500_x16.parquet",
    "gsm8k": "math__gsm8k_test_x7.parquet",
    "math500": "math__math_500_x16.parquet",
}
EXPECTED_TRAIN_ROWS = {"easy": 9_500, "medium": 19_500}
EXPECTED_BASE_VAL_ROWS = {"easy": 500, "medium": 500, "gsm8k": 1_319, "math500": 500}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".parquet", delete=False) as handle:
        tmp_path = Path(handle.name)
    try:
        df.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".parquet", delete=False) as handle:
        tmp_path = Path(handle.name)
    try:
        shutil.copyfile(source, tmp_path)
        os.replace(tmp_path, destination)
    finally:
        tmp_path.unlink(missing_ok=True)


@contextmanager
def _data_lock(data_dir: Path):
    import fcntl

    data_dir.mkdir(parents=True, exist_ok=True)
    lock_path = data_dir / ".prepare_deepscaler_easy_medium.lock"
    with lock_path.open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def repeat_to_min_rows(
    base: pd.DataFrame,
    *,
    min_rows: int,
    uid_prefix: str,
) -> tuple[pd.DataFrame, int]:
    """Repeat every base question equally until the output reaches ``min_rows``."""
    if base.empty:
        raise ValueError(f"cannot repeat an empty validation set: {uid_prefix}")
    if min_rows < 1:
        raise ValueError("min_rows must be positive")

    normalized = base.copy().reset_index(drop=True)
    normalized["uid"] = [f"{uid_prefix}-{index}" for index in range(len(normalized))]
    repeat = math.ceil(min_rows / len(normalized))
    repeated = pd.concat([normalized] * repeat, ignore_index=True)
    return repeated, repeat


def _math_row(index: int, question: str, answer: str, data_source: str, tag: str) -> dict[str, Any]:
    question = question.strip()
    if not question.endswith(BOXED):
        question = f"{question} {BOXED}"
    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": question}],
        "reward_model": {"style": "rule", "ground_truth": str(answer)},
        "extra_info": {"index": str(index), "split": "test"},
        "uid": f"{tag}-{index}",
    }


def _load_public_math_validation() -> tuple[pd.DataFrame, pd.DataFrame]:
    import datasets

    gsm8k = datasets.load_dataset(GSM8K_REPO, "main", split="test", revision=GSM8K_REVISION)
    gsm_rows = [
        _math_row(
            index,
            row["question"],
            row["answer"].split("####")[-1].strip(),
            "gsm8k",
            "math__gsm8k_test",
        )
        for index, row in enumerate(gsm8k)
    ]

    math500 = datasets.load_dataset(MATH500_REPO, split="test", revision=MATH500_REVISION)
    math_rows = [
        _math_row(index, row["problem"], row["answer"], "math500", "math__math_500")
        for index, row in enumerate(math500)
    ]
    return pd.DataFrame(gsm_rows), pd.DataFrame(math_rows)


def _download_split(repo_id: str, revision: str, filename: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            filename=filename,
        )
    )


def _prompt_text(row: pd.Series) -> str:
    prompt = row["prompt"]
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    return json.dumps(prompt, sort_keys=True, ensure_ascii=False)


def validate_prepared(data_dir: Path, min_validation_rows: int) -> dict[str, Any]:
    details: dict[str, Any] = {"files": {}}
    for difficulty, expected_rows in EXPECTED_TRAIN_ROWS.items():
        path = data_dir / TRAIN_FILES[difficulty]
        frame = pd.read_parquet(path)
        if len(frame) != expected_rows:
            raise ValueError(f"{path}: expected {expected_rows} rows, found {len(frame)}")
        details["files"][path.name] = {"rows": len(frame), "sha256": _sha256(path)}

    for difficulty in ("easy", "medium"):
        train = pd.read_parquet(data_dir / TRAIN_FILES[difficulty])
        base_val = pd.read_parquet(data_dir / BASE_VAL_FILES[difficulty])
        overlap = {_prompt_text(row) for _, row in train.iterrows()} & {
            _prompt_text(row) for _, row in base_val.iterrows()
        }
        if overlap:
            raise ValueError(f"{difficulty}: train/validation prompt overlap detected ({len(overlap)})")

    expected_unique = {
        "easy": EXPECTED_BASE_VAL_ROWS["easy"],
        "medium": EXPECTED_BASE_VAL_ROWS["medium"],
        "gsm8k": EXPECTED_BASE_VAL_ROWS["gsm8k"],
        "math500": EXPECTED_BASE_VAL_ROWS["math500"],
    }
    for name, filename in REPEATED_VAL_FILES.items():
        path = data_dir / filename
        frame = pd.read_parquet(path)
        if len(frame) < min_validation_rows:
            raise ValueError(f"{path}: expected at least {min_validation_rows} rows, found {len(frame)}")
        if "uid" not in frame:
            raise ValueError(f"{path}: missing stable uid column")
        unique = frame["uid"].nunique()
        if unique != expected_unique[name]:
            raise ValueError(f"{path}: expected {expected_unique[name]} unique uids, found {unique}")
        counts = frame.groupby("uid", sort=False).size()
        if counts.nunique() != 1:
            raise ValueError(f"{path}: validation questions do not have an equal repeat count")
        details["files"][path.name] = {
            "rows": len(frame),
            "unique_questions": unique,
            "repeat": int(counts.iloc[0]),
            "sha256": _sha256(path),
        }
    return details


def prepare(data_dir: Path, min_validation_rows: int) -> dict[str, Any]:
    with _data_lock(data_dir):
        manifest_path = data_dir / "gemma4_difficulty_rl_data_manifest.json"
        expected_sources = {
            EASY_REPO: EASY_REVISION,
            MEDIUM_REPO: MEDIUM_REVISION,
            GSM8K_REPO: GSM8K_REVISION,
            MATH500_REPO: MATH500_REVISION,
        }
        if manifest_path.exists():
            try:
                existing = json.loads(manifest_path.read_text())
                if (
                    existing.get("schema_version") == 1
                    and existing.get("min_validation_rows") == min_validation_rows
                    and existing.get("sources") == expected_sources
                ):
                    details = validate_prepared(data_dir, min_validation_rows)
                    print(f"GEMMA4_DIFFICULTY_DATA_REUSED manifest={manifest_path}", flush=True)
                    return {**existing, **details}
            except (OSError, ValueError, json.JSONDecodeError):
                # Rebuild from immutable sources if a partial/stale runtime
                # directory was left by a preempted preparation step.
                pass

        easy_train = _download_split(EASY_REPO, EASY_REVISION, TRAIN_FILES["easy"])
        easy_val = _download_split(EASY_REPO, EASY_REVISION, BASE_VAL_FILES["easy"])
        medium_train = _download_split(MEDIUM_REPO, MEDIUM_REVISION, TRAIN_FILES["medium"])
        medium_val = _download_split(MEDIUM_REPO, MEDIUM_REVISION, BASE_VAL_FILES["medium"])

        for source, filename in (
            (easy_train, TRAIN_FILES["easy"]),
            (easy_val, BASE_VAL_FILES["easy"]),
            (medium_train, TRAIN_FILES["medium"]),
            (medium_val, BASE_VAL_FILES["medium"]),
        ):
            _atomic_copy(source, data_dir / filename)

        gsm8k, math500 = _load_public_math_validation()
        bases = {
            "easy": pd.read_parquet(data_dir / BASE_VAL_FILES["easy"]),
            "medium": pd.read_parquet(data_dir / BASE_VAL_FILES["medium"]),
            "gsm8k": gsm8k,
            "math500": math500,
        }
        prefixes = {
            "easy": "deepscaler_easy_10k_val500",
            "medium": "deepscaler_medium_20k_val500",
            "gsm8k": "math__gsm8k_test",
            "math500": "math__math_500",
        }
        for name, base in bases.items():
            repeated, repeat = repeat_to_min_rows(
                base,
                min_rows=min_validation_rows,
                uid_prefix=prefixes[name],
            )
            destination = data_dir / REPEATED_VAL_FILES[name]
            _atomic_parquet(repeated, destination)
            print(
                f"prepared {destination}: {len(base)} unique x{repeat} = {len(repeated)} rows",
                flush=True,
            )

        details = validate_prepared(data_dir, min_validation_rows)
        manifest = {
            "schema_version": 1,
            "min_validation_rows": min_validation_rows,
            "sources": expected_sources,
            **details,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(f"GEMMA4_DIFFICULTY_DATA_READY manifest={manifest_path}", flush=True)
        return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/tmp/verl/data")
    parser.add_argument("--min-validation-rows", type=int, default=8_000)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    if args.validate_only:
        details = validate_prepared(data_dir, args.min_validation_rows)
        print(json.dumps(details, indent=2, sort_keys=True))
        print("GEMMA4_DIFFICULTY_DATA_VALID", flush=True)
        return
    prepare(data_dir, args.min_validation_rows)


if __name__ == "__main__":
    main()
