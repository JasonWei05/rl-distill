#!/usr/bin/env python3
"""Materialize the pinned 26B-teacher difficulty bands for verl RL.

Each 300-question validation split is expanded to exactly 16 rows per stable
UID. The rollout sampler still generates one response per row, so verl emits
native ``mean@16`` and ``maj@16`` metrics without changing question identity.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import os
import shutil
from pathlib import Path

import pandas as pd

REPO_ID = "JWei05/DeepScaleR-Easy-Medium-Hard-Gemma-26B-PT-10k"
REVISION = "a0ba3c3dc07c7bc27e901670ceb1a0b0ceeaa8db"
VALIDATION_REPEATS = 16
BANDS = ("easy", "medium", "hard")
EXPECTED_COLUMNS = {"uid", "data_source", "prompt", "reward_model", "extra_info"}
EXPECTED_FILES = {
    "easy": {
        "train": (3000, "f48a50a0527faf788170e23071b7c54e81f63e9542f3cc84a965f8758d7d55bf"),
        "validation": (300, "faad7f8a1ba7f9d523cd3cc95d2d51e034567c3877c65fd895fd3f48f1e6e057"),
    },
    "medium": {
        "train": (3000, "ae8fc66ec846330d2f65591d929971c8f4a37218e8b55d062486d1c4da53f2b5"),
        "validation": (300, "8ac6a8f0dcee17867fb32a23b3a78a1cf667bba4e49efa192063b4fe79ffb2d6"),
    },
    "hard": {
        "train": (3000, "c4be71387f382fbeb4fbc10dcd4a547012996d00168e59b5f3bda6cfb73628c1"),
        "validation": (300, "7dd10c2a983f9386ae9509571bca3e22447b884bf3ab0aecaba5ad678f045d90"),
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def output_paths(data_dir: Path, band: str) -> tuple[Path, Path]:
    prefix = f"deepscaler_gemma4_26b_{band}"
    return data_dir / f"{prefix}_train.parquet", data_dir / f"{prefix}_val300_x16.parquet"


def validate_source(frame: pd.DataFrame, *, band: str, split: str) -> None:
    expected_rows, _ = EXPECTED_FILES[band][split]
    if len(frame) != expected_rows:
        raise ValueError(f"{band}/{split} has {len(frame):,} rows, expected {expected_rows:,}")
    if set(frame.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"{band}/{split} columns differ from the published schema: {list(frame.columns)}")
    if frame["uid"].isna().any() or not frame["uid"].is_unique:
        raise ValueError(f"{band}/{split} must contain one non-null row per UID")
    if not frame["data_source"].eq("math").all():
        raise ValueError(f"{band}/{split} contains a non-math data_source")
    for uid, extra_info in frame[["uid", "extra_info"]].itertuples(index=False, name=None):
        if extra_info["difficulty_band"] != band or extra_info["split"] != split:
            raise ValueError(f"published provenance mismatch for uid={uid!r}")


def repeat_validation(frame: pd.DataFrame, repeats: int = VALIDATION_REPEATS) -> pd.DataFrame:
    if repeats < 1:
        raise ValueError("validation repeats must be positive")
    if frame["uid"].isna().any() or not frame["uid"].is_unique:
        raise ValueError("validation input must contain unique non-null UIDs")
    repeated = frame.loc[frame.index.repeat(repeats)].reset_index(drop=True)
    per_uid = repeated.groupby("uid", sort=False).size()
    if len(per_uid) != len(frame) or set(per_uid.tolist()) != {repeats}:
        raise AssertionError("validation expansion did not preserve exactly one group per source UID")
    return repeated


def validate_outputs(train_path: Path, val_path: Path, *, band: str, repeats: int) -> bool:
    if not train_path.is_file() or not val_path.is_file():
        return False
    try:
        train = pd.read_parquet(train_path)
        validation = pd.read_parquet(val_path)
        validate_source(train, band=band, split="train")
        expected_validation_rows = EXPECTED_FILES[band]["validation"][0] * repeats
        if len(validation) != expected_validation_rows:
            return False
        if set(validation.columns) != EXPECTED_COLUMNS or not validation["data_source"].eq("math").all():
            return False
        val_counts = validation.groupby("uid", sort=False).size()
        if len(val_counts) != EXPECTED_FILES[band]["validation"][0] or set(val_counts.tolist()) != {repeats}:
            return False
        if set(train["uid"]).intersection(validation["uid"]):
            return False
    except (KeyError, OSError, TypeError, ValueError):
        return False
    return True


def prepare_band(data_dir: Path, band: str, *, repeats: int = VALIDATION_REPEATS) -> tuple[Path, Path]:
    from huggingface_hub import hf_hub_download

    if band not in BANDS:
        raise ValueError(f"unknown difficulty band {band!r}; expected one of {BANDS}")
    train_path, val_path = output_paths(data_dir, band)
    if validate_outputs(train_path, val_path, band=band, repeats=repeats):
        print(f"DATASET_READY band={band} train={train_path} validation={val_path}", flush=True)
        return train_path, val_path

    source_frames: dict[str, pd.DataFrame] = {}
    source_paths: dict[str, Path] = {}
    for split in ("train", "validation"):
        source_path = Path(
            hf_hub_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                filename=f"{band}/{split}.parquet",
                revision=REVISION,
            )
        )
        expected_rows, expected_sha = EXPECTED_FILES[band][split]
        actual_sha = sha256_file(source_path)
        if actual_sha != expected_sha:
            raise ValueError(f"published {band}/{split} SHA-256 mismatch: got {actual_sha}, expected {expected_sha}")
        frame = pd.read_parquet(source_path)
        if len(frame) != expected_rows:
            raise ValueError(f"published {band}/{split} row-count mismatch")
        validate_source(frame, band=band, split=split)
        source_paths[split] = source_path
        source_frames[split] = frame

    if set(source_frames["train"]["uid"]).intersection(source_frames["validation"]["uid"]):
        raise ValueError(f"published {band} train and validation UIDs overlap")

    repeated_validation = repeat_validation(source_frames["validation"], repeats)
    tmp_train = train_path.with_name(f".{train_path.name}.{os.getpid()}.tmp")
    tmp_val = val_path.with_name(f".{val_path.name}.{os.getpid()}.tmp")
    try:
        shutil.copyfile(source_paths["train"], tmp_train)
        repeated_validation.to_parquet(tmp_val, index=False)
        os.replace(tmp_train, train_path)
        os.replace(tmp_val, val_path)
    finally:
        tmp_train.unlink(missing_ok=True)
        tmp_val.unlink(missing_ok=True)

    if not validate_outputs(train_path, val_path, band=band, repeats=repeats):
        raise RuntimeError(f"materialized {band} outputs failed validation")
    print(f"DATASET_READY band={band} train={train_path} validation={val_path}", flush=True)
    return train_path, val_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--band", action="append", choices=BANDS, dest="bands")
    parser.add_argument("--validation-repeats", type=int, default=VALIDATION_REPEATS)
    args = parser.parse_args()

    args.data_dir.mkdir(parents=True, exist_ok=True)
    lock_path = args.data_dir / ".deepscaler_gemma4_26b_difficulty.lock"
    with lock_path.open("w") as lock_handle:
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
        for band in args.bands or BANDS:
            prepare_band(args.data_dir, band, repeats=args.validation_repeats)


if __name__ == "__main__":
    main()
