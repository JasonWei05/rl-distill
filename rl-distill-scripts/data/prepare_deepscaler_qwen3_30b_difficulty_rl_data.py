#!/usr/bin/env python3
"""Materialize the pinned Qwen3-30B-A3B-Base difficulty bands for verl RL.

Same contract as prepare_deepscaler_gemma4_26b_difficulty_rl_data.py (pinned Hub files verified by SHA-256, each
300-question validation split expanded to exactly 16 rows per UID so verl reports mean@16 / maj@16), plus the prompt
the teacher was sampled with: the published rows hold the plain problem, and the teacher saw
"<problem> <extra_info.teacher_prompt_instruction>" in its native ChatML template. The instruction is appended here
(train and validation alike) so RL rollouts see exactly the teacher's prompt; the chat template itself is applied by
the RL config (rl-distill-scripts/data/qwen3_base_native_chat.jinja).
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import os
from pathlib import Path

import pandas as pd

REPO_ID = "JWei05/DeepScaleR-Easy-Medium-Hard-Qwen3-30B-A3B-Base-10k"
REVISION = "be4dcd8b1fad7d5eb7bf721f0210baf78277aaa9"
INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
VALIDATION_REPEATS = 16
BANDS = ("easy", "medium", "hard")
EXPECTED_COLUMNS = {"uid", "data_source", "prompt", "reward_model", "extra_info"}
EXPECTED_FILES = {
    "easy": {
        "train": (3000, "6d6e3ad4e170550e83ab785f10b3155cf415849d3ab5d8a310a07370d3bdb7a3"),
        "validation": (300, "9d57d82b2ddfd1ce2e264c54410c47ffcf3c2c5ee363490b981493fe05033e5f"),
    },
    "medium": {
        "train": (3000, "d885a2a5a1a5042553b636b06376283802571a8a8439da6b85aba12613613ec8"),
        "validation": (300, "90f8a5e540c6bddb8e75581a52a63a79c4ea3f89b27aa720001eef6ff955c507"),
    },
    "hard": {
        "train": (3000, "a04790ce5c142fdbe50308ccd33bb122ed89b1acb179387592edb50854ff77de"),
        "validation": (300, "21755b21aeb1826482d719f58edb2cdfdc3954473b09c1058a9fd5016cc0b6d7"),
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def output_paths(data_dir: Path, band: str) -> tuple[Path, Path]:
    prefix = f"deepscaler_qwen3_30b_{band}"
    return data_dir / f"{prefix}_train.parquet", data_dir / f"{prefix}_val300_x16.parquet"


def with_instruction(frame: pd.DataFrame) -> pd.DataFrame:
    """Append the teacher's instruction to the single user message, exactly as the sampler built its prompt."""
    out = frame.copy()

    def prompt(messages, extra_info):
        if extra_info["teacher_prompt_instruction"] != INSTRUCTION:
            raise ValueError(f"unexpected teacher instruction for uid={extra_info['source_uid']!r}")
        (message,) = list(messages)
        return [{"role": "user", "content": f"{str(message['content']).strip()} {INSTRUCTION}"}]

    out["prompt"] = [prompt(m, e) for m, e in zip(out["prompt"], out["extra_info"])]
    return out


def validate(frame: pd.DataFrame, *, band: str, split: str, rows: int, per_uid: int) -> None:
    if len(frame) != rows * per_uid or set(frame.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"{band}/{split}: {len(frame):,} rows / columns {sorted(frame.columns)}")
    counts = frame.groupby("uid", sort=False).size()
    if len(counts) != rows or set(counts.tolist()) != {per_uid}:
        raise ValueError(f"{band}/{split}: expected {rows} UIDs x {per_uid}")
    if not frame["data_source"].eq("math").all():
        raise ValueError(f"{band}/{split} contains a non-math data_source")
    for uid, messages, extra_info in frame[["uid", "prompt", "extra_info"]].itertuples(index=False, name=None):
        if extra_info["difficulty_band"] != band or extra_info["split"] != split:
            raise ValueError(f"provenance mismatch for uid={uid!r}")
        (message,) = list(messages)
        if message["role"] != "user" or not str(message["content"]).endswith(f" {INSTRUCTION}"):
            raise ValueError(f"prompt for uid={uid!r} does not end with the teacher instruction")


def prepare_band(data_dir: Path, band: str, *, repeats: int = VALIDATION_REPEATS) -> tuple[Path, Path]:
    from huggingface_hub import hf_hub_download

    train_path, val_path = output_paths(data_dir, band)
    frames = {}
    for split in ("train", "validation"):
        source = Path(hf_hub_download(REPO_ID, f"{band}/{split}.parquet", repo_type="dataset", revision=REVISION))
        rows, sha = EXPECTED_FILES[band][split]
        if sha256_file(source) != sha:
            raise ValueError(f"published {band}/{split} SHA-256 mismatch")
        frames[split] = with_instruction(pd.read_parquet(source))
        validate(frames[split], band=band, split=split, rows=rows, per_uid=1)
    if set(frames["train"]["uid"]) & set(frames["validation"]["uid"]):
        raise ValueError(f"published {band} train and validation UIDs overlap")
    validation = frames["validation"].loc[frames["validation"].index.repeat(repeats)].reset_index(drop=True)
    validate(validation, band=band, split="validation", rows=EXPECTED_FILES[band]["validation"][0], per_uid=repeats)

    for path, frame in ((train_path, frames["train"]), (val_path, validation)):
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        frame.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    print(f"DATASET_READY band={band} train={train_path} validation={val_path}", flush=True)
    return train_path, val_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--band", action="append", choices=BANDS, dest="bands")
    parser.add_argument("--validation-repeats", type=int, default=VALIDATION_REPEATS)
    args = parser.parse_args()

    args.data_dir.mkdir(parents=True, exist_ok=True)
    with (args.data_dir / ".deepscaler_qwen3_30b_difficulty.lock").open("w") as lock_handle:
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
        for band in args.bands or BANDS:
            prepare_band(args.data_dir, band, repeats=args.validation_repeats)


if __name__ == "__main__":
    main()
