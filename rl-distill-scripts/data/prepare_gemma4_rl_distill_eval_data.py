#!/usr/bin/env python3
"""Materialize the four pinned math datasets for the expanded Gemma 4 matrix."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

from prepare_gemma4_three_model_eval_data import (
    DEFAULT_CHAT_TEMPLATE,
    DEFAULT_THRESHOLD,
    HF_SOURCES,
    _dataset_manifest_entry,
    _validate_rows,
    _write_parquet,
    convert_hf_rows,
    question_text,
    sha256_file,
)
from prepare_deepscaler_easy_medium_rl_data import (
    BASE_VAL_FILES,
    EASY_REPO,
    EASY_REVISION,
    MEDIUM_REPO,
    MEDIUM_REVISION,
)

DEFAULT_OUTPUT_DIR = Path("/lambda/nfs/Jason-scale/rl-distill-evals/gemma4-rl-distill-base-v1/data")
SAMPLES_PER_QUESTION = {"id_easy": 16, "id_medium": 16, "math500": 16, "gsm8k": 8}
ID_SOURCES = {
    "id_easy": {
        "repo_id": EASY_REPO,
        "revision": EASY_REVISION,
        "filename": BASE_VAL_FILES["easy"],
        "difficulty": "easy",
    },
    "id_medium": {
        "repo_id": MEDIUM_REPO,
        "revision": MEDIUM_REVISION,
        "filename": BASE_VAL_FILES["medium"],
        "difficulty": "medium",
    },
}


def _normalize_id_rows(name: str, frame: pd.DataFrame) -> list[dict[str, Any]]:
    if len(frame) != 500:
        raise ValueError(f"{name} expected 500 held-out questions, found {len(frame)}")
    rows = []
    for index, raw in enumerate(frame.to_dict(orient="records")):
        row = dict(raw)
        row["prompt"] = [{"role": "user", "content": question_text(row["prompt"])}]
        row["uid"] = str(row.get("uid") or f"{name}:{index}")
        rows.append(row)
    _validate_rows(name, rows, 500)
    return rows


def materialize(
    *,
    output_dir: Path,
    chat_template_path: Path,
    threshold: int,
    overwrite: bool,
    hf_hub_download_fn: Callable[..., str],
    load_dataset_fn: Callable[..., Any],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for name, source in ID_SOURCES.items():
        downloaded = Path(
            hf_hub_download_fn(
                repo_id=source["repo_id"],
                repo_type="dataset",
                revision=source["revision"],
                filename=source["filename"],
            )
        )
        rows = _normalize_id_rows(name, pd.read_parquet(downloaded))
        output_path = output_dir / f"{name}.parquet"
        _write_parquet(rows, output_path, overwrite=overwrite)
        entry = _dataset_manifest_entry(
                name=name,
                rows=rows,
                output_path=output_path,
                threshold=threshold,
                source={
                    "repo_id": source["repo_id"],
                    "revision": source["revision"],
                    "filename": source["filename"],
                    "source_sha256": sha256_file(downloaded),
                },
                role=f"in_distribution_{source['difficulty']}",
            )
        entry["samples_per_question"] = SAMPLES_PER_QUESTION[name]
        entry["total_requests"] = len(rows) * SAMPLES_PER_QUESTION[name]
        entries.append(entry)

    for name in ("math500", "gsm8k"):
        source: Mapping[str, Any] = HF_SOURCES[name]
        dataset = load_dataset_fn(
            source["repo_id"],
            source["config"],
            split=source["split"],
            revision=source["revision"],
        )
        rows = convert_hf_rows(name, list(dataset))
        _validate_rows(name, rows, int(source["expected_rows"]))
        output_path = output_dir / f"{name}.parquet"
        _write_parquet(rows, output_path, overwrite=overwrite)
        entry = _dataset_manifest_entry(
                name=name,
                rows=rows,
                output_path=output_path,
                threshold=threshold,
                source={
                    "repo_id": source["repo_id"],
                    "config": source["config"],
                    "split": source["split"],
                    "revision": source["revision"],
                    "datasets_fingerprint": getattr(dataset, "_fingerprint", None),
                },
                role="out_of_distribution_math",
            )
        entry["samples_per_question"] = SAMPLES_PER_QUESTION[name]
        entry["total_requests"] = len(rows) * SAMPLES_PER_QUESTION[name]
        entries.append(entry)

    manifest = {
        "schema_version": 1,
        "protocol": "gemma4_rl_distill_math_eval_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repetition_rule": {
            "policy": "fixed_by_dataset",
            "samples_per_question": SAMPLES_PER_QUESTION,
            "allowed_factors": "powers_of_two",
        },
        "chat_template": {
            "path": str(chat_template_path.resolve()),
            "sha256": sha256_file(chat_template_path),
        },
        "sampling": {
            "temperature": 1.0,
            "top_k": -1,
            "top_p": 1.0,
            "max_response_tokens": 8192,
            "max_prompt_tokens": 4096,
            "max_model_len": 12288,
            "predictive_topk_width": 128,
        },
        "datasets": sorted(entries, key=lambda entry: entry["name"]),
    }
    manifest_path = output_dir / "math_eval_manifest.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {manifest_path}; pass --overwrite to replace it")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--chat-template", type=Path, default=DEFAULT_CHAT_TEMPLATE)
    parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    manifest = materialize(
        output_dir=args.output_dir.resolve(),
        chat_template_path=args.chat_template.resolve(),
        threshold=args.threshold,
        overwrite=args.overwrite,
        hf_hub_download_fn=hf_hub_download,
        load_dataset_fn=load_dataset,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
