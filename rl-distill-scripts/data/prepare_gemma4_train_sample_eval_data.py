#!/usr/bin/env python3
"""Materialize a random 300-question sample of a Gemma 4 band's TRAIN split as an eval dataset.

Same schema / manifest protocol as prepare_gemma4_rl_distill_eval_data.py (the id_easy/id_medium/id_hard validation sets),
so eval_math_passk.py runs it with the registered Gemma 4 sampling settings; 16 samples per question. Questions are
deduplicated by text before sampling (the train split repeats some texts).

    python prepare_gemma4_train_sample_eval_data.py --band medium --output-dir /tmp/gemma4_train_sample_eval/data
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from prepare_gemma4_rl_distill_eval_data import BAND_REPO_ID, BAND_REVISION, PROTOCOL, _normalize_id_rows
from prepare_gemma4_three_model_eval_data import DEFAULT_CHAT_TEMPLATE, _dataset_manifest_entry, _write_parquet, question_text, sha256_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--band", default="medium", choices=("easy", "medium", "hard"))
    parser.add_argument("--questions", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-question", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chat-template", type=Path, default=DEFAULT_CHAT_TEMPLATE)
    args = parser.parse_args()
    from huggingface_hub import hf_hub_download

    filename = f"{args.band}/train.parquet"
    downloaded = Path(hf_hub_download(repo_id=BAND_REPO_ID, repo_type="dataset", revision=BAND_REVISION, filename=filename))
    frame = pd.read_parquet(downloaded)
    frame = frame.assign(_text=frame["prompt"].map(question_text)).drop_duplicates("_text").drop(columns="_text")
    sample = frame.sample(n=args.questions, random_state=args.seed).reset_index(drop=True)
    name = f"train_{args.band}"
    rows = _normalize_id_rows(name, sample, args.questions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{name}.parquet"
    _write_parquet(rows, output_path, overwrite=True)
    entry = _dataset_manifest_entry(
        name=name, rows=rows, output_path=output_path, threshold=0,
        source={"repo_id": BAND_REPO_ID, "revision": BAND_REVISION, "filename": filename, "source_sha256": sha256_file(downloaded),
                "sample": {"questions": args.questions, "seed": args.seed, "dedup": "question_text"}},
        role=f"train_sample_{args.band}",
    )
    entry["samples_per_question"] = args.samples_per_question
    entry["total_requests"] = len(rows) * args.samples_per_question
    manifest = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repetition_rule": {"policy": "fixed_by_dataset", "samples_per_question": {name: args.samples_per_question}, "allowed_factors": "powers_of_two"},
        "chat_template": {"path": str(args.chat_template.resolve()), "sha256": sha256_file(args.chat_template)},
        "sampling": {"temperature": 1.0, "top_k": -1, "top_p": 1.0, "max_response_tokens": 8192, "max_prompt_tokens": 4096,
                     "max_model_len": 12288, "predictive_topk_width": 128},
        "datasets": [entry],
    }
    (args.output_dir / "math_eval_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(entry, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
