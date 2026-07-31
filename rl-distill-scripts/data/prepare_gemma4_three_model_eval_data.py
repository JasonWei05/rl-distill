#!/usr/bin/env python3
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Materialize the pinned Gemma 4 three-model math evaluation inputs.

The output parquets contain one row per unique question. The manifest records
the independent samples-per-question value that makes each evaluation contain
strictly more than 2,000 generated responses by repeated powers of two. The
evaluator consumes that manifest and performs the repetition through seeded
sampling, so repeated parquet rows are unnecessary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

BOXED_INSTRUCTION = "Please output the final answer within \\boxed{}."
DEFAULT_THRESHOLD = 2_000
DEFAULT_OUTPUT_DIR = Path("/lambda/nfs/Jason-scale/rl-distill-evals/gemma4-three-model/data")
DEFAULT_ID_VALIDATION = Path("/tmp/verl/data/deepscaler_4of4strict_rl_val200_x16.parquet")
DEFAULT_ID_TRAIN = Path("/tmp/verl/data/deepscaler_4of4strict_rl_train.parquet")
DEFAULT_CHAT_TEMPLATE = Path(__file__).with_name("gemma3_it_fewshot_math.jinja")

HF_SOURCES: dict[str, dict[str, Any]] = {
    "math500": {
        "repo_id": "HuggingFaceH4/MATH-500",
        "config": None,
        "split": "test",
        "revision": "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
        "expected_rows": 500,
    },
    "gsm8k": {
        "repo_id": "openai/gsm8k",
        "config": "main",
        "split": "test",
        "revision": "740312add88f781978c0658806c59bc2815b9866",
        "expected_rows": 1_319,
    },
    "olympiadbench": {
        "repo_id": "math-ai/OlympiadBench",
        "config": None,
        "split": "test",
        "revision": "4faaf1e6ec17d11a4218a9bf4c049ecaf954dd84",
        "expected_rows": 674,
    },
    "minervamath": {
        "repo_id": "math-ai/minervamath",
        "config": None,
        "split": "test",
        "revision": "ee46ddc498933b1977577953250ca5c66be64f96",
        "expected_rows": 272,
    },
    "aime2025": {
        "repo_id": "MathArena/aime_2025",
        "config": None,
        "split": "train",
        "revision": "c94da77eb22bbd6439e62a323bec18493a421302",
        "expected_rows": 30,
    },
    "aime2026": {
        "repo_id": "MathArena/aime_2026",
        "config": None,
        "split": "train",
        "revision": "d2de22f3c656b4f56cf8981212186377d1e23bc3",
        "expected_rows": 30,
    },
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def samples_per_question(unique_questions: int, threshold: int = DEFAULT_THRESHOLD) -> int:
    """Return the smallest power of two making requests strictly exceed ``threshold``."""

    if isinstance(unique_questions, bool) or not isinstance(unique_questions, int) or unique_questions <= 0:
        raise ValueError("unique_questions must be a positive integer")
    if isinstance(threshold, bool) or not isinstance(threshold, int) or threshold < 0:
        raise ValueError("threshold must be a non-negative integer")
    samples = 1
    while unique_questions * samples <= threshold:
        samples *= 2
    return samples


def question_text(prompt: Any) -> str:
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    if isinstance(prompt, Sequence) and not isinstance(prompt, str) and prompt and isinstance(prompt[-1], Mapping):
        return str(prompt[-1]["content"])
    return str(prompt)


def with_boxed_instruction(question: Any) -> str:
    text = str(question).strip()
    return text if text.endswith(BOXED_INSTRUCTION) else f"{text} {BOXED_INSTRUCTION}"


def make_row(
    *,
    name: str,
    source_id: Any,
    question: Any,
    answer: Any,
    extra_info: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    uid = f"{name}:{source_id}"
    return {
        "data_source": name,
        "prompt": [{"role": "user", "content": with_boxed_instruction(question)}],
        "reward_model": {"ground_truth": str(answer).strip(), "style": "rule"},
        "extra_info": {"source_id": str(source_id), "split": "test", **dict(extra_info or {})},
        "uid": uid,
    }


def convert_hf_rows(name: str, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    converted = []
    for index, row in enumerate(rows):
        if name == "math500":
            converted.append(
                make_row(
                    name=name,
                    source_id=row["unique_id"],
                    question=row["problem"],
                    answer=row["answer"],
                    extra_info={"subject": row["subject"], "level": int(row["level"])},
                )
            )
        elif name == "gsm8k":
            converted.append(
                make_row(
                    name=name,
                    source_id=index,
                    question=row["question"],
                    answer=str(row["answer"]).split("####")[-1].strip(),
                )
            )
        elif name == "olympiadbench":
            if row.get("language") != "English" or row.get("modality") != "Text-only":
                raise ValueError("the pinned OlympiadBench revision is no longer English text-only")
            if row.get("question_type") != "Open-ended":
                raise ValueError("the pinned OlympiadBench revision contains a non-open-ended question")
            if any(row.get(f"image_{image_index}") is not None for image_index in range(1, 6)):
                raise ValueError("the pinned OlympiadBench revision contains an image-dependent question")
            final_answers = row.get("final_answer") or []
            if len(final_answers) != 1:
                raise ValueError(f"OlympiadBench row {row.get('id')} does not have exactly one final_answer entry")
            answer = str(final_answers[0]).strip()
            if answer.startswith("$") and answer.endswith("$"):
                answer = answer[1:-1].strip()
            converted.append(
                make_row(
                    name=name,
                    source_id=row["id"],
                    question=row["question"],
                    answer=answer,
                    extra_info={
                        "subject": row.get("subject"),
                        "subfield": row.get("subfield"),
                        "difficulty": row.get("difficulty"),
                        "answer_type": row.get("answer_type"),
                        "is_multiple_answer": bool(row.get("is_multiple_answer")),
                    },
                )
            )
        elif name == "minervamath":
            converted.append(
                make_row(
                    name=name,
                    source_id=index,
                    question=row["question"],
                    answer=row["answer"],
                )
            )
        elif name in {"aime2025", "aime2026"}:
            converted.append(
                make_row(
                    name=name,
                    source_id=row.get("problem_idx", index + 1),
                    question=row["problem"],
                    answer=row["answer"],
                    extra_info={"year": int(name[-4:]), "split": "train"},
                )
            )
        else:
            raise ValueError(f"unsupported dataset {name!r}")
    return converted


def _validate_rows(name: str, rows: Sequence[Mapping[str, Any]], expected_rows: int) -> None:
    if len(rows) != expected_rows:
        raise ValueError(f"{name} expected {expected_rows} rows, found {len(rows)}")
    uids = [str(row["uid"]) for row in rows]
    if len(set(uids)) != len(uids):
        raise ValueError(f"{name} contains duplicate UIDs")
    if any(not str(row["reward_model"]["ground_truth"]).strip() for row in rows):
        raise ValueError(f"{name} contains an empty ground-truth answer")


def _write_parquet(rows: Sequence[Mapping[str, Any]], output_path: Path, *, overwrite: bool) -> None:
    import pandas as pd

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {output_path}; pass --overwrite to replace generated data")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + f".partial-{os.getpid()}")
    pd.DataFrame(list(rows)).to_parquet(temporary, index=False)
    os.replace(temporary, output_path)


def _id_validation_rows(validation_path: Path, train_path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    import pandas as pd

    validation = pd.read_parquet(validation_path)
    train = pd.read_parquet(train_path)
    by_uid: dict[str, dict[str, Any]] = {}
    for raw in validation.to_dict(orient="records"):
        uid = str(raw["uid"])
        normalized = dict(raw)
        normalized["prompt"] = [{"role": "user", "content": question_text(raw["prompt"])}]
        normalized["uid"] = uid
        previous = by_uid.get(uid)
        if previous is not None and (
            question_text(previous["prompt"]) != question_text(normalized["prompt"])
            or previous["reward_model"] != normalized["reward_model"]
        ):
            raise ValueError(f"validation UID {uid!r} maps to conflicting rows")
        by_uid[uid] = normalized
    rows = [by_uid[uid] for uid in sorted(by_uid)]
    train_questions = {question_text(prompt) for prompt in train["prompt"]}
    overlap_hashes = sorted(
        {
            hashlib.sha256(question_text(row["prompt"]).encode()).hexdigest()
            for row in rows
            if question_text(row["prompt"]) in train_questions
        }
    )
    return rows, overlap_hashes


def _dataset_manifest_entry(
    *,
    name: str,
    rows: Sequence[Mapping[str, Any]],
    output_path: Path,
    threshold: int,
    source: Mapping[str, Any],
    role: str,
) -> dict[str, Any]:
    sample_count = samples_per_question(len(rows), threshold)
    return {
        "name": name,
        "role": role,
        "source": dict(source),
        "unique_questions": len(rows),
        "samples_per_question": sample_count,
        "total_requests": len(rows) * sample_count,
        "output_path": str(output_path.resolve()),
        "output_sha256": sha256_file(output_path),
    }


def materialize(
    *,
    output_dir: Path,
    id_validation_path: Path,
    id_train_path: Path,
    chat_template_path: Path,
    threshold: int,
    overwrite: bool,
    load_dataset_fn: Callable[..., Any],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_entries = []

    id_rows, overlap_hashes = _id_validation_rows(id_validation_path, id_train_path)
    _validate_rows("id_validation_full", id_rows, 200)
    overlap_set = set(overlap_hashes)
    clean_rows = [
        row for row in id_rows if hashlib.sha256(question_text(row["prompt"]).encode()).hexdigest() not in overlap_set
    ]
    _validate_rows("id_validation_clean", clean_rows, 193)
    local_source = {
        "validation_path": str(id_validation_path.resolve()),
        "validation_sha256": sha256_file(id_validation_path),
        "train_path": str(id_train_path.resolve()),
        "train_sha256": sha256_file(id_train_path),
        "train_validation_overlap_questions": len(overlap_hashes),
    }
    for name, rows, role in (
        ("id_validation_full", id_rows, "in_distribution_primary_candidate"),
        ("id_validation_clean", clean_rows, "in_distribution_overlap_free_diagnostic"),
    ):
        output_path = output_dir / f"{name}.parquet"
        _write_parquet(rows, output_path, overwrite=overwrite)
        dataset_entries.append(
            _dataset_manifest_entry(
                name=name,
                rows=rows,
                output_path=output_path,
                threshold=threshold,
                source=local_source,
                role=role,
            )
        )

    overlap_index = {
        "cross_split_question_text_overlap_count": len(overlap_hashes),
        "cross_split_question_text_overlap_sha256s": overlap_hashes,
        "validation_source_sha256": local_source["validation_sha256"],
        "train_source_sha256": local_source["train_sha256"],
    }
    overlap_path = output_dir / "id_validation_overlap_index.json"
    overlap_path.write_text(json.dumps(overlap_index, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    for name, source in HF_SOURCES.items():
        dataset = load_dataset_fn(
            source["repo_id"],
            source["config"],
            split=source["split"],
            revision=source["revision"],
        )
        if len(dataset) != source["expected_rows"]:
            raise ValueError(
                f"{name} source row count changed: expected {source['expected_rows']}, found {len(dataset)}"
            )
        rows = convert_hf_rows(name, list(dataset))
        _validate_rows(name, rows, source["expected_rows"])
        output_path = output_dir / f"{name}.parquet"
        _write_parquet(rows, output_path, overwrite=overwrite)
        dataset_entries.append(
            _dataset_manifest_entry(
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
        )

    manifest = {
        "schema_version": 1,
        "protocol": "gemma4_three_model_math_eval_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repetition_rule": {
            "threshold": threshold,
            "comparison": "strictly_greater_than",
            "allowed_factors": "powers_of_two",
            "implementation": "smallest power of two k such that unique_questions * k > threshold",
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
        "overlap_index": {
            "path": str(overlap_path.resolve()),
            "sha256": sha256_file(overlap_path),
        },
        "datasets": sorted(dataset_entries, key=lambda entry: entry["name"]),
    }
    manifest_path = output_dir / "math_eval_manifest.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {manifest_path}; pass --overwrite to replace it")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--id-validation", type=Path, default=DEFAULT_ID_VALIDATION)
    parser.add_argument("--id-train", type=Path, default=DEFAULT_ID_TRAIN)
    parser.add_argument("--chat-template", type=Path, default=DEFAULT_CHAT_TEMPLATE)
    parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    from datasets import load_dataset

    manifest = materialize(
        output_dir=args.output_dir.resolve(),
        id_validation_path=args.id_validation.resolve(),
        id_train_path=args.id_train.resolve(),
        chat_template_path=args.chat_template.resolve(),
        threshold=args.threshold,
        overwrite=args.overwrite,
        load_dataset_fn=load_dataset,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
