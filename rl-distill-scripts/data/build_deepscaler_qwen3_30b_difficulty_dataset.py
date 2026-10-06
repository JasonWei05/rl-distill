#!/usr/bin/env python3
"""Build and optionally upload the Qwen3-30B-A3B-Base DeepScaleR difficulty data.

A replica of build_deepscaler_gemma4_26b_difficulty_dataset.py (same bands, sizes, hash-rank selection, seeds,
schema, manifest and README layout) with the Qwen3-30B-A3B-Base n=8 sampling run as the teacher.

The Hugging Face dataset has three configurations:

* hard:   current RL grader score == 1/8
* medium: current RL grader score in {2/8, 3/8}
* easy:   current RL grader score in {4/8, ..., 8/8}

Each configuration contains 3,300 unique questions: 3,000 train and 300
validation. Selection and validation assignment use stable SHA-256 ranks so
the result is independent of Python, NumPy, and pandas RNG implementations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QUESTIONS = Path(
    "/mnt/efs/jasonwei/verl/data/deepscaler_it_gen/"
    "deepscaler_it_gen_dedup_v2.parquet"
)
DEFAULT_GRADES = Path(
    "/mnt/efs/jasonwei/verl/data/"
    "deepscaler_qwen3_30b_a3b_base_n8_dedup_v2_20261005/"
    "questions_with_current_rl_grades.parquet"
)
DEFAULT_OUTPUT_DIR = Path(
    "/mnt/efs/jasonwei/verl/data/"
    "deepscaler_easy_medium_hard_qwen3_30b_a3b_base_10k_20261006"
)
DEFAULT_REPO_ID = "JWei05/DeepScaleR-Easy-Medium-Hard-Qwen3-30B-A3B-Base-10k"

UPSTREAM_DATASET = "agentica-org/DeepScaleR-Preview-Dataset"
TEACHER_MODEL = "Qwen/Qwen3-30B-A3B-Base"
TEACHER_REVISION = "1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9"
PROMPT_TEMPLATE = "qwen3_base_native_chat.jinja"
PROMPT_TEMPLATE_SHA256 = "87a2728cb8dc9fe424d624542f6060ec05a1d285ebbec578bb078900e33396b5"
PROMPT_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
TEACHER_RUN_S3 = "s3://scale-ml/genai/rl-distill/qwen3-deepscaler-samples/qwen3-30b-a3b-base-0shot-qwenmath-n8-8k/"
QUESTIONS_SHA256 = "4de0a409372f3af6cb30fc47cd9e12f98114e4a8d51664fefd87a08be4c62b5a"
GRADES_SHA256 = "84965510b81040036720ac3bbed81378d9b98d9c5200a1e69dbfa80e67496513"
MATH_VERIFY_SHA256 = "2b991c039ca497cd89658c308f81c492179bfa2faa65c33185674b989892801d"
MILES_SYMPY_SHA256 = "8fe9cd48189bd03ce667ac24193e9f3394c22defe78076e5ff5244c831628795"

TEACHER_SAMPLES = 8
TEACHER_TEMPERATURE = 1.0
TEACHER_TOP_P = 1.0
TEACHER_TOP_K = -1
TEACHER_MAX_RESPONSE_TOKENS = 8192
TEACHER_MAX_PROMPT_TOKENS = 4096
TEACHER_MAX_CONTEXT_TOKENS = 12288
PROMPT_SHOTS = 0
MATH_VERIFY_TIMEOUT_SECONDS = 30.0
MILES_SYMPY_TIMEOUT_SECONDS = 5.0
GRADER_NAME = "current_rl_last_box_math_verify_or_miles_sympy"
GRADER_CONTRACT = (
    "last well-formed boxed/fbox answer; correct if math_verify OR bounded Miles SymPy"
)

BANDS: dict[str, tuple[int, int]] = {
    "easy": (4, 8),
    "medium": (2, 3),
    "hard": (1, 1),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_rank(purpose: str, seed: int, band: str, uid: str) -> str:
    return sha256_text(f"{purpose}\0{seed}\0{band}\0{uid}")


def uid_set_sha256(uids: Iterable[str]) -> str:
    return sha256_text("\n".join(sorted(str(uid) for uid in uids)))


def git_head() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def load_env_file() -> None:
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def normalize_bool_list(value: Any) -> list[bool]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [bool(item) for item in value]


def load_joined_source(questions_path: Path, grades_path: Path) -> pd.DataFrame:
    questions = pd.read_parquet(questions_path, columns=["uid", "problem", "answer"])
    grades = pd.read_parquet(
        grades_path,
        columns=["uid", "current_rl_accs", "current_rl_n_correct", "current_rl_pass_rate"],
    )

    if len(questions) != 38_796 or len(grades) != 38_796:
        raise ValueError(
            f"Expected 38,796 rows in each source; got questions={len(questions):,}, "
            f"grades={len(grades):,}"
        )
    for name, frame in (("questions", questions), ("grades", grades)):
        if frame["uid"].isna().any() or not frame["uid"].is_unique:
            raise ValueError(f"{name} must contain one non-null row per uid")

    question_uids = set(questions["uid"])
    grade_uids = set(grades["uid"])
    if question_uids != grade_uids:
        raise ValueError(
            f"Question/grade uid sets differ: questions_only={len(question_uids - grade_uids)}, "
            f"grades_only={len(grade_uids - question_uids)}"
        )

    joined = questions.merge(grades, on="uid", how="inner", validate="one_to_one")
    if joined["problem"].isna().any() or joined["answer"].isna().any():
        raise ValueError("Problems and gold answers must be non-null")
    if not joined["current_rl_n_correct"].between(0, TEACHER_SAMPLES).all():
        raise ValueError("current_rl_n_correct must be in [0, 8]")
    expected_rates = joined["current_rl_n_correct"].astype(float) / TEACHER_SAMPLES
    if not expected_rates.eq(joined["current_rl_pass_rate"].astype(float)).all():
        raise ValueError("current_rl_pass_rate does not equal current_rl_n_correct / 8")
    for uid, accs, n_correct in joined[
        ["uid", "current_rl_accs", "current_rl_n_correct"]
    ].itertuples(index=False, name=None):
        normalized = normalize_bool_list(accs)
        if len(normalized) != TEACHER_SAMPLES or sum(normalized) != int(n_correct):
            raise ValueError(f"Invalid per-trace grade vector for uid={uid}")
    return joined


def select_band(
    source: pd.DataFrame,
    band: str,
    selection_size: int,
    validation_size: int,
    selection_seed: int,
    validation_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    low, high = BANDS[band]
    candidates = source[source["current_rl_n_correct"].between(low, high)].copy()
    available = len(candidates)
    if available < selection_size:
        raise ValueError(f"Band {band!r} has {available:,} rows, need {selection_size:,}")
    if not 0 < validation_size < selection_size:
        raise ValueError("validation_size must be positive and smaller than selection_size")

    candidates["selection_rank_sha256"] = [
        stable_rank("selection", selection_seed, band, uid) for uid in candidates["uid"]
    ]
    selected = candidates.sort_values(["selection_rank_sha256", "uid"], kind="stable").head(
        selection_size
    )
    selected = selected.copy()
    selected["validation_rank_sha256"] = [
        stable_rank("validation", validation_seed, band, uid) for uid in selected["uid"]
    ]

    validation_uids = set(
        selected.sort_values(["validation_rank_sha256", "uid"], kind="stable")
        .head(validation_size)["uid"]
        .tolist()
    )
    validation = selected[selected["uid"].isin(validation_uids)].copy()
    train = selected[~selected["uid"].isin(validation_uids)].copy()
    train = train.sort_values(["selection_rank_sha256", "uid"], kind="stable")
    validation = validation.sort_values(["validation_rank_sha256", "uid"], kind="stable")
    return train.reset_index(drop=True), validation.reset_index(drop=True), available


def to_verl_rows(
    selected: pd.DataFrame,
    band: str,
    split: str,
    selection_seed: int,
    validation_seed: int,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    low, high = BANDS[band]
    definition = f"{low}/8" if low == high else f"{low}/8-{high}/8"
    for split_index, row in enumerate(selected.itertuples(index=False)):
        n_correct = int(row.current_rl_n_correct)
        records.append(
            {
                "uid": str(row.uid),
                "data_source": "math",
                "prompt": [{"role": "user", "content": str(row.problem)}],
                "reward_model": {"style": "rule", "ground_truth": str(row.answer)},
                "extra_info": {
                    "index": f"{band}-{split}-{split_index}",
                    "source_uid": str(row.uid),
                    "upstream_dataset": UPSTREAM_DATASET,
                    "difficulty_band": band,
                    "difficulty_definition": definition,
                    "split": split,
                    "teacher_model": TEACHER_MODEL,
                    "teacher_revision": TEACHER_REVISION,
                    "teacher_n_correct": n_correct,
                    "teacher_n_samples": TEACHER_SAMPLES,
                    "teacher_pass_rate": n_correct / TEACHER_SAMPLES,
                    "teacher_current_rl_accs": normalize_bool_list(row.current_rl_accs),
                    "teacher_temperature": TEACHER_TEMPERATURE,
                    "teacher_top_p": TEACHER_TOP_P,
                    "teacher_top_k": TEACHER_TOP_K,
                    "teacher_max_response_tokens": TEACHER_MAX_RESPONSE_TOKENS,
                    "teacher_max_context_tokens": TEACHER_MAX_CONTEXT_TOKENS,
                    "prompt_template": PROMPT_TEMPLATE,
                    "prompt_template_sha256": PROMPT_TEMPLATE_SHA256,
                    "prompt_shots": PROMPT_SHOTS,
                    "teacher_prompt_instruction": PROMPT_INSTRUCTION,
                    "grader": GRADER_NAME,
                    "grader_contract": GRADER_CONTRACT,
                    "math_verify_timeout_seconds": MATH_VERIFY_TIMEOUT_SECONDS,
                    "miles_sympy_timeout_seconds": MILES_SYMPY_TIMEOUT_SECONDS,
                    "selection_seed": selection_seed,
                    "validation_seed": validation_seed,
                    "selection_rank_sha256": str(row.selection_rank_sha256),
                    "validation_rank_sha256": str(row.validation_rank_sha256),
                },
            }
        )
    return pd.DataFrame.from_records(records)


def score_counts(frame: pd.DataFrame) -> dict[str, int]:
    values = frame["extra_info"].map(lambda info: int(info["teacher_n_correct"]))
    counts = values.value_counts().sort_index()
    return {str(score): int(count) for score, count in counts.items()}


def validate_materialized(
    frames: dict[str, dict[str, pd.DataFrame]],
    selection_size: int,
    validation_size: int,
) -> None:
    all_band_uids: dict[str, set[str]] = {}
    expected_columns = {"uid", "data_source", "prompt", "reward_model", "extra_info"}
    for band, splits in frames.items():
        train = splits["train"]
        validation = splits["validation"]
        if len(train) != selection_size - validation_size or len(validation) != validation_size:
            raise ValueError(f"Unexpected split sizes for {band}")
        train_uids = set(train["uid"])
        validation_uids = set(validation["uid"])
        if len(train_uids) != len(train) or len(validation_uids) != len(validation):
            raise ValueError(f"Duplicate uid within {band}")
        if train_uids & validation_uids:
            raise ValueError(f"Train/validation overlap in {band}")
        all_band_uids[band] = train_uids | validation_uids

        low, high = BANDS[band]
        for split, frame in splits.items():
            if set(frame.columns) != expected_columns:
                raise ValueError(f"Unexpected columns for {band}/{split}: {list(frame.columns)}")
            if not frame["data_source"].eq("math").all():
                raise ValueError(f"Unexpected data_source in {band}/{split}")
            for record in frame.to_dict(orient="records"):
                prompt = record["prompt"]
                reward = record["reward_model"]
                extra = record["extra_info"]
                if len(prompt) != 1 or prompt[0]["role"] != "user" or not prompt[0]["content"]:
                    raise ValueError(f"Invalid prompt for uid={record['uid']}")
                if reward["style"] != "rule" or not reward["ground_truth"]:
                    raise ValueError(f"Invalid reward model for uid={record['uid']}")
                if extra["source_uid"] != record["uid"] or extra["split"] != split:
                    raise ValueError(f"Invalid provenance for uid={record['uid']}")
                if not low <= int(extra["teacher_n_correct"]) <= high:
                    raise ValueError(f"Band violation for uid={record['uid']}")

    bands = list(all_band_uids)
    for index, left in enumerate(bands):
        for right in bands[index + 1 :]:
            overlap = all_band_uids[left] & all_band_uids[right]
            if overlap:
                raise ValueError(f"Cross-band overlap between {left} and {right}: {len(overlap)}")


def render_readme(
    available_counts: dict[str, int],
    frames: dict[str, dict[str, pd.DataFrame]],
    selection_size: int,
    validation_size: int,
    selection_seed: int,
    validation_seed: int,
) -> str:
    train_size = selection_size - validation_size
    config_metadata = "\n".join(
        [
            f"- config_name: {band}\n"
            "  data_files:\n"
            f"  - split: train\n    path: {band}/train.parquet\n"
            f"  - split: validation\n    path: {band}/validation.parquet"
            for band in BANDS
        ]
    )
    split_rows = "\n".join(
        f"| `{band}` | {BANDS[band][0]}/8"
        f"{'-' + str(BANDS[band][1]) + '/8' if BANDS[band][0] != BANDS[band][1] else ''} "
        f"| {available_counts[band]:,} | {train_size:,} | {validation_size:,} | {selection_size:,} |"
        for band in BANDS
    )
    score_rows: list[str] = []
    for band in BANDS:
        for split in ("train", "validation"):
            counts = score_counts(frames[band][split])
            score_rows.append(
                f"| `{band}` | `{split}` | "
                + ", ".join(f"{score}/8: {count:,}" for score, count in counts.items())
                + " |"
            )

    return f"""---
pretty_name: DeepScaleR Easy/Medium/Hard by Qwen3-30B-A3B-Base
license: mit
language:
- en
task_categories:
- text-generation
tags:
- reinforcement-learning
- mathematics
- qwen3
- verl
size_categories:
- 1K<n<10K
configs:
{config_metadata}
---

# DeepScaleR Easy/Medium/Hard — Qwen3-30B-A3B-Base

This dataset contains 9,900 unique, deduplicated DeepScaleR math questions for
reinforcement-learning experiments. Difficulty is defined by how often the
pretrained `Qwen/Qwen3-30B-A3B-Base` teacher solved each question across eight
temperature-1 samples under the same rule-based grader used by the RL training
pipeline.

The Hub dataset has three configurations—`easy`, `medium`, and `hard`—and each
configuration has a `train` split with {train_size:,} questions and a disjoint
`validation` split with {validation_size:,} questions.

## Configurations and splits

| Configuration | Teacher score band | Available before sampling | Train | Validation | Total selected |
|---|---:|---:|---:|---:|---:|
{split_rows}

The 0/8 bucket is intentionally excluded. In this dataset, **hard means exactly
1/8**. Bands, sizes, seeds, selection procedure and schema are identical to
[`JWei05/DeepScaleR-Easy-Medium-Hard-Gemma-26B-PT-10k`](https://huggingface.co/datasets/JWei05/DeepScaleR-Easy-Medium-Hard-Gemma-26B-PT-10k);
only the teacher (and its prompt/sampler) differs.

### Exact score composition after sampling

| Configuration | Split | Number of rows by teacher score |
|---|---|---|
{chr(10).join(score_rows)}

## Construction

1. Start from `agentica-org/DeepScaleR-Preview-Dataset` (40,315 original rows,
   MIT license).
2. Use the existing dedup-v2 view with 38,796 unique questions. That pipeline
   groups questions under strong text/LaTeX normalization, removes duplicate
   groups with conflicting non-equivalent gold answers, and keeps one
   deterministic representative from each surviving group. The exact local
   source Parquet has SHA-256 `{QUESTIONS_SHA256}`.
3. Grade eight independently sampled teacher responses per question using the
   current RL-training grading contract described below. The exact question-level
   grade Parquet has SHA-256 `{GRADES_SHA256}`.
4. Form disjoint score bands: hard = 1/8, medium = 2–3/8, easy = 4–8/8.
5. Within each band, rank every UID by
   `SHA256("selection\\0{selection_seed}\\0<band>\\0<uid>")` and take the lowest
   {selection_size:,} ranks without replacement.
6. Within each selected set, rank UIDs by
   `SHA256("validation\\0{validation_seed}\\0<band>\\0<uid>")`; the lowest
   {validation_size:,} become validation and the remaining {train_size:,} become train.

Stable hash ranking makes the result independent of pandas, NumPy, and Python
random-number-generator versions. `manifest.json` contains the output file
hashes, UID-set hashes, source hashes, score counts, and all construction settings.

## Teacher generation protocol

- Model: `Qwen/Qwen3-30B-A3B-Base` (pretrained/base MoE, 128 experts, 8 active), revision
  `{TEACHER_REVISION}`.
- Eight responses per question.
- Sampling: temperature {TEACHER_TEMPERATURE:g}, top-p {TEACHER_TOP_P:g},
  top-k {TEACHER_TOP_K}.
- Maximum response length: {TEACHER_MAX_RESPONSE_TOKENS:,} tokens.
- Maximum prompt length: {TEACHER_MAX_PROMPT_TOKENS:,} tokens; maximum model context: {TEACHER_MAX_CONTEXT_TOKENS:,} tokens.
- Prompt: 0-shot, the model's native ChatML chat template (`{PROMPT_TEMPLATE}`,
  SHA-256 `{PROMPT_TEMPLATE_SHA256}`) with a single user turn
  `<problem> {PROMPT_INSTRUCTION}` and the assistant turn opened for generation.
- Generation stops: the `<|im_end|>`, `<|endoftext|>` and `<|im_start|>` token ids, plus the plain-text
  turn starts `\\nHuman:` and `\\nuser\\n` (base models otherwise continue with invented chat turns).
- Inference: vLLM 0.15.1, tensor parallel 2 × data parallel 2. Raw responses: `{TEACHER_RUN_S3}`.

The data rows store the plain problem as the single user message, exactly as in the
Gemma version. The teacher saw the problem followed by the instruction above
(`extra_info.teacher_prompt_instruction`); to match the teacher's prompt, append it
(or apply an equivalent template) at tokenization time. Teacher strict mean@8 over all
38,796 questions: 33.91%, pass@8: 76.33%.

## RL grading contract

For each response, the grader extracts the **last well-formed** `\\boxed{{...}}`
or `\\fbox{{...}}` answer. A response is correct if either Hugging Face
`math_verify` or the bounded Miles-style SymPy fallback accepts equivalence with
the gold answer.

- `math_verify` timeout: {MATH_VERIFY_TIMEOUT_SECONDS:g} seconds.
- Miles-SymPy timeout: {MILES_SYMPY_TIMEOUT_SECONDS:g} seconds.
- Evaluation-time `math_verify.py` SHA-256: `{MATH_VERIFY_SHA256}`.
- Evaluation-time `miles_sympy.py` SHA-256: `{MILES_SYMPY_SHA256}`.

This is the current RL grader, not the earlier whole-output lenient grader or the
earlier exactly-one-box strict grader.

## Schema

Each row is directly usable by verl-style rule-reward math training:

- `uid`: stable UID inherited from the deduplicated source.
- `data_source`: always `math`.
- `prompt`: one plain user message containing the problem.
- `reward_model`: `{{"style": "rule", "ground_truth": <gold answer>}}`.
- `extra_info`: split/band provenance, teacher model and revision, all eight
  per-trace correctness flags, teacher pass count/rate, prompt and sampler
  settings, grader contract, deterministic ranks, and sampling seeds.

## Reproduction

From the `rl-distill` repository, using its `uv` environment:

```bash
.venv/bin/python rl-distill-scripts/data/build_deepscaler_qwen3_30b_difficulty_dataset.py
```

The default build writes the six Parquet files, this README, and `manifest.json`
locally. Pass `--upload` to publish the same files to
`{DEFAULT_REPO_ID}`.

## Intended use and limitations

These labels measure empirical solvability by one specific pretrained teacher,
prompt, sampler, response limit, and grader. They are useful curriculum bands,
not universal or human difficulty labels. The candidate pools contain
{available_counts['easy']:,} (easy), {available_counts['medium']:,} (medium) and {available_counts['hard']:,} (hard)
questions, so each {selection_size:,}-question sample is a random subset of its band. Validation splits are disjoint
from training within and across configurations, but they are not intended as a
replacement for established external math benchmarks.
"""


def write_outputs(
    output_dir: Path,
    frames: dict[str, dict[str, pd.DataFrame]],
    available_counts: dict[str, int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, Any] = {}
    for band, splits in frames.items():
        band_dir = output_dir / band
        band_dir.mkdir(parents=True, exist_ok=True)
        files[band] = {}
        for split, frame in splits.items():
            path = band_dir / f"{split}.parquet"
            frame.to_parquet(path, index=False)
            reloaded = pd.read_parquet(path)
            if len(reloaded) != len(frame) or set(reloaded["uid"]) != set(frame["uid"]):
                raise ValueError(f"Round-trip validation failed for {path}")
            files[band][split] = {
                "path": str(path.relative_to(output_dir)),
                "rows": len(frame),
                "sha256": sha256_file(path),
                "uid_set_sha256": uid_set_sha256(frame["uid"]),
                "teacher_n_correct_counts": score_counts(frame),
            }

    readme = render_readme(
        available_counts,
        frames,
        args.selection_size,
        args.validation_size,
        args.selection_seed,
        args.validation_seed,
    )
    (output_dir / "README.md").write_text(readme)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "repo_id": args.repo_id,
        "builder": {
            "path": str(Path(__file__).relative_to(REPO_ROOT)),
            "sha256": sha256_file(Path(__file__)),
            "rl_distill_git_head": git_head(),
        },
        "sources": {
            "upstream_dataset": UPSTREAM_DATASET,
            "deduplicated_questions": {
                "rows": 38_796,
                "sha256": sha256_file(args.questions),
                "expected_sha256": QUESTIONS_SHA256,
            },
            "current_rl_grades": {
                "rows": 38_796,
                "sha256": sha256_file(args.grades),
                "expected_sha256": GRADES_SHA256,
            },
        },
        "teacher": {
            "model": TEACHER_MODEL,
            "revision": TEACHER_REVISION,
            "samples_per_question": TEACHER_SAMPLES,
            "temperature": TEACHER_TEMPERATURE,
            "top_p": TEACHER_TOP_P,
            "top_k": TEACHER_TOP_K,
            "max_response_tokens": TEACHER_MAX_RESPONSE_TOKENS,
            "max_context_tokens": TEACHER_MAX_CONTEXT_TOKENS,
            "prompt_template": PROMPT_TEMPLATE,
            "prompt_template_sha256": PROMPT_TEMPLATE_SHA256,
            "prompt_shots": PROMPT_SHOTS,
            "prompt_instruction": PROMPT_INSTRUCTION,
            "max_prompt_tokens": TEACHER_MAX_PROMPT_TOKENS,
            "stop_strings": ["<|im_end|>", "\nHuman:", "\nuser\n"],
            "stop_token_ids": [151645, 151643, 151644],
            "sampling_run_s3": TEACHER_RUN_S3,
        },
        "grader": {
            "name": GRADER_NAME,
            "contract": GRADER_CONTRACT,
            "math_verify_timeout_seconds": MATH_VERIFY_TIMEOUT_SECONDS,
            "miles_sympy_timeout_seconds": MILES_SYMPY_TIMEOUT_SECONDS,
            "math_verify_py_sha256": MATH_VERIFY_SHA256,
            "miles_sympy_py_sha256": MILES_SYMPY_SHA256,
        },
        "sampling": {
            "algorithm": "ascending SHA-256 rank over purpose, seed, band, and uid",
            "selection_seed": args.selection_seed,
            "validation_seed": args.validation_seed,
            "selection_size_per_band": args.selection_size,
            "validation_size_per_band": args.validation_size,
            "train_size_per_band": args.selection_size - args.validation_size,
            "bands": {
                band: {
                    "min_n_correct": limits[0],
                    "max_n_correct": limits[1],
                    "available": available_counts[band],
                }
                for band, limits in BANDS.items()
            },
        },
        "files": files,
        "validation": {
            "exact_split_counts": True,
            "unique_uids_within_splits": True,
            "no_train_validation_overlap": True,
            "no_cross_band_overlap": True,
            "all_rows_satisfy_band_definition": True,
            "verl_schema_valid": True,
            "parquet_round_trip_valid": True,
        },
    }
    if manifest["sources"]["deduplicated_questions"]["sha256"] != QUESTIONS_SHA256:
        raise ValueError("Question source SHA-256 does not match the audited artifact")
    if manifest["sources"]["current_rl_grades"]["sha256"] != GRADES_SHA256:
        raise ValueError("Grade source SHA-256 does not match the audited artifact")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def upload_dataset(output_dir: Path, repo_id: str, private: bool) -> str:
    load_env_file()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY")
    if not token:
        raise RuntimeError("Missing HF_TOKEN or HUGGINGFACE_API_KEY")

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    commit = api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=output_dir,
        path_in_repo="",
        allow_patterns=[
            "README.md",
            "manifest.json",
            "easy/*.parquet",
            "medium/*.parquet",
            "hard/*.parquet",
        ],
        commit_message="Add Qwen3-30B-A3B-Base Easy/Medium/Hard RL datasets",
    )
    return commit.oid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--grades", type=Path, default=DEFAULT_GRADES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--selection-size", type=int, default=3300)
    parser.add_argument("--validation-size", type=int, default=300)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--validation-seed", type=int, default=43)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--private", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing known generated files in a non-empty output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.questions = args.questions.expanduser().resolve()
    args.grades = args.grades.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if not args.questions.is_file() or not args.grades.is_file():
        raise FileNotFoundError("Both --questions and --grades must be existing Parquet files")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is non-empty: {args.output_dir}. Pass --overwrite to rebuild."
        )

    print("[1/5] Loading and validating audited source artifacts")
    source = load_joined_source(args.questions, args.grades)

    print("[2/5] Selecting deterministic difficulty bands and validation splits")
    selected_source: dict[str, dict[str, pd.DataFrame]] = {}
    available_counts: dict[str, int] = {}
    for band in BANDS:
        train, validation, available = select_band(
            source,
            band,
            args.selection_size,
            args.validation_size,
            args.selection_seed,
            args.validation_seed,
        )
        selected_source[band] = {"train": train, "validation": validation}
        available_counts[band] = available
        print(
            f"  {band}: available={available:,}, train={len(train):,}, "
            f"validation={len(validation):,}"
        )

    print("[3/5] Converting to verl rule-reward schema")
    frames = {
        band: {
            split: to_verl_rows(
                frame,
                band,
                split,
                args.selection_seed,
                args.validation_seed,
            )
            for split, frame in splits.items()
        }
        for band, splits in selected_source.items()
    }
    validate_materialized(frames, args.selection_size, args.validation_size)

    print(f"[4/5] Writing and round-trip validating files in {args.output_dir}")
    manifest = write_outputs(args.output_dir, frames, available_counts, args)
    for band in BANDS:
        print(
            f"  {band}: train scores={manifest['files'][band]['train']['teacher_n_correct_counts']}; "
            f"validation scores={manifest['files'][band]['validation']['teacher_n_correct_counts']}"
        )

    if args.upload:
        visibility = "private" if args.private else "public"
        print(f"[5/5] Uploading {visibility} dataset to {args.repo_id}")
        commit_sha = upload_dataset(args.output_dir, args.repo_id, args.private)
        print(f"  commit={commit_sha}")
    else:
        print("[5/5] Upload skipped (pass --upload to publish)")
    print("DONE")


if __name__ == "__main__":
    main()
