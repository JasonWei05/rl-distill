#!/usr/bin/env python3
"""Add a `very_hard` configuration to the Qwen3-30B-A3B-Base DeepScaleR difficulty dataset.

very_hard = questions the teacher solved 0/8 in the original n=8 run (excluded from easy/medium/hard) that it then solved at
least once in a 16-sample resample with a new seed (identical protocol otherwise). From those candidates, 3,300 are selected and
split 3,000 train / 300 validation with the same stable SHA-256 rank procedure, seeds and verl schema as the other bands
(build_deepscaler_qwen3_30b_difficulty_dataset.py); extra_info additionally records the 16 resample correctness flags.

    python build_deepscaler_qwen3_30b_very_hard_band.py --resample-grades <grades_strict_n16_accs.parquet> [--upload]

Writes a copy of the existing build plus very_hard/{train,validation}.parquet, README.md and manifest.json to --output-dir; --upload
publishes only the two new Parquet files and the updated README/manifest in one commit on top of --parent-revision.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from build_deepscaler_qwen3_30b_difficulty_dataset import (
    DEFAULT_GRADES,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_QUESTIONS,
    DEFAULT_REPO_ID,
    GRADER_CONTRACT,
    GRADER_NAME,
    MATH_VERIFY_TIMEOUT_SECONDS,
    MILES_SYMPY_TIMEOUT_SECONDS,
    PROMPT_INSTRUCTION,
    PROMPT_SHOTS,
    PROMPT_TEMPLATE,
    PROMPT_TEMPLATE_SHA256,
    REPO_ROOT,
    TEACHER_MAX_CONTEXT_TOKENS,
    TEACHER_MAX_RESPONSE_TOKENS,
    TEACHER_MODEL,
    TEACHER_REVISION,
    TEACHER_SAMPLES,
    TEACHER_TEMPERATURE,
    TEACHER_TOP_K,
    TEACHER_TOP_P,
    UPSTREAM_DATASET,
    git_head,
    load_env_file,
    load_joined_source,
    normalize_bool_list,
    sha256_file,
    stable_rank,
    uid_set_sha256,
)

BAND = "very_hard"
RESAMPLE_SAMPLES = 16
RESAMPLE_SEED = 1042
RESAMPLE_RUN_S3 = (
    "s3://scale-ml/genai/rl-distill/qwen3-deepscaler-samples/qwen3-30b-a3b-base-0shot-qwenmath-zero8-n16-seed1042-8k/"
)
DEFINITION = "0/8 (original n=8 run), then >=1/16 in a 16-sample resample (seed 1042)"
PARENT_REVISION = "be4dcd8b1fad7d5eb7bf721f0210baf78277aaa9"
DEFAULT_NEW_OUTPUT_DIR = Path(
    "/mnt/efs/jasonwei/verl/data/deepscaler_easy_medium_hard_very_hard_qwen3_30b_a3b_base_20261010"
)


def load_resample(path: Path, zero_uids: set[str]) -> pd.DataFrame:
    grades = pd.read_parquet(path, columns=["uid", "resample_accs", "resample_n_correct"])
    if len(grades) != len(zero_uids) or set(grades["uid"]) != zero_uids or not grades["uid"].is_unique:
        raise ValueError(f"Resample grades must cover exactly the {len(zero_uids):,} 0/8 uids once each")
    for uid, accs, n_correct in grades.itertuples(index=False, name=None):
        flags = normalize_bool_list(accs)
        if len(flags) != RESAMPLE_SAMPLES or sum(flags) != int(n_correct):
            raise ValueError(f"Invalid resample grade vector for uid={uid}")
    return grades


def select(candidates: pd.DataFrame, selection_size: int, validation_size: int, selection_seed: int, validation_seed: int):
    if len(candidates) < selection_size:
        raise ValueError(f"{BAND} has {len(candidates):,} candidates, need {selection_size:,}")
    candidates = candidates.copy()
    candidates["selection_rank_sha256"] = [stable_rank("selection", selection_seed, BAND, uid) for uid in candidates["uid"]]
    selected = candidates.sort_values(["selection_rank_sha256", "uid"], kind="stable").head(selection_size).copy()
    selected["validation_rank_sha256"] = [stable_rank("validation", validation_seed, BAND, uid) for uid in selected["uid"]]
    validation_uids = set(selected.sort_values(["validation_rank_sha256", "uid"], kind="stable").head(validation_size)["uid"])
    validation = selected[selected["uid"].isin(validation_uids)].sort_values(["validation_rank_sha256", "uid"], kind="stable")
    train = selected[~selected["uid"].isin(validation_uids)].sort_values(["selection_rank_sha256", "uid"], kind="stable")
    return train.reset_index(drop=True), validation.reset_index(drop=True)


def to_rows(selected: pd.DataFrame, split: str, selection_seed: int, validation_seed: int) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for split_index, row in enumerate(selected.itertuples(index=False)):
        resample_n = int(row.resample_n_correct)
        records.append({
            "uid": str(row.uid),
            "data_source": "math",
            "prompt": [{"role": "user", "content": str(row.problem)}],
            "reward_model": {"style": "rule", "ground_truth": str(row.answer)},
            "extra_info": {
                "index": f"{BAND}-{split}-{split_index}",
                "source_uid": str(row.uid),
                "upstream_dataset": UPSTREAM_DATASET,
                "difficulty_band": BAND,
                "difficulty_definition": DEFINITION,
                "split": split,
                "teacher_model": TEACHER_MODEL,
                "teacher_revision": TEACHER_REVISION,
                "teacher_n_correct": int(row.current_rl_n_correct),
                "teacher_n_samples": TEACHER_SAMPLES,
                "teacher_pass_rate": int(row.current_rl_n_correct) / TEACHER_SAMPLES,
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
                "resample_n_correct": resample_n,
                "resample_n_samples": RESAMPLE_SAMPLES,
                "resample_pass_rate": resample_n / RESAMPLE_SAMPLES,
                "resample_accs": normalize_bool_list(row.resample_accs),
                "resample_seed": RESAMPLE_SEED,
            },
        })
    return pd.DataFrame.from_records(records)


def counts(frame: pd.DataFrame, key: str) -> dict[str, int]:
    values = frame["extra_info"].map(lambda info: int(info[key])).value_counts().sort_index()
    return {str(k): int(v) for k, v in values.items()}


def render_counts(c: dict[str, int], denom: int) -> str:
    return ", ".join(f"{k}/{denom}: {v:,}" for k, v in c.items())


def update_readme(text: str, n_zero: int, n_candidates: int, frames: dict[str, pd.DataFrame], resample_dist: dict[str, int]) -> str:
    def rep(old: str, new: str) -> None:
        nonlocal text
        if text.count(old) != 1:
            raise ValueError(f"README anchor not found exactly once: {old[:60]!r}")
        text = text.replace(old, new)

    rep("pretty_name: DeepScaleR Easy/Medium/Hard by Qwen3-30B-A3B-Base", "pretty_name: DeepScaleR Easy/Medium/Hard/Very-Hard by Qwen3-30B-A3B-Base")
    rep("- 1K<n<10K", "- 10K<n<100K")
    rep("    path: hard/validation.parquet\n---",
        "    path: hard/validation.parquet\n- config_name: very_hard\n  data_files:\n  - split: train\n    path: very_hard/train.parquet\n"
        "  - split: validation\n    path: very_hard/validation.parquet\n---")
    rep("# DeepScaleR Easy/Medium/Hard — Qwen3-30B-A3B-Base", "# DeepScaleR Easy/Medium/Hard/Very-Hard — Qwen3-30B-A3B-Base")
    rep("This dataset contains 9,900 unique, deduplicated DeepScaleR math questions",
        "This dataset contains 13,200 unique, deduplicated DeepScaleR math questions")
    rep("The Hub dataset has three configurations—`easy`, `medium`, and `hard`—and each",
        "The Hub dataset has four configurations—`easy`, `medium`, `hard`, and `very_hard`—and each")
    rep("| `hard` | 1/8 | 6,440 | 3,000 | 300 | 3,300 |",
        f"| `hard` | 1/8 | 6,440 | 3,000 | 300 | 3,300 |\n| `very_hard` | 0/8, then ≥1/16 on resample | {n_candidates:,} | 3,000 | 300 | 3,300 |")
    rep("The 0/8 bucket is intentionally excluded. In this dataset, **hard means exactly\n1/8**.",
        "The `easy`/`medium`/`hard` bands exclude the 0/8 bucket, and **hard means exactly\n1/8**. "
        "`very_hard` is drawn from that 0/8 bucket: only questions the teacher then solved at least once\n"
        "in a separate 16-sample resample (see *Very-hard configuration* below), so every `very_hard` question is\n"
        "verifiably solvable by the teacher but rarely so.")
    tr, va = frames["train"], frames["validation"]
    rep("| `hard` | `validation` | 1/8: 300 |",
        "| `hard` | `validation` | 1/8: 300 |\n"
        f"| `very_hard` | `train` | 0/8: 3,000; resample {render_counts(counts(tr, 'resample_n_correct'), 16)} |\n"
        f"| `very_hard` | `validation` | 0/8: 300; resample {render_counts(counts(va, 'resample_n_correct'), 16)} |")
    section = f"""
## Very-hard configuration

The `very_hard` configuration (added 2026-10-10) is built from the {n_zero:,} questions the teacher solved **0/8** in the
original run above, which the other three bands exclude.

1. Resample each 0/8 question **16 more times** with the identical teacher protocol (same model, revision, prompt, template,
   stops, sampling settings and limits) but a new sampling seed ({RESAMPLE_SEED}), so the 16 responses are fresh draws rather than
   repeats of the original eight. Inference: vLLM 0.15.1, data parallel 2 × tensor parallel 1. Raw responses:
   `{RESAMPLE_RUN_S3}`.
2. Grade all 16 responses with the same RL grading contract (last well-formed box; `math_verify` OR bounded Miles SymPy).
3. Keep the questions with **at least one correct response out of 16**: {n_candidates:,} of {n_zero:,}
   ({100 * n_candidates / n_zero:.1f}%). The other {n_zero - n_candidates:,} stayed 0/16 and are not used.
   Resample score distribution over all {n_zero:,} questions: {render_counts(resample_dist, 16)}.
4. Select 3,300 of the {n_candidates:,} candidates and split them 3,000 train / 300 validation exactly as for the other bands,
   with band name `very_hard`: rank by `SHA256("selection\\042\\0very_hard\\0<uid>")`, take the lowest 3,300; rank those by
   `SHA256("validation\\043\\0very_hard\\0<uid>")`, the lowest 300 are validation.

Rows use the same schema as the other configurations. `extra_info.teacher_n_correct` is 0 (the original eight flags are
all false), and `extra_info` additionally carries `resample_n_correct`, `resample_n_samples` (16), `resample_pass_rate`,
`resample_accs` (the 16 per-response flags) and `resample_seed`. Mean per-sample accuracy of the resample over the selected
questions is low (most are 1/16 or 2/16), so `very_hard` sits below `hard` (1/8) in expected solve rate. Selection on ≥1/16
is a noisy filter: the 0/8 condition over-selects long or truncated responses, and fresh responses on these questions are
shorter (mean 967 vs 1,213 tokens).

Reproduce the configuration with
`python rl-distill-scripts/data/build_deepscaler_qwen3_30b_very_hard_band.py --resample-grades <resample grade parquet>`
(the grade Parquet is in the resample prefix above as `grades_strict_n16_accs.parquet`).
"""
    rep("\n## Teacher generation protocol\n", section + "\n## Teacher generation protocol\n")
    rep("13,859 (easy), 9,315 (medium) and 6,440 (hard)\nquestions, so each 3,300-question sample is a random subset of its band.",
        f"13,859 (easy), 9,315 (medium), 6,440 (hard) and {n_candidates:,} (very_hard)\n"
        "questions, so each 3,300-question sample is a random subset of its band.")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--grades", type=Path, default=DEFAULT_GRADES)
    parser.add_argument("--resample-grades", type=Path, required=True)
    parser.add_argument("--base-build-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="existing easy/medium/hard build")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_NEW_OUTPUT_DIR)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--parent-revision", default=PARENT_REVISION)
    parser.add_argument("--selection-size", type=int, default=3300)
    parser.add_argument("--validation-size", type=int, default=300)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--validation-seed", type=int, default=43)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    source = load_joined_source(args.questions, args.grades)
    zero = source[source["current_rl_n_correct"] == 0]
    resample = load_resample(args.resample_grades, set(zero["uid"]))
    joined = zero.merge(resample, on="uid", how="inner", validate="one_to_one")
    resample_dist = {str(k): int(v) for k, v in joined["resample_n_correct"].value_counts().sort_index().items()}
    candidates = joined[joined["resample_n_correct"] >= 1]
    print(f"0/8 questions {len(zero):,}; >=1/16 candidates {len(candidates):,}")
    train, validation = select(candidates, args.selection_size, args.validation_size, args.selection_seed, args.validation_seed)
    frames = {"train": to_rows(train, "train", args.selection_seed, args.validation_seed),
              "validation": to_rows(validation, "validation", args.selection_seed, args.validation_seed)}

    # validation against the existing build: same top-level schema, extra_info a superset, no uid overlap with any band
    existing_uids: set[str] = set()
    for band in ("easy", "medium", "hard"):
        for split in ("train", "validation"):
            frame = pd.read_parquet(args.base_build_dir / band / f"{split}.parquet")
            existing_uids |= set(frame["uid"])
            if list(frame.columns) != list(frames[split].columns):
                raise ValueError(f"column mismatch vs {band}/{split}")
            if not set(frame["extra_info"].iloc[0]) <= set(frames[split]["extra_info"].iloc[0]):
                raise ValueError(f"extra_info is missing keys present in {band}/{split}")
    if len(frames["train"]) != args.selection_size - args.validation_size or len(frames["validation"]) != args.validation_size:
        raise ValueError("unexpected split sizes")
    train_uids, val_uids = set(frames["train"]["uid"]), set(frames["validation"]["uid"])
    if train_uids & val_uids or (train_uids | val_uids) & existing_uids:
        raise ValueError("very_hard overlaps itself or an existing band")
    for frame in frames.values():
        if not frame["extra_info"].map(lambda i: i["teacher_n_correct"] == 0 and i["resample_n_correct"] >= 1).all():
            raise ValueError("a row violates the very_hard definition")

    out = args.output_dir
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(args.base_build_dir, out)
    files: dict[str, Any] = {}
    (out / BAND).mkdir()
    for split, frame in frames.items():
        path = out / BAND / f"{split}.parquet"
        frame.to_parquet(path, index=False)
        reloaded = pd.read_parquet(path)
        if len(reloaded) != len(frame) or set(reloaded["uid"]) != set(frame["uid"]):
            raise ValueError(f"round-trip failed for {path}")
        files[split] = {"path": f"{BAND}/{split}.parquet", "rows": len(frame), "sha256": sha256_file(path),
                        "uid_set_sha256": uid_set_sha256(frame["uid"]), "teacher_n_correct_counts": counts(frame, "teacher_n_correct"),
                        "resample_n_correct_counts": counts(frame, "resample_n_correct")}
        print(f"  {BAND}/{split}: {len(frame)} rows, resample scores {files[split]['resample_n_correct_counts']}")

    (out / "README.md").write_text(update_readme((args.base_build_dir / "README.md").read_text(), len(zero), len(candidates), frames, resample_dist))
    manifest = json.loads((args.base_build_dir / "manifest.json").read_text())
    manifest["files"][BAND] = files
    manifest["sampling"]["bands"][BAND] = {"definition": DEFINITION, "original_n_correct": 0, "min_resample_n_correct": 1,
                                           "zero_of_8_questions": len(zero), "available": len(candidates)}
    manifest["very_hard"] = {
        "added_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "builder": {"path": str(Path(__file__).resolve().relative_to(REPO_ROOT)), "sha256": sha256_file(Path(__file__)),
                    "rl_distill_git_head": git_head()},
        "resample": {"samples_per_question": RESAMPLE_SAMPLES, "seed": RESAMPLE_SEED, "run_s3": RESAMPLE_RUN_S3,
                     "grades_sha256": sha256_file(args.resample_grades), "n_correct_distribution": resample_dist,
                     "protocol": "identical to the teacher protocol above except the sampling seed"},
        "validation": {"exact_split_counts": True, "no_train_validation_overlap": True, "no_overlap_with_easy_medium_hard": True,
                       "all_rows_satisfy_definition": True, "schema_matches_other_bands": True},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out}")

    if args.upload:
        load_env_file()
        from huggingface_hub import CommitOperationAdd, HfApi
        api = HfApi(token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY"))
        ops = [CommitOperationAdd(path_in_repo=p, path_or_fileobj=str(out / p))
               for p in ("very_hard/train.parquet", "very_hard/validation.parquet", "README.md", "manifest.json")]
        commit = api.create_commit(repo_id=args.repo_id, repo_type="dataset", operations=ops, parent_commit=args.parent_revision,
                                   commit_message="Add very_hard configuration (0/8 questions solved >=1/16 on a 16-sample resample)")
        print(f"UPLOADED commit={commit.oid}")


if __name__ == "__main__":
    main()
