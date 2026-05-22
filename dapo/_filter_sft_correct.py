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

"""Filter JWei05/DAPO-Gemma3-27B-IT-RL-SFT-Data down to teacher responses that
the math_verify scorer marks as correct.

Pipeline:
  1. Download the teacher SFT parquet from HF.
  2. Load ~/verl/data/dapo-math-17k.parquet for ground truths, keyed by prompt_idx.
  3. Score each row in parallel via a ProcessPoolExecutor (spawn workers), using
     math_verify's LatexExtractionConfig path — the same scorer used during
     RL training and the math eval suite (only `\\boxed{}` answers count).
  4. Keep rows where score > 0.5.
  5. Write `teacher_27b_step40_n4_correct.parquet` and upload it as a new
     HF dataset: JWei05/DAPO-Gemma3-27B-IT-RL-SFT-Data-correct
"""

import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from multiprocessing import get_context
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, create_repo, hf_hub_download
from tqdm import tqdm

SRC_REPO = os.environ.get("SRC_REPO", "JWei05/DAPO-Gemma3-27B-IT-RL-SFT-Data")
SRC_FILE = os.environ.get("SRC_FILE", "teacher_27b_step40_n4.parquet")
DST_REPO = os.environ.get("DST_REPO", "JWei05/DAPO-Gemma3-27B-IT-RL-SFT-Data-correct")
DAPO_PARQUET = os.environ.get("DAPO_PARQUET", str(Path.home() / "verl/data/dapo-math-17k.parquet"))
OUT_PARQUET = os.environ.get("OUT_PARQUET", str(Path.home() / "verl/data/teacher_27b_step40_n4_correct.parquet"))
WORKERS = int(os.environ.get("WORKERS", "32"))
TIMEOUT = float(os.environ.get("TIMEOUT", "20"))


def _score_one(gt_boxed: str, model_output: str) -> float:
    """Subprocess-safe scorer. Same logic as verl/utils/reward_score/math_verify.py
    but callable from a parent ProcessPoolExecutor."""
    try:
        from math_verify.grader import verify
        from math_verify.parser import LatexExtractionConfig, parse
    except Exception:
        return 0.0
    try:
        gold = parse(gt_boxed, (LatexExtractionConfig(),))
        pred = parse(model_output, (LatexExtractionConfig(),))
        if gold and pred:
            return float(max(1.0 if any(verify(g, p) for g in gold) else 0.0 for p in pred))
    except Exception:
        pass
    return 0.0


def main():
    t0 = time.time()

    print(f"[filter] reading DAPO-Math-17k ground truths from {DAPO_PARQUET}")
    dapo = pd.read_parquet(DAPO_PARQUET)
    # Ground truth is nested under `reward_model.ground_truth`. Row index is prompt_idx.
    gt_by_idx = {}
    for i, row in dapo.iterrows():
        rm = row.get("reward_model", None)
        gt = None
        if isinstance(rm, dict):
            gt = rm.get("ground_truth")
        else:
            try:
                gt = rm["ground_truth"]
            except Exception:
                gt = None
        if gt is not None:
            gt_by_idx[i] = str(gt)
    print(f"[filter] {len(gt_by_idx)} / {len(dapo)} prompts have ground_truth")

    print(f"[filter] downloading {SRC_REPO}/{SRC_FILE}")
    sft_path = hf_hub_download(repo_id=SRC_REPO, filename=SRC_FILE, repo_type="dataset")
    sft = pd.read_parquet(sft_path)
    print(f"[filter] loaded {len(sft)} rows, cols={list(sft.columns)}")

    print(f"[filter] submitting {len(sft)} scoring jobs to {WORKERS} workers (timeout={TIMEOUT}s per call)")
    ctx = get_context("spawn")
    scores = [0.0] * len(sft)
    n_skipped_no_gt = 0
    n_timeout = 0
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=ctx) as pool:
        futures = []
        for i, row in sft.iterrows():
            gt = gt_by_idx.get(row["prompt_idx"])
            if gt is None:
                n_skipped_no_gt += 1
                continue
            gt_boxed = "\\boxed{" + gt + "}"
            resp = row["messages"][1]["content"]
            futures.append((i, pool.submit(_score_one, gt_boxed, resp)))

        print(f"[filter] {len(futures)} rows queued ({n_skipped_no_gt} skipped: no ground_truth)")
        for i, fut in tqdm(futures, total=len(futures), smoothing=0.05):
            try:
                scores[i] = float(fut.result(timeout=TIMEOUT))
            except (FuturesTimeoutError, Exception):
                scores[i] = 0.0
                n_timeout += 1

    sft["score"] = scores
    correct = sft[sft["score"] > 0.5].copy()

    # Per-prompt correctness breakdown
    per_prompt = correct.groupby("prompt_idx").size()
    fully_correct_prompts = (per_prompt == 4).sum()
    at_least_one = per_prompt.shape[0]
    total_prompts = sft["prompt_idx"].nunique()

    t_elapsed = time.time() - t0
    print("\n=== filter summary ===")
    print(f"  elapsed:           {t_elapsed:.0f}s  ({t_elapsed / 60:.1f} min)")
    print(f"  total rows:        {len(sft):,}")
    print(f"  correct rows:      {len(correct):,}  ({100 * len(correct) / len(sft):.1f}%)")
    print(f"  total prompts:     {total_prompts:,}")
    print(f"  prompts ≥1 right:  {at_least_one:,}  ({100 * at_least_one / total_prompts:.1f}%)")
    print(f"  prompts 4/4 right: {fully_correct_prompts:,}  ({100 * fully_correct_prompts / total_prompts:.1f}%)")
    print(f"  timeouts/errors:   {n_timeout}")
    print(f"  skipped no GT:     {n_skipped_no_gt}")

    # Write filtered parquet
    out_df = correct.drop(columns=["score"])
    os.makedirs(os.path.dirname(OUT_PARQUET), exist_ok=True)
    out_df.to_parquet(OUT_PARQUET, index=False)
    print(f"\n[filter] wrote {OUT_PARQUET} ({os.path.getsize(OUT_PARQUET) / 1e6:.1f} MB)")

    # Upload
    print(f"[upload] creating/ensuring dataset repo {DST_REPO}")
    api = HfApi()
    create_repo(repo_id=DST_REPO, repo_type="dataset", private=False, exist_ok=True)
    api.upload_file(
        path_or_fileobj=OUT_PARQUET,
        path_in_repo=os.path.basename(OUT_PARQUET),
        repo_id=DST_REPO,
        repo_type="dataset",
        commit_message="add filtered SFT parquet (teacher responses math_verify-correct)",
    )

    readme = f"""---
license: gemma
task_categories:
- text-generation
- question-answering
language:
- en
tags:
- math
- gemma3
- sft
- distillation
size_categories:
- 10K<n<100K
---

# DAPO-Gemma3-27B-IT-RL-SFT-Data-correct

Filtered subset of
[JWei05/DAPO-Gemma3-27B-IT-RL-SFT-Data](https://huggingface.co/datasets/{SRC_REPO}):
only the teacher responses whose final answer is `math_verify`-correct against
the original DAPO-Math-17k ground truth.

## Stats
- Source rows: {len(sft):,} (17,398 prompts × 4 teacher responses)
- Kept rows: {len(correct):,} ({100 * len(correct) / len(sft):.1f}%)
- Prompts with ≥1 correct response: {at_least_one:,} / {total_prompts:,}  ({100 * at_least_one / total_prompts:.1f}%)
- Prompts with 4/4 correct responses: {fully_correct_prompts:,}  ({100 * fully_correct_prompts / total_prompts:.1f}%)

## Scoring

Same function as used during RL training + math evals: `math_verify.parser`
with `LatexExtractionConfig` on both ground-truth and model output (only
`\\boxed{{…}}` answers are credited), `math_verify.grader.verify` for equality.
Rows with `score > 0.5` are kept.

## Columns
Same as the source:
- `messages`: [{{"role":"user","content":...}},{{"role":"assistant","content":...}}]
- `teacher_log_probs`: list[float], one per generated token
- `teacher_token_ids`: list[int], 1:1 with `teacher_log_probs`
- `prompt_idx`: index in the original DAPO-Math-17k dataset

## Use

Same as the unfiltered dataset — drop-in SFT data, but every response is
ground-truth-correct. Useful for reject-sampling SFT, where you want to train
the student only on responses that reach the right answer.

## License

Gemma derivative — subject to the [Gemma terms](https://ai.google.dev/gemma/terms).
"""
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        f.write(readme)
        rp = f.name
    api.upload_file(
        path_or_fileobj=rp,
        path_in_repo="README.md",
        repo_id=DST_REPO,
        repo_type="dataset",
        commit_message="add dataset card",
    )
    print(f"[upload] DONE → https://huggingface.co/datasets/{DST_REPO}")


if __name__ == "__main__":
    sys.exit(main() or 0)
