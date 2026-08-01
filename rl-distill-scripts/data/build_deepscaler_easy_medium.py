#!/usr/bin/env python3
"""Build DeepScaleR-Easy-10k and DeepScaleR-Medium-20k RL datasets (2026-08-01).

Source: the DEDUPED v2 generation set (`dedup_deepscaler_it_gen.py`: max-strength
question normalization; conflicting-gold duplicate groups removed entirely via
math_verify equivalence; one random copy kept per surviving group; seed 42).
Buckets here use the LENIENT grading (`n_correct`).

- Easy-10k:   10,000 random questions from lenient 4/4 -> train 9,500 / val 500.
- Medium-20k: the SAME 10,000 + 3,000 from 3/4 + 3,000 from 2/4 + 4,000 from 1/4
              -> 500 random val from the 20,000, train 19,500.

Rows are verl-format (data_source/prompt/reward_model/extra_info), matching the
existing 4of4 parquets. Each val also gets a x16-replicated variant for mean@16.
Uploads to HF hub: JWei05/DeepScaleR-Easy-10k, JWei05/DeepScaleR-Medium-20k.
"""

import argparse
import os

import numpy as np
import pandas as pd

INSTR = None  # appended instruction is baked into the chat template, not the prompt


def to_verl(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "data_source": "math",
            "prompt": [[{"role": "user", "content": p}] for p in df["problem"]],
            "reward_model": [{"style": "rule", "ground_truth": str(a)} for a in df["answer"]],
            "extra_info": [
                {"lenient_n_correct": int(n), "strict_n_correct": int(s)}
                for n, s in zip(df["n_correct"], df["strict_n_correct"])
            ],
        }
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="~/verl/data/deepscaler_it_gen/deepscaler_it_gen_dedup_v2.parquet")
    ap.add_argument("--outdir", default="~/verl/data")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--upload", action="store_true")
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    df = pd.read_parquet(os.path.expanduser(args.inp))
    outdir = os.path.expanduser(args.outdir)

    # ---- Easy-10k: 10k random from lenient 4/4 ----
    b4 = df[df["n_correct"] == 4]
    easy = b4.sample(n=10_000, random_state=rng)
    easy_val = easy.sample(n=500, random_state=rng)
    easy_train = easy.drop(easy_val.index)

    # ---- Medium-20k: same 10k + 3k from 3/4 + 3k from 2/4 + 4k from 1/4 ----
    parts = [easy]
    for k, n in ((3, 3_000), (2, 3_000), (1, 4_000)):
        parts.append(df[df["n_correct"] == k].sample(n=n, random_state=rng))
    medium = pd.concat(parts)
    med_val = medium.sample(n=500, random_state=rng)
    med_train = medium.drop(med_val.index)

    sets = {
        "DeepScaleR-Easy-10k": (easy_train, easy_val),
        "DeepScaleR-Medium-20k": (med_train, med_val),
    }
    files = {}
    for name, (tr, va) in sets.items():
        stem = name.lower().replace("-", "_")
        f_tr = os.path.join(outdir, f"{stem}_train.parquet")
        f_va = os.path.join(outdir, f"{stem}_val500.parquet")
        f_vx = os.path.join(outdir, f"{stem}_val500_x16.parquet")
        to_verl(tr).to_parquet(f_tr, index=False)
        va_v = to_verl(va)
        va_v.to_parquet(f_va, index=False)
        pd.concat([va_v] * 16, ignore_index=True).to_parquet(f_vx, index=False)
        files[name] = [f_tr, f_va]  # x16 replica stays LOCAL only (mean@16 prep artifact)
        vb = va["n_correct"].value_counts()
        tb = tr["n_correct"].value_counts()
        print(f"{name}: train {len(tr)} val {len(va)}")
        print(f"  train buckets {{k/4: n}}: { {k: int(tb.get(k, 0)) for k in range(5)} }")
        print(f"  val buckets:   { {k: int(vb.get(k, 0)) for k in range(5)} }")

    if args.upload:
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ["HF_TOKEN"])
        readmes = {
            "DeepScaleR-Easy-10k": """---
license: mit
---
# DeepScaleR-Easy-10k

RL training set of 10,000 DeepScaleR questions that Gemma-3-4B-IT solved **4/4**
(temp 1.0, 4 samples, LENIENT math_verify grading), split train 9,500 / val 500.

Creation (seed 42 throughout):
1. Gemma-3-4B-IT generated 4 responses per DeepScaleR question (40,315 rows).
2. **Dedup v2**: questions grouped under max-strength normalization (case,
   whitespace, prose punctuation, LaTeX formatting); duplicate groups whose gold
   answers are not all math_verify-equivalent were removed entirely (167 groups /
   567 rows — label conflicts); one random copy kept per surviving group
   -> 38,796 unique questions. Script: `rl-distill-scripts/data/dedup_deepscaler_it_gen.py`.
3. 10,000 sampled uniformly from the lenient 4/4 bucket (10,053 available).
4. 500 sampled uniformly as validation; remaining 9,500 are train.

Files: `*_train.parquet` and `*_val500.parquet` (500 unique held-out questions).
For mean@16 validation, replicate the val rows 16x locally (verl samples one
response per row). verl format: data_source "math" (routes to the repo's
math_verify scorer), prompt = single user message, reward_model.ground_truth,
extra_info carries lenient/strict pass counts.

Built by `rl-distill-scripts/data/build_deepscaler_easy_medium.py` in
JasonWei05/rl-distill.
""",
            "DeepScaleR-Medium-20k": """---
license: mit
---
# DeepScaleR-Medium-20k

RL training set of 20,000 DeepScaleR questions spanning difficulty (Gemma-3-4B-IT
pass rate at temp 1.0, 4 samples, LENIENT math_verify grading): the **same 10,000
4/4 questions as DeepScaleR-Easy-10k** + 3,000 from 3/4 + 3,000 from 2/4 + 4,000
from 1/4. 500 random questions (across all difficulties) held out as validation;
train 19,500.

Creation (seed 42 throughout): identical dedup-v2 pipeline as DeepScaleR-Easy-10k
(see that README): max-strength question dedup, conflicting-gold groups removed
via math_verify equivalence, one random copy per group -> 38,796 unique questions;
buckets sampled uniformly without replacement.

NOTE: because the 10k easy questions are shared, DeepScaleR-Easy-10k's val
questions may appear in this set's TRAIN split (and vice versa). Do not evaluate
a Medium-trained model on Easy-10k's val (or vice versa); each set is
self-consistent only with its own split.

Files: `*_train.parquet` and `*_val500.parquet` (500 unique held-out questions).
For mean@16 validation, replicate the val rows 16x locally. verl format:
data_source "math", prompt = single user message, reward_model.ground_truth,
extra_info carries lenient/strict pass counts.

Built by `rl-distill-scripts/data/build_deepscaler_easy_medium.py` in
JasonWei05/rl-distill.
""",
        }
        for name, fl in files.items():
            repo = f"JWei05/{name}"
            api.create_repo(repo, repo_type="dataset", private=False, exist_ok=True)
            for f in fl:
                api.upload_file(path_or_fileobj=f, path_in_repo=os.path.basename(f), repo_id=repo, repo_type="dataset")
            api.upload_file(path_or_fileobj=readmes[name].encode(), path_in_repo="README.md", repo_id=repo, repo_type="dataset")
            print(f"uploaded https://huggingface.co/datasets/{repo}")


if __name__ == "__main__":
    main()
