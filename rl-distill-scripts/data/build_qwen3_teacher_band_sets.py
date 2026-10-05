#!/usr/bin/env python3
"""Build teacher-defined DeepScaleR difficulty eval sets from Qwen3 base n=8 sampling grades.

For each teacher (Qwen3-8B-Base, Qwen3-30B-A3B-Base), questions are bucketed by the teacher's own strict (boxed-only)
k/8 from the full-dataset sampling run: easy = 4..7, medium = 2..3, hard = 1 (0/8 and 8/8 excluded). 300 questions
per band are drawn uniformly (numpy seed 42, independently per teacher/band). Output parquets have uid / problem /
answer / teacher_k (+ source_index into deepscaler_it_gen_dedup_v2.parquet), the input format of
generate_deepscaler_4bpt_samples.py.

    python build_qwen3_teacher_band_sets.py --grades-8b <dir> --grades-30b <dir> --out <dir>
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

BANDS = {"easy": range(4, 8), "medium": range(2, 4), "hard": range(1, 2)}
SOURCE = "/mnt/efs/jasonwei/verl/data/deepscaler_it_gen/deepscaler_it_gen_dedup_v2.parquet"


def load_k(root: str) -> pd.Series:
    rows = [r for f in sorted(glob.glob(f"{root}/**/part_*.parquet.grades.json", recursive=True)) for r in json.load(open(f))]
    k = pd.DataFrame(rows).set_index("source_index")["k"]
    assert len(k) == 38796 and k.index.is_unique, len(k)
    return k


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grades-8b", required=True)
    parser.add_argument("--grades-30b", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-band", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    source = pd.read_parquet(SOURCE, columns=["uid", "problem", "answer"]).reset_index(drop=True)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for teacher, root in (("8b", args.grades_8b), ("30b", args.grades_30b)):
        k = load_k(root)
        for band, ks in BANDS.items():
            pool = np.sort(k.index[k.isin(list(ks))].to_numpy())
            pick = np.random.default_rng(args.seed).choice(pool, size=args.per_band, replace=False)
            df = source.loc[np.sort(pick)].copy()
            df.insert(0, "source_index", df.index.astype(int))
            df["teacher_k"] = k.loc[df.index].to_numpy()
            path = args.out / f"teacher{teacher}_{band}.parquet"
            df.reset_index(drop=True).to_parquet(path, index=False)
            manifest[path.stem] = {"pool": int(len(pool)), "n": len(df), "teacher_mean_at_8": float(df.teacher_k.mean() / 8)}
            print(f"{path.stem}: pool={len(pool)} picked={len(df)} teacher mean@8={100*df.teacher_k.mean()/8:.1f}%")
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
