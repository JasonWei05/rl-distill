#!/usr/bin/env python3
"""Single-pass helpers for run_qwen3_math_suite_st.sh (SINGLE_PASS=1).

combine <data dir> <out parquet> <set...>   one generator input over all sets: uid "<set>::<uid>", per-row n_samples
                                            (AIME_N for aime*, SET_N otherwise); prints the row count
split <samples parquet> <out root> <set...> per-set samples.parquet under <out root>/<set>/ with the original uids
"""

import os
import sys
from pathlib import Path

import pandas as pd


def combine(data: str, out: str, sets: list[str]) -> None:
    parts = []
    for s in sets:
        df = pd.read_parquet(f"{data}/{s}.parquet").reset_index(drop=True)
        if {"problem", "answer"}.issubset(df.columns):
            d = pd.DataFrame({"uid": df["uid"].astype(str), "problem": df["problem"], "answer": df["answer"].astype(str)})
        else:  # band sets: prompt/reward_model rows (same extraction as the generator's normalize_input)
            d = pd.DataFrame(
                {
                    "uid": df["uid"].astype(str),
                    "problem": df["prompt"].map(lambda messages: messages[-1]["content"]),
                    "answer": df["reward_model"].map(lambda reward: reward["ground_truth"]),
                }
            )
        d["uid"] = s + "::" + d["uid"]
        d["n_samples"] = int(os.environ["AIME_N"] if s.startswith("aime") else os.environ["SET_N"])
        parts.append(d)
    combo = pd.concat(parts, ignore_index=True)
    if combo["uid"].duplicated().any():
        raise ValueError("duplicate uids across sets")
    combo.to_parquet(out, index=False)
    print(len(combo))


def split(samples: str, root: str, sets: list[str]) -> None:
    df = pd.read_parquet(samples)
    parts = df["uid"].str.split("::", n=1)
    df["set"], df["uid"] = parts.str[0], parts.str[1]
    for s in sets:
        part = df[df["set"] == s].drop(columns="set").reset_index(drop=True)
        if part.empty:
            raise ValueError(f"no samples for set {s}")
        (Path(root) / s).mkdir(parents=True, exist_ok=True)
        part.to_parquet(Path(root) / s / "samples.parquet", index=False)
        print("SPLIT", s, len(part), int(part["response_texts"].map(len).iloc[0]), flush=True)


if __name__ == "__main__":
    {"combine": combine, "split": split}[sys.argv[1]](sys.argv[2], sys.argv[3], sys.argv[4:])
