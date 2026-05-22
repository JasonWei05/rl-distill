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

"""Compile eval summary JSONs into a single markdown table report."""

import glob
import json
import os
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/home/tiger/verl/data/eval_results"
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "EVAL_RESULTS.md")

DATASET_ORDER = [
    ("math__aime2024_repeated_32x_960", "AIME 2024"),
    ("math__aime2025_repeated_32x_960", "AIME 2025"),
    ("math__aime2026_repeated_32x_960", "AIME 2026"),
    ("math__math_500_repeated_2x_1000", "MATH500"),
    ("math__olympiadbench_repeated_2x", "OlympiadBench"),
    ("math__minervamath_repeated_4x", "MinervaMath"),
    ("math__gsm8k_test", "GSM8K"),
]


def model_key(summary):
    repo = summary["model"].split("/")[-1]
    sub = summary["subfolder"]
    if summary["model"].startswith("google/"):
        return f"{repo} (base IT, no SFT)"
    return f"{repo} / {sub}"


def model_sort_key(s):
    is_base = s["model"].startswith("google/")
    size = 0 if "4b" in s["model"].lower() else 1
    if is_base:
        step = -1  # baselines first within each size group
    elif s["subfolder"] == "root":
        step = 10**9
    else:
        step = int(s["subfolder"].replace("step_", ""))
    return (size, step)


def main():
    summaries = []
    for j in sorted(glob.glob(os.path.join(ROOT, "*__summary.json"))):
        with open(j) as f:
            summaries.append(json.load(f))
    summaries.sort(key=model_sort_key)

    if not summaries:
        print(f"no summary JSONs in {ROOT}", file=sys.stderr)
        return 1

    lines = []
    lines.append("# Eval: DAPO-Gemma3-27B off-policy distilled students on math val sets")
    lines.append("")
    first = summaries[0]
    lines.append(
        f"Generation config: T={first.get('temperature', 0.7)}, "
        f"top_p={first.get('top_p', 0.95)}, max_tokens={first.get('max_tokens', 20480)}."
    )
    lines.append("")
    lines.append("All scores are `math_verify`-judged per-sample mean (`val-core/<dataset>/acc/mean@1` convention).")
    lines.append("")

    # Main accuracy table
    lines.append("## Accuracy")
    lines.append("")
    header = ["Model / Checkpoint"] + [disp for _, disp in DATASET_ORDER] + ["Mean"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for s in summaries:
        row = [model_key(s)]
        accs = []
        for ds_key, _ in DATASET_ORDER:
            e = s["per_dataset"].get(ds_key, {})
            acc = e.get("acc", None)
            if acc is None:
                row.append("—")
            else:
                row.append(f"{100 * acc:.1f}")
                accs.append(acc)
        row.append(f"{100 * sum(accs) / len(accs):.1f}" if accs else "—")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Response length table
    lines.append("## Mean response length (tokens)")
    lines.append("")
    lines.append("| " + " | ".join(header[:-1]) + " |")
    lines.append("|" + "|".join(["---"] * (len(header) - 1)) + "|")
    for s in summaries:
        row = [model_key(s)]
        for ds_key, _ in DATASET_ORDER:
            e = s["per_dataset"].get(ds_key, {})
            ln = e.get("response_length_mean", None)
            row.append("—" if ln is None else f"{ln:.0f}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Per-model wall-clock (total gen time across 7 sets)
    lines.append("## Wall-clock (generation seconds, summed over 7 datasets)")
    lines.append("")
    lines.append("| Model / Checkpoint | Total gen seconds |")
    lines.append("|---|---|")
    for s in summaries:
        t = sum(s["per_dataset"].get(k, {}).get("gen_seconds", 0) for k, _ in DATASET_ORDER)
        lines.append(f"| {model_key(s)} | {t:.0f} |")
    lines.append("")

    with open(OUT, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {OUT}")
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    sys.exit(main() or 0)
