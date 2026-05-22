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

"""Update the repo-level EVAL_RESULTS.md from distillation eval summaries."""

from __future__ import annotations

import argparse
import fcntl
import json
import shutil
import time
from pathlib import Path

START = "<!-- DISTILL_SFT_RESULTS_START -->"
END = "<!-- DISTILL_SFT_RESULTS_END -->"

BENCHES = [
    ("DAPO val", "dapo_openmath2_mix_val_compat"),
    ("AIME 2024", "math__aime2024_repeated_32x_960_compat"),
    ("AIME 2025", "math__aime2025_repeated_32x_960_compat"),
    ("AIME 2026", "math__aime2026_repeated_32x_960_compat"),
    ("MATH500", "math__math_500_repeated_2x_1000_compat"),
    ("OlympiadBench", "math__olympiadbench_repeated_2x_compat"),
    ("MinervaMath", "math__minervamath_repeated_4x_compat"),
    ("GSM8K", "math__gsm8k_test_compat"),
]


def infer_label(repo: str) -> str:
    tail = repo.split("/")[-1]
    if tail == "gemma3-4b-pt-sft-distill-from-27b-rl-step40-seed43":
        return "4B PT <- 27B RL step40"
    if tail == "gemma3-4b-pt-sft-distill-from-27b-rl-step40-seed43-n4-32k":
        return "4B PT <- 27B RL step40, 32k x4"
    if tail == "gemma3-4b-pt-sft-distill-from-12b-rl-step20-seed43":
        return "4B PT <- 12B RL step20"
    if tail == "gemma3-4b-pt-sft-distill-from-12b-rl-step20-seed43-n4-32k":
        return "4B PT <- 12B RL step20, 32k x4"
    if tail == "gemma3-12b-pt-sft-distill-from-27b-rl-step40-seed43-lr2p5e-6":
        return "12B PT <- 27B RL step40, lr 2.5e-6"
    if tail == "gemma3-4b-pt-sft-distill-from-27b-rl-step40-seed43-32k-n4":
        return "4B PT <- 27B RL step40, 32k x4"
    if tail == "gemma3-4b-pt-sft-distill-from-12b-rl-step20-seed43-32k-n4":
        return "4B PT <- 12B RL step20, 32k x4"
    if tail == "gemma3-12b-pt-sft-distill-from-27b-rl-step40-seed43-32k-n4-lr2p5e-6":
        return "12B PT <- 27B RL step40, 32k x4, lr 2.5e-6"
    if tail == "gemma3-12b-pt-sft-distill-from-27b-rl-step40-seed43-n4-32k-lr2p5e-6":
        return "12B PT <- 27B RL step40, 32k x4, lr 2.5e-6"
    if tail == "gemma3-4b-pt-sft-distill-from-27b-rl-step40-seed43-all33296-n4":
        return "4B PT <- 27B RL step40, all33296 x4"
    if tail == "gemma3-12b-pt-sft-distill-from-27b-rl-step40-seed43-all33296-n4":
        return "12B PT <- 27B RL step40, all33296 x4"
    if tail == "gemma3-4b-pt-sft-distill-from-12b-rl-step20-seed43-all33296-n4":
        return "4B PT <- 12B RL step20, all33296 x4"
    return tail


def step_num(subfolder: str) -> int:
    try:
        return int(subfolder.rsplit("_", 1)[-1])
    except Exception:
        return 0


def fmt_pct(value) -> str:
    if not isinstance(value, int | float):
        return ""
    return f"{100.0 * value:.2f}"


def fmt_int(value) -> str:
    if not isinstance(value, int | float):
        return ""
    return f"{value:.0f}"


def mean_acc(per_dataset: dict) -> float | None:
    vals = []
    for _, key in BENCHES:
        val = per_dataset.get(key, {}).get("acc")
        if isinstance(val, int | float):
            vals.append(float(val))
    if not vals:
        return None
    return sum(vals) / len(vals)


def collect_summaries(shared_json_dir: Path) -> list[dict]:
    records_by_key = {}
    for path in sorted(shared_json_dir.glob("**/*__summary.json")):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        repo = data.get("model", "")
        subfolder = data.get("subfolder", "")
        per_dataset = data.get("per_dataset", {})
        key = (repo, subfolder)
        record = {
            "repo": repo,
            "label": infer_label(repo),
            "subfolder": subfolder,
            "step": step_num(subfolder),
            "per_dataset": per_dataset,
            "path": path,
        }
        old = records_by_key.get(key)
        if old is None or path.stat().st_mtime >= old["path"].stat().st_mtime:
            records_by_key[key] = record
    records = list(records_by_key.values())
    records.sort(key=lambda r: (r["label"], r["step"], r["repo"]))
    return records


def copy_results(results_dirs: list[Path], shared_json_dir: Path, run_name: str | None) -> None:
    shared_json_dir.mkdir(parents=True, exist_ok=True)
    for results_dir in results_dirs:
        if not results_dir.exists():
            continue
        dest_dir = shared_json_dir / (run_name or results_dir.name)
        dest_dir.mkdir(parents=True, exist_ok=True)
        for src in sorted(results_dir.glob("*__summary.json")):
            shutil.copy2(src, dest_dir / src.name)


def make_table(records: list[dict], metric: str) -> list[str]:
    if metric == "acc":
        title = "### Accuracy (%)"
        format_value = fmt_pct
    elif metric == "length":
        title = "### Mean Response Length (tokens)"
        format_value = fmt_int
    else:
        raise ValueError(metric)

    lines = [title, ""]
    headers = ["Run / Checkpoint"] + [name for name, _ in BENCHES]
    if metric == "acc":
        headers.append("Mean (8)")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")

    for rec in records:
        row = [f"{rec['label']} / {rec['subfolder']}"]
        for _, key in BENCHES:
            item = rec["per_dataset"].get(key, {})
            value = item.get("acc") if metric == "acc" else item.get("response_length_mean")
            row.append(format_value(value))
        if metric == "acc":
            row.append(fmt_pct(mean_acc(rec["per_dataset"])))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return lines


def make_section(records: list[dict], shared_json_dir: Path) -> str:
    lines = [
        START,
        "## Accuracy - Current PT SFT Distillation (8 eval sets)",
        "",
        "Generation config: T=1.0, top_p=0.7, top_k=-1, max_tokens=20480, mean@1, `math_verify` scoring.",
        "Checkpoint coverage depends on the run; the 32k x4 reruns evaluate "
        "`step_000250`, `step_000500`, `step_000750`, and `step_001000`.",
        f"Updated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}.",
        "",
        f"Summary JSON archive: `{shared_json_dir}`.",
        "",
    ]
    if not records:
        lines += ["No completed eval summaries have been written yet.", "", END, ""]
        return "\n".join(lines)

    lines += make_table(records, "acc")
    lines += make_table(records, "length")
    lines += [
        "### HF Repos",
        "",
    ]
    seen = set()
    for rec in records:
        key = (rec["label"], rec["repo"])
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- {rec['label']}: `{rec['repo']}`")
    lines += ["", END, ""]
    return "\n".join(lines)


def update_md(md_path: Path, section: str) -> None:
    md_path.parent.mkdir(parents=True, exist_ok=True)
    old = md_path.read_text() if md_path.exists() else "# Evaluation Results\n"
    if START in old and END in old:
        before = old.split(START, 1)[0].rstrip()
        after = old.split(END, 1)[1].lstrip()
        new = before + "\n\n" + section.rstrip() + "\n\n" + after
    else:
        new = old.rstrip() + "\n\n" + section.rstrip() + "\n"
    md_path.write_text(new)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir", action="append", default=[], help="Eval output directory containing *__summary.json files."
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--md", default=None)
    parser.add_argument("--shared-json-dir", default=None)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    md_path = Path(args.md) if args.md else project_root / "EVAL_RESULTS.md"
    shared_json_dir = (
        Path(args.shared_json_dir) if args.shared_json_dir else project_root / "eval_results" / "distill_sft"
    )
    results_dirs = [Path(p) for p in args.results_dir]

    lock_path = md_path.with_suffix(md_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        copy_results(results_dirs, shared_json_dir, args.run_name)
        records = collect_summaries(shared_json_dir)
        section = make_section(records, shared_json_dir)
        update_md(md_path, section)
        fcntl.flock(lock, fcntl.LOCK_UN)

    print(f"[eval-md] wrote {md_path} with {len(records)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
