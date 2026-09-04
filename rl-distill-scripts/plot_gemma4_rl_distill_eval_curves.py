#!/usr/bin/env python3
"""Plot comparison and per-model pass@k/maj@k curves from completed evals."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

GROUPS = {
    "e2b_easy": (
        "base_e2b",
        "rl_e2b_easy_total_step360",
        "distill_e4b_easy_to_e2b_step1000",
        "distill_12b_easy_to_e2b_step1000",
    ),
    "e2b_medium": (
        "base_e2b",
        "rl_e2b_medium_step180",
        "distill_e4b_medium_to_e2b_step1000",
        "distill_12b_medium_to_e2b_step1000",
    ),
    "e4b_easy_from_12b": ("base_e4b", "rl_e4b_easy_step160", "distill_12b_easy_to_e4b_step1000"),
    "e4b_medium_from_12b": ("base_e4b", "rl_e4b_medium_step060", "distill_12b_medium_to_e4b_step1000"),
}

TEACHER_BY_STUDENT = {
    "distill_e4b_easy_to_e2b_step1000": "rl_e4b_easy_step160",
    "distill_e4b_medium_to_e2b_step1000": "rl_e4b_medium_step060",
    "distill_12b_easy_to_e2b_step1000": "rl_12b_easy_step160",
    "distill_12b_medium_to_e2b_step1000": "rl_12b_medium_first_step140",
    "distill_12b_easy_to_e4b_step1000": "rl_12b_easy_step160",
    "distill_12b_medium_to_e4b_step1000": "rl_12b_medium_first_step140",
}

TEACHER_ALPHA = 0.32

DATASET_ORDER = ("id_easy", "id_medium", "math500", "gsm8k")


def _curve_values(result: dict) -> tuple[list[int], list[float], list[float]]:
    ks = sorted(int(k) for k in result["by_k"])
    pass_values = [100 * result["by_k"][str(k)]["pass_at_k"] for k in ks]
    maj_values = [100 * result["by_k"][str(k)]["maj_at_k"] for k in ks]
    return ks, pass_values, maj_values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--audit-report", type=Path)
    source.add_argument("--results-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison_rows = []
    if args.audit_report is not None:
        report = json.loads(args.audit_report.read_text(encoding="utf-8"))
        if report.get("protocol") != "gemma4_rl_distill_eval_final_audit_v1" or report.get("status") != "pass":
            raise ValueError("--audit-report must be a passing final Gemma 4 evaluation audit")
        cache = {model["tag"]: model["math"] for model in report["models"]}
    else:
        cache = {}
        required_tags = {tag for tags in GROUPS.values() for tag in tags}
        required_tags.update(TEACHER_BY_STUDENT.values())
        for tag in sorted(required_tags):
            path = args.results_root / tag / "math" / "metrics.json"
            cache[tag] = json.loads(path.read_text(encoding="utf-8"))["results"]

    teacher_curve_rows = 0
    teacher_overlays = 0
    for group, tags in GROUPS.items():
        difficulty = "easy" if "_easy_" in f"_{group}_" else "medium"
        for dataset in (f"id_{difficulty}", "math500", "gsm8k"):
            figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True, sharey=True)
            all_ks = set()
            for tag in tags:
                result = cache[tag][dataset]
                ks, pass_values, maj_values = _curve_values(result)
                all_ks.update(ks)
                (pass_line,) = axes[0].plot(ks, pass_values, marker="o", label=tag, zorder=3)
                color = pass_line.get_color()
                axes[1].plot(ks, maj_values, marker="o", color=color, label=tag, zorder=3)
                for k, pass_value, maj_value in zip(ks, pass_values, maj_values, strict=True):
                    comparison_rows.append(
                        {
                            "group": group,
                            "dataset": dataset,
                            "model": tag,
                            "k": k,
                            "pass_at_k": pass_value,
                            "maj_at_k": maj_value,
                        }
                    )

                teacher_tag = TEACHER_BY_STUDENT.get(tag)
                if teacher_tag is None:
                    continue
                teacher_result = cache[teacher_tag][dataset]
                teacher_ks, teacher_pass, teacher_maj = _curve_values(teacher_result)
                all_ks.update(teacher_ks)
                teacher_label = f"{teacher_tag} (teacher)"
                teacher_style = {
                    "alpha": TEACHER_ALPHA,
                    "color": color,
                    "linestyle": "--",
                    "linewidth": 1.5,
                    "marker": "o",
                    "markersize": 4,
                    "zorder": 1,
                }
                axes[0].plot(teacher_ks, teacher_pass, label=teacher_label, **teacher_style)
                axes[1].plot(teacher_ks, teacher_maj, label=teacher_label, **teacher_style)
                teacher_overlays += 1
                for k, pass_value, maj_value in zip(
                    teacher_ks, teacher_pass, teacher_maj, strict=True
                ):
                    comparison_rows.append(
                        {
                            "group": group,
                            "dataset": dataset,
                            "model": teacher_tag,
                            "k": k,
                            "pass_at_k": pass_value,
                            "maj_at_k": maj_value,
                        }
                    )
                    teacher_curve_rows += 1
            for axis, title in zip(axes, ("pass@k", "maj@k"), strict=True):
                axis.set_title(title)
                axis.set_xlabel("k")
                axis.set_ylabel("accuracy (%)")
                axis.grid(alpha=0.3)
                axis.set_xticks(sorted(all_ks))
            axes[1].legend(fontsize=7)
            figure.suptitle(f"{group}: {dataset}")
            figure.tight_layout()
            figure.savefig(args.output_dir / f"{group}__{dataset}.png", dpi=180)
            plt.close(figure)

    with (args.output_dir / "curves.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0]))
        writer.writeheader()
        writer.writerows(comparison_rows)

    individual_dir = args.output_dir / "by_model_dataset"
    individual_dir.mkdir(parents=True, exist_ok=True)
    individual_rows = []
    individual_figures = 0
    for tag, results in cache.items():
        datasets = [dataset for dataset in DATASET_ORDER if dataset in results]
        datasets.extend(sorted(set(results) - set(datasets)))
        for dataset in datasets:
            ks, pass_values, maj_values = _curve_values(results[dataset])
            figure, axis = plt.subplots(figsize=(7.5, 4.8))
            axis.plot(ks, pass_values, marker="o", linewidth=2, label="pass@k")
            axis.plot(ks, maj_values, marker="o", linewidth=2, label="maj@k")
            axis.set_title(f"{tag}: {dataset}")
            axis.set_xlabel("k")
            axis.set_ylabel("accuracy (%)")
            axis.set_xticks(ks)
            axis.set_ylim(0, 100)
            axis.grid(alpha=0.3)
            axis.legend()
            figure.tight_layout()
            figure.savefig(individual_dir / f"{tag}__{dataset}.png", dpi=180)
            plt.close(figure)
            individual_figures += 1
            for k, pass_value, maj_value in zip(ks, pass_values, maj_values, strict=True):
                individual_rows.append(
                    {
                        "model": tag,
                        "dataset": dataset,
                        "k": k,
                        "pass_at_k": pass_value,
                        "maj_at_k": maj_value,
                    }
                )

    with (individual_dir / "curves.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(individual_rows[0]))
        writer.writeheader()
        writer.writerows(individual_rows)

    (args.output_dir / "plot_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "comparison_figures": len(GROUPS) * 3,
                "comparison_curve_rows": len(comparison_rows),
                "comparison_teacher_curve_rows": teacher_curve_rows,
                "comparison_teacher_overlays": teacher_overlays,
                "individual_figures": individual_figures,
                "individual_curve_rows": len(individual_rows),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
