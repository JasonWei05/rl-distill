#!/usr/bin/env python3
"""Copy finished evaluation numbers into DISTILLATION_EXPERIMENTS.md (§8) as models complete.

Scans one or more results roots for the per-model files the eval runners write
(``<root>/<tag>/math/metrics.json`` and ``<root>/<tag>/ood/<benchmark>/**/results_*.json``; the queue
gives every model its own root ``<base>/<tag>/`` -- pass ``--results-base``),
orders rows by the study registry, and rewrites the doc between the ``<!-- results:start -->``
and ``<!-- results:end -->`` markers. Partial results are shown (missing cells as "—"), so it can
run after every completed model. Math (repo ``\\boxed{}`` verifier) and OOD (lm-eval-harness) are
rendered as separate tables because their scorers are not comparable.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DOC = REPO_ROOT / "rl-distill-scripts/DISTILLATION_EXPERIMENTS.md"
REGISTRY = REPO_ROOT / "rl-distill-scripts/config/gemma4_distill_study_eval_sources.json"
START, END = "<!-- results:start -->", "<!-- results:end -->"

MATH_COLUMNS = [("id_easy", 16), ("id_medium", 16), ("id_hard", 16), ("math500", 16), ("gsm8k", 8)]
OOD_COLUMNS = [  # benchmark dir name -> (label, lm-eval accuracy keys in preference order)
    ("mmlu_pro", "MMLU-Pro", ("exact_match,custom-extract", "exact_match,none", "acc,none")),
    ("gpqa", "GPQA-Diamond", ("exact_match,flexible-extract", "exact_match,none", "acc,none")),
    ("mmmlu14k", "MMLU-14k", ("acc,none", "acc_norm,none", "exact_match,none")),
]


def _math_results(metrics_path: Path) -> dict[str, dict[str, Any]]:
    """Return {dataset: {"mean@k", "pass@k", ...}} regardless of list/dict layout."""
    payload = json.loads(metrics_path.read_text())
    datasets = payload.get("datasets", payload.get("results", payload))
    out: dict[str, dict[str, Any]] = {}
    if isinstance(datasets, dict):
        for name, entry in datasets.items():
            if isinstance(entry, dict) and "mean@k" in entry:
                out[name] = entry
    elif isinstance(datasets, list):
        for entry in datasets:
            if isinstance(entry, dict) and "mean@k" in entry:
                out[str(entry.get("dataset") or entry.get("name"))] = entry
    return out


def _ood_accuracy(bench_dir: Path, keys: tuple[str, ...]) -> float | None:
    """Accuracy of the benchmark's *aggregate* result (not its first subtask).

    MMLU-Pro and MMLU-14k are lm-eval groups with 14 subtasks each; the harness results file lists the
    subtasks alongside the group, so the first key is e.g. ``mmlu_pro_biology``. The benchmark's
    ``complete.json`` records the aggregate key (``artifacts.result_key``) and the results file it refers to.
    """
    complete_path = bench_dir / "complete.json"
    result_file: Path | None = None
    result_key: str | None = None
    if complete_path.is_file():
        try:
            artifacts = json.loads(complete_path.read_text(encoding="utf-8")).get("artifacts", {})
            result_key = artifacts.get("result_key") or None
            if artifacts.get("result_path"):
                candidate = bench_dir / artifacts["result_path"]
                if candidate.is_file():
                    result_file = candidate
        except (OSError, ValueError):
            pass
    if result_file is None:
        files = sorted(bench_dir.glob("**/results_*.json"))
        if not files:
            return None
        result_file = files[-1]
    try:
        results = json.loads(result_file.read_text(encoding="utf-8")).get("results", {})
    except (OSError, ValueError):
        return None
    if not results:
        return None
    task = None
    if result_key and result_key in results:
        task = result_key
    else:
        # Prefer a group/aggregate entry: the key that is a prefix of the other keys, else a lone task.
        candidates = [t for t in results if not t.startswith("_")]
        groups = [t for t in candidates if sum(1 for u in candidates if u != t and u.startswith(t)) > 0]
        task = groups[0] if groups else (candidates[0] if len(candidates) == 1 else None)
        if task is None:
            return None
    for key in keys:
        value = results[task].get(key)
        if isinstance(value, (int, float)):
            return float(value) * 100.0
    return None


def collect(roots: list[Path], tags: list[str]) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {tag: {"math": {}, "ood": {}} for tag in tags}
    for root in roots:
        for tag in tags:
            metrics = root / tag / "math" / "metrics.json"
            if metrics.exists():
                found[tag]["math"].update(_math_results(metrics))
            for bench, _label, keys in OOD_COLUMNS:
                bench_dir = root / tag / "ood" / bench
                if bench_dir.exists() and (bench_dir / "complete.json").exists():
                    acc = _ood_accuracy(bench_dir, keys)
                    if acc is not None:
                        found[tag]["ood"][bench] = acc
    return found


def _cell(entry: dict[str, Any] | None, k: int, bold: bool) -> str:
    if not entry:
        return "—"
    text = f"{entry['mean@k']:.1f} / {entry['pass@k']:.1f}"
    return f"**{text}**" if bold else text


def render(models: list[dict[str, Any]], found: dict[str, dict[str, Any]]) -> str:
    lines = []
    math_done = sum(1 for m in models if all(d in found[m["tag"]]["math"] for d, _ in MATH_COLUMNS))
    ood_done = sum(1 for m in models if all(b in found[m["tag"]]["ood"] for b, _, _ in OOD_COLUMNS))
    lines.append(
        f"_Updated {datetime.now(timezone.utc):%Y-%m-%d %H:%M}Z — math complete for {math_done}/{len(models)} "
        f"models, OOD complete for {ood_done}/{len(models)}. Partial rows are shown as they finish._"
    )
    lines.append("")
    lines.append("**Math family** — `mean@k / pass@k` (%), repo `\\boxed{}` verifier (= RL reward). Bold = own band.")
    lines.append("")
    lines.append("| Model | Category | Trained on | id_easy (16) | id_medium (16) | id_hard (16) | MATH500 (16) | GSM8K (8) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for m in models:
        row = found[m["tag"]]["math"]
        cells = [_cell(row.get(d), k, bold=(d == f"id_{m.get('trained_on')}")) for d, k in MATH_COLUMNS]
        lines.append(f"| `{m['tag']}` | {m['category']} | {m.get('trained_on') or '—'} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("**Out-of-domain** — accuracy (%), lm-eval-harness 5-shot CoT (different scorer; not comparable to the math family).")
    lines.append("")
    lines.append("| Model | " + " | ".join(label for _, label, _ in OOD_COLUMNS) + " |")
    lines.append("|---|" + "---|" * len(OOD_COLUMNS))
    for m in models:
        row = found[m["tag"]]["ood"]
        cells = [f"{row[b]:.1f}" if b in row else "—" for b, _, _ in OOD_COLUMNS]
        lines.append(f"| `{m['tag']}` | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, action="append", default=None,
                        help="results root(s) laid out as <root>/<tag>/{math,ood}; repeatable")
    parser.add_argument("--results-base", type=Path, default=None,
                        help="directory whose immediate subdirectories are per-model results roots "
                             "(the queue layout; default /tmp/gemma4_distill_study_eval/results when no --results-root)")
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--doc", type=Path, default=DOC)
    parser.add_argument("--dry-run", action="store_true", help="print the rendered section instead of editing the doc")
    args = parser.parse_args()
    roots = list(args.results_root or [])
    base = args.results_base or (None if roots else Path("/tmp/gemma4_distill_study_eval/results"))
    if base is not None and base.is_dir():
        roots.extend(sorted(p for p in base.iterdir() if p.is_dir()))

    models = json.loads(args.registry.read_text())["models"]
    order = {"base": 0, "rl": 1, "distilled": 2}
    models.sort(key=lambda m: (order.get(m["category"], 9), m.get("architecture", ""), m.get("trained_on") or "", m["tag"]))
    section = render(models, collect(roots, [m["tag"] for m in models]))
    if args.dry_run:
        print(section)
        return 0
    text = args.doc.read_text()
    head, rest = text.split(START, 1)
    _old, tail = rest.split(END, 1)
    args.doc.write_text(f"{head}{START}\n{section}\n{END}{tail}")
    print(f"updated {args.doc} ({len(models)} models)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
