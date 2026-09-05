#!/usr/bin/env python3
"""Write a compact, committable summary of the OOD eval results (accuracies + provenance, no samples).

The queue leaves ~60 MB of lm-eval JSON per 22 models under the results base; this keeps only what §8
needs so the numbers can travel with the repo (``update_distill_study_results_doc.py --summary``) to a
machine that holds the math results.

    python rl-distill-scripts/summarize_gemma4_ood_results.py \
        --results-base /tmp/gemma4_distill_study_eval/results \
        --output rl-distill-scripts/config/gemma4_distill_study_ood_summary.json
"""

from __future__ import annotations

import argparse
import json
import socket
from datetime import datetime, timezone
from pathlib import Path

from update_distill_study_results_doc import OOD_COLUMNS, _ood_accuracy  # noqa: E402  (same directory)


def summarize(roots: list[Path]) -> dict:
    models: dict[str, dict] = {}
    for root in roots:
        tag = root.name
        ood: dict[str, dict] = {}
        for bench, _label, keys in OOD_COLUMNS:
            bench_dir = root / tag / "ood" / bench
            complete = bench_dir / "complete.json"
            if not complete.is_file():
                continue
            acc = _ood_accuracy(bench_dir, keys)
            if acc is None:
                continue
            c = json.loads(complete.read_text(encoding="utf-8"))
            art = c.get("artifacts", {})
            ood[bench] = {
                "accuracy": round(acc, 4),
                "task": art.get("result_key"),
                "effective_samples": art.get("effective_samples"),
                "result_sha256": art.get("result_sha256"),
                "model_identity_sha256": c.get("model_identity_sha256"),
                "completed_at_utc": c.get("completed_at_utc"),
                "protocol": c.get("protocol"),
            }
        if ood:
            entry: dict = {"ood": ood}
            rc = root / "RUN_COMPLETE.json"
            if rc.is_file():
                r = json.loads(rc.read_text(encoding="utf-8"))
                entry["run_complete"] = {"phases": r.get("phases"), "completed_at_utc": r.get("completed_at_utc")}
            models[tag] = entry
    return {
        "schema_version": 1,
        "protocol": "gemma4_distill_study_ood_summary_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "benchmarks": [bench for bench, _l, _k in OOD_COLUMNS],
        "models": dict(sorted(models.items())),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-base", type=Path, default=Path("/tmp/gemma4_distill_study_eval/results"))
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    roots = sorted(d for d in a.results_base.iterdir() if d.is_dir() and not d.name.startswith("_"))
    summary = summarize(roots)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.output.is_file():
        try:
            previous = json.loads(a.output.read_text(encoding="utf-8"))
        except ValueError:
            previous = {}
        if previous.get("models") == summary["models"] and previous.get("benchmarks") == summary["benchmarks"]:
            print(f"unchanged {a.output}: {len(summary['models'])} models with OOD results")
            return 0  # keep the previous generated_at/host so a no-op refresh leaves git clean
    a.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    n_full = sum(1 for m in summary["models"].values() if len(m["ood"]) == len(summary["benchmarks"]))
    print(f"wrote {a.output}: {len(summary['models'])} models with OOD results ({n_full} complete)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
