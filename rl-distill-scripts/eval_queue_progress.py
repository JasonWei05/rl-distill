#!/usr/bin/env python3
"""Progress and ETA table for the distillation-study eval queue.

Reads, per model root under the results base (``<base>/<tag>/``): the math shard log (start time),
the trace files (``<tag>/math/traces/*.jsonl`` complete, ``*.jsonl.partial`` in progress; one line
per finished request, written per generation group), ``<tag>/math/metrics.json`` (math done),
``<tag>/ood/<bench>/complete.json`` (OOD benchmarks done) and ``RUN_COMPLETE.json``. The math ETA
is remaining requests / observed request rate, so it is rough early on and pessimistic when the
cheap datasets (GSM8K, easy band) run first.

    python rl-distill-scripts/eval_queue_progress.py [--results-base DIR] [--registry JSON] [--queue-log LOG]
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

REQUESTS = {"id_easy": 4800, "id_medium": 4800, "id_hard": 4800, "math500": 8000, "gsm8k": 10552}
TOTAL = sum(REQUESTS.values())
OOD = ("mmlu_pro", "gpqa", "mmmlu14k")
STAMP = re.compile(r"^\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})")


def _fmt(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    m = int(seconds // 60)
    return f"{m // 60}h{m % 60:02d}m" if m >= 60 else f"{m}m"


def _count_lines(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(buf.count(b"\n") for buf in iter(lambda: handle.read(1 << 20), b""))


def model_status(root: Path, tag: str, now: datetime) -> dict:
    status: dict = {"tag": tag, "phase": "init", "done": 0, "elapsed": None, "rate": None, "eta": None, "ood": 0, "datasets_done": 0}
    log = root / "logs" / "math" / f"{tag}__shard_00.log"
    start = None
    if log.exists():  # the runner stamps "[<utc>] CUDA_VISIBLE_DEVICES=..." at every (re)start; take the last one
        with log.open(errors="replace") as handle:
            for line in handle:
                m = STAMP.match(line)
                if m:
                    start = datetime.fromisoformat(m.group(1)).replace(tzinfo=timezone.utc)
    if start is None:
        start = datetime.fromtimestamp(root.stat().st_mtime, timezone.utc)
    status["elapsed"] = (now - start).total_seconds()
    traces = root / tag / "math" / "traces"
    if traces.exists():
        for path in traces.iterdir():
            if path.suffix == ".jsonl":
                status["datasets_done"] += 1
            if path.name.endswith(".jsonl") or path.name.endswith(".partial"):
                status["done"] += _count_lines(path)
    if (root / "RUN_COMPLETE.json").exists():
        status["phase"] = "complete"
    elif (root / tag / "math" / "metrics.json").exists():
        status["phase"] = "ood"
    elif status["done"] > 0:
        status["phase"] = "math"
    status["ood"] = sum((root / tag / "ood" / bench / "complete.json").exists() for bench in OOD)
    if status["phase"] == "math" and status["elapsed"] and status["elapsed"] > 0:
        status["rate"] = status["done"] / status["elapsed"]  # requests/s incl. startup -> conservative
        status["eta"] = (TOTAL - status["done"]) / status["rate"] if status["rate"] > 0 else None
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-base", type=Path, default=Path("/tmp/gemma4_distill_study_eval/results"))
    parser.add_argument("--registry", type=Path, default=Path(__file__).resolve().parent / "config/gemma4_distill_study_eval_sources.json")
    parser.add_argument("--queue-log", type=Path, default=Path("/tmp/gemma4_distill_study_eval/eval_queue.log"))
    args = parser.parse_args()
    now = datetime.now(timezone.utc)
    roster = [m["tag"] for m in json.loads(args.registry.read_text())["models"]] if args.registry.exists() else []
    launched = set(re.findall(r"EVAL_QUEUE launch model=(\S+)", args.queue_log.read_text())) if args.queue_log.exists() else set()
    rows = [model_status(args.results_base / tag, tag, now) for tag in sorted(launched) if (args.results_base / tag).exists()]
    print(f"{now:%Y-%m-%d %H:%M}Z  math requests per model = {TOTAL:,} (3 bands x 4800, MATH500 8000, GSM8K 10552)")
    print(f"{'model':<28}{'phase':<10}{'math done':>12}{'%':>6}{'sets':>6}{'req/min':>9}{'ETA math':>10}{'OOD':>6}{'elapsed':>9}")
    for s in rows:
        pct = 100 * s["done"] / TOTAL
        print(f"{s['tag']:<28}{s['phase']:<10}{s['done']:>12,}{pct:>6.1f}{s['datasets_done']:>4}/5{(s['rate'] or 0) * 60:>9.1f}{_fmt(s['eta']):>10}{s['ood']:>4}/3{_fmt(s['elapsed']):>9}")
    running = [s for s in rows if s["phase"] not in ("complete",)]
    complete = [s for s in rows if s["phase"] == "complete"]
    queued = [t for t in roster if t not in launched]
    etas = [s["eta"] for s in running if s["eta"]]
    print(f"\nrunning={len(running)} complete={len(complete)} queued={len(queued)}" + (f": {', '.join(queued)}" if queued else ""))
    if etas:
        print(f"math ETA of running models: min {_fmt(min(etas))}, median {_fmt(sorted(etas)[len(etas)//2])}, max {_fmt(max(etas))} (OOD phase adds ~1h each; queued models start as slots free)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
