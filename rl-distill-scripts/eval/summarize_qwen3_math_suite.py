#!/usr/bin/env python3
"""Write QWEN3_MATH_SUITE_RESULTS.md: every Qwen3 math-suite score (s3://.../qwen3-math-suite-v1/<model>/<set>/summary.json).

One table per student size, cells "mean@n / pass@n" (n = 16 / AIME 32), rows grouped base / direct RL best / off-policy
distilled (step 1000) / on-policy distilled (best val step <= 120), plus the checkpoint each row was evaluated from.

    AWS_PROFILE=ml-worker python rl-distill-scripts/eval/summarize_qwen3_math_suite.py
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

S3 = "s3://scale-ml/genai/rl-distill/qwen3-math-suite-v1"
OUT = Path(__file__).resolve().parents[1] / "docs" / "QWEN3_MATH_SUITE_RESULTS.md"
SETS = ["easy", "medium", "hard", "very_hard", "gsm8k", "math500", "aime2025", "aime2026"]
BANDS = [("medium", "m"), ("hard", "h"), ("very_hard", "vh")]


def rows_for(size: str) -> list[tuple[str, str, str]]:
    """(group, label, model name) rows for one student size."""
    tag = {"1.7B": "q1p7b", "4B": "q4b"}[size]
    rows = [("base", f"Qwen3-{size}-Base", f"{tag}-base")]
    rows += [("RL (best val ckpt)", f"RL {b}", f"{tag}-rl-{b}-best") for b, _ in BANDS]
    teachers = ["q4b", "q30b"] if size == "1.7B" else ["q30b"]
    for kind, suffix in (("off-policy distill (step 1000)", "s1000"), ("on-policy distill (best val <= 120)", "opd120")):
        for t in teachers:
            for b, short in BANDS:
                rows.append((kind, f"{t[1:].upper()} {b} -> {size}", f"{t}{short}-to-{tag}-{suffix}"))
    return rows


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["aws", "s3", "sync", "--only-show-errors", "--exclude", "*", "--include", "*/summary.json", S3, tmp], check=True)
        results = {(p.parent.parent.name, p.parent.name): json.loads(p.read_text()) for p in Path(tmp).glob("*/*/summary.json")}
    lines = [
        "# Qwen3 math suite results",
        "",
        f"_Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%MZ} by `rl-distill-scripts/eval/summarize_qwen3_math_suite.py` from `{S3}/<model>/<set>/summary.json`._",
        "",
        "Protocol: 0-shot native ChatML + \"Please reason step by step, and put your final answer within \\boxed{}.\", temp 1, top-p 1,",
        "8192 response tokens, strict last-box grading (= the RL reward). n = 16 samples per question (AIME 32); mean@n = average",
        "accuracy, pass@n = unbiased any-of-n. Sets: the Qwen3-30B-defined DeepScaleR easy / medium / hard / very_hard validation",
        "splits (300 q each), GSM8K test (1,319), MATH500 (500), AIME 2025 / 2026 (30 each). avg = plain mean over the 8 sets.",
        "Teachers are the RL best checkpoints (30B-A3B: medium @320, hard @410, very_hard @460; 4B: @510 / @560 / @620).",
        "Noise: ~0.4-0.6 pts SE on a 300-q set's mean@16; AIME pass@32 moves 3.3 pts per question.",
        "",
    ]
    for size in ("1.7B", "4B"):
        lines += [f"## {size} students — mean@n / pass@n", "", "| group | model | " + " | ".join(SETS) + " | avg |", "|---" * (len(SETS) + 3) + "|"]
        for group, label, name in rows_for(size):
            cells, means, passes = [], [], []
            for s in SETS:
                r = results.get((name, s))
                if r is None:
                    cells.append("—"); continue
                m, p = r["mean_at_n"], r["pass_at_k"][str(r["n"])]
                means.append(m); passes.append(p); cells.append(f"{m:.1f} / {p:.1f}")
            avg = f"{sum(means) / len(SETS):.1f} / {sum(passes) / len(SETS):.1f}" if len(means) == len(SETS) else "—"
            lines.append(f"| {group} | {label} | {' | '.join(cells)} | {avg} |")
        lines.append("")
    lines += ["## Checkpoints evaluated", "", "| model | source |", "|---|---|"]
    for size in ("1.7B", "4B"):
        for _, _, name in rows_for(size):
            r = results.get((name, "easy"))
            lines.append(f"| `{name}` | `{r['source'] if r else 'not evaluated'}` |")
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
