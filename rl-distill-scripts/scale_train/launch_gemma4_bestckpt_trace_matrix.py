#!/usr/bin/env python3
"""Resolve or launch off-policy trace collectors for the completed difficulty-band
RL checkpoints (e4b/12b/26b easy/medium/hard that finished training).

One ScaleTrain job per completed checkpoint: 8 responses per training question +
1 per validation question, training sampling, RL few-shot prompt, top-128 teacher
logprobs + token ids. e4b runs on 1 GPU; 12b/26b on 2 GPUs (DP-2, TP-1).
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TraceJob:
    key: str
    job_name: str
    trace_spec: str
    gpus: int
    run_file: str = "run_gemma4_bestckpt_trace_collection.sh"


# All twelve (model, band) best RL checkpoints. e4b/12b/26b-easy come from S3; e2b and
# 26b-medium/hard from their Hub exports. Best step is baked into the run-file per spec.
#
# "queue" packs all twelve collections into ONE 8-GPU job with an async GPU-pool
# scheduler (2-GPU runs first, then 1-GPU, refilling as each finishes). The per-spec
# jobs remain available for launching a single collection standalone.
TRACE_JOBS = (
    TraceJob("queue", "g4tr-queue", "", 8, run_file="run_gemma4_bestckpt_trace_queue.sh"),
    TraceJob("e4b-easy", "g4tr-e4b-easy", "e4b-easy", 1),
    TraceJob("e4b-medium", "g4tr-e4b-med", "e4b-medium", 1),
    TraceJob("e4b-hard", "g4tr-e4b-hard", "e4b-hard", 1),
    TraceJob("12b-easy", "g4tr-12b-easy", "12b-easy", 2),
    TraceJob("12b-medium", "g4tr-12b-med", "12b-medium", 2),
    TraceJob("12b-hard", "g4tr-12b-hard", "12b-hard", 2),
    TraceJob("26b-easy", "g4tr-26b-easy", "26b-easy", 2),
    # HF-sourced teachers (runs continued off-cluster / rerun locally; see the collection script).
    TraceJob("26b-medium", "g4tr-26b-med", "26b-medium", 2),
    TraceJob("26b-hard", "g4tr-26b-hard", "26b-hard", 2),
    TraceJob("e2b-easy", "g4tr-e2b-easy", "e2b-easy", 1),
    TraceJob("e2b-medium", "g4tr-e2b-med", "e2b-medium", 1),
    TraceJob("e2b-hard", "g4tr-e2b-hard", "e2b-hard", 1),
)


def launch_command(
    job: TraceJob,
    *,
    dry_run: bool,
    allow_borrowing: bool,
    priority: str = "high",
    image: str | None = None,
    build_env: str = "remote",
) -> list[str]:
    launcher = Path(__file__).with_name("launch_st_job.py")
    command = [
        sys.executable,
        str(launcher),
        "--cluster",
        "eks",
        "--build-env",
        build_env,
        "--n-instances",
        "1",
        "--gpus-per-instance",
        str(job.gpus),
        "--priority",
        priority,
        "--team",
        "egp",
        "--product",
        "train.enterprise_rlvr",
        "--build-config-key",
        "train-rl-distill-gemma4-trace",
        "--active-deadline-hours",
        "240",
        "--job-name",
        job.job_name,
        "--run-file",
        job.run_file,
    ]
    if job.trace_spec:
        command += ["--env-vars", f"TRACE_SPEC={job.trace_spec}"]
    if allow_borrowing:
        command.append("--allow-borrowing")
    if image:
        command.extend(["--image", image])
    if dry_run:
        command.append("--dry-run")
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", action="store_true", help="Submit; without it, only resolved dry-run commands are printed.")
    parser.add_argument("--allow-borrowing", action="store_true")
    parser.add_argument("--priority", choices=("normal", "high"), default="high")
    parser.add_argument("--build-env", choices=("local", "remote"), default="remote")
    parser.add_argument("--image", default=None, help="Reuse an existing remote image for every selected job (skips build).")
    parser.add_argument("--select", action="append", default=[], help="Job key(s); default is all twelve.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    available = {job.key for job in TRACE_JOBS}
    # Default (no --select) is the twelve per-spec jobs; the packed "queue" is opt-in
    # (and is mutually exclusive with the per-spec jobs — do not run both).
    selected = set(args.select) if args.select else (available - {"queue"})
    unknown = selected.difference(available)
    if unknown:
        raise SystemExit(f"unknown --select values: {sorted(unknown)}; available: {sorted(available)}")
    jobs = [job for job in TRACE_JOBS if job.key in selected]
    print(
        f"Resolved {len(jobs)} trace jobs on EKS: priority={args.priority}, "
        f"borrowing={args.allow_borrowing}, build_env={args.build_env}, launch={args.launch}."
    )
    for job in jobs:
        command = launch_command(
            job,
            dry_run=not args.launch,
            allow_borrowing=args.allow_borrowing,
            priority=args.priority,
            image=args.image,
            build_env=args.build_env,
        )
        print(f"\n[{job.key}] gpus={job.gpus} spec={job.trace_spec}")
        print(shlex.join(command))
        subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
