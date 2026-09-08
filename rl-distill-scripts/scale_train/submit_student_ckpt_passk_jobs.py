#!/usr/bin/env python3
"""Poll the distilled-student Hub repos from a CPU box and submit one ScaleTrain eval job per new checkpoint.

For every ``step_NNNNNN/`` export of each ``--repo`` that has no result under ``--s3-root`` yet and no live job, submit
``run_gemma4_student_ckpt_passk_st.sh`` with ``STUDENT=<student>,STEP=<step>,BANDS=<band>`` on 2 GPUs (12B -> dp 2,
26B -> tp 2; priority high, borrowing off by default) through ``launch_st_with_code.sh`` (current HEAD as the code
tarball). Job names: ``g4e4b-pk-<band[:3]>-<student>-s<step>`` (+ ``-<user>``, 32-char platform limit). Submitted jobs
are tracked in ``--state-file``; a job that ends FAILED/CANCELLED without a result is resubmitted up to ``--max-attempts``.
Plots come from ``eval_student_checkpoints_passk.py --plot-from-s3`` (separate loop, same S3 root).

    python rl-distill-scripts/scale_train/submit_student_ckpt_passk_jobs.py --poll-minutes 10 \\
        --repo JWei05/Distill-gemma4-e4b-base-medium-to-12b-base --repo JWei05/Distill-gemma4-e4b-base-medium-to-26b-base
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from eval_student_checkpoints_passk import REPO_NAME, step_tag  # noqa: E402

DEFAULT_ST_PYTHON = os.path.expanduser("~/.local/pipx/venvs/scaletrain-cli/bin/python")
LIVE = {"QUEUED", "PENDING", "IN_PROGRESS", "RUNNING", "STARTING", "CREATED"}
STATUS_SNIPPET = """
import json, sys
from scaletrain_cli.commands.list.commands import ListJobsUseCase, FilesystemGlobalContextRepository, LiveTrainGateway, FilesystemConfigurationRepository
jobs = ListJobsUseCase(global_context_repository=FilesystemGlobalContextRepository(),
    train_gateway=LiveTrainGateway(FilesystemConfigurationRepository().read().get_train_service_host_url())).execute(user_id=None, limit=200, no_limit=None)
out = [{k: str(d.get(k)) for k in ("name", "status", "id", "created_at", "status_detail")} for d in (j.to_dict() for j in jobs)
       if str(d.get("name")).startswith(sys.argv[1])]
print(json.dumps(out))
"""


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def st_jobs(st_python: str, prefix: str) -> list[dict]:
    proc = subprocess.run([st_python, "-c", STATUS_SNIPPET, prefix], capture_output=True, text=True, timeout=180)
    if proc.returncode != 0:
        print(f"[{now()}] job listing failed: {proc.stderr.strip()[-300:]}", flush=True)
        return []
    return json.loads(proc.stdout)


def s3_result_exists(s3_root: str, tag: str) -> bool:
    proc = subprocess.run(["aws", "s3", "ls", f"{s3_root.rstrip('/')}/{tag}/metrics.json"], capture_output=True, text=True)
    return proc.returncode == 0 and "metrics.json" in proc.stdout


def job_name(band: str, student: str, step: str) -> str:
    return f"g4e4b-pk-{band[:3]}-{student}-s{int(step.split('_')[-1]):04d}"


def submit(args, band: str, student: str, step: str) -> bool:
    name = job_name(band, student, step)
    cmd = ["bash", str(HERE / "launch_st_with_code.sh"), "--gpus-per-instance", str(args.gpus_per_instance), "--priority", args.priority,
           "--active-deadline-hours", str(args.deadline_hours), "--run-file", "run_gemma4_student_ckpt_passk_st.sh",
           "--job-name", name, "--env-vars", f"STUDENT={student},STEP={step},BANDS={band}"]
    if args.allow_borrowing:
        cmd.append("--allow-borrowing")
    print(f"[{now()}] submit {name}: {' '.join(cmd[1:])}", flush=True)
    if args.dry_run:
        return True
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=HERE)
    tail = "\n".join(line for line in proc.stdout.splitlines() if "HF_TOKEN" not in line and "WANDB" not in line)[-800:]
    if proc.returncode != 0:
        print(f"[{now()}] submit FAILED rc={proc.returncode}\n{tail}\n{proc.stderr[-800:]}", flush=True)
        return False
    print(tail, flush=True)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", action="append", required=True)
    parser.add_argument("--poll-minutes", type=float, default=10.0, help="0 = single pass")
    parser.add_argument("--s3-root", default="s3://scale-ml/genai/rl-distill/gemma4-e4b-base-student-passk-v1")
    parser.add_argument("--state-file", type=Path, default=Path("/tmp/gemma4_e4b_val32/passk_jobs_state.json"))
    parser.add_argument("--gpus-per-instance", type=int, default=2)
    parser.add_argument("--priority", choices=["normal", "high"], default="high")
    parser.add_argument("--deadline-hours", type=int, default=12)
    parser.add_argument("--allow-borrowing", action="store_true", help="preemptible capacity (off: reserved queue, no restarts)")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--st-python", default=DEFAULT_ST_PYTHON, help="interpreter with the scaletrain_cli package (job status)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("AWS_PROFILE", "ml-worker")   # the instance role cannot read/write the bucket
    api = HfApi()
    state: dict = json.loads(args.state_file.read_text()) if args.state_file.exists() else {}
    args.state_file.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        args.state_file.write_text(json.dumps(state, indent=2, sort_keys=True))

    while True:
        jobs = {j["name"]: j for j in st_jobs(args.st_python, "g4e4b-pk-")}
        for repo in args.repo:
            m = REPO_NAME.match(repo)
            if not m:
                raise SystemExit(f"repo name not recognised: {repo}")
            band, student = m["band"], m["student"]
            try:
                steps = sorted(e.path for e in api.list_repo_tree(repo, revision="main") if e.path.startswith("step_"))
            except RepositoryNotFoundError:
                print(f"[{now()}] {repo}: not on the Hub yet", flush=True)
                continue
            summary = []
            for step in steps:
                tag = step_tag(repo, step)
                entry = state.setdefault(tag, {"repo": repo, "step": step, "attempts": 0, "jobs": []})
                if entry.get("done") or s3_result_exists(args.s3_root, tag):
                    entry["done"] = True
                    summary.append(f"{step}=done")
                    continue
                name = job_name(band, student, step)
                live = next((j for j in jobs.values() if j["name"].startswith(name + "-") and j["status"] in LIVE), None)
                if live:
                    summary.append(f"{step}={live['status'].lower()}")
                    if live["id"] not in entry["jobs"]:
                        entry["jobs"].append(live["id"])
                    continue
                if entry["attempts"] >= args.max_attempts:
                    summary.append(f"{step}=GAVE_UP({entry['attempts']})")
                    continue
                ended = [j for j in jobs.values() if j["name"].startswith(name + "-")]
                if ended:
                    print(f"[{now()}] {tag}: previous job(s) {[(j['id'], j['status']) for j in ended]} ended without a result; resubmitting", flush=True)
                entry["attempts"] += 1
                entry["last_submit"] = now()
                if submit(args, band, student, step):
                    summary.append(f"{step}=submitted#{entry['attempts']}")
                else:
                    summary.append(f"{step}=submit_error")
                save()
            print(f"[{now()}] {repo}: {' '.join(summary) if summary else 'no step exports yet'}", flush=True)
        save()
        if args.poll_minutes <= 0:
            return 0
        time.sleep(args.poll_minutes * 60)


if __name__ == "__main__":
    raise SystemExit(main())
