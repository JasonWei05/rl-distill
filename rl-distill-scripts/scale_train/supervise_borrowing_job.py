#!/usr/bin/env python3
"""Launch and relaunch one borrowing ScaleTrain job after preemption."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import IO

JOB_ID_RE = re.compile(r"Created job:\s*(job_[a-z0-9]+)")
IMAGE_RE = re.compile(r"^image:\s*(\S+)\s*$", re.MULTILINE)
CANCEL_SUCCESS_RE = re.compile(r"Result:\s*True", re.IGNORECASE)
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
SCALE_TRAIN_STATUS_RE = re.compile(
    r"\bstatus:\s*(QUEUED|IN_PROGRESS|COMPLETED|SUCCEEDED|FAILED|ERROR|"
    r"CANCELED|CANCELLED|PREEMPTED)\b",
    re.IGNORECASE,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_state(path: Path, state: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _log(path: Path, message: str) -> None:
    line = f"{_utc_now()} {message}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def _pseudo_tty_command(command: list[str]) -> list[str]:
    """Give ScaleTrain a controlling terminal while retaining captured output.

    The CLI can consult ``/dev/tty`` even when the devbox instance role is
    already sufficient. Supervisors capture output to avoid persisting runtime
    credentials, so wrap the child in util-linux ``script`` when available.
    """

    script = shutil.which("script")
    if script is None:
        return list(command)
    return [script, "--quiet", "--return", "--command", shlex.join(command), "/dev/null"]


def _launch(command: list[str], launch_log: Path) -> tuple[str, str | None]:
    """Launch without persisting the credential-bearing ScaleTrain rendering."""
    launch_log.parent.mkdir(parents=True, exist_ok=True)
    launch_log.touch(mode=0o600, exist_ok=True)
    launch_log.chmod(0o600)
    completed = subprocess.run(
        _pseudo_tty_command(command),
        text=True,
        capture_output=True,
        check=False,
    )
    output = f"{completed.stdout}\n{completed.stderr}"
    clean_output = ANSI_ESCAPE_RE.sub("", output).replace("\r", "")
    if completed.returncode != 0:
        with launch_log.open("a", encoding="utf-8") as stream:
            stream.write(f"{_utc_now()} launcher_returncode={completed.returncode}\n")
        raise RuntimeError(f"launcher exited with code {completed.returncode}")
    matches = JOB_ID_RE.findall(clean_output)
    if not matches:
        with launch_log.open("a", encoding="utf-8") as stream:
            stream.write(f"{_utc_now()} launcher_returncode=0 job_id=missing\n")
        raise RuntimeError("launcher succeeded but no ScaleTrain job ID was found")
    images = IMAGE_RE.findall(clean_output)
    job_id = matches[-1]
    image_uri = images[-1] if images else None
    with launch_log.open("a", encoding="utf-8") as stream:
        stream.write(
            f"{_utc_now()} launcher_returncode=0 job_id={job_id} "
            f"image={image_uri or 'existing-command-image'}\n"
        )
    return job_id, image_uri


def _with_image(command: list[str], image_uri: str | None) -> list[str]:
    if not image_uri or "--image" in command:
        return list(command)
    return [*command, "--image", image_uri]


def _command_image(command: list[str]) -> str | None:
    if "--image" not in command:
        return None
    index = command.index("--image") + 1
    if index >= len(command) or not command[index].strip():
        return None
    return command[index]


def _resolved_image_uri(command: list[str], launcher_image: str | None) -> str | None:
    """Prefer the exact submitted image over a rendered launcher summary.

    ScaleTrain's rich output can truncate a long ECR URI when it is captured
    through a narrow pseudo-terminal.  If the command already contains an
    explicit ``--image``, that argument is authoritative for both state and
    subsequent relaunches.
    """
    return _command_image(command) or launcher_image


def _read_completion_receipt(uri: str) -> dict[str, object] | None:
    return _read_s3_json_object(f"{uri.rstrip('/')}/run_complete.json")


def _read_best_hf_completion(uri: str) -> dict[str, object] | None:
    return _read_s3_json_object(f"{uri.rstrip('/')}/best_hf/_REMOTE_COMPLETE.json")


def _read_s3_json_object(object_uri: str) -> dict[str, object] | None:
    # `aws s3 cp` retries a missing object for tens of seconds on this devbox.
    # The exact-object listing returns a clean miss quickly, so only download
    # after the receipt is known to exist.
    try:
        exists = subprocess.run(
            ["aws", "s3", "ls", object_uri, "--region", "us-west-2"],
            text=True,
            capture_output=True,
            check=False,
            timeout=15,
        )
    except subprocess.TimeoutExpired:
        return None
    if exists.returncode != 0:
        return None
    try:
        completed = subprocess.run(
            [
                "aws",
                "s3",
                "cp",
                object_uri,
                "-",
                "--region",
                "us-west-2",
                "--no-progress",
            ],
            text=True,
            capture_output=True,
            check=False,
            timeout=45,
        )
    except subprocess.TimeoutExpired:
        return None
    if completed.returncode != 0:
        return None
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _completion_receipts_status(
    uris: list[str],
    *,
    expected_step: int | None,
    max_step: int | None,
    expected_world_size: int,
) -> tuple[bool, str]:
    if not uris:
        return False, "completion receipt checking is not configured"
    incomplete: list[str] = []
    for uri in uris:
        normalized = uri.rstrip("/")
        receipt = _read_completion_receipt(normalized)
        expected = {
            "protocol": "gemma4_rl_run_complete_v1",
            "status": "complete",
            "checkpoint_s3_uri": normalized,
            "checkpoint_world_size": expected_world_size,
        }
        if expected_step is not None:
            expected["checkpoint_step"] = expected_step
        if receipt is None:
            incomplete.append(f"{normalized}:missing")
            continue
        mismatches = [key for key, value in expected.items() if receipt.get(key) != value]
        if max_step is not None:
            checkpoint_step = receipt.get("checkpoint_step")
            if type(checkpoint_step) is not int or not 1 <= checkpoint_step <= max_step:
                mismatches.append("checkpoint_step")
        if mismatches:
            incomplete.append(f"{normalized}:mismatch({','.join(sorted(set(mismatches)))})")
    if incomplete:
        return False, ";".join(incomplete)
    return True, f"verified {len(uris)} durable run completion receipts"


def _best_hf_completions_status(uris: list[str]) -> tuple[bool, str]:
    if not uris:
        return False, "best-HF completion checking is not configured"
    incomplete: list[str] = []
    for uri in uris:
        normalized = uri.rstrip("/")
        manifest = _read_best_hf_completion(normalized)
        expected = {
            "protocol": "gemma4_rl_best_hf_v1",
            "status": "complete",
            "checkpoint_s3_uri": normalized,
        }
        if manifest is None:
            incomplete.append(f"{normalized}:missing")
            continue
        mismatches = [key for key, value in expected.items() if manifest.get(key) != value]
        if mismatches:
            incomplete.append(f"{normalized}:mismatch({','.join(sorted(mismatches))})")
    if incomplete:
        return False, ";".join(incomplete)
    return True, f"verified {len(uris)} durable best-HF completion markers"


def _durable_completion_status(
    *,
    receipt_uris: list[str],
    best_hf_uris: list[str],
    expected_step: int | None,
    max_step: int | None,
    expected_world_size: int | None,
) -> tuple[bool, str]:
    checks: list[tuple[bool, str]] = []
    if receipt_uris:
        assert expected_world_size is not None
        checks.append(
            _completion_receipts_status(
                receipt_uris,
                expected_step=expected_step,
                max_step=max_step,
                expected_world_size=expected_world_size,
            )
        )
    if best_hf_uris:
        checks.append(_best_hf_completions_status(best_hf_uris))
    if not checks:
        return False, "durable completion checking is not configured"
    incomplete = [detail for complete, detail in checks if not complete]
    if incomplete:
        return False, ";".join(incomplete)
    return True, ";".join(detail for _, detail in checks)


def _classify_pod_payload(payload: dict[str, object]) -> str:
    items = payload.get("items", [])
    if not isinstance(items, list):
        return "QUEUED"
    phases = [item.get("status", {}).get("phase") for item in items if isinstance(item, dict)]
    phases = [phase for phase in phases if phase]
    if not phases:
        return "QUEUED"
    if any(phase in {"Pending", "Running", "Unknown"} for phase in phases):
        return "IN_PROGRESS"
    if any(phase == "Succeeded" for phase in phases):
        return "COMPLETED"
    if any(phase == "Failed" for phase in phases):
        for item in items:
            if not isinstance(item, dict):
                continue
            status = item.get("status", {})
            if not isinstance(status, dict):
                continue
            if status.get("reason") in {"Evicted", "Preempted", "Shutdown"}:
                return "PREEMPTED"
            conditions = status.get("conditions", [])
            if isinstance(conditions, list) and any(
                isinstance(condition, dict)
                and condition.get("type") == "DisruptionTarget"
                and condition.get("status") == "True"
                for condition in conditions
            ):
                return "PREEMPTED"
            container_statuses = status.get("containerStatuses", [])
            if not isinstance(container_statuses, list):
                continue
            for container_status in container_statuses:
                if not isinstance(container_status, dict):
                    continue
                terminated = container_status.get("state", {}).get("terminated", {})
                if not isinstance(terminated, dict):
                    continue
                exit_code = terminated.get("exitCode")
                reason = terminated.get("reason")
                if exit_code in {137, 143} and reason != "OOMKilled":
                    return "PREEMPTED"
        return "FAILED"
    return "UNKNOWN"


def _pod_status(job_id: str) -> str | None:
    """Use the Kubernetes pod phase when ScaleTrain's CLI is unavailable.

    The shared devbox occasionally leaves ``scale-train get`` blocked in an
    uninterruptible EFS read.  Pod phases are enough to distinguish active,
    successful, and failed single-node training attempts without allowing a
    stuck CLI process to disable preemption recovery.
    """
    try:
        completed = subprocess.run(
            [
                "kubectl",
                "get",
                "pods",
                "-n",
                "train",
                "-l",
                f"scaletrain/job_id={job_id}",
                "-o",
                "json",
            ],
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return None
    if completed.returncode != 0:
        return None
    try:
        payload = json.loads(completed.stdout)
    except (AttributeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return _classify_pod_payload(payload)


def _scale_train_status(job_id: str) -> str | None:
    """Read the control-plane status without logging credential-bearing output."""
    try:
        completed = subprocess.run(
            _pseudo_tty_command(["scale-train", "get", "job", job_id]),
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return None
    if completed.returncode != 0:
        return None
    clean_output = ANSI_ESCAPE_RE.sub("", f"{completed.stdout}\n{completed.stderr}")
    matches = SCALE_TRAIN_STATUS_RE.findall(clean_output)
    if not matches:
        return None
    status = matches[-1].upper()
    return {
        "SUCCEEDED": "COMPLETED",
        "CANCELLED": "CANCELED",
    }.get(status, status)


def _job_status(job_id: str) -> str | None:
    pod_status = _pod_status(job_id)
    if pod_status not in {None, "QUEUED"}:
        return pod_status
    # An empty pod list usually means queued, but it also occurs after a
    # pre-admission cancellation or control-plane failure.  Consult the
    # ScaleTrain record only in this ambiguous state, with a bounded timeout.
    return _scale_train_status(job_id) or pod_status


def _pod_name(job_id: str) -> str | None:
    try:
        completed = subprocess.run(
            [
                "kubectl",
                "get",
                "pods",
                "-n",
                "train",
                "-l",
                f"scaletrain/job_id={job_id}",
                "-o",
                "json",
            ],
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return None
    if completed.returncode != 0:
        return None
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return None
    items = payload.get("items", []) if isinstance(payload, dict) else []
    names = sorted(
        item.get("metadata", {}).get("name")
        for item in items
        if isinstance(item, dict) and item.get("metadata", {}).get("name")
    )
    return names[0] if names else None


def _start_pod_log_capture(
    *, job_id: str, attempt: int, log_dir: Path
) -> tuple[subprocess.Popen[str], IO[str], str] | None:
    pod_name = _pod_name(job_id)
    if pod_name is None:
        return None
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"attempt-{attempt:03d}-{job_id}-{pod_name}.log"
    stream = log_path.open("a", encoding="utf-8")
    log_path.chmod(0o600)
    process = subprocess.Popen(
        ["kubectl", "logs", "-n", "train", pod_name, "--follow", "--timestamps"],
        text=True,
        stdout=stream,
        stderr=subprocess.STDOUT,
    )
    return process, stream, pod_name


def _stop_pod_log_capture(capture: tuple[subprocess.Popen[str], IO[str], str] | None) -> None:
    if capture is None:
        return
    process, stream, _ = capture
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
    stream.close()


def _cancel_job(job_id: str) -> tuple[bool, str]:
    """Cancel one exact ScaleTrain job without exposing CLI output or credentials."""
    try:
        completed = subprocess.run(
            ["scale-train", "cancel", "job", job_id],
            text=True,
            capture_output=True,
            check=False,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return False, "timeout"
    combined = f"{completed.stdout}\n{completed.stderr}"
    verified = completed.returncode == 0 and CANCEL_SUCCESS_RE.search(combined) is not None
    return verified, f"returncode={completed.returncode} result_verified={verified}"


def _retire_job_before_relaunch(
    job_id: str,
    *,
    retry_seconds: int,
    max_attempts: int = 3,
) -> tuple[bool, str]:
    """Ensure a terminal-looking job cannot later re-enter the queue.

    ScaleTrain can transiently report a preempted borrowing job as COMPLETED
    before moving it back to QUEUED. Launching a replacement during that window
    creates two jobs that can restore and write the same checkpoint prefix.
    Require a verified cancellation (or an already-CANCELED status) before a
    replacement is submitted.
    """

    details: list[str] = []
    for attempt in range(1, max_attempts + 1):
        canceled, cancel_detail = _cancel_job(job_id)
        if canceled:
            return True, f"cancellation verified ({cancel_detail})"
        status = _job_status(job_id)
        details.append(
            f"attempt={attempt} cancel={cancel_detail} status={status or 'unavailable'}"
        )
        if status == "CANCELED":
            return True, f"cancellation observed after unverified response ({cancel_detail})"
        if attempt < max_attempts:
            time.sleep(retry_seconds)
    return False, "; ".join(details)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--launch-log", type=Path, required=True)
    parser.add_argument("--monitor-log", type=Path, required=True)
    parser.add_argument("--state-file", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("--retry-seconds", type=int, default=5)
    parser.add_argument(
        "--completion-s3-uri",
        action="append",
        default=[],
        help="Logical-run checkpoint root whose durable run_complete.json is required; repeat per run.",
    )
    parser.add_argument(
        "--completion-best-hf-s3-uri",
        action="append",
        default=[],
        help="Logical-run artifact root whose durable best-HF marker is required; repeat per run.",
    )
    parser.add_argument(
        "--pod-log-dir",
        type=Path,
        help="Persist a live kubectl log stream for each attempted pod in this directory.",
    )
    parser.add_argument("--expected-completion-step", type=int)
    parser.add_argument(
        "--max-completion-step",
        type=int,
        help="Accept a positive terminal checkpoint step at or below this limit (for early stopping).",
    )
    parser.add_argument("--expected-completion-world-size", type=int)
    parser.add_argument(
        "--initial-job-id",
        help="Adopt an already-submitted job before launching any replacement attempt.",
    )
    parser.add_argument(
        "--relaunch-on-cancel",
        action="store_true",
        help="Treat an external CANCELED like a preemption: relaunch (the run-file resumes from the newest S3 checkpoint).",
    )
    parser.add_argument(
        "--relaunch-on-failure",
        action="store_true",
        help="Relaunch after FAILED/ERROR as well (with backoff) instead of stopping for diagnosis.",
    )
    parser.add_argument(
        "--max-relaunches",
        type=int,
        default=20,
        help="Stop after this many relaunches without a durable completion (guards against crash loops).",
    )
    parser.add_argument(
        "--failure-backoff-seconds",
        type=int,
        default=120,
        help="Wait this long times the number of consecutive failures (capped at 10x) before relaunching after FAILED/ERROR.",
    )
    parser.add_argument("launch_command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.launch_command)
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("a launch command is required after --")
    if args.expected_completion_step is not None and args.max_completion_step is not None:
        parser.error("use only one of --expected-completion-step or --max-completion-step")
    if args.completion_s3_uri and (
        (args.expected_completion_step is None and args.max_completion_step is None)
        or args.expected_completion_world_size is None
    ):
        parser.error(
            "--completion-s3-uri requires either --expected-completion-step or "
            "--max-completion-step, plus --expected-completion-world-size"
        )
    if args.max_completion_step is not None and args.max_completion_step < 1:
        parser.error("--max-completion-step must be positive")

    attempt = 0
    consecutive_failures = 0
    initial_job_id = args.initial_job_id
    image_uri = _command_image(command)
    while not args.stop_file.exists():
        if attempt - 1 >= args.max_relaunches and initial_job_id is None:
            _log(args.monitor_log, f"name={args.name} relaunch budget exhausted (max_relaunches={args.max_relaunches}); supervisor stopping")
            return
        completion_configured = bool(args.completion_s3_uri or args.completion_best_hf_s3_uri)
        if completion_configured:
            complete, detail = _durable_completion_status(
                receipt_uris=args.completion_s3_uri,
                best_hf_uris=args.completion_best_hf_s3_uri,
                expected_step=args.expected_completion_step,
                max_step=args.max_completion_step,
                expected_world_size=args.expected_completion_world_size,
            )
            if complete:
                _log(args.monitor_log, f"name={args.name} status=COMPLETED_RECEIPTS detail={detail}")
                return
        attempt += 1
        if initial_job_id is not None:
            job_id = initial_job_id
            initial_job_id = None
            _log(args.monitor_log, f"name={args.name} attempt={attempt} job_id={job_id} adopted")
        else:
            try:
                launch_command = _with_image(command, image_uri)
                job_id, launched_image = _launch(launch_command, args.launch_log)
                image_uri = _resolved_image_uri(launch_command, launched_image)
            except Exception as error:
                _log(args.monitor_log, f"name={args.name} launch_attempt={attempt} error={error}")
                time.sleep(args.retry_seconds)
                continue

        state: dict[str, object] = {
            "name": args.name,
            "attempt": attempt,
            "job_id": job_id,
            "status": "SUBMITTED",
            "image": image_uri,
            "updated_at": _utc_now(),
        }
        _write_state(args.state_file, state)
        _log(args.monitor_log, f"name={args.name} attempt={attempt} job_id={job_id} submitted")

        last_status = None
        missing_status_polls = 0
        pod_log_capture: tuple[subprocess.Popen[str], IO[str], str] | None = None
        while not args.stop_file.exists():
            completion_detail = "completion receipt checking is not configured"
            if pod_log_capture is not None and pod_log_capture[0].poll() is not None:
                _stop_pod_log_capture(pod_log_capture)
                pod_log_capture = None
            if args.pod_log_dir is not None and pod_log_capture is None:
                pod_log_capture = _start_pod_log_capture(
                    job_id=job_id,
                    attempt=attempt,
                    log_dir=args.pod_log_dir,
                )
                if pod_log_capture is not None:
                    _log(
                        args.monitor_log,
                        f"name={args.name} job_id={job_id} pod={pod_log_capture[2]} "
                        f"log_capture=started",
                    )
            if completion_configured:
                complete, completion_detail = _durable_completion_status(
                    receipt_uris=args.completion_s3_uri,
                    best_hf_uris=args.completion_best_hf_s3_uri,
                    expected_step=args.expected_completion_step,
                    max_step=args.max_completion_step,
                    expected_world_size=args.expected_completion_world_size,
                )
                if complete:
                    _log(
                        args.monitor_log,
                        f"name={args.name} job_id={job_id} status=COMPLETED_RECEIPTS "
                        f"detail={completion_detail}",
                    )
                    state.update(status="COMPLETED_RECEIPTS", updated_at=_utc_now())
                    _write_state(args.state_file, state)
                    _stop_pod_log_capture(pod_log_capture)
                    return
            status = _job_status(job_id)
            if status is None:
                missing_status_polls += 1
                if missing_status_polls in {1, 5, 20}:
                    _log(
                        args.monitor_log,
                        f"name={args.name} job_id={job_id} status_unavailable polls={missing_status_polls}",
                    )
                time.sleep(args.poll_seconds)
                continue
            missing_status_polls = 0
            if status != last_status:
                if status == "IN_PROGRESS" and last_status is not None:
                    consecutive_failures = 0
                _log(args.monitor_log, f"name={args.name} job_id={job_id} status={status}")
                state.update(status=status, updated_at=_utc_now())
                _write_state(args.state_file, state)
                last_status = status
            if status == "COMPLETED":
                if completion_configured:
                    retired, retirement_detail = _retire_job_before_relaunch(
                        job_id,
                        retry_seconds=args.retry_seconds,
                    )
                    if not retired:
                        _log(
                            args.monitor_log,
                            f"name={args.name} job_id={job_id} terminal=COMPLETED but durable "
                            f"completion is incomplete: {completion_detail}; old job retirement "
                            f"is unverified: {retirement_detail}; continuing to observe the "
                            "existing job without launching a duplicate",
                        )
                        state.update(
                            status="WAITING_FOR_OLD_JOB_RETIREMENT",
                            updated_at=_utc_now(),
                        )
                        _write_state(args.state_file, state)
                        time.sleep(args.poll_seconds)
                        continue
                    _log(
                        args.monitor_log,
                        f"name={args.name} job_id={job_id} terminal=COMPLETED but durable "
                        f"completion is incomplete: {completion_detail}; {retirement_detail}; "
                        "relaunching from latest complete checkpoints",
                    )
                    state.update(status="RELAUNCHING_MISSING_COMPLETION", updated_at=_utc_now())
                    _write_state(args.state_file, state)
                    _stop_pod_log_capture(pod_log_capture)
                    time.sleep(args.retry_seconds)
                    break
                _stop_pod_log_capture(pod_log_capture)
                return
            if status == "CANCELED":
                if args.relaunch_on_cancel:
                    _log(
                        args.monitor_log,
                        f"name={args.name} job_id={job_id} terminal=CANCELED (external); "
                        "relaunching from latest complete checkpoint (--relaunch-on-cancel)",
                    )
                    _stop_pod_log_capture(pod_log_capture)
                    time.sleep(args.retry_seconds)
                    break
                _log(args.monitor_log, f"name={args.name} job_id={job_id} canceled; supervisor stopping")
                _stop_pod_log_capture(pod_log_capture)
                return
            if status == "PREEMPTED":
                consecutive_failures = 0
                _log(
                    args.monitor_log,
                    f"name={args.name} job_id={job_id} terminal={status}; "
                    "relaunching from latest complete checkpoint",
                )
                _stop_pod_log_capture(pod_log_capture)
                time.sleep(args.retry_seconds)
                break
            if status in {"FAILED", "ERROR"}:
                if args.relaunch_on_failure:
                    consecutive_failures += 1
                    backoff = args.failure_backoff_seconds * min(consecutive_failures, 10)
                    _log(
                        args.monitor_log,
                        f"name={args.name} job_id={job_id} terminal={status}; consecutive_failures={consecutive_failures}; "
                        f"relaunching from latest complete checkpoint after {backoff}s (--relaunch-on-failure)",
                    )
                    _stop_pod_log_capture(pod_log_capture)
                    time.sleep(backoff)
                    break
                _log(
                    args.monitor_log,
                    f"name={args.name} job_id={job_id} terminal={status}; "
                    "non-preemptive failure requires diagnosis, supervisor stopping",
                )
                _stop_pod_log_capture(pod_log_capture)
                return
            time.sleep(args.poll_seconds)
        _stop_pod_log_capture(pod_log_capture)


if __name__ == "__main__":
    main()
