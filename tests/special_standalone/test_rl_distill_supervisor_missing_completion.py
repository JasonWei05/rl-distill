from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPOSITORY_ROOT / "rl-distill-scripts/scale_train/supervise_borrowing_job.py"
SPEC = importlib.util.spec_from_file_location("scale_train_job_supervisor", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
supervisor = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = supervisor
SPEC.loader.exec_module(supervisor)



def test_supervisor_stops_after_repeated_quick_completions_without_marker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    launches: list[str] = []
    statuses: dict[str, list[str]] = {}

    def launch(command, launch_log):
        job_id = f"job_{len(launches)}"
        launches.append(job_id)
        statuses[job_id] = ["IN_PROGRESS", "COMPLETED"]
        return job_id, None

    def job_status(job_id):
        queue = statuses[job_id]
        return queue.pop(0) if len(queue) > 1 else queue[0]

    monkeypatch.setattr(supervisor, "_launch", launch)
    monkeypatch.setattr(supervisor, "_job_status", job_status)
    monkeypatch.setattr(supervisor, "_durable_completion_status", lambda **kwargs: (False, "best_hf marker missing"))
    monkeypatch.setattr(supervisor, "_retire_job_before_relaunch", lambda job_id, retry_seconds: (True, "retired"))
    monkeypatch.setattr(supervisor.time, "sleep", lambda seconds: None)
    monitor_log = tmp_path / "monitor.log"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "supervise_borrowing_job.py",
            "--name", "demo",
            "--launch-log", str(tmp_path / "launch.log"),
            "--monitor-log", str(monitor_log),
            "--state-file", str(tmp_path / "state.json"),
            "--stop-file", str(tmp_path / "STOP"),
            "--completion-best-hf-s3-uri", "s3://bucket/run",
            "--max-relaunches", "200",
            "--",
            "echo", "launch",
        ],
    )

    supervisor.main()

    assert launches == ["job_0", "job_1", "job_2"]
    assert "3 consecutive quick missing completions" in monitor_log.read_text()
    assert '"STOPPED_MISSING_COMPLETION"' in (tmp_path / "state.json").read_text()
