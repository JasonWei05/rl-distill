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

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPOSITORY_ROOT / "rl-distill-scripts" / "gemma4_topk_distill_fsdp2.sh"
TEACHER_IDENTITY = "a" * 64
STUDENT_IDENTITY = "b" * 64


def _write_executable(path: Path, contents: str) -> None:
    path.write_text(contents, encoding="utf-8")
    path.chmod(0o755)


def _base_environment(tmp_path: Path) -> tuple[dict[str, str], Path]:
    student = tmp_path / "student"
    student.mkdir()
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(fake_bin / "torchrun", "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\"\n")
    environment = dict(os.environ)
    environment.update(
        {
            "ALLOW_QUESTION_OVERLAP": "false",
            "DATASET_INDEX": str(tmp_path / "dataset-index.json"),
            "DISTILL_DIRECTION": "e4b_rl100_to_e2b",
            "EXPECTED_STUDENT_IDENTITY_SHA256": STUDENT_IDENTITY,
            "EXPECTED_TEACHER_IDENTITY_SHA256": TEACHER_IDENTITY,
            "HF_PUSH_ENABLE": "false",
            "LOAD_DOTENV": "false",
            "MODEL_PATH": str(student),
            "NPROC_PER_NODE": "1",
            "PATH": f"{fake_bin}:{environment['PATH']}",
            "PREFLIGHT_LOCAL_FILES_ONLY": "true",
            "TRAIN_LOGGER": '["console"]',
            "VENV": str(tmp_path / "missing-venv"),
        }
    )
    return environment, fake_bin


def _preflight_output(train_files: list[str], validation_files: list[str], **overrides: str) -> str:
    values = {
        "TRAIN_FILES_HYDRA": json.dumps(train_files, separators=(",", ":")),
        "VAL_FILES_HYDRA": json.dumps(validation_files, separators=(",", ":")),
        "TOPK_WIDTH": "128",
        "TOPK_VALIDATION_TOLERANCE": "0.0025",
        "DATASET_INDEX_SHA256": "c" * 64,
        "GENERATION_EXPERIMENT_SHA256": "d" * 64,
        "DIRECTION": "e4b_rl100_to_e2b",
        "TEACHER_IDENTITY_SHA256": TEACHER_IDENTITY,
        "STUDENT_IDENTITY_SHA256": STUDENT_IDENTITY,
        "STUDENT_TOKENIZER_SHA256": "e" * 64,
    }
    values.update(overrides)
    return "".join(f"{key}={value}\n" for key, value in values.items())


def _run_launcher(environment: dict[str, str], *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(LAUNCHER), *arguments],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_launcher_uses_preflight_hydra_lists_without_eval(tmp_path: Path) -> None:
    environment, fake_bin = _base_environment(tmp_path)
    sentinel = tmp_path / "must-not-be-created"
    train_files = [f"$(touch {sentinel})", "/tmp/train with spaces.parquet"]
    validation_files = ["/tmp/validation.parquet"]
    fake_python = fake_bin / "preflight-python"
    _write_executable(
        fake_python,
        "#!/usr/bin/env python3\n" + "print(" + repr(_preflight_output(train_files, validation_files)) + ", end='')\n",
    )
    environment["PYTHON_BIN"] = str(fake_python)

    result = _run_launcher(environment)

    assert result.returncode == 0, result.stderr
    assert not sentinel.exists()
    argv = result.stdout.splitlines()
    assert f"data.train_files={json.dumps(train_files, separators=(',', ':'))}" in argv
    assert f"data.val_files={json.dumps(validation_files, separators=(',', ':'))}" in argv
    assert "teacher_model.top_k=128" in argv
    assert "data.teacher_topk_validation_tolerance=0.0025" in argv


def test_launcher_rejects_identity_changed_after_preflight(tmp_path: Path) -> None:
    environment, fake_bin = _base_environment(tmp_path)
    fake_python = fake_bin / "preflight-python"
    _write_executable(
        fake_python,
        "#!/usr/bin/env python3\n"
        + "print("
        + repr(
            _preflight_output(
                ["/tmp/train.parquet"],
                ["/tmp/validation.parquet"],
                STUDENT_IDENTITY_SHA256="f" * 64,
            )
        )
        + ", end='')\n",
    )
    environment["PYTHON_BIN"] = str(fake_python)

    result = _run_launcher(environment)

    assert result.returncode == 2
    assert "unexpected student identity" in result.stderr


def test_direct_file_bypass_is_limited_to_non_uploading_smoke(tmp_path: Path) -> None:
    environment, _ = _base_environment(tmp_path)
    environment.pop("DATASET_INDEX")
    environment.update(
        {
            "SMOKE_ONLY_ALLOW_DIRECT_FILES": "true",
            "TRAIN_FILE": "/tmp/train.parquet",
            "VAL_FILE": "/tmp/validation.parquet",
            "TOTAL_TRAINING_STEPS": "3",
        }
    )

    result = _run_launcher(environment)

    assert result.returncode == 2
    assert "between 1 and 2" in result.stderr


def test_launcher_rejects_post_preflight_hydra_overrides(tmp_path: Path) -> None:
    environment, fake_bin = _base_environment(tmp_path)
    marker = tmp_path / "torchrun-was-called"
    _write_executable(fake_bin / "torchrun", f"#!/usr/bin/env bash\ntouch {marker}\n")

    result = _run_launcher(environment, "data.train_files=/tmp/unvalidated.parquet")

    assert result.returncode == 2
    assert "Hydra overrides are disabled" in result.stderr
    assert not marker.exists()
