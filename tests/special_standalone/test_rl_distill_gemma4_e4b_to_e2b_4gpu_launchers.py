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
OVERLAY_LAUNCHER = REPOSITORY_ROOT / "rl-distill-scripts" / "prepare_gemma4_e4b_rl_to_e2b_topk_overlay_4gpu.sh"


def _write_executable(path: Path, contents: str) -> None:
    path.write_text(contents, encoding="utf-8")
    path.chmod(0o755)


def test_overlay_status_is_read_only_when_inputs_are_missing(tmp_path: Path) -> None:
    environment = dict(os.environ)
    environment.update(
        {
            "OVERLAY_ROOT": str(tmp_path / "missing-overlay"),
            "PYTHON_BIN": str(tmp_path / "missing-python"),
            "RESCORER_SCRIPT": str(tmp_path / "missing-rescorer.py"),
            "SOURCE_DATASET_INDEX": str(tmp_path / "missing-index.json"),
            "TEACHER_MODEL_PATH": str(tmp_path / "missing-teacher"),
        }
    )

    result = subprocess.run(
        ["bash", str(OVERLAY_LAUNCHER), "status"],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "dataset_index.json=missing" in result.stdout


def test_overlay_score_assigns_one_worker_to_each_local_gpu(tmp_path: Path) -> None:
    source_index = tmp_path / "source" / "dataset_index.json"
    teacher = tmp_path / "teacher"
    overlay = tmp_path / "overlay"
    source_index.parent.mkdir()
    teacher.mkdir()
    source_index.write_text(
        json.dumps({"dataset_index_sha256": "8b5712e0f5dea3388340a9bc91a6ceee40ff2ff990e66b7238090e240daeda6c"}) + "\n",
        encoding="utf-8",
    )
    fake_rescorer = tmp_path / "fake-rescorer.py"
    fake_rescorer.write_text("# test fixture\n", encoding="utf-8")
    fake_python = tmp_path / "fake-python"
    _write_executable(
        fake_python,
        "#!/usr/bin/env python3\n"
        "import json, os, pathlib, sys\n"
        "worker = sys.argv[sys.argv.index('--worker-id') + 1]\n"
        "record = pathlib.Path(os.environ['TEST_RECORD_DIR']) / f'{worker}.json'\n"
        "record.parent.mkdir(parents=True, exist_ok=True)\n"
        "record.write_text(json.dumps({'gpu': os.environ['CUDA_VISIBLE_DEVICES'], 'argv': sys.argv[1:]}))\n",
    )
    record_dir = tmp_path / "records"
    environment = dict(os.environ)
    environment.update(
        {
            "GEMMA4_E4B_RL_TO_E2B_RESCORE_AUTHORIZED": "YES",
            "OVERLAY_ROOT": str(overlay),
            "PYTHON_BIN": str(fake_python),
            "RESCORER_SCRIPT": str(fake_rescorer),
            "SOURCE_DATASET_INDEX": str(source_index),
            "TEACHER_MODEL_PATH": str(teacher),
            "TEST_RECORD_DIR": str(record_dir),
        }
    )

    result = subprocess.run(
        ["bash", str(OVERLAY_LAUNCHER), "score"],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    records = [json.loads((record_dir / f"{worker}.json").read_text(encoding="utf-8")) for worker in range(4)]
    assert [record["gpu"] for record in records] == ["0", "1", "2", "3"]
    for worker, record in enumerate(records):
        argv = record["argv"]
        assert argv[1] == "score"
        assert argv[argv.index("--worker-id") + 1] == str(worker)
        assert argv[argv.index("--num-workers") + 1] == "4"
        assert argv[argv.index("--lm-head-chunk-tokens") + 1] == "8192"
