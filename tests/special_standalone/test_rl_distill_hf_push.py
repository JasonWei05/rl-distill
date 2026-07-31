# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).parents[2] / "rl-distill-scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from hf_push import HFPusher, wait_for_hf_pusher  # noqa: E402


def test_wait_propagates_background_upload_failure(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    pusher = HFPusher(repo_id="test/repo", token="test-token", enable_hf_transfer=False, max_retries=1)

    monkeypatch.setattr(pusher, "_ensure_repo", lambda: None)
    monkeypatch.setattr(pusher, "_upload_with_retry", lambda *args, **kwargs: False)
    pusher.push_async(str(checkpoint), step=250)

    with pytest.raises(RuntimeError, match="failed 1 queued upload"):
        pusher.wait(timeout=5)


def test_push_async_rejects_missing_checkpoint(tmp_path):
    pusher = HFPusher(repo_id="test/repo", token="test-token", enable_hf_transfer=False)
    try:
        with pytest.raises(FileNotFoundError, match="checkpoint directory does not exist"):
            pusher.push_async(str(tmp_path / "missing"), step=250)
    finally:
        pusher.wait(timeout=5)


def test_wait_rejects_new_work_after_close(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    pusher = HFPusher(repo_id="test/repo", token="test-token", enable_hf_transfer=False)
    pusher.wait(timeout=5)

    with pytest.raises(RuntimeError, match="closed"):
        pusher.push_async(str(checkpoint), step=250)


def test_wait_helper_propagates_upload_failure_after_success():
    class FailedPusher:
        def wait(self, timeout=None):
            raise RuntimeError("upload failed")

    with pytest.raises(RuntimeError, match="upload failed"):
        wait_for_hf_pusher(FailedPusher(), timeout=5)


def test_wait_helper_preserves_primary_training_failure(capsys):
    class FailedPusher:
        def wait(self, timeout=None):
            raise RuntimeError("upload failed")

    def fail_training():
        try:
            raise ValueError("training failed")
        finally:
            wait_for_hf_pusher(FailedPusher(), timeout=5)

    with pytest.raises(ValueError, match="training failed") as exc_info:
        fail_training()

    assert "upload failed" in capsys.readouterr().err
    if hasattr(exc_info.value, "__notes__"):
        assert any("upload failed" in note for note in exc_info.value.__notes__)
