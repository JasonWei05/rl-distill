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


class _FakeHubApi:
    """Minimal stand-in for HfApi covering what HFPusher._prune touches."""

    def __init__(self, step_folders):
        self.step_folders = set(step_folders)
        self.deleted = []
        self.squashes = 0

    def list_repo_tree(self, repo_id, repo_type):
        class _Entry:
            def __init__(self, path):
                self.path = path

        return [_Entry(f"step_{s:06d}") for s in sorted(self.step_folders)] + [_Entry("README.md")]

    def delete_folder(self, path_in_repo, repo_id, repo_type, commit_message):
        self.step_folders.discard(int(path_in_repo.removeprefix("step_")))
        self.deleted.append(path_in_repo)

    def super_squash_history(self, repo_id, repo_type):
        self.squashes += 1


def _pusher_with_fake_hub(step_folders, max_to_keep):
    pusher = HFPusher(repo_id="test/repo", token="test-token", enable_hf_transfer=False, max_to_keep=max_to_keep)
    pusher._api = _FakeHubApi(step_folders)
    return pusher


def test_prune_lists_hub_keeps_newest_and_squashes(monkeypatch):
    monkeypatch.delenv("HF_PUSH_SQUASH_AFTER_PRUNE", raising=False)
    pusher = _pusher_with_fake_hub([50, 60, 70, 80, 90], max_to_keep=3)
    pusher._pushed_steps = [90]  # only this process's push is tracked; older folders came from an earlier run
    pusher._prune()
    assert pusher._api.deleted == ["step_000050", "step_000060"]
    assert pusher._remote_step_folders() == [70, 80, 90]
    assert pusher._api.squashes == 1
    pusher.wait(timeout=5)


def test_prune_squash_can_be_disabled(monkeypatch):
    monkeypatch.setenv("HF_PUSH_SQUASH_AFTER_PRUNE", "0")
    pusher = _pusher_with_fake_hub([10, 20, 30], max_to_keep=2)
    pusher._prune()
    assert pusher._api.deleted == ["step_000010"]
    assert pusher._api.squashes == 0
    pusher.wait(timeout=5)


def test_prune_is_noop_when_within_limit_or_unbounded():
    bounded = _pusher_with_fake_hub([10, 20], max_to_keep=2)
    bounded._prune()
    unbounded = _pusher_with_fake_hub([10, 20, 30, 40], max_to_keep=None)
    unbounded._prune()
    assert bounded._api.deleted == [] and unbounded._api.deleted == []
    bounded.wait(timeout=5)
    unbounded.wait(timeout=5)


def test_make_room_before_upload_prunes_to_one_below_limit(monkeypatch):
    monkeypatch.setenv("HF_PUSH_SQUASH_AFTER_PRUNE", "0")
    pusher = _pusher_with_fake_hub([70, 80, 90], max_to_keep=3)
    pusher._make_room_before_upload()
    assert pusher._api.deleted == ["step_000070"]
    single = _pusher_with_fake_hub([70, 80, 90], max_to_keep=1)
    single._make_room_before_upload()  # keep=1 must never delete everything before the upload
    assert single._api.deleted == []
    pusher.wait(timeout=5)
    single.wait(timeout=5)
