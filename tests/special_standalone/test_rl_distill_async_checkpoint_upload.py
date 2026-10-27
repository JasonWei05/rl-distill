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

import threading
from types import SimpleNamespace

import pytest

from dapo import dapo_ray_trainer


def _trainer(tmp_path, step):
    trainer = object.__new__(dapo_ray_trainer.RayDAPOTrainer)
    trainer.global_steps = step
    trainer.config = SimpleNamespace(
        trainer=SimpleNamespace(default_local_dir=str(tmp_path)),
        actor_rollout_ref=SimpleNamespace(rollout={"enable_sleep_mode": False}),
    )
    trainer._validation_early_stopping = None
    return trainer


@pytest.fixture
def helper(monkeypatch):
    events = []
    release = threading.Event()

    def upload_checkpoint(_root, step, _uri):
        events.append(("upload_start", step))
        assert release.wait(5)
        events.append(("upload_done", step))

    def fake_base_save(self, *, save_hf_model=None):
        events.append(("local_save", self.global_steps))

    helper = SimpleNamespace(
        events=events,
        release=release,
        upload_checkpoint=upload_checkpoint,
        retire_rolling_checkpoint=lambda *_args: events.append(("retire",)),
    )
    monkeypatch.setattr(dapo_ray_trainer.RayPPOTrainer, "_save_checkpoint", fake_base_save)
    monkeypatch.setattr(dapo_ray_trainer.RayDAPOTrainer, "_full_checkpoint_helper", staticmethod(lambda: helper))
    monkeypatch.setenv("FULL_CHECKPOINT_S3_URI", "s3://bucket/run")
    monkeypatch.setenv("ASYNC_CHECKPOINT_UPLOAD", "True")
    monkeypatch.delenv("ROLLING_CHECKPOINT_ENABLED", raising=False)
    return helper


def test_async_upload_returns_before_upload_and_next_save_joins_it(tmp_path, helper):
    trainer = _trainer(tmp_path, 10)
    trainer._save_checkpoint(permanent=True)
    assert trainer._pending_checkpoint_upload is not None
    assert not trainer._pending_checkpoint_upload.done()  # training continues while the upload is blocked

    trainer.global_steps = 20
    threading.Timer(0.2, helper.release.set).start()
    trainer._save_checkpoint(permanent=True)
    # step 10's upload committed before step 20's local save started; no rolling slot to retire
    assert helper.events[:4] == [("local_save", 10), ("upload_start", 10), ("upload_done", 10), ("local_save", 20)]
    trainer._wait_for_checkpoint_upload()
    assert ("retire",) not in helper.events
    assert helper.events[-1] == ("upload_done", 20)


def test_async_upload_failure_is_raised_at_the_next_join(tmp_path, helper):
    def failing_upload(*_args):
        raise RuntimeError("s3 down")

    helper.upload_checkpoint = failing_upload
    trainer = _trainer(tmp_path, 10)
    trainer._save_checkpoint(permanent=True)
    with pytest.raises(RuntimeError, match="s3 down"):
        trainer._save_checkpoint(permanent=True)


def test_run_outcome_waits_for_terminal_upload(tmp_path, helper, monkeypatch):
    trainer = _trainer(tmp_path, 10)
    trainer._save_checkpoint(permanent=True)
    joined = []
    monkeypatch.setattr(
        trainer, "_wait_for_checkpoint_upload", lambda: (helper.release.set(), joined.append(True), 1/0)
    )
    with pytest.raises(ZeroDivisionError):  # stop right after the join; the outcome body is covered elsewhere
        trainer._write_run_outcome(early_stopping=None, stop_reason="max_steps")
    assert joined == [True]


def test_sync_upload_is_the_default(tmp_path, helper, monkeypatch):
    monkeypatch.delenv("ASYNC_CHECKPOINT_UPLOAD")
    helper.release.set()
    trainer = _trainer(tmp_path, 10)
    trainer._save_checkpoint(permanent=True)
    assert getattr(trainer, "_pending_checkpoint_upload", None) is None
    assert helper.events == [("local_save", 10), ("upload_start", 10), ("upload_done", 10)]
