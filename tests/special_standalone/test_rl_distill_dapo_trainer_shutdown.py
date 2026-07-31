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

from unittest.mock import Mock

import pytest

from dapo import dapo_ray_trainer


def test_dapo_fit_drains_pending_uploads_after_success(monkeypatch):
    trainer = Mock()
    trainer._fit_impl.return_value = "done"
    trainer._hf_pusher = Mock()
    wait = Mock()
    monkeypatch.setattr(dapo_ray_trainer, "wait_for_hf_pusher", wait)

    assert dapo_ray_trainer.RayDAPOTrainer.fit(trainer) == "done"

    wait.assert_called_once_with(trainer._hf_pusher, timeout=1800)


def test_dapo_fit_drains_pending_uploads_after_training_failure(monkeypatch):
    trainer = Mock()
    trainer._fit_impl.side_effect = ValueError("training failed")
    trainer._hf_pusher = Mock()
    wait = Mock()
    monkeypatch.setattr(dapo_ray_trainer, "wait_for_hf_pusher", wait)

    with pytest.raises(ValueError, match="training failed"):
        dapo_ray_trainer.RayDAPOTrainer.fit(trainer)

    wait.assert_called_once_with(trainer._hf_pusher, timeout=1800)
