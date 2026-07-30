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

import pytest
import torch
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.workers.config import ActorConfig
from verl.workers.utils.losses import ppo_loss


def _nested(tensors: list[torch.Tensor]) -> torch.Tensor:
    return torch.nested.as_nested_tensor(tensors, layout=torch.jagged)


def test_ppo_loss_reports_token_weighted_ratio_health_metrics():
    ratios = [torch.tensor([0.5, 1.0, 2.0]), torch.tensor([0.9, 1.1])]
    log_probs = _nested([torch.cat([ratio.log(), torch.zeros(1)]) for ratio in ratios])
    input_ids = _nested([torch.zeros(4, dtype=torch.long), torch.zeros(3, dtype=torch.long)])
    prompts = _nested([torch.zeros(1, dtype=torch.long), torch.zeros(1, dtype=torch.long)])
    responses = _nested([torch.zeros(3, dtype=torch.long), torch.zeros(2, dtype=torch.long)])

    data = TensorDict(
        {
            "input_ids": input_ids,
            "prompts": prompts,
            "responses": responses,
            "attention_mask": input_ids,
            "response_mask": torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool),
            "old_log_probs": torch.zeros(2, 3),
            "advantages": torch.ones(2, 3),
        },
        batch_size=[2],
    )
    tu.assign_non_tensor(
        data,
        dp_size=1,
        batch_num_tokens=5,
        global_batch_size=2,
        max_response_len=3,
    )

    config = ActorConfig(strategy="fsdp2", rollout_n=1, ppo_micro_batch_size_per_gpu=1)
    _, metrics = ppo_loss(config, {"log_probs": log_probs}, data)

    assert metrics["actor/probs_ratio"].aggregate() == pytest.approx(1.1)
    assert metrics["actor/probs_ratio_clamped"].aggregate() == pytest.approx(1.0)
    assert metrics["actor/probs_ratio_min"].aggregate() == pytest.approx(0.5)
    assert metrics["actor/probs_ratio_max"].aggregate() == pytest.approx(2.0)
    assert metrics["actor/probs_ratio_clamped_min"].aggregate() == pytest.approx(0.8)
    assert metrics["actor/probs_ratio_clamped_max"].aggregate() == pytest.approx(1.2)
