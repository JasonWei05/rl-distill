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

"""Forward KL distillation loss for verl's SFT trainer.

Loss per token: teacher_log_prob(x_t) - student_log_prob(x_t)
where x_t is the teacher's sampled token. Averaged over response tokens.

Gradients are identical to cross-entropy (teacher log prob is constant w.r.t.
student params). The expected sampled loss is the forward KL, but individual
token estimates can be negative because they use the teacher's sampled token
instead of the full teacher distribution.
"""

import torch
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.torch_functional import masked_sum


def forward_kl_loss(config, model_output, data: TensorDict, dp_group=None):
    pad_mode = tu.get_non_tensor_data(data=data, key="pad_mode", default=DatasetPadMode.NO_PADDING)
    dp_size = data["dp_size"]
    batch_num_tokens = data["batch_num_tokens"]

    student_log_prob = model_output["log_probs"]
    teacher_log_prob = data["teacher_log_probs"]

    if pad_mode == DatasetPadMode.NO_PADDING:
        student_lp = student_log_prob.values()
        teacher_lp = teacher_log_prob.values()
        loss_mask = data["loss_mask"].values()

        # Left-shift by one token to align with next-token log probs
        loss_mask = torch.roll(loss_mask, shifts=-1, dims=0)
        teacher_lp = torch.roll(teacher_lp, shifts=-1, dims=0)

        # Forward KL: teacher_lp - student_lp, masked to response tokens
        per_token_kl = (teacher_lp - student_lp) * loss_mask
        loss = masked_sum(per_token_kl, loss_mask) / batch_num_tokens * dp_size
    else:
        response_mask = data["response_mask"].to(bool)
        per_token_kl = (teacher_log_prob - student_log_prob) * response_mask
        loss = masked_sum(per_token_kl, response_mask) / batch_num_tokens * dp_size

    with torch.no_grad():
        masked_kl = (
            per_token_kl[loss_mask.bool()] if pad_mode == DatasetPadMode.NO_PADDING else per_token_kl[response_mask]
        )
        metrics = {
            "forward_kl/mean": masked_kl.mean().item(),
            "forward_kl/max": masked_kl.max().item(),
        }

    return loss, metrics
