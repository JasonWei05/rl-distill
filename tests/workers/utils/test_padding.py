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

from verl.workers.utils.padding import make_padded_attention_mask


def test_make_padded_attention_mask_uses_full_sequence_lengths():
    max_response_length = 8192
    prompt_length = 1569
    sequence_lengths = torch.tensor([2048, max_response_length, max_response_length + prompt_length])

    mask = make_padded_attention_mask(sequence_lengths, int(sequence_lengths.max().item()))

    assert mask.dtype == torch.int32
    assert mask.sum(dim=-1).tolist() == sequence_lengths.tolist()
    # Regression for the bad Gemma-4 sample: its response was at the 8192-token
    # cap, but the model input also contained a 1569-token prompt. Building this
    # mask from response-width loss_mask hid the final 1569 response positions.
    assert mask[2, max_response_length:].all()


def test_make_padded_attention_mask_validates_inputs():
    with pytest.raises(ValueError, match="must be 1-D"):
        make_padded_attention_mask(torch.ones(2, 2), 2)

    with pytest.raises(ValueError, match="non-negative"):
        make_padded_attention_mask(torch.ones(2), -1)
