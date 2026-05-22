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

"""SFT dataset that also loads per-token teacher log probs for forward KL distillation.

Expects parquet with:
  - 'messages': list of {role, content} dicts (standard SFT format)
  - 'teacher_log_probs': list of floats (one per response token from teacher generation)

Returns everything MultiTurnSFTDataset returns, plus:
  - 'teacher_log_probs': tensor aligned to the full sequence (0 at prompt positions,
    teacher log prob values at assistant/response positions matching loss_mask=1)
"""

import torch

from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset


class DistillSFTDataset(MultiTurnSFTDataset):
    def _read_files_and_process(self):
        super()._read_files_and_process()
        self.teacher_log_probs_col = self.dataframe["teacher_log_probs"].tolist()

    def __getitem__(self, item):
        res = super().__getitem__(item)

        teacher_lps_raw = self.teacher_log_probs_col[item]
        if hasattr(teacher_lps_raw, "tolist"):
            teacher_lps_raw = teacher_lps_raw.tolist()

        loss_mask = res["loss_mask"]
        teacher_lp_tensor = torch.zeros_like(loss_mask, dtype=torch.float32)

        response_positions = (loss_mask == 1).nonzero(as_tuple=True)[0]
        n_response_tokens = len(response_positions)
        n_teacher_tokens = len(teacher_lps_raw)

        # Align: truncate or warn if lengths differ (tokenizer mismatch)
        n = min(n_response_tokens, n_teacher_tokens)
        if n_teacher_tokens != n_response_tokens and item < 3:
            print(
                f"[DistillSFTDataset] sample {item}: teacher has {n_teacher_tokens} tokens, "
                f"response has {n_response_tokens} tokens (using first {n})"
            )

        teacher_lp_tensor[response_positions[:n]] = torch.tensor(teacher_lps_raw[:n], dtype=torch.float32)
        res["teacher_log_probs"] = teacher_lp_tensor
        return res
