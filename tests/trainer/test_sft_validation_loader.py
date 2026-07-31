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

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from verl.trainer.sft_trainer import SFTTrainer


class _SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, size: int):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        del index
        return {
            "input_ids": torch.tensor([1, 2, 3], dtype=torch.long),
            "position_ids": torch.arange(3),
            "loss_mask": torch.tensor([0, 1, 1], dtype=torch.long),
        }


def _trainer(*, rank: int, val_size: int, require_exact: bool = True) -> SFTTrainer:
    trainer = object.__new__(SFTTrainer)
    trainer.config = OmegaConf.create(
        {
            "data": {
                "train_batch_size": 4,
                "pad_mode": "no_padding",
                "num_workers": 0,
                "require_exact_val_coverage": require_exact,
            }
        }
    )
    trainer.engine = SimpleNamespace(
        get_data_parallel_rank=lambda: rank,
        get_data_parallel_size=lambda: 2,
    )
    trainer.train_dataset = _SequenceDataset(4)
    trainer.val_dataset = _SequenceDataset(val_size)
    return trainer


@pytest.mark.parametrize("rank", [0, 1])
def test_validation_loader_keeps_partial_final_batch(rank):
    trainer = _trainer(rank=rank, val_size=6)
    trainer._build_dataloader()

    local_batch_sizes = [batch["input_ids"].shape[0] for batch in trainer.val_dataloader]
    assert local_batch_sizes == [2, 1]


def test_exact_validation_coverage_rejects_distributed_sampler_padding():
    trainer = _trainer(rank=0, val_size=5)

    with pytest.raises(ValueError, match="divisible by the data-parallel size"):
        trainer._build_dataloader()


@pytest.mark.parametrize("rank", [0, 1])
def test_generic_validation_preserves_drop_last_behavior(rank):
    trainer = _trainer(rank=rank, val_size=6, require_exact=False)
    trainer._build_dataloader()

    local_batch_sizes = [batch["input_ids"].shape[0] for batch in trainer.val_dataloader]
    assert local_batch_sizes == [2]
