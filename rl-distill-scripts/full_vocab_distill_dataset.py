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

"""Pretokenized SFT dataset for full-vocabulary off-policy distillation.

The teacher-response dataset already stores Gemma token ids, so this dataset
uses `input_ids` and `response_mask` directly instead of re-applying a chat
template. `loss_mask` is the unshifted response-token mask; the loss shifts it
left to align next-token logits with the response token being predicted.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset

from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.fs import copy_local_path_from_hdfs


def _to_1d_list(value: Any) -> list[int]:
    if hasattr(value, "as_py"):
        value = value.as_py()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and len(value) == 1 and hasattr(value[0], "tolist"):
        value = value[0].tolist()
    return list(value)


class FullVocabDistillDataset(Dataset):
    """Dataset that consumes precomputed `input_ids` and `response_mask` columns."""

    def __init__(
        self,
        parquet_files: str | list[str],
        tokenizer,
        config: DictConfig,
        processor=None,
        max_samples: int = -1,
    ):
        del processor
        self.tokenizer = tokenizer
        self.pad_mode = config.get("pad_mode", "no_padding")
        assert self.pad_mode in ["right", "no_padding"], f"Unsupported pad_mode={self.pad_mode}"
        self.max_length = int(config.get("max_length", 20480))
        self.truncation = config.get("truncation", "right")
        assert self.truncation in ["error", "left", "right"]
        self.shuffle = bool(config.get("shuffle", False))
        self.seed = config.get("seed", None)
        self.input_ids_key = config.get("input_ids_key", "input_ids")
        self.response_mask_key = config.get("response_mask_key", "response_mask")
        self.max_samples = int(max_samples)

        if not isinstance(parquet_files, list | ListConfig):
            parquet_files = [parquet_files]
        self.parquet_files = [copy_local_path_from_hdfs(path, verbose=True) for path in parquet_files]
        self._read_files()

    def _read_files(self):
        dataframes = []
        for parquet_file in self.parquet_files:
            dataframes.append(
                pd.read_parquet(
                    parquet_file,
                    columns=[self.input_ids_key, self.response_mask_key],
                    dtype_backend="pyarrow",
                )
            )
        self.dataframe = pd.concat(dataframes, ignore_index=True)
        total = len(self.dataframe)
        print(f"[FullVocabDistillDataset] dataset len: {total}", flush=True)

        if self.max_samples > 0 and self.max_samples < total:
            if self.shuffle:
                rng_args = (self.seed,) if self.seed is not None else ()
                rng = np.random.default_rng(*rng_args)
                indices = rng.choice(total, size=self.max_samples, replace=False)
            else:
                indices = np.arange(self.max_samples)
            self.dataframe = self.dataframe.iloc[indices.tolist()].reset_index(drop=True)
            print(f"[FullVocabDistillDataset] selected {self.max_samples} samples out of {total}", flush=True)

    def __len__(self):
        return len(self.dataframe)

    def _truncate(self, input_ids: torch.Tensor, loss_mask: torch.Tensor):
        sequence_length = input_ids.shape[0]
        if sequence_length <= self.max_length:
            return input_ids, loss_mask
        if self.truncation == "error":
            raise ValueError(f"{sequence_length=} is larger than max_length={self.max_length}")
        if self.truncation == "left":
            return input_ids[-self.max_length :], loss_mask[-self.max_length :]
        return input_ids[: self.max_length], loss_mask[: self.max_length]

    def __getitem__(self, item):
        row = self.dataframe.iloc[item]
        input_ids = torch.tensor(_to_1d_list(row[self.input_ids_key]), dtype=torch.long)
        loss_mask = torch.tensor(_to_1d_list(row[self.response_mask_key]), dtype=torch.long)
        if input_ids.shape != loss_mask.shape:
            raise ValueError(
                f"input_ids and response_mask shape mismatch at row {item}: {input_ids.shape} vs {loss_mask.shape}"
            )

        input_ids, loss_mask = self._truncate(input_ids, loss_mask)
        position_ids = torch.arange(input_ids.shape[0], dtype=torch.long)

        if self.pad_mode == DatasetPadMode.NO_PADDING:
            return {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "loss_mask": loss_mask,
            }

        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        sequence_length = input_ids.shape[0]
        if sequence_length < self.max_length:
            pad_len = self.max_length - sequence_length
            input_ids = F.pad(input_ids, (0, pad_len), value=pad_token_id)
            loss_mask = F.pad(loss_mask, (0, pad_len), value=0)
            position_ids = F.pad(position_ids, (0, pad_len), value=0)
            attention_mask = torch.cat(
                [torch.ones(sequence_length, dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)]
            )
        else:
            attention_mask = torch.ones(sequence_length, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }
