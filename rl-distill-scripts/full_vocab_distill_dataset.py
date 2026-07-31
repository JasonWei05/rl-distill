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

For precomputed top-k distillation, parquet rows store one ``[response_len, k]``
array for teacher token ids and normalized log probabilities.  The dataset
flattens each array before collation because verl's generic no-padding SFT
collator only handles one jagged sequence dimension.  The loss restores the
``[response_len, k]`` shape after the batch has been split into micro-batches.
"""

from __future__ import annotations

import os
import warnings
from bisect import bisect_right
from collections import OrderedDict
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
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


def _to_2d_array(value: Any, *, dtype: np.dtype, column: str, item: int) -> np.ndarray:
    if hasattr(value, "as_py"):
        value = value.as_py()
    if hasattr(value, "tolist"):
        value = value.tolist()
    array = np.asarray(value, dtype=dtype)
    if array.ndim != 2:
        raise ValueError(f"{column} must be rank 2 at row {item}, got shape {array.shape}")
    return array


class _ParquetRowGroup:
    """Loader-safe row-group metadata container with no module registration dependency."""

    __slots__ = ("file_index", "row_group_index", "first_row", "num_rows", "payload_bytes")

    def __init__(self, file_index: int, row_group_index: int, first_row: int, num_rows: int, payload_bytes: int):
        self.file_index = file_index
        self.row_group_index = row_group_index
        self.first_row = first_row
        self.num_rows = num_rows
        self.payload_bytes = payload_bytes


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
        self.use_precomputed_topk = bool(config.get("use_precomputed_topk", False))
        self.teacher_topk_token_ids_key = config.get("teacher_topk_token_ids_key", "teacher_topk_token_ids")
        self.teacher_topk_logprobs_key = config.get("teacher_topk_logprobs_key", "teacher_topk_logprobs")
        self.teacher_topk_width = int(config.get("teacher_topk_width", 128))
        self.teacher_topk_validation_tolerance = float(config.get("teacher_topk_validation_tolerance", 5e-4))
        self.parquet_row_group_cache_size = int(config.get("parquet_row_group_cache_size", 2))
        self.parquet_row_group_cache_max_bytes = int(
            config.get("parquet_row_group_cache_max_bytes", 1024 * 1024 * 1024)
        )
        self.parquet_max_row_group_bytes = int(config.get("parquet_max_row_group_bytes", 512 * 1024 * 1024))
        self.parquet_oversized_row_group_policy = str(config.get("parquet_oversized_row_group_policy", "warn")).lower()
        self.parquet_cache_diagnostics_interval = int(config.get("parquet_cache_diagnostics_interval", 0))
        self.max_samples = int(max_samples)

        if self.use_precomputed_topk and self.teacher_topk_width <= 0:
            raise ValueError(f"teacher_topk_width must be positive, got {self.teacher_topk_width}")
        if self.parquet_row_group_cache_size < 0:
            raise ValueError("parquet_row_group_cache_size must be non-negative")
        if self.parquet_row_group_cache_max_bytes <= 0:
            raise ValueError("parquet_row_group_cache_max_bytes must be positive")
        if self.parquet_max_row_group_bytes < 0:
            raise ValueError("parquet_max_row_group_bytes must be non-negative")
        if self.parquet_oversized_row_group_policy not in {"warn", "error", "ignore"}:
            raise ValueError("parquet_oversized_row_group_policy must be one of: warn, error, ignore")
        if self.parquet_cache_diagnostics_interval < 0:
            raise ValueError("parquet_cache_diagnostics_interval must be non-negative")

        if not isinstance(parquet_files, list | ListConfig):
            parquet_files = [parquet_files]
        self.parquet_files = [copy_local_path_from_hdfs(path, verbose=True) for path in parquet_files]
        self._reset_row_group_cache()
        self._read_files()

    def _read_files(self):
        columns = [self.input_ids_key, self.response_mask_key]
        if self.use_precomputed_topk:
            columns.extend([self.teacher_topk_token_ids_key, self.teacher_topk_logprobs_key])
            self._index_parquet_row_groups(columns)
            return

        dataframes = []
        for parquet_file in self.parquet_files:
            dataframes.append(
                pd.read_parquet(
                    parquet_file,
                    columns=columns,
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

    @staticmethod
    def _row_group_payload_bytes(row_group_metadata, columns: set[str]) -> int:
        """Estimate uncompressed bytes for selected root columns without reading their payloads."""
        total = 0
        for column_index in range(row_group_metadata.num_columns):
            column = row_group_metadata.column(column_index)
            root_column = column.path_in_schema.split(".", maxsplit=1)[0]
            if root_column in columns:
                total += max(0, int(column.total_uncompressed_size))
        return total

    def _index_parquet_row_groups(self, columns: list[str]) -> None:
        self._parquet_columns = tuple(columns)
        self._parquet_row_groups: list[_ParquetRowGroup] = []
        self._parquet_row_group_stops: list[int] = []
        required_columns = set(columns)
        total = 0

        for file_index, parquet_path in enumerate(self.parquet_files):
            parquet_file = pq.ParquetFile(parquet_path)
            try:
                missing_columns = required_columns.difference(parquet_file.schema_arrow.names)
                if missing_columns:
                    missing = ", ".join(sorted(missing_columns))
                    raise ValueError(f"missing required parquet columns in {parquet_path}: {missing}")

                metadata = parquet_file.metadata
                for row_group_index in range(metadata.num_row_groups):
                    row_group_metadata = metadata.row_group(row_group_index)
                    num_rows = int(row_group_metadata.num_rows)
                    if num_rows == 0:
                        continue
                    payload_bytes = self._row_group_payload_bytes(row_group_metadata, required_columns)
                    location = _ParquetRowGroup(
                        file_index=file_index,
                        row_group_index=row_group_index,
                        first_row=total,
                        num_rows=num_rows,
                        payload_bytes=payload_bytes,
                    )
                    self._check_row_group_size(parquet_path, location)
                    self._parquet_row_groups.append(location)
                    total += num_rows
                    self._parquet_row_group_stops.append(total)
            finally:
                parquet_file.close()

        self._total_rows = total
        self._selected_indices: np.ndarray | None = None
        print(f"[FullVocabDistillDataset] dataset len: {total}", flush=True)
        print(
            "[FullVocabDistillDataset] lazily indexed "
            f"{len(self._parquet_row_groups)} parquet row groups across {len(self.parquet_files)} files; "
            "nested payload columns remain unloaded",
            flush=True,
        )

        if self.max_samples > 0 and self.max_samples < total:
            if self.shuffle:
                rng_args = (self.seed,) if self.seed is not None else ()
                rng = np.random.default_rng(*rng_args)
                self._selected_indices = rng.choice(total, size=self.max_samples, replace=False)
            else:
                self._selected_indices = np.arange(self.max_samples, dtype=np.int64)
            print(f"[FullVocabDistillDataset] selected {self.max_samples} samples out of {total}", flush=True)

    def _check_row_group_size(self, parquet_path: str, location: _ParquetRowGroup) -> None:
        threshold = self.parquet_max_row_group_bytes
        if threshold == 0 or self.parquet_oversized_row_group_policy == "ignore" or location.payload_bytes <= threshold:
            return

        message = (
            "precomputed top-k parquet row group is oversized: "
            f"file={parquet_path}, row_group={location.row_group_index}, rows={location.num_rows}, "
            f"estimated_uncompressed_payload_bytes={location.payload_bytes}, configured_max_bytes={threshold}. "
            "Write production traces with smaller parquet row groups or raise parquet_max_row_group_bytes explicitly."
        )
        if self.parquet_oversized_row_group_policy == "error":
            raise ValueError(message)
        warnings.warn(message, UserWarning, stacklevel=3)

    def _reset_row_group_cache(self) -> None:
        self._row_group_cache: OrderedDict[tuple[int, int], pa.Table] = OrderedDict()
        self._row_group_cache_bytes = 0
        self._row_group_cache_pid = os.getpid()
        self._row_group_cache_hits = 0
        self._row_group_cache_misses = 0
        self._row_group_cache_evictions = 0
        self._row_group_cache_bypasses = 0

    def _ensure_process_local_cache(self) -> None:
        if self._row_group_cache_pid != os.getpid():
            self._reset_row_group_cache()

    def __getstate__(self):
        state = self.__dict__.copy()
        # Arrow tables may own large C++ buffers. Never serialize or transfer a
        # populated cache to a DataLoader worker; each process starts empty.
        state["_row_group_cache"] = OrderedDict()
        state["_row_group_cache_bytes"] = 0
        state["_row_group_cache_pid"] = None
        state["_row_group_cache_hits"] = 0
        state["_row_group_cache_misses"] = 0
        state["_row_group_cache_evictions"] = 0
        state["_row_group_cache_bypasses"] = 0
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._reset_row_group_cache()

    def cache_diagnostics(self) -> dict[str, int | float]:
        """Return process-local cache residency and hit/miss counters."""
        self._ensure_process_local_cache()
        requests = self._row_group_cache_hits + self._row_group_cache_misses
        return {
            "pid": self._row_group_cache_pid,
            "requests": requests,
            "hits": self._row_group_cache_hits,
            "misses": self._row_group_cache_misses,
            "hit_rate": self._row_group_cache_hits / requests if requests else 0.0,
            "evictions": self._row_group_cache_evictions,
            "bypasses": self._row_group_cache_bypasses,
            "resident_row_groups": len(self._row_group_cache),
            "resident_bytes": self._row_group_cache_bytes,
            "max_row_groups": self.parquet_row_group_cache_size,
            "max_bytes": self.parquet_row_group_cache_max_bytes,
        }

    def _maybe_log_cache_diagnostics(self) -> None:
        interval = self.parquet_cache_diagnostics_interval
        requests = self._row_group_cache_hits + self._row_group_cache_misses
        if interval > 0 and requests % interval == 0:
            diagnostics = self.cache_diagnostics()
            print(f"[FullVocabDistillDataset] parquet cache: {diagnostics}", flush=True)

    def _read_row_group(self, location: _ParquetRowGroup) -> pa.Table:
        self._ensure_process_local_cache()
        cache_key = (location.file_index, location.row_group_index)
        cached = self._row_group_cache.pop(cache_key, None)
        if cached is not None:
            self._row_group_cache[cache_key] = cached
            self._row_group_cache_hits += 1
            self._maybe_log_cache_diagnostics()
            return cached

        self._row_group_cache_misses += 1
        parquet_file = pq.ParquetFile(self.parquet_files[location.file_index])
        try:
            table = parquet_file.read_row_group(
                location.row_group_index,
                columns=list(self._parquet_columns),
                use_threads=False,
            )
        finally:
            parquet_file.close()

        table_bytes = int(table.nbytes)
        can_cache = self.parquet_row_group_cache_size > 0 and table_bytes <= self.parquet_row_group_cache_max_bytes
        if can_cache:
            while self._row_group_cache and (
                len(self._row_group_cache) >= self.parquet_row_group_cache_size
                or self._row_group_cache_bytes + table_bytes > self.parquet_row_group_cache_max_bytes
            ):
                _, evicted = self._row_group_cache.popitem(last=False)
                self._row_group_cache_bytes -= int(evicted.nbytes)
                self._row_group_cache_evictions += 1
            self._row_group_cache[cache_key] = table
            self._row_group_cache_bytes += table_bytes
        else:
            self._row_group_cache_bypasses += 1

        self._maybe_log_cache_diagnostics()
        return table

    def _get_precomputed_row(self, item: int) -> dict[str, Any]:
        dataset_length = len(self)
        if item < 0:
            item += dataset_length
        if item < 0 or item >= dataset_length:
            raise IndexError(f"dataset index out of range: {item}")

        physical_item = int(self._selected_indices[item]) if self._selected_indices is not None else item
        row_group_position = bisect_right(self._parquet_row_group_stops, physical_item)
        location = self._parquet_row_groups[row_group_position]
        table = self._read_row_group(location)
        local_row = physical_item - location.first_row
        return {column: table[column][local_row] for column in self._parquet_columns}

    def __len__(self):
        if self.use_precomputed_topk:
            return len(self._selected_indices) if self._selected_indices is not None else self._total_rows
        return len(self.dataframe)

    def _truncate(
        self,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        teacher_topk_token_ids: torch.Tensor | None = None,
        teacher_topk_logprobs: torch.Tensor | None = None,
    ):
        sequence_length = input_ids.shape[0]
        if sequence_length > self.max_length:
            if self.truncation == "error":
                raise ValueError(f"{sequence_length=} is larger than max_length={self.max_length}")
            sequence_slice = (
                slice(sequence_length - self.max_length, sequence_length)
                if self.truncation == "left"
                else slice(0, self.max_length)
            )
        else:
            sequence_slice = slice(0, sequence_length)

        slice_start = sequence_slice.start or 0
        dropped_response_prefix = int(loss_mask[:slice_start].sum().item())
        input_ids = input_ids[sequence_slice]
        loss_mask = loss_mask[sequence_slice]
        if teacher_topk_token_ids is not None:
            # Teacher rows correspond to response tokens, while the causal loss is
            # attached to their preceding positions.  If left truncation makes a
            # response token the first retained token, that token no longer has a
            # predecessor in the sample and must not retain a teacher target.
            missing_predecessor = int(loss_mask.numel() > 0 and bool(loss_mask[0].item()))
            teacher_start = dropped_response_prefix + missing_predecessor
            active_response_count = int(loss_mask[1:].sum().item())
            teacher_slice = slice(teacher_start, teacher_start + active_response_count)
            teacher_topk_token_ids = teacher_topk_token_ids[teacher_slice]
            teacher_topk_logprobs = teacher_topk_logprobs[teacher_slice]
        return input_ids, loss_mask, teacher_topk_token_ids, teacher_topk_logprobs

    def __getitem__(self, item):
        row = self._get_precomputed_row(item) if self.use_precomputed_topk else self.dataframe.iloc[item]
        input_ids = torch.tensor(_to_1d_list(row[self.input_ids_key]), dtype=torch.long)
        loss_mask = torch.tensor(_to_1d_list(row[self.response_mask_key]), dtype=torch.long)
        if input_ids.shape != loss_mask.shape:
            raise ValueError(
                f"input_ids and response_mask shape mismatch at row {item}: {input_ids.shape} vs {loss_mask.shape}"
            )
        if not torch.all((loss_mask == 0) | (loss_mask == 1)):
            raise ValueError(f"response_mask must be binary at row {item}")
        response_start = int((loss_mask == 1).nonzero(as_tuple=True)[0][0]) if loss_mask.any() else len(loss_mask)
        expected_mask = torch.zeros_like(loss_mask)
        expected_mask[response_start:] = 1
        if not torch.equal(loss_mask, expected_mask):
            raise ValueError(f"response_mask must be zero-prefix/one-suffix at row {item}")

        response_count = int(loss_mask.sum().item())
        if self.use_precomputed_topk:
            if response_start == 0:
                raise ValueError(f"precomputed top-k trace must retain at least one prompt token at row {item}")
            if response_count == 0:
                raise ValueError(f"precomputed top-k trace must contain at least one response token at row {item}")
        teacher_topk_token_ids = None
        teacher_topk_logprobs = None
        if self.use_precomputed_topk:
            teacher_ids_array = _to_2d_array(
                row[self.teacher_topk_token_ids_key],
                dtype=np.int64,
                column=self.teacher_topk_token_ids_key,
                item=item,
            )
            teacher_logprobs_array = _to_2d_array(
                row[self.teacher_topk_logprobs_key],
                dtype=np.float32,
                column=self.teacher_topk_logprobs_key,
                item=item,
            )
            expected_shape = (response_count, self.teacher_topk_width)
            if teacher_ids_array.shape != expected_shape or teacher_logprobs_array.shape != expected_shape:
                raise ValueError(
                    f"precomputed top-k shape mismatch at row {item}: expected {expected_shape}, "
                    f"got ids={teacher_ids_array.shape}, logprobs={teacher_logprobs_array.shape}"
                )
            if not np.isfinite(teacher_logprobs_array).all():
                raise ValueError(f"non-finite teacher top-k log probability at row {item}")
            tolerance = self.teacher_topk_validation_tolerance
            if teacher_logprobs_array.size and teacher_logprobs_array.max() > tolerance:
                raise ValueError(f"teacher top-k log probabilities must be <= 0 at row {item}")
            if teacher_logprobs_array.shape[1] > 1 and np.any(np.diff(teacher_logprobs_array, axis=1) > tolerance):
                raise ValueError(f"teacher top-k log probabilities must be in descending rank order at row {item}")
            if teacher_logprobs_array.size and np.any(
                np.exp(teacher_logprobs_array, dtype=np.float64).sum(axis=1) > 1.0 + tolerance
            ):
                raise ValueError(f"teacher top-k probability mass exceeds 1 at row {item}")
            if teacher_ids_array.size:
                vocab_size = int(self.tokenizer.vocab_size)
                if teacher_ids_array.min() < 0 or teacher_ids_array.max() >= vocab_size:
                    raise ValueError(f"teacher top-k token id outside [0, {vocab_size}) at row {item}")
                sorted_ids = np.sort(teacher_ids_array, axis=1)
                if np.any(np.diff(sorted_ids, axis=1) == 0):
                    raise ValueError(f"teacher top-k token ids must be unique at every position in row {item}")
            teacher_topk_token_ids = torch.from_numpy(teacher_ids_array.astype(np.int32, copy=False))
            teacher_topk_logprobs = torch.from_numpy(teacher_logprobs_array.astype(np.float16, copy=False))

        input_ids, loss_mask, teacher_topk_token_ids, teacher_topk_logprobs = self._truncate(
            input_ids,
            loss_mask,
            teacher_topk_token_ids,
            teacher_topk_logprobs,
        )
        if self.use_precomputed_topk and (
            loss_mask.numel() < 2 or bool(loss_mask[0].item()) or not bool(loss_mask.any().item())
        ):
            raise ValueError(
                f"precomputed top-k trace must retain both prompt and response tokens after truncation at row {item}"
            )
        position_ids = torch.arange(input_ids.shape[0], dtype=torch.long)

        if self.pad_mode == DatasetPadMode.NO_PADDING:
            sample = {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "loss_mask": loss_mask,
            }
            if self.use_precomputed_topk:
                sample["teacher_topk_token_ids"] = teacher_topk_token_ids.flatten()
                sample["teacher_topk_logprobs"] = teacher_topk_logprobs.flatten()
            return sample

        if self.use_precomputed_topk:
            raise NotImplementedError("precomputed top-k distillation requires pad_mode=no_padding")

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
