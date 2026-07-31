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

import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from verl.utils.import_utils import load_extern_object

SCRIPTS_DIR = Path(__file__).parents[3] / "rl-distill-scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import full_vocab_distill_dataset as dataset_module  # noqa: E402
from full_vocab_distill_dataset import FullVocabDistillDataset  # noqa: E402

_PARQUET_COLUMNS = [
    "input_ids",
    "response_mask",
    "teacher_topk_token_ids",
    "teacher_topk_logprobs",
]


def _write_trace_file(path: Path, markers: list[int], *, row_group_size: int) -> None:
    rows = [
        {
            "input_ids": [1, marker, marker + 100, marker + 200],
            "response_mask": [0, 0, 1, 1],
            "teacher_topk_token_ids": [[1, 2], [3, 4]],
            "teacher_topk_logprobs": [[-0.7, -1.7], [-0.8, -1.8]],
        }
        for marker in markers
    ]
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=row_group_size)


def _config(**overrides):
    values = {
        "pad_mode": "no_padding",
        "max_length": 16,
        "truncation": "error",
        "shuffle": False,
        "seed": 123,
        "use_precomputed_topk": True,
        "teacher_topk_width": 2,
        "parquet_max_row_group_bytes": 0,
    }
    values.update(overrides)
    return OmegaConf.create(values)


def _dataset(files, **config_overrides):
    max_samples = config_overrides.pop("max_samples", -1)
    return FullVocabDistillDataset(
        files,
        SimpleNamespace(vocab_size=10_000, pad_token_id=0),
        _config(**config_overrides),
        max_samples=max_samples,
    )


def _marker(dataset, item: int) -> int:
    return int(dataset[item]["input_ids"][1].item())


def test_dataset_class_loads_through_verl_external_loader():
    dataset_class = load_extern_object(str(SCRIPTS_DIR / "full_vocab_distill_dataset.py"), "FullVocabDistillDataset")
    assert dataset_class.__name__ == "FullVocabDistillDataset"


def test_lazy_initialization_reads_only_parquet_metadata(tmp_path, monkeypatch):
    parquet = tmp_path / "traces.parquet"
    _write_trace_file(parquet, [10, 11, 12], row_group_size=1)

    original_parquet_file = pq.ParquetFile
    payload_reads = []

    class MetadataOnlyParquetFile:
        def __init__(self, *args, **kwargs):
            self.delegate = original_parquet_file(*args, **kwargs)

        @property
        def metadata(self):
            return self.delegate.metadata

        @property
        def schema_arrow(self):
            return self.delegate.schema_arrow

        def read_row_group(self, *args, **kwargs):
            payload_reads.append((args, kwargs))
            raise AssertionError("dataset initialization loaded parquet payload columns")

        def close(self):
            self.delegate.close()

    monkeypatch.setattr(dataset_module.pq, "ParquetFile", MetadataOnlyParquetFile)
    dataset = _dataset(str(parquet))

    assert len(dataset) == 3
    assert payload_reads == []
    assert dataset.cache_diagnostics()["resident_bytes"] == 0


def test_random_access_crosses_files_and_row_groups(tmp_path):
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    _write_trace_file(first, [10, 11, 12, 13, 14], row_group_size=2)
    _write_trace_file(second, [20, 21, 22, 23], row_group_size=1)
    dataset = _dataset([str(first), str(second)], parquet_row_group_cache_size=2)

    access_order = [8, 0, 5, 3, 7, 2, 4, 1, 6]
    expected = [23, 10, 20, 13, 22, 12, 14, 11, 21]
    assert [_marker(dataset, item) for item in access_order] == expected


def test_max_samples_selection_is_seeded_and_deterministic(tmp_path):
    parquet = tmp_path / "traces.parquet"
    markers = list(range(10, 20))
    _write_trace_file(parquet, markers, row_group_size=2)

    first = _dataset(str(parquet), shuffle=True, seed=987, max_samples=4)
    second = _dataset(str(parquet), shuffle=True, seed=987, max_samples=4)
    expected_indices = np.random.default_rng(987).choice(len(markers), size=4, replace=False)
    expected_markers = [markers[index] for index in expected_indices]

    assert [_marker(first, item) for item in range(len(first))] == expected_markers
    assert [_marker(second, item) for item in range(len(second))] == expected_markers

    prefix = _dataset(str(parquet), shuffle=False, max_samples=4)
    assert [_marker(prefix, item) for item in range(len(prefix))] == markers[:4]


def test_row_group_cache_is_byte_bounded_and_pickle_starts_empty(tmp_path):
    parquet = tmp_path / "traces.parquet"
    _write_trace_file(parquet, [10, 11, 12, 13], row_group_size=2)
    parquet_file = pq.ParquetFile(parquet)
    try:
        row_group_bytes = [
            parquet_file.read_row_group(index, columns=_PARQUET_COLUMNS, use_threads=False).nbytes
            for index in range(parquet_file.metadata.num_row_groups)
        ]
    finally:
        parquet_file.close()
    cache_max_bytes = max(row_group_bytes)

    dataset = _dataset(
        str(parquet),
        parquet_row_group_cache_size=2,
        parquet_row_group_cache_max_bytes=cache_max_bytes,
    )
    assert _marker(dataset, 0) == 10
    assert _marker(dataset, 1) == 11
    assert _marker(dataset, 2) == 12

    diagnostics = dataset.cache_diagnostics()
    assert diagnostics["hits"] == 1
    assert diagnostics["misses"] == 2
    assert diagnostics["evictions"] == 1
    assert diagnostics["resident_row_groups"] == 1
    assert diagnostics["resident_bytes"] <= cache_max_bytes

    restored = pickle.loads(pickle.dumps(dataset))
    restored_diagnostics = restored.cache_diagnostics()
    assert restored_diagnostics["requests"] == 0
    assert restored_diagnostics["resident_row_groups"] == 0
    assert restored_diagnostics["resident_bytes"] == 0
    assert _marker(restored, 3) == 13


def test_dataset_is_safe_with_spawned_dataloader_worker(tmp_path):
    parquet = tmp_path / "traces.parquet"
    markers = [10, 11, 12]
    _write_trace_file(parquet, markers, row_group_size=1)
    dataset = _dataset(str(parquet), parquet_row_group_cache_size=1)
    assert _marker(dataset, 0) == markers[0]

    loader = DataLoader(dataset, batch_size=None, num_workers=1, multiprocessing_context="spawn")
    assert [int(sample["input_ids"][1].item()) for sample in loader] == markers


def test_oversized_row_group_can_fail_before_payload_read(tmp_path):
    parquet = tmp_path / "traces.parquet"
    _write_trace_file(parquet, [10, 11], row_group_size=2)

    with pytest.raises(ValueError, match="parquet row group is oversized"):
        _dataset(
            str(parquet),
            parquet_max_row_group_bytes=1,
            parquet_oversized_row_group_policy="error",
        )
