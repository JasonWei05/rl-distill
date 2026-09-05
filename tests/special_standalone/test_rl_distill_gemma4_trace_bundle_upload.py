# Copyright 2026 rl-distill contributors
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
"""Completeness gate and upload filters of upload_gemma4_trace_bundle_hf.py (no network)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPOSITORY_ROOT / "rl-distill-scripts" / "data"
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

import upload_gemma4_trace_bundle_hf as uploader  # noqa: E402


def _bundle(tmp_path: Path, *, complete: bool = True, roster: bool = True, rows: int = 24300) -> Path:
    root = tmp_path / "26b-hard"
    for sub in ("train", "validation", "source", "logs"):
        (root / sub).mkdir(parents=True)
    (root / "dataset_index.json").write_text(json.dumps({"total_rows": rows, "total_response_tokens": 7}))
    if complete:
        (root / "COMPLETE.json").write_text("{}")
    if roster:
        (root / "source" / "roster_train.parquet").write_bytes(b"")
    (root / "train" / ".traces-train-000000.parquet.lock").write_bytes(b"")
    return root


def test_complete_bundle_passes_and_returns_the_index(tmp_path: Path):
    index = uploader.bundle_is_complete(_bundle(tmp_path), 24300)
    assert index["total_rows"] == 24300


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"complete": False}, "no COMPLETE.json"),
        ({"roster": False}, "source/ has no roster"),
        ({"rows": 24000}, "expected 24300"),
    ],
)
def test_incomplete_bundles_are_rejected(tmp_path: Path, kwargs: dict, message: str):
    with pytest.raises(SystemExit, match=message):
        uploader.bundle_is_complete(_bundle(tmp_path, **kwargs), 24300)


def test_row_count_check_can_be_disabled(tmp_path: Path):
    assert uploader.bundle_is_complete(_bundle(tmp_path, rows=5), None)["total_rows"] == 5


def test_upload_filters_exclude_generator_lock_files_and_keep_the_bundle_contract():
    assert "*.lock" in uploader.IGNORE_PATTERNS
    assert ".cache/*" in uploader.IGNORE_PATTERNS
    for required in ("train/*", "validation/*", "source/*", "dataset_index.json", "COMPLETE.json"):
        assert required in uploader.ALLOW_PATTERNS


def test_dry_run_does_not_touch_the_hub(tmp_path: Path):
    class NoNetwork:
        def __getattr__(self, name):  # any API call is a test failure
            raise AssertionError(f"unexpected HfApi.{name} call in dry run")

    result = uploader.upload_bundle(
        NoNetwork(), root=_bundle(tmp_path), repo="x/y", private=False, index={}, dry_run=True
    )
    assert result == {"repo": "x/y", "dry_run": True}
