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
"""Per-row provenance vs identity checks for Gemma 4 trace records.

The generator keeps the git commit out of the hashed semantic config (it is recorded under
``generator_repository`` instead) and stamps each row with the commit that was checked out when the
shard was produced. A resumed collection therefore legitimately mixes commits, and the resume path
validates existing shards against a semantic config that has no ``generator.commit`` key. The row
identity is ``generator_source_sha256``; the commit is provenance only.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPOSITORY_ROOT / "rl-distill-scripts" / "data"
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

from gemma4_distill_trace_schema import (  # noqa: E402
    RESPONSE_TEXT_NORMALIZATION,
    TOPK_WIDTH,
    TraceValidationError,
    validate_trace_record,
)
from test_rl_distill_gemma4_training_view import DIRECTION, GLOBAL_SEED, _row  # noqa: E402

GEN_SHA = "1" * 64
SOURCE_SHA = "0" * 64


def _expected_semantic(*, generator_source_sha256: str = SOURCE_SHA) -> dict:
    """Semantic config as the generator records it since 6c426fd5: no ``generator.commit``."""
    return {
        "direction": DIRECTION,
        "source_dataset": "ds",
        "source_dataset_sha256": "e" * 64,
        "teacher": {"model": "teacher", "revision": "rev", "content_sha256": "b" * 64},
        "tokenizer": {"model": "tok", "revision": "rev", "sha256": "c" * 64, "vocab_size": 1000},
        "chat_template": {"path": "t.jinja", "sha256": "d" * 64},
        "global_seed": GLOBAL_SEED,
        "generator": {"source_sha256": generator_source_sha256},
        "sampling": {},
        "environment_versions": {},
    }


def _record(**overrides) -> dict:
    """A synthetic row that satisfies every per-row check (uniform, normalized top-k targets)."""
    row = _row(gen_sha=GEN_SHA, split="train", uid="q1", sample_index=0, question="1+1")
    uniform = math.log(1.0 / TOPK_WIDTH)
    row.update(
        teacher_topk_rank_order=f"1..{TOPK_WIDTH}",
        teacher_topk_logprobs=[[uniform] * TOPK_WIDTH for _ in row["response_token_ids"]],
        sampled_token_logprobs=[uniform] * len(row["response_token_ids"]),
        response_text_normalization=RESPONSE_TEXT_NORMALIZATION,
    )
    row.update(overrides)
    return row


def test_row_commit_is_not_part_of_the_identity_check():
    record = _record(generator_commit="82d526ee41ef2111c54609c53b62a67eaf6d5fea")
    mass = validate_trace_record(
        record,
        decoder=None,
        expected_config_sha256=GEN_SHA,
        expected_direction=DIRECTION,
        expected_split="train",
        expected_semantic_config=_expected_semantic(),
    )
    assert len(mass) == record["response_length"]


def test_generator_source_hash_is_still_enforced():
    record = _record(generator_source_sha256="f" * 64)
    with pytest.raises(TraceValidationError, match="generator_source_sha256"):
        validate_trace_record(
            record,
            decoder=None,
            expected_config_sha256=GEN_SHA,
            expected_direction=DIRECTION,
            expected_split="train",
            expected_semantic_config=_expected_semantic(),
        )
