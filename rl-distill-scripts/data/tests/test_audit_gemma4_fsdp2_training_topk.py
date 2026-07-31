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

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from torch.utils.data import DistributedSampler

DATA_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATA_DIR))

import audit_gemma4_fsdp2_training_topk as audit  # noqa: E402


def test_distributed_train_batch_indices_matches_opening_production_batches() -> None:
    actual = audit.distributed_train_batch_indices(
        dataset_size=101,
        world_size=4,
        rank=2,
        global_batch_size=8,
        seed=42,
        batch_count=3,
    )
    sampler = DistributedSampler(
        range(101),
        num_replicas=4,
        rank=2,
        shuffle=True,
        seed=42,
        drop_last=True,
    )
    sampler.set_epoch(0)
    expected = list(sampler)[:6]

    assert actual == [expected[0:2], expected[2:4], expected[4:6]]


def test_distributed_train_batch_indices_rejects_incomplete_contract() -> None:
    with pytest.raises(audit.FSDP2TopKAuditError, match="smaller than 3"):
        audit.distributed_train_batch_indices(
            dataset_size=16,
            world_size=4,
            rank=0,
            global_batch_size=8,
            seed=42,
            batch_count=3,
        )


def test_distributed_validation_indices_matches_sft_trainer() -> None:
    actual = audit.distributed_validation_indices(dataset_size=16, world_size=4, rank=2)
    sampler = DistributedSampler(
        range(16),
        num_replicas=4,
        rank=2,
        shuffle=False,
        drop_last=False,
    )

    assert actual == list(sampler)


def test_distributed_validation_indices_requires_exact_coverage() -> None:
    with pytest.raises(audit.FSDP2TopKAuditError, match="divisible"):
        audit.distributed_validation_indices(dataset_size=17, world_size=4, rank=0)


def _gate_inputs() -> tuple[dict, dict, list[list[dict]], SimpleNamespace]:
    aggregate = {
        "top1_tie_safe": {"mean": 1.0},
        "top10_overlap_fraction": {"mean": 1.0},
        "topk_overlap_fraction": {"mean": 1.0},
        "stored_support_weighted_abs_logprob_delta": {"mean": 0.0001},
        "stored_support_probability_l1": {"mean": 0.0001},
        "sampled_token_abs_logprob_delta": {"p95": 0.0001},
        "stored_only_topk_mass": {"p99": 0.0},
        "reference_only_topk_mass": {"p99": 0.0},
    }
    exact = {
        "ordered_topk_exact": {"mean": 1.0},
        "fp16_support_exact_fraction": {"mean": 1.0},
        "fp16_sampled_exact": {"mean": 1.0},
    }
    production_batches = []
    global_index = 0
    for batch_index in range(1, 4):
        rank_batches = []
        for rank in range(2):
            rank_batches.append(
                {
                    "batch_index": batch_index,
                    "rank": rank,
                    "global_indices": [global_index, global_index + 1],
                    "microbatch_count": 2,
                    "microbatches": [
                        {"indices": [0], "padded_tokens": 2000},
                        {"indices": [1], "padded_tokens": 2500},
                    ],
                }
            )
            global_index += 2
        production_batches.append(rank_batches)
    args = SimpleNamespace(
        min_top1_tie_safe=0.999,
        min_top10_overlap=0.995,
        min_topk_overlap=0.995,
        max_weighted_abs_logprob_delta=0.003,
        max_support_probability_l1=0.003,
        max_sampled_token_abs_delta_p95=0.01,
        max_membership_delta_mass_p99=0.001,
        train_batches=3,
        train_batch_size=4,
        micro_batch_size_per_gpu=1,
        max_padded_tokens_per_microbatch=4096,
        max_grad_norm=50.0,
        validate_before_train=True,
        validation_every=1,
    )
    return aggregate, exact, production_batches, args


def test_evaluate_gate_accepts_three_singleton_batches() -> None:
    aggregate, exact, production_batches, args = _gate_inputs()

    gate = audit.evaluate_gate(aggregate, exact, [12.0, 16.0, 14.0], production_batches, [0, 1, 2, 3], args)

    assert gate["status"] == "pass"
    assert gate["checks"]["production_batch_count"]["observed"] == 3
    assert gate["checks"]["backward_grad_norm_max"]["observed"] == 16.0
    assert gate["checks"]["validation_event_count"]["observed"] == 4


def test_evaluate_gate_fails_closed_on_missing_rank_or_bad_gradient() -> None:
    aggregate, exact, production_batches, args = _gate_inputs()
    production_batches[2].pop()

    gate = audit.evaluate_gate(aggregate, exact, [12.0, 16.0, 60.0], production_batches, [0, 1, 2], args)

    assert gate["status"] == "fail"
    assert gate["checks"]["production_rank_count_spread"]["passed"] is False
    assert gate["checks"]["backward_grad_norm_max"]["passed"] is False
    assert gate["checks"]["validation_event_count"]["passed"] is False
