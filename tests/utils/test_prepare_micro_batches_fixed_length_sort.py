import pytest
import torch
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.workers.engine import utils as engine_utils
from verl.workers.engine.utils import postprocess_batch_func, prepare_micro_batches


def test_fixed_padded_token_budget_isolates_long_sequences():
    lengths = [1610, 1611, 1616, 1619, 1647, 1658, 1667, 1676, 1687, 1703, 1743, 1811, 1813, 1891, 2021, 3117]
    input_ids = torch.nested.as_nested_tensor(
        [torch.arange(length) for length in lengths],
        layout=torch.jagged,
    )
    batch = TensorDict(
        {"input_ids": input_ids, "sample_id": torch.arange(len(lengths))},
        batch_size=[len(lengths)],
    )
    tu.assign_non_tensor(
        batch,
        use_dynamic_bsz=False,
        micro_batch_size_per_gpu=2,
        force_group_size=1,
        max_padded_tokens_per_microbatch=5120,
    )

    micro_batches, indices = prepare_micro_batches(batch)

    assert len(micro_batches) == 9
    assert [15] in indices
    assert sorted(index for partition in indices for index in partition) == list(range(len(lengths)))
    for micro_batch in micro_batches:
        microbatch_lengths = micro_batch["input_ids"].offsets().diff().tolist()
        assert len(microbatch_lengths) <= 2
        if len(microbatch_lengths) > 1:
            assert max(microbatch_lengths) * len(microbatch_lengths) <= 5120


def test_fixed_padded_token_budget_supports_dense_padded_inputs():
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
        ]
    )
    batch = TensorDict(
        {
            "input_ids": torch.arange(24).reshape(4, 6),
            "attention_mask": attention_mask,
            "sample_id": torch.arange(4),
        },
        batch_size=[4],
    )
    tu.assign_non_tensor(
        batch,
        use_dynamic_bsz=False,
        micro_batch_size_per_gpu=2,
        force_group_size=1,
        max_padded_tokens_per_microbatch=6,
    )

    micro_batches, indices = prepare_micro_batches(batch)

    assert indices == [[1, 3], [2], [0]]
    assert [micro_batch["sample_id"].tolist() for micro_batch in micro_batches] == [[1, 3], [2], [0]]


def test_fixed_padded_token_budget_rejects_force_groups():
    input_ids = torch.nested.as_nested_tensor(
        [torch.arange(length) for length in (4, 1, 3, 2)],
        layout=torch.jagged,
    )
    batch = TensorDict({"input_ids": input_ids}, batch_size=[4])
    tu.assign_non_tensor(
        batch,
        use_dynamic_bsz=False,
        micro_batch_size_per_gpu=1,
        force_group_size=2,
        max_padded_tokens_per_microbatch=5120,
    )

    with pytest.raises(ValueError, match="requires force_group_size=1"):
        prepare_micro_batches(batch)


def test_fixed_padded_token_budget_equalizes_microbatch_count(monkeypatch):
    input_ids = torch.nested.as_nested_tensor(
        [torch.arange(length) for length in (1, 2, 3, 4)],
        layout=torch.jagged,
    )
    batch = TensorDict({"input_ids": input_ids}, batch_size=[4])
    tu.assign_non_tensor(
        batch,
        use_dynamic_bsz=False,
        micro_batch_size_per_gpu=2,
        force_group_size=1,
        max_padded_tokens_per_microbatch=16,
    )

    monkeypatch.setattr(engine_utils.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(engine_utils, "get_device_name", lambda: "cpu")

    def fake_all_reduce(value, *, op, group):
        assert op == engine_utils.dist.ReduceOp.MAX
        assert group == "dp-group"
        value.fill_(3)

    monkeypatch.setattr(engine_utils.dist, "all_reduce", fake_all_reduce)

    micro_batches, indices = prepare_micro_batches(batch, dp_group="dp-group")

    assert len(micro_batches) == 3
    assert sorted(index for partition in indices for index in partition) == [0, 1, 2, 3]
    assert all(1 <= len(partition) <= 2 for partition in indices)


def test_fixed_padded_token_budget_restores_model_output_order():
    lengths = [4, 1, 3, 2]
    input_ids = torch.nested.as_nested_tensor(
        [torch.arange(length) for length in lengths],
        layout=torch.jagged,
    )
    batch = TensorDict({"input_ids": input_ids}, batch_size=[4])
    tu.assign_non_tensor(
        batch,
        use_dynamic_bsz=False,
        micro_batch_size_per_gpu=2,
        force_group_size=1,
        max_padded_tokens_per_microbatch=6,
        pad_mode="no_padding",
    )

    _, indices = prepare_micro_batches(batch)
    output_lst = []
    for partition in indices:
        rows = [torch.full((lengths[index],), index, dtype=torch.long) for index in partition]
        output_lst.append(
            {
                "model_output": {
                    "sample_id": torch.nested.as_nested_tensor(rows, layout=torch.jagged),
                }
            }
        )

    output = postprocess_batch_func(output_lst=output_lst, indices=indices, data=batch)

    restored = output["model_output"]["sample_id"].unbind()
    assert [row.tolist() for row in restored] == [[index] * length for index, length in enumerate(lengths)]
