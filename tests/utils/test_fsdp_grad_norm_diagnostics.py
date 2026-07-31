import pytest
import torch

from verl.utils.fsdp_utils import fsdp2_grad_norm_diagnostics, gradient_norm_anomaly_reason


def test_fsdp2_grad_norm_diagnostics_plain_parameters_without_distributed_init():
    first = torch.nn.Parameter(torch.zeros(2))
    second = torch.nn.Parameter(torch.zeros(1))
    first.grad = torch.tensor([3.0, 4.0])
    second.grad = torch.tensor([12.0])

    diagnostics = fsdp2_grad_norm_diagnostics(
        [("first", first), ("second", second)],
        top_k=2,
    )

    assert diagnostics["total_norm"] == pytest.approx(13.0)
    assert diagnostics["top"] == [("second", 12.0), ("first", 5.0)]
    assert diagnostics["per_parameter"] == {"first": 5.0, "second": 12.0}


def test_fsdp2_grad_norm_diagnostics_ignores_missing_gradients():
    parameter = torch.nn.Parameter(torch.zeros(1))

    assert fsdp2_grad_norm_diagnostics([("parameter", parameter)]) == {
        "total_norm": 0.0,
        "top": [],
        "per_parameter": {},
    }


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_gradient_norm_anomaly_reason_rejects_nonfinite_values(value):
    assert "non-finite" in gradient_norm_anomaly_reason(torch.tensor(value))


def test_gradient_norm_anomaly_reason_enforces_optional_threshold():
    assert gradient_norm_anomaly_reason(torch.tensor(10.0), max_norm="100") is None
    assert "exceeds fail-closed threshold" in gradient_norm_anomaly_reason(torch.tensor(150.0), max_norm="100")


@pytest.mark.parametrize("threshold", [0, -1, float("nan"), float("inf")])
def test_gradient_norm_anomaly_reason_rejects_invalid_threshold(threshold):
    with pytest.raises(ValueError, match="finite and positive"):
        gradient_norm_anomaly_reason(1.0, max_norm=threshold)
