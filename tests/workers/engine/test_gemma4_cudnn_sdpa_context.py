from types import SimpleNamespace

import pytest
import torch

from verl.workers.engine.fsdp import transformer_impl


class _Model(torch.nn.Module):
    def __init__(self, model_type: str):
        super().__init__()
        self.config = SimpleNamespace(model_type=model_type)


def _engine(model_type: str):
    engine = object.__new__(transformer_impl.FSDPEngine)
    engine.module = _Model(model_type)
    engine.ulysses_parallel_group = object()
    engine.engine_config = SimpleNamespace(fsdp_size=1)
    engine.optimizer_zero_grad = lambda: None
    return engine


@pytest.fixture
def cudnn_sdpa_state(monkeypatch):
    state = {"enabled": False, "calls": []}

    def enable(value: bool) -> None:
        state["enabled"] = value
        state["calls"].append(value)

    monkeypatch.setattr(transformer_impl, "device_name", "cuda")
    monkeypatch.setattr(torch.backends.cuda, "cudnn_sdp_enabled", lambda: state["enabled"])
    monkeypatch.setattr(torch.backends.cuda, "enable_cudnn_sdp", enable)
    monkeypatch.setattr(transformer_impl, "get_ulysses_sequence_parallel_group", lambda: None)
    monkeypatch.setattr(transformer_impl, "set_ulysses_sequence_parallel_group", lambda _group: None)
    return state


@pytest.mark.parametrize(
    "context_type",
    [transformer_impl.EngineTrainModeCtx, transformer_impl.EngineEvalModeCtx],
)
def test_gemma4_context_temporarily_restores_cudnn_sdpa(context_type, cudnn_sdpa_state):
    engine = _engine("gemma4_text")

    with context_type(engine, disable_auto_offload=True):
        assert cudnn_sdpa_state["enabled"] is True

    assert cudnn_sdpa_state["enabled"] is False
    assert cudnn_sdpa_state["calls"] == [True, False]


def test_gemma4_context_preserves_already_enabled_state(cudnn_sdpa_state):
    cudnn_sdpa_state["enabled"] = True
    engine = _engine("gemma4_text")

    with transformer_impl.EngineTrainModeCtx(engine, disable_auto_offload=True):
        assert cudnn_sdpa_state["enabled"] is True

    assert cudnn_sdpa_state["enabled"] is True
    assert cudnn_sdpa_state["calls"] == []


def test_non_gemma_context_does_not_change_cudnn_sdpa(cudnn_sdpa_state):
    engine = _engine("llama")

    with transformer_impl.EngineTrainModeCtx(engine, disable_auto_offload=True):
        assert cudnn_sdpa_state["enabled"] is False

    assert cudnn_sdpa_state["calls"] == []
