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


def test_gemma4_context_can_disable_cudnn_sdpa(monkeypatch, cudnn_sdpa_state):
    cudnn_sdpa_state["enabled"] = True
    monkeypatch.setenv("VERL_GEMMA4_CUDNN_SDPA", "0")
    engine = _engine("gemma4_text")

    with transformer_impl.EngineTrainModeCtx(engine, disable_auto_offload=True):
        assert cudnn_sdpa_state["enabled"] is False

    assert cudnn_sdpa_state["enabled"] is True
    assert cudnn_sdpa_state["calls"] == [False, True]


def test_gemma4_eval_context_can_use_separate_backend(monkeypatch, cudnn_sdpa_state):
    cudnn_sdpa_state["enabled"] = True
    monkeypatch.setenv("VERL_GEMMA4_CUDNN_SDPA", "1")
    monkeypatch.setenv("VERL_GEMMA4_EVAL_CUDNN_SDPA", "0")
    engine = _engine("gemma4_text")

    with transformer_impl.EngineEvalModeCtx(engine, disable_auto_offload=True):
        assert cudnn_sdpa_state["enabled"] is False

    assert cudnn_sdpa_state["enabled"] is True
    assert cudnn_sdpa_state["calls"] == [False, True]


def test_gemma4_context_rejects_invalid_cudnn_sdpa_override(monkeypatch, cudnn_sdpa_state):
    monkeypatch.setenv("VERL_GEMMA4_CUDNN_SDPA", "invalid")
    engine = _engine("gemma4_text")

    with pytest.raises(ValueError, match="must be 0 or 1"):
        with transformer_impl.EngineTrainModeCtx(engine, disable_auto_offload=True):
            pass
