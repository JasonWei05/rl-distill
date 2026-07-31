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

from pathlib import Path
from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf

from dapo import main_dapo

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("address", ["auto", "local"])
def test_initialize_dapo_ray_forwards_driver_secrets_without_logging_them(monkeypatch, capsys, address):
    wandb_secret = "wandb-secret-value"
    hf_secret = "hf-secret-value"
    monkeypatch.setenv("WANDB_API_KEY", wandb_secret)
    monkeypatch.setenv("HF_TOKEN", hf_secret)
    monkeypatch.setattr(main_dapo.ray, "is_initialized", Mock(return_value=False))
    ray_init = Mock()
    monkeypatch.setattr(main_dapo.ray, "init", ray_init)

    config = OmegaConf.create(
        {
            "ray_kwargs": {
                "ray_init": {
                    "address": address,
                    "runtime_env": {"env_vars": {"KEEP_ME": "yes"}},
                }
            }
        }
    )
    main_dapo._initialize_dapo_ray(config)

    ray_init.assert_called_once()
    kwargs = ray_init.call_args.kwargs
    assert kwargs["address"] == address
    assert kwargs["runtime_env"]["env_vars"]["KEEP_ME"] == "yes"
    assert kwargs["runtime_env"]["env_vars"]["WANDB_API_KEY"] == wandb_secret
    assert kwargs["runtime_env"]["env_vars"]["HF_TOKEN"] == hf_secret

    logged = capsys.readouterr().out
    assert wandb_secret not in logged
    assert hf_secret not in logged
    assert logged.count(main_dapo._REDACTED) >= 2


def test_driver_environment_is_the_only_credential_source(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("HF_TOKEN", "driver-hf-token")
    config = OmegaConf.create(
        {
            "ray_kwargs": {
                "ray_init": {
                    "runtime_env": {
                        "env_vars": {
                            "WANDB_API_KEY": "unsafe-config-wandb-token",
                            "HF_TOKEN": "unsafe-config-hf-token",
                        }
                    }
                }
            }
        }
    )

    kwargs = main_dapo._build_dapo_ray_init_kwargs(config)

    env_vars = kwargs["runtime_env"]["env_vars"]
    assert "WANDB_API_KEY" not in env_vars
    assert env_vars["HF_TOKEN"] == "driver-hf-token"


def test_redaction_scrubs_secret_keys_and_interpolated_values():
    environ = {
        "WANDB_API_KEY": "wandb-sensitive",
        "HF_TOKEN": "hf-sensitive",
    }
    logged = main_dapo._redact_for_logging(
        {
            "runtime_env": {"env_vars": dict(environ)},
            "unrelated": "prefix-wandb-sensitive-suffix",
            "nested": ["hf-sensitive"],
        },
        environ=environ,
    )

    assert logged["runtime_env"]["env_vars"] == {
        "WANDB_API_KEY": main_dapo._REDACTED,
        "HF_TOKEN": main_dapo._REDACTED,
    }
    assert logged["unrelated"] == f"prefix-{main_dapo._REDACTED}-suffix"
    assert logged["nested"] == [main_dapo._REDACTED]


def test_null_runtime_env_still_receives_driver_secrets(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "driver-wandb-token")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    config = OmegaConf.create(
        {
            "ray_kwargs": {
                "ray_init": {
                    "address": "local",
                    "runtime_env": None,
                }
            }
        }
    )

    kwargs = main_dapo._build_dapo_ray_init_kwargs(config)

    assert kwargs["address"] == "local"
    assert kwargs["runtime_env"]["env_vars"]["WANDB_API_KEY"] == "driver-wandb-token"
    assert "HF_TOKEN" not in kwargs["runtime_env"]["env_vars"]


def test_transfer_queue_enable_is_preserved_when_dapo_initializes_ray(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    config = OmegaConf.create(
        {
            "ray_kwargs": {"ray_init": {"runtime_env": {"env_vars": {"KEEP_ME": "yes"}}}},
            "transfer_queue": {"enable": True},
        }
    )

    kwargs = main_dapo._build_dapo_ray_init_kwargs(config)

    assert kwargs["runtime_env"]["env_vars"]["KEEP_ME"] == "yes"
    assert kwargs["runtime_env"]["env_vars"]["TRANSFER_QUEUE_ENABLE"] == "1"


def test_fewshot_launcher_does_not_put_credentials_in_hydra_overrides():
    launcher = (PROJECT_ROOT / "rl-distill-scripts" / "gemma3_pt_fewshot_math_rl.sh").read_text()

    assert "runtime_env.env_vars.WANDB_API_KEY" not in launcher
    assert "runtime_env.env_vars.HF_TOKEN" not in launcher
