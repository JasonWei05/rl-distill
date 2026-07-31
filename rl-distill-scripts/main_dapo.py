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
"""
DAPO training entry point. Reuses the standard PPO TaskRunner and run_ppo,
only replacing the trainer class with RayDAPOTrainer.
"""

import os
import socket
from collections.abc import Mapping
from copy import deepcopy
from pprint import pprint

import hydra
import ray
from omegaconf import OmegaConf

from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.trainer.main_ppo import (
    TaskRunner,
    create_rl_dataset,
    create_rl_sampler,
    get_ppo_ray_runtime_env,
    run_ppo,
)
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device

_RAY_SECRET_ENV_VARS = ("WANDB_API_KEY", "HF_TOKEN")
_REDACTED = "<redacted>"


def _to_plain_container(value):
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return deepcopy(value)


def _redact_for_logging(value, environ=None):
    """Return a log-safe copy with Ray credentials removed.

    Secret keys are always redacted. Secret values are also scrubbed wherever
    they occur so an interpolation or a composite config value cannot leak a
    credential under an unrelated key.
    """
    environ = os.environ if environ is None else environ
    secret_values = tuple(secret_value for key in _RAY_SECRET_ENV_VARS if (secret_value := environ.get(key)))

    def redact(item, key=None):
        if key is not None and str(key).upper() in _RAY_SECRET_ENV_VARS:
            return _REDACTED
        if isinstance(item, Mapping):
            return {map_key: redact(map_value, map_key) for map_key, map_value in item.items()}
        if isinstance(item, list):
            return [redact(element) for element in item]
        if isinstance(item, tuple):
            return tuple(redact(element) for element in item)
        if isinstance(item, str):
            for secret_value in secret_values:
                item = item.replace(secret_value, _REDACTED)
        return item

    return redact(_to_plain_container(value))


def _build_dapo_ray_init_kwargs(config, environ=None):
    """Build Ray init kwargs, sourcing credentials only from the driver env."""
    environ = os.environ if environ is None else environ
    configured_ray_init = config.ray_kwargs.get("ray_init", {})
    ray_init_kwargs = _to_plain_container(configured_ray_init) or {}
    if not isinstance(ray_init_kwargs, Mapping):
        raise TypeError("ray_kwargs.ray_init must be a mapping")
    ray_init_kwargs = dict(ray_init_kwargs)

    configured_runtime_env = ray_init_kwargs.pop("runtime_env", {}) or {}
    if not isinstance(configured_runtime_env, Mapping):
        raise TypeError("ray_kwargs.ray_init.runtime_env must be a mapping or null")

    runtime_env = OmegaConf.to_container(
        OmegaConf.merge(get_ppo_ray_runtime_env(), configured_runtime_env),
        resolve=True,
    )
    env_vars = dict(runtime_env.get("env_vars") or {})

    # ``run_ppo`` normally injects this before it initializes Ray. DAPO now
    # initializes Ray itself so credentials can come from the driver environment
    # without entering Hydra's resolved config, and must preserve that standard
    # transfer-queue behavior explicitly.
    transfer_queue = config.get("transfer_queue", None)
    if transfer_queue is not None and bool(transfer_queue.get("enable", False)):
        env_vars["TRANSFER_QUEUE_ENABLE"] = "1"

    # Credentials in Hydra config are deliberately ignored. This keeps the
    # command line, resolved config, and Hydra output files credential-free.
    for env_var in _RAY_SECRET_ENV_VARS:
        env_vars.pop(env_var, None)
        if secret_value := environ.get(env_var):
            env_vars[env_var] = secret_value

    runtime_env["env_vars"] = env_vars
    ray_init_kwargs["runtime_env"] = runtime_env
    return ray_init_kwargs


def _initialize_dapo_ray(config):
    """Initialize Ray with DAPO's credential-safe job runtime environment."""
    if ray.is_initialized():
        return
    ray_init_kwargs = _build_dapo_ray_init_kwargs(config)
    print("ray init kwargs:")
    pprint(_redact_for_logging(ray_init_kwargs))
    ray.init(**ray_init_kwargs)


def _materialize_custom_chat_template(config):
    template = config.actor_rollout_ref.model.get("custom_chat_template", None)
    if not isinstance(template, str):
        return

    path = None
    if template.startswith("@"):
        path = template[1:]
    elif template.startswith("file://"):
        path = template[len("file://") :]

    if path:
        path = os.path.expandvars(os.path.expanduser(path))
        with open(path, encoding="utf-8") as f:
            config.actor_rollout_ref.model.custom_chat_template = f.read()


def _apply_custom_chat_template(tokenizer, processor, template):
    if not template:
        return
    tokenizer.chat_template = template
    if processor is not None:
        processor.chat_template = template
        if getattr(processor, "tokenizer", None) is not None:
            processor.tokenizer.chat_template = template


class DAPOTaskRunner(TaskRunner):
    """TaskRunner that uses RayDAPOTrainer instead of RayPPOTrainer."""

    def run(self, config):
        from verl.utils.fs import copy_to_local

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        _materialize_custom_chat_template(config)
        pprint(_redact_for_logging(config))
        OmegaConf.resolve(config)

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)
        self.add_reward_model_resource_pool(config)
        self.add_teacher_model_resource_pool(config)
        self.add_ref_policy_worker(config, actor_rollout_cls)

        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(config),
            use_critic=need_critic(config),
        )

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )

        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)
        _apply_custom_chat_template(
            tokenizer,
            processor,
            config.actor_rollout_ref.model.get("custom_chat_template", None),
        )

        resource_pool_manager = self.init_resource_pool_mgr(config)

        from verl.utils.dataset.rl_dataset import collate_fn

        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            is_train=True,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            is_train=False,
            max_samples=config.data.get("val_max_samples", -1),
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)

        # Only difference from base TaskRunner: use RayDAPOTrainer
        from .dapo_ray_trainer import RayDAPOTrainer

        trainer = RayDAPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        trainer.init_workers()
        trainer.fit()


@hydra.main(config_path="config", config_name="dapo_trainer", version_base=None)
def main(config):
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)
    _initialize_dapo_ray(config)
    if os.environ.get("DAPO_LOCAL_TASK_RUNNER") == "1":
        DAPOTaskRunner().run(config)
        return
    run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(DAPOTaskRunner))


if __name__ == "__main__":
    main()
