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

"""Off-policy distillation entry point.

Uses verl's SFT trainer with:
  - DistillSFTDataset: loads teacher_log_probs from parquet
  - forward_kl_loss:   loss = teacher_log_prob - student_log_prob per token
  - HFPusher hook:     on each checkpoint save, async-push the HF-format
                       directory to HF Hub (if trainer.hf_push.enable=True)
"""

import os
from functools import partial

import hydra

# hf_push.py lives alongside this script in rl-distill-scripts/.
from hf_push import HFPusher  # noqa: E402

from verl.trainer.sft_trainer import SFTTrainer
from verl.utils.device import auto_set_device
from verl.workers.engine_workers import TrainingWorker, TrainingWorkerConfig


class DistillSFTTrainer(SFTTrainer):
    _hf_pusher = None

    def _build_engine(self):
        from forward_kl_loss import forward_kl_loss

        self.loss_fn = partial(forward_kl_loss, config=None)

        config = TrainingWorkerConfig(
            model_type="language_model",
            model_config=self.model_config,
            engine_config=self.engine_config,
            optimizer_config=self.optimizer_config,
            checkpoint_config=self.checkpoint_config,
            profiler_config=self.profiler_config,
        )

        self.training_client = TrainingWorker(config=config)
        self.training_client.set_loss_fn(loss_fn=self.loss_fn)
        self.engine = self.training_client.engine

    def _build_dataloader(self):
        super()._build_dataloader()
        # SFTTrainer constructs DistributedSampler without a `seed=` kwarg, so
        # the base shuffle seed is PyTorch's default 0 regardless of
        # `trainer.seed`. Force it to `trainer.seed` (default 42) so the
        # epoch-to-epoch shuffle order is reproducible at the cadence you
        # expect. `set_epoch(epoch)` is already called by the parent fit()
        # loop, so within-epoch order will be seed + epoch as intended.
        import random

        import numpy as np
        import torch

        seed = int(getattr(self.config.trainer, "seed", 42))
        self.train_sampler.seed = seed
        if getattr(self, "val_sampler", None) is not None:
            self.val_sampler.seed = seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)
        print(f"[DistillSFT] DistributedSampler + torch/numpy/random seeded with {seed}", flush=True)

    def _build_ckpt_handler(self):
        super()._build_ckpt_handler()
        self._init_hf_pusher()

    def _init_hf_pusher(self):
        import torch.distributed as dist

        cfg = self.config.trainer.get("hf_push", None)
        if cfg is None or not cfg.get("enable", False):
            self._hf_pusher = None
            return
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            self._hf_pusher = None
            return
        if not cfg.get("repo_id"):
            print("[HFPusher] hf_push.enable=True but hf_push.repo_id is empty; disabling.", flush=True)
            self._hf_pusher = None
            return
        self._hf_pusher = HFPusher(
            repo_id=cfg["repo_id"],
            private=bool(cfg.get("private", False)),
            max_to_keep=cfg.get("max_to_keep", None),
        )
        # Wrap the checkpoint handler so each successful save fires an async push.
        _orig_save = self.ckpt_handler.save_checkpoint
        trainer = self

        def _save_and_push(step):
            _orig_save(step=step)
            trainer._maybe_push_hf(step)

        self.ckpt_handler.save_checkpoint = _save_and_push
        print(
            f"[HFPusher] enabled; each save -> {cfg['repo_id']} "
            f"(private={cfg.get('private', False)}, "
            f"delete_local_after={cfg.get('delete_local_after', False)})",
            flush=True,
        )

    def _maybe_push_hf(self, step: int):
        if self._hf_pusher is None:
            return
        # SFT layout (no "actor/" subdir — that's PPO-only).
        hf_dir = os.path.join(
            self.config.trainer.default_local_dir,
            f"global_step_{step}",
            "huggingface",
        )
        if not os.path.isdir(hf_dir):
            print(
                f"[HFPusher] skip step {step}: {hf_dir} missing (ensure checkpoint.save_contents includes 'hf_model')",
                flush=True,
            )
            return
        cfg = self.config.trainer.hf_push
        self._hf_pusher.push_async(
            local_dir=hf_dir,
            step=step,
            delete_local_after=bool(cfg.get("delete_local_after", False)),
        )

    def fit(self):
        try:
            super().fit()
        finally:
            if getattr(self, "_hf_pusher", None) is not None:
                print("[HFPusher] waiting for pending uploads before exit...", flush=True)
                self._hf_pusher.wait(timeout=1800)


@hydra.main(config_path="config", config_name="distill_offpolicy", version_base=None)
def main(config):
    # SFTTrainer.__init__ calls torch.distributed.get_rank() — need the process
    # group up first. This mirrors verl's own `run_sft()` helper.
    from verl.utils.distributed import initialize_global_process_group

    initialize_global_process_group()
    auto_set_device(config)
    trainer = DistillSFTTrainer(config=config)
    trainer.fit()


if __name__ == "__main__":
    main()
