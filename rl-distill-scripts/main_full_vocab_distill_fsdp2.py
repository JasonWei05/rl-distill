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

"""FSDP2 full-vocabulary off-policy distillation entry point."""

import hashlib
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path

import hydra
import torch
from hf_push import HFPusher, wait_for_hf_pusher

from verl.trainer.sft_trainer import SFTTrainer
from verl.utils.device import auto_set_device
from verl.workers.engine_workers import TrainingWorker, TrainingWorkerConfig


class FullVocabDistillTrainer(SFTTrainer):
    _hf_pusher = None

    def __init__(self, config):
        self._restore_remote_checkpoint(config)
        super().__init__(config=config)

    @staticmethod
    def _remote_checkpoint_config(config):
        cfg = config.trainer.get("remote_checkpoint", None)
        if cfg is None or not bool(cfg.get("enable", False)):
            return None
        s3_uri = cfg.get("s3_uri", None)
        if not isinstance(s3_uri, str) or not s3_uri.startswith("s3://"):
            raise ValueError("trainer.remote_checkpoint.s3_uri must be an s3:// URI when enabled")
        return cfg

    def _restore_remote_checkpoint(self, config):
        import torch.distributed as dist

        cfg = self._remote_checkpoint_config(config)
        if cfg is None:
            return
        from full_checkpoint_s3 import restore_latest_checkpoint

        result = [None]
        if dist.get_rank() == 0:
            try:
                restored_step = restore_latest_checkpoint(
                    Path(config.trainer.default_local_dir),
                    str(cfg.s3_uri),
                )
                result[0] = {"ok": True, "step": restored_step}
            except Exception as error:
                result[0] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
        dist.broadcast_object_list(result, src=0)
        if not result[0]["ok"]:
            raise RuntimeError(f"remote checkpoint restore failed: {result[0]['error']}")
        if dist.get_rank() == 0:
            print(
                f"[FullCheckpointS3] startup restore complete; step={result[0]['step']}",
                flush=True,
            )

    def _build_engine(self):
        from full_vocab_kl_loss import FullVocabKLLoss

        teacher_cfg = self.config.teacher_model
        if bool(teacher_cfg.get("precomputed_topk", False)):
            configured_top_k = int(teacher_cfg.get("top_k", 0))
            stored_top_k = int(self.config.data.get("teacher_topk_width", 0))
            if configured_top_k <= 0 or configured_top_k != stored_top_k:
                raise ValueError(
                    "Precomputed top-k width must be positive and identical in teacher_model.top_k and "
                    f"data.teacher_topk_width; got {configured_top_k=} and {stored_top_k=}"
                )
        self.loss_fn = FullVocabKLLoss(
            teacher_model_path=teacher_cfg.get("path", None),
            temperature=float(teacher_cfg.get("temperature", 1.0)),
            chunk_size=int(teacher_cfg.get("chunk_size", 64)),
            top_k=int(teacher_cfg.get("top_k", 0)),
            teacher_dtype=str(teacher_cfg.get("dtype", "bfloat16")),
            trust_remote_code=bool(teacher_cfg.get("trust_remote_code", False)),
            attn_implementation=str(teacher_cfg.get("attn_implementation", "flash_attention_2")),
            use_teacher_hidden_states=bool(teacher_cfg.get("use_hidden_states", True)),
            precomputed_topk=bool(teacher_cfg.get("precomputed_topk", False)),
            clamp_min_kl=bool(teacher_cfg.get("clamp_min_kl", False)),
            checkpoint_student_chunks=bool(teacher_cfg.get("checkpoint_student_chunks", True)),
        )

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
        import random

        import numpy as np

        seed = int(getattr(self.config.trainer, "seed", 42))
        self.train_sampler.seed = seed
        if getattr(self, "val_sampler", None) is not None:
            self.val_sampler.seed = seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)
        print(f"[FullVocabDistill] DistributedSampler + torch/numpy/random seeded with {seed}", flush=True)

    def _load_validated_file_list(self):
        manifest_path = self.config.data.get("validated_file_list", None)
        if not manifest_path:
            return
        expected_dataset_sha256 = self.config.data.get("validated_dataset_index_sha256", None)
        if not isinstance(expected_dataset_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_dataset_sha256):
            raise ValueError("validated_dataset_index_sha256 must be a SHA256 digest")
        expected_file_list_sha256 = self.config.data.get("validated_file_list_sha256", None)
        if not isinstance(expected_file_list_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", expected_file_list_sha256
        ):
            raise ValueError("validated_file_list_sha256 must be a SHA256 digest")
        path = Path(str(manifest_path)).expanduser().resolve(strict=True)
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or value.get("schema_version") != "gemma4-validated-file-list-v1":
            raise ValueError(f"unsupported validated file-list manifest: {path}")
        claimed = value.get("file_list_sha256")
        if not isinstance(claimed, str) or not re.fullmatch(r"[0-9a-f]{64}", claimed):
            raise ValueError("validated file-list manifest has no valid self-hash")
        unhashed = dict(value)
        del unhashed["file_list_sha256"]
        canonical = json.dumps(unhashed, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        actual = hashlib.sha256(canonical).hexdigest()
        if actual != claimed:
            raise ValueError(f"validated file-list manifest self-hash mismatch: {actual} != {claimed}")
        if claimed != expected_file_list_sha256:
            raise ValueError("validated file-list manifest does not match the launcher-bound file-list identity")
        if value.get("dataset_index_sha256") != expected_dataset_sha256:
            raise ValueError("validated file-list manifest does not match the preflight dataset identity")
        train_files = value.get("train_files")
        validation_files = value.get("validation_files")
        if (
            not isinstance(train_files, list)
            or not train_files
            or not all(isinstance(item, str) for item in train_files)
        ):
            raise ValueError("validated train file list is malformed")
        if (
            not isinstance(validation_files, list)
            or not validation_files
            or not all(isinstance(item, str) for item in validation_files)
        ):
            raise ValueError("validated validation file list is malformed")
        self.config.data.train_files = train_files
        self.config.data.val_files = validation_files
        print(
            f"[FullVocabDistill] loaded validated file list: train={len(train_files)} "
            f"validation={len(validation_files)} dataset={expected_dataset_sha256}",
            flush=True,
        )

    def _build_dataset(self):
        self._load_validated_file_list()
        super()._build_dataset()

    def _build_ckpt_handler(self):
        super()._build_ckpt_handler()
        self._init_hf_pusher()
        self._init_remote_checkpoint_persistence()

    def _init_remote_checkpoint_persistence(self):
        import torch.distributed as dist

        cfg = self._remote_checkpoint_config(self.config)
        if cfg is None:
            return
        from full_checkpoint_s3 import upload_checkpoint

        original_save = self.ckpt_handler.save_checkpoint
        checkpoint_root = Path(self.config.trainer.default_local_dir)
        s3_uri = str(cfg.s3_uri)

        def save_and_upload(step):
            original_save(step=step)
            result = [None]
            if dist.get_rank() == 0:
                try:
                    upload_checkpoint(checkpoint_root, int(step), s3_uri)
                    result[0] = {"ok": True}
                except Exception as error:
                    result[0] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
            dist.broadcast_object_list(result, src=0)
            if not result[0]["ok"]:
                raise RuntimeError(f"remote checkpoint upload failed at step {step}: {result[0]['error']}")

        self.ckpt_handler.save_checkpoint = save_and_upload
        if dist.get_rank() == 0:
            print(f"[FullCheckpointS3] enabled: {s3_uri}", flush=True)

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

        orig_save = self.ckpt_handler.save_checkpoint
        trainer = self

        def save_and_push(step):
            orig_save(step=step)
            trainer._maybe_push_hf(step)

        self.ckpt_handler.save_checkpoint = save_and_push
        print(
            f"[HFPusher] enabled; each save -> {cfg['repo_id']} "
            f"(private={cfg.get('private', False)}, delete_local_after={cfg.get('delete_local_after', False)})",
            flush=True,
        )

    def _maybe_push_hf(self, step: int):
        if self._hf_pusher is None:
            return
        hf_dir = os.path.join(
            self.config.trainer.default_local_dir,
            f"global_step_{step}",
            "huggingface",
        )
        if not os.path.isdir(hf_dir):
            raise FileNotFoundError(
                f"HF push is enabled, but the required step-{step} checkpoint directory is missing: {hf_dir}"
            )
        cfg = self.config.trainer.hf_push
        self._hf_pusher.push_async(
            local_dir=hf_dir,
            step=step,
            delete_local_after=bool(cfg.get("delete_local_after", False)),
        )

    def _write_completion_receipt(self):
        """Record completion only after the final checkpoint and uploads succeed."""
        import torch.distributed as dist

        result = [None]
        if dist.get_rank() == 0:
            try:
                checkpoint_root = Path(self.config.trainer.default_local_dir)
                final_step = int(self.total_training_steps)
                tracker = checkpoint_root / "latest_checkpointed_iteration.txt"
                try:
                    checkpoint_step = int(tracker.read_text(encoding="utf-8").strip())
                except (FileNotFoundError, ValueError) as error:
                    raise RuntimeError(f"cannot prove final checkpoint completion from {tracker}") from error
                if checkpoint_step != final_step:
                    raise RuntimeError(
                        f"training returned before its final checkpoint: tracker={checkpoint_step}, expected={final_step}"
                    )
                checkpoint_dir = checkpoint_root / f"global_step_{final_step}"
                if not checkpoint_dir.is_dir():
                    raise RuntimeError(f"final checkpoint directory is missing: {checkpoint_dir}")

                hf_cfg = self.config.trainer.get("hf_push", None)
                hf_push_enabled = bool(hf_cfg is not None and hf_cfg.get("enable", False))
                if hf_push_enabled and self._hf_pusher is None:
                    raise RuntimeError("HF push was required but no rank-0 HFPusher was initialized")

                remote_cfg = self._remote_checkpoint_config(self.config)
                payload = {
                    "checkpoint_step": final_step,
                    "completed_at": datetime.now(UTC).isoformat(),
                    "hf_push_enabled": hf_push_enabled,
                    "hf_repo": hf_cfg.get("repo_id") if hf_push_enabled else None,
                    "remote_checkpoint_s3_uri": str(remote_cfg.s3_uri) if remote_cfg is not None else None,
                    "wandb_run_id": os.environ.get("WANDB_RUN_ID"),
                    "cudnn_sdpa": os.environ.get("VERL_GEMMA4_CUDNN_SDPA"),
                    "eval_cudnn_sdpa": os.environ.get("VERL_GEMMA4_EVAL_CUDNN_SDPA"),
                }
                receipt = checkpoint_root / "run_complete.json"
                temporary = receipt.with_name(f".{receipt.name}.tmp.{os.getpid()}")
                temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                os.replace(temporary, receipt)
                print(f"[FullVocabDistill] wrote verified completion receipt: {receipt}", flush=True)
                if remote_cfg is not None:
                    from full_checkpoint_s3 import upload_completion_receipt

                    upload_completion_receipt(checkpoint_root, str(remote_cfg.s3_uri))
                result[0] = {"ok": True}
            except Exception as error:
                result[0] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
        dist.broadcast_object_list(result, src=0)
        if not result[0]["ok"]:
            raise RuntimeError(f"run completion receipt failed: {result[0]['error']}")

    def fit(self):
        try:
            super().fit()
        finally:
            if getattr(self, "_hf_pusher", None) is not None:
                print("[HFPusher] waiting for pending uploads before exit...", flush=True)
                wait_for_hf_pusher(self._hf_pusher, timeout=3600)
        self._write_completion_receipt()


@hydra.main(config_path="config", config_name="full_vocab_distill_fsdp2", version_base=None)
def main(config):
    from verl.utils.distributed import initialize_global_process_group

    initialize_global_process_group()
    auto_set_device(config)
    trainer = FullVocabDistillTrainer(config=config)
    trainer.fit()


if __name__ == "__main__":
    main()
