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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import importlib.util
import json
import os
import random
import traceback
import uuid
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from pprint import pprint

import numpy as np
import torch
from tqdm import tqdm

from dapo.hf_push import HFPusher, wait_for_hf_pusher
from dapo.validation_early_stopping import EARLY_STOPPING_STATE_FILENAME, ValidationEarlyStopping
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.reward import extract_reward
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.metric import reduce_metrics
from verl.utils.profiler import marked_timer
from verl.utils.rollout_skip import RolloutSkip

_REWARD_GROUP_KINDS = ("mixed", "all_zero", "all_one")


def count_binary_reward_groups(uids, accuracies) -> dict[str, int]:
    """Count prompt groups by their raw binary-correctness composition."""
    grouped_accuracies = defaultdict(list)
    for uid, accuracy in zip(uids, accuracies, strict=True):
        accuracy_array = np.asarray(accuracy)
        if accuracy_array.size != 1:
            raise ValueError(f"Expected scalar accuracy for uid={uid!r}, got shape={accuracy_array.shape}")
        accuracy_value = accuracy_array.item()
        if accuracy_value not in (0, 1, False, True):
            raise ValueError(f"Expected binary accuracy for uid={uid!r}, got {accuracy_value!r}")
        grouped_accuracies[uid].append(bool(accuracy_value))

    counts = {"total": len(grouped_accuracies), **{kind: 0 for kind in _REWARD_GROUP_KINDS}}
    for group_accuracies in grouped_accuracies.values():
        if all(group_accuracies):
            counts["all_one"] += 1
        elif any(group_accuracies):
            counts["mixed"] += 1
        else:
            counts["all_zero"] += 1
    return counts


def binary_reward_group_metrics(counts: dict[str, int]) -> dict[str, int | float]:
    """Build count and 0-100 percentage metrics from accumulated group counts."""
    total = int(counts.get("total", 0))
    if total <= 0:
        return {}

    metrics: dict[str, int | float] = {"train/reward_group/total/count": total}
    for kind in _REWARD_GROUP_KINDS:
        count = int(counts.get(kind, 0))
        metrics[f"train/reward_group/{kind}/count"] = count
        metrics[f"train/reward_group/{kind}/percent"] = 100.0 * count / total
    return metrics


class RayDAPOTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    @staticmethod
    def _full_checkpoint_helper():
        # This trainer is copied from ``rl-distill-scripts`` into ``dapo`` in
        # the production image. Resolve the helper from the repository root so
        # both the source-tree and deployed locations work.
        helper_path = Path(__file__).resolve().parents[1] / "rl-distill-scripts" / "full_checkpoint_s3.py"
        spec = importlib.util.spec_from_file_location("rl_distill_full_checkpoint_s3", helper_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load full-checkpoint helper from {helper_path}")
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        return helper

    @staticmethod
    def _rolling_checkpoint_frequency() -> int:
        enabled = os.environ.get("ROLLING_CHECKPOINT_ENABLED", "False").strip().lower()
        if enabled in {"", "0", "false", "no", "off"}:
            return 0
        if enabled not in {"1", "true", "yes", "on"}:
            raise ValueError(f"invalid ROLLING_CHECKPOINT_ENABLED value: {enabled!r}")
        frequency = int(os.environ.get("ROLLING_CHECKPOINT_FREQ", "1"))
        if frequency < 1:
            raise ValueError(f"ROLLING_CHECKPOINT_FREQ must be positive when enabled, got {frequency}")
        if not os.environ.get("FULL_CHECKPOINT_S3_URI", "").strip():
            raise ValueError("ROLLING_CHECKPOINT_ENABLED requires FULL_CHECKPOINT_S3_URI")
        return frequency

    def _should_save_rolling_checkpoint(self) -> bool:
        frequency = self._rolling_checkpoint_frequency()
        return frequency > 0 and self.global_steps % frequency == 0

    def _save_checkpoint(self, *, permanent: bool = True):
        """Save locally, then publish a permanent or rolling resumable checkpoint."""
        early_stopping = getattr(self, "_validation_early_stopping", None)
        if early_stopping is not None:
            checkpoint_dir = Path(self.config.trainer.default_local_dir) / f"global_step_{self.global_steps}"
            state_path = early_stopping.save(checkpoint_dir)
            print(
                "EARLY_STOPPING_STATE_SAVED "
                f"step={self.global_steps} best={early_stopping.best_score} "
                f"best_step={early_stopping.best_step} "
                f"misses={early_stopping.non_improving_rounds} path={state_path}",
                flush=True,
            )
        # Rolling checkpoints need the sharded model, Adam, scheduler/RNG, and
        # dataloader cursor, but not a second consolidated HF copy every step.
        # Permanent checkpoints retain the existing HF-export behavior.
        super()._save_checkpoint(save_hf_model=permanent)
        s3_uri = os.environ.get("FULL_CHECKPOINT_S3_URI", "").strip()
        if not s3_uri:
            return
        helper = self._full_checkpoint_helper()
        checkpoint_root = Path(self.config.trainer.default_local_dir)
        if permanent:
            helper.upload_checkpoint(checkpoint_root, self.global_steps, s3_uri)
            if self._rolling_checkpoint_frequency() > 0:
                helper.retire_rolling_checkpoint(s3_uri, self.global_steps)
        else:
            helper.upload_rolling_checkpoint(checkpoint_root, self.global_steps, s3_uri)

    def _resume_checkpoint_dir(self) -> Path:
        if self.global_steps <= 0:
            raise ValueError("a positive restored global step is required")
        if self.config.trainer.resume_mode == "resume_path":
            checkpoint_dir = Path(str(self.config.trainer.resume_from_path))
            if not checkpoint_dir.is_absolute():
                checkpoint_dir = Path.cwd() / checkpoint_dir
            return checkpoint_dir
        return Path(self.config.trainer.default_local_dir) / f"global_step_{self.global_steps}"

    @staticmethod
    def _early_stopping_migrate_patience_from() -> int | None:
        raw_value = os.environ.get("EARLY_STOPPING_MIGRATE_PATIENCE_FROM", "").strip()
        if not raw_value:
            return None
        value = int(raw_value)
        if value < 1:
            raise ValueError(f"EARLY_STOPPING_MIGRATE_PATIENCE_FROM must be a positive integer, got {raw_value!r}")
        return value

    def _restore_early_stopping_state(self, early_stopping: ValidationEarlyStopping | None) -> bool:
        """Restore stopping history and report whether this is a logical resume."""
        self._validation_early_stopping = early_stopping
        if early_stopping is None:
            return False

        encoded_migration = os.environ.get("EARLY_STOPPING_LEGACY_STATE_B64", "").strip()
        if self.global_steps <= 0:
            if not encoded_migration:
                return False
            source_step = early_stopping.load_legacy_history(
                encoded_payload=encoded_migration,
                current_step=0,
                model=os.environ.get("GEMMA4_MODEL", ""),
                difficulty=os.environ.get("DIFFICULTY_DATASET", ""),
            )
            print(
                "EARLY_STOPPING_HISTORY_SEEDED "
                f"source_step={source_step} best={early_stopping.best_score} "
                f"best_step={early_stopping.best_step} misses={early_stopping.non_improving_rounds}",
                flush=True,
            )
            return True

        checkpoint_dir = self._resume_checkpoint_dir()
        state_path = checkpoint_dir / EARLY_STOPPING_STATE_FILENAME
        if not state_path.is_file():
            if not encoded_migration:
                raise FileNotFoundError(
                    "resume checkpoint is missing validation early-stopping state; refusing to reset the "
                    "all-time best or patience counter silently. Resume from a checkpoint produced by the "
                    f"updated trainer or explicitly migrate the legacy checkpoint: {state_path}"
                )
            early_stopping.migrate_legacy_state(
                encoded_payload=encoded_migration,
                checkpoint_dir=checkpoint_dir,
                checkpoint_step=int(self.global_steps),
                model=os.environ.get("GEMMA4_MODEL", ""),
                difficulty=os.environ.get("DIFFICULTY_DATASET", ""),
            )
            print(
                "EARLY_STOPPING_STATE_MIGRATED "
                f"checkpoint_step={self.global_steps} best={early_stopping.best_score} "
                f"best_step={early_stopping.best_step} misses={early_stopping.non_improving_rounds} "
                f"path={state_path}",
                flush=True,
            )
        state_path, patience_migrated = early_stopping.load(
            checkpoint_dir,
            migrate_patience_from=self._early_stopping_migrate_patience_from(),
        )
        if patience_migrated:
            print(
                "EARLY_STOPPING_PATIENCE_MIGRATED "
                f"checkpoint_step={self.global_steps} active_patience={early_stopping.patience} "
                f"misses={early_stopping.non_improving_rounds} "
                f"triggered={early_stopping.last_observed_triggered} path={state_path}",
                flush=True,
            )
        if encoded_migration:
            historical = ValidationEarlyStopping(
                metric=early_stopping.metric,
                patience=early_stopping.patience,
                mode=early_stopping.mode,
                min_delta=early_stopping.min_delta,
                include_initial_validation=early_stopping.include_initial_validation,
            )
            source_step = historical.load_legacy_history(
                encoded_payload=encoded_migration,
                current_step=int(self.global_steps),
                model=os.environ.get("GEMMA4_MODEL", ""),
                difficulty=os.environ.get("DIFFICULTY_DATASET", ""),
            )
            if early_stopping.merge_historical_state(historical):
                early_stopping.save(checkpoint_dir)
                print(
                    "EARLY_STOPPING_HISTORY_MERGED "
                    f"checkpoint_step={self.global_steps} source_step={source_step} "
                    f"best={early_stopping.best_score} best_step={early_stopping.best_step} "
                    f"misses={early_stopping.non_improving_rounds} path={state_path}",
                    flush=True,
                )
        if early_stopping.last_observed_step is not None and early_stopping.last_observed_step > self.global_steps:
            raise ValueError(
                "early-stopping state is newer than the restored trainer checkpoint: "
                f"last_observed_step={early_stopping.last_observed_step} global_steps={self.global_steps}"
            )
        print(
            "EARLY_STOPPING_STATE_RESTORED "
            f"checkpoint_step={self.global_steps} best={early_stopping.best_score} "
            f"best_step={early_stopping.best_step} misses={early_stopping.non_improving_rounds} "
            f"last_observed_step={early_stopping.last_observed_step} path={state_path}",
            flush=True,
        )
        return True

    def _finalize_restored_early_stop(self, early_stopping: ValidationEarlyStopping | None) -> bool:
        """Finalize a checkpoint whose restored patience state is already terminal."""
        if early_stopping is None or not early_stopping.last_observed_triggered:
            return False
        print(
            "EARLY_STOP_ALREADY_TRIGGERED_ON_RESUME "
            f"step={self.global_steps} best={early_stopping.best_score} "
            f"best_step={early_stopping.best_step} "
            f"misses={early_stopping.non_improving_rounds} metric={early_stopping.metric}",
            flush=True,
        )
        self._save_checkpoint(permanent=True)
        self._write_run_outcome(early_stopping=early_stopping, stop_reason="early_stopping")
        return True

    def _should_run_initial_validation(self, *, restored_early_stopping_history: bool) -> bool:
        if not self.config.trainer.get("val_before_train", True):
            return False
        if self.config.trainer.get("val_only", False):
            return True
        return self.global_steps <= 0 and not restored_early_stopping_history

    def _fast_forward_fresh_dataloader(self) -> None:
        """Advance a weight-only continuation to the requested historical data cursor.

        This is intentionally separate from true checkpoint resume. It is used only
        when an old run retained an HF model export but lost its optimizer checkpoint.
        """
        skip_batches = int(os.environ.get("DATALOADER_SKIP_BATCHES", "0"))
        if skip_batches <= 0:
            return
        if self.global_steps != 0:
            raise ValueError(
                "DATALOADER_SKIP_BATCHES is valid only for a fresh optimizer continuation; "
                f"a checkpoint already restored global_steps={self.global_steps}"
            )

        print(f"Fast-forwarding fresh StatefulDataLoader by {skip_batches} historical batches", flush=True)
        iterator = iter(self.train_dataloader)
        for _ in range(skip_batches):
            try:
                next(iterator)
            except StopIteration:
                iterator = iter(self.train_dataloader)
                next(iterator)
        dataloader_state = self.train_dataloader.state_dict()
        self.train_dataloader.load_state_dict(dataloader_state)
        print(f"Fast-forwarded StatefulDataLoader by {skip_batches} historical batches", flush=True)

    def _write_run_outcome(
        self,
        *,
        early_stopping: ValidationEarlyStopping | None,
        stop_reason: str,
    ) -> Path:
        """Persist the terminal step and all-time-best checkpoint selection.

        A preemption restores only the newest complete checkpoint locally, so
        recover an older all-time-best HF export from the full-checkpoint S3
        prefix before writing the final outcome when necessary.
        """

        if early_stopping is None or early_stopping.best_step is None:
            best_step = int(self.global_steps)
            best_score = None
            metric = None
        else:
            best_step = int(early_stopping.best_step)
            best_score = float(early_stopping.best_score)
            metric = early_stopping.metric

        checkpoint_root = Path(self.config.trainer.default_local_dir)
        if best_step > 0:
            best_hf_dir = checkpoint_root / f"global_step_{best_step}" / "actor" / "huggingface"
            s3_uri = os.environ.get("FULL_CHECKPOINT_S3_URI", "").strip()
            if not best_hf_dir.is_dir() and s3_uri:
                self._full_checkpoint_helper().restore_hf_export(checkpoint_root, s3_uri, best_step)
            if not best_hf_dir.is_dir():
                raise FileNotFoundError(
                    f"all-time-best HF checkpoint was not retained: best_step={best_step} expected={best_hf_dir}"
                )

        wandb_run_id = os.environ.get("WANDB_RUN_ID", "").strip()
        if not wandb_run_id:
            try:
                import wandb

                if wandb.run is not None:
                    wandb_run_id = str(wandb.run.id)
            except Exception:
                pass

        outcome = {
            "protocol": "gemma4_rl_run_outcome_v1",
            "status": "complete",
            "stop_reason": stop_reason,
            "final_step": int(self.global_steps),
            "best_step": best_step,
            "best_score": best_score,
            "early_stopping_metric": metric,
            "model": os.environ.get("GEMMA4_MODEL", ""),
            "difficulty": os.environ.get("DIFFICULTY_DATASET", ""),
            "seed": int(os.environ.get("DATA_SEED", "0")),
            "wandb_run_id": wandb_run_id,
            "experiment_name": str(self.config.trainer.experiment_name),
            "hf_repo": str(self.config.trainer.get("hf_push", {}).get("repo_id", "")),
        }
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        outcome_path = checkpoint_root / "run_outcome.json"
        temporary = outcome_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(outcome, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        temporary.replace(outcome_path)
        print(
            "RUN_OUTCOME_WRITTEN "
            f"reason={stop_reason} final_step={self.global_steps} best_step={best_step} "
            f"path={outcome_path}",
            flush=True,
        )
        return outcome_path

    def _maybe_push_to_hf(self, step: int):
        """If trainer.hf_push.enable is True, async-push the HF-format checkpoint to HF Hub.

        Expected config shape (all under `trainer.hf_push`):
            enable: bool
            repo_id: str                  # e.g. "JWei05/dapo-gemma3-27b-it"
            private: bool = True
            delete_local_after: bool = False
            max_to_keep: Optional[int] = None   # keep N most-recent step_* folders on hub
            freq: Optional[int] = None           # only upload every N steps
        """
        cfg = self.config.trainer.get("hf_push", None)
        if cfg is None or not cfg.get("enable", False):
            return
        upload_freq = cfg.get("freq", None)
        if upload_freq is None:
            upload_freq = cfg.get("upload_freq", None)
        if upload_freq is not None and int(upload_freq) > 0 and step % int(upload_freq) != 0:
            print(f"[HFPusher] skip step {step}: upload freq is {upload_freq}", flush=True)
            return
        # NOTE: we do NOT check isdir() here because in multi-node FSDP runs
        # rank-0 of the actor_rollout worker group may live on a *different* Ray
        # node than this driver. The files land on that node's local /tmp, not
        # the driver's. `push_cluster` broadcasts the upload task to every alive
        # node; whichever node holds the files performs the upload.
        hf_dir = os.path.join(
            self.config.trainer.default_local_dir,
            f"global_step_{step}",
            "actor",
            "huggingface",
        )
        if getattr(self, "_hf_pusher", None) is None:
            self._hf_pusher = HFPusher(
                repo_id=cfg["repo_id"],
                private=cfg.get("private", True),
                max_to_keep=cfg.get("max_to_keep", None),
            )
        self._hf_pusher.push_cluster(
            local_dir=hf_dir,
            step=step,
            delete_local_after=cfg.get("delete_local_after", False),
        )

    def _save_trajectories(self, batch: DataProto, step: int, sample_prob: float = 0.01):
        """Save a random sample of trajectories to disk for debugging/analysis."""
        traj_dir = os.path.join(self.config.trainer.default_local_dir, "trajectories")
        os.makedirs(traj_dir, exist_ok=True)

        for i in range(len(batch)):
            if random.random() > sample_prob:
                continue

            item = batch[i]
            prompt_ids = item.batch["prompts"]
            response_ids = item.batch["responses"]
            attention_mask = item.batch["attention_mask"]
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = attention_mask[:prompt_length].sum().item()
            valid_response_length = attention_mask[prompt_length:].sum().item()

            prompt_str = self.tokenizer.decode(prompt_ids[-valid_prompt_length:], skip_special_tokens=True)
            response_str = self.tokenizer.decode(response_ids[:valid_response_length], skip_special_tokens=True)

            reward = item.batch["token_level_scores"].sum().item()

            trajectory = {
                "step": step,
                "index": i,
                "prompt": prompt_str,
                "response": response_str,
                "reward": reward,
                "response_length": valid_response_length,
            }

            # Add extra info if available
            for key in ["acc", "data_source", "ground_truth"]:
                if key in item.non_tensor_batch:
                    val = item.non_tensor_batch[key]
                    if isinstance(val, np.generic):
                        val = val.item()
                    trajectory[key] = val
            if "reward_model" in item.non_tensor_batch:
                rm = item.non_tensor_batch["reward_model"]
                if isinstance(rm, dict) and "ground_truth" in rm:
                    trajectory["ground_truth"] = rm["ground_truth"]

            filename = f"step{step:06d}_idx{i:05d}.json"
            with open(os.path.join(traj_dir, filename), "w") as f:
                json.dump(trajectory, f, indent=2, ensure_ascii=False)

    def compute_kl_related_metrics(self, batch: DataProto, metrics: dict, timing_raw: dict):
        batch.batch["response_mask"] = compute_response_mask(batch)

        # recompute old_log_probs
        with marked_timer("old_log_prob", timing_raw, "blue"):
            old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
            old_log_prob_metrics = {
                "perf/mfu/actor_infer": old_log_prob_mfu,
            }
            if "entropys" in old_log_prob.batch:
                entropys = old_log_prob.batch["entropys"]
                response_masks = batch.batch["response_mask"]
                actor_config = self.config.actor_rollout_ref.actor
                entropy_agg = agg_loss(
                    loss_mat=entropys,
                    loss_mask=response_masks,
                    loss_agg_mode=actor_config.loss_agg_mode,
                    loss_scale_factor=actor_config.loss_scale_factor,
                )
                old_log_prob_metrics["actor/entropy"] = entropy_agg.detach().item()
                old_log_prob.batch.pop("entropys")
            metrics.update(old_log_prob_metrics)
            batch = batch.union(old_log_prob)

        if self.use_reference_policy:
            # compute reference log_prob
            with marked_timer("ref", timing_raw, "olive"):
                ref_log_prob = self._compute_ref_log_prob(batch)
                batch = batch.union(ref_log_prob)

        return batch

    def _log_train_generations_to_wandb(self, batch):
        """Upload a random sample of this step's train rollouts to wandb (fork addition).

        Mirrors the val-generations logging but as a FRESH per-step table (one row per trace)
        under 'train/generations', so it doesn't accumulate/re-upload a growing table every step.
        Gated on trainer.log_train_generations (N traces); no-op if 0 or wandb isn't active.
        """
        n = int(self.config.trainer.get("log_train_generations", 0) or 0)
        if n <= 0 or "wandb" not in self.config.trainer.logger:
            return
        import numpy as np

        import wandb

        if wandb.run is None:
            return
        inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
        outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
        scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
        rng = np.random.RandomState(self.global_steps)  # vary the sample each step
        idx = rng.permutation(len(inputs))[:n]
        table = wandb.Table(columns=["step", "input", "output", "score"])
        for i in idx:
            table.add_data(self.global_steps, inputs[i], outputs[i], scores[i])
        wandb.log({"train/generations": table}, step=self.global_steps)

    def fit(self):
        try:
            return self._fit_impl()
        finally:
            if getattr(self, "_hf_pusher", None) is not None:
                print("[HFPusher] waiting for pending uploads before exit...", flush=True)
                try:
                    wait_for_hf_pusher(self._hf_pusher, timeout=1800)
                except Exception as error:
                    hf_push_cfg = self.config.trainer.get("hf_push", {})
                    if bool(hf_push_cfg.get("required", True)):
                        raise
                    print(
                        f"[HFPusher] optional publication failed; preserving successful training outcome: {error}",
                        flush=True,
                    )
                    traceback.print_exception(type(error), error, error.__traceback__)

    def _fit_impl(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self.gen_steps = 0
        self.max_steps_duration = 0

        # load checkpoint before doing anything
        self._load_checkpoint()
        self._fast_forward_fresh_dataloader()

        early_stopping = ValidationEarlyStopping.from_trainer_config(self.config.trainer)
        if early_stopping is not None and self.config.trainer.test_freq <= 0:
            raise ValueError("validation early stopping requires trainer.test_freq > 0")
        restored_early_stopping_history = self._restore_early_stopping_state(early_stopping)
        if self._finalize_restored_early_stop(early_stopping):
            return
        self.checkpoint_manager.update_weights()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        initial_early_stop_triggered = False
        if self._should_run_initial_validation(restored_early_stopping_history=restored_early_stopping_history):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            if early_stopping is not None:
                if early_stopping.should_observe_initial_validation(step=self.global_steps):
                    initial_early_stop_triggered, initial_early_metrics = early_stopping.observe(
                        val_metrics, step=self.global_steps
                    )
                    val_metrics.update(initial_early_metrics)
                else:
                    print(
                        "EARLY_STOPPING_INITIAL_VALIDATION_IGNORED "
                        f"checkpoint_step={self.global_steps} "
                        f"last_observed_step={early_stopping.last_observed_step}",
                        flush=True,
                    )
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return
            if initial_early_stop_triggered:
                print(
                    "EARLY_STOP_TRIGGERED_ON_RESUME "
                    f"step={self.global_steps} best={early_stopping.best_score} "
                    f"best_step={early_stopping.best_step} "
                    f"misses={early_stopping.non_improving_rounds} metric={early_stopping.metric}",
                    flush=True,
                )
                self._save_checkpoint(permanent=True)
                self._write_run_outcome(early_stopping=early_stopping, stop_reason="early_stopping")
                return
        elif self.config.trainer.get("val_before_train", True):
            print(
                "INITIAL_VALIDATION_SKIPPED_ON_RESUME "
                f"checkpoint_step={self.global_steps} "
                f"restored_early_stopping_history={restored_early_stopping_history}",
                flush=True,
            )

        # A pod can be preempted after uploading its terminal checkpoint but
        # before the wrapper publishes the run outcome and best-HF marker. A
        # resumed terminal checkpoint must finalize idempotently, not execute
        # an unintended step beyond the configured maximum.
        if self.global_steps >= self.total_training_steps:
            print(
                "TRAINING_ALREADY_AT_MAX_STEPS_ON_RESUME "
                f"checkpoint_step={self.global_steps} total_training_steps={self.total_training_steps}",
                flush=True,
            )
            self._save_checkpoint(permanent=True)
            self._write_run_outcome(early_stopping=early_stopping, stop_reason="max_steps")
            return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.async_rollout_manager)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        self.gen_steps += 1
        last_val_metrics = None
        stop_reason = "max_steps"

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        timing_raw = defaultdict(float)
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        all_gen_acc = []  # Track accuracy across ALL generated responses (before filtering)
        all_gen_reward_group_counts = defaultdict(int)
        current_epoch = self.global_steps // len(self.train_dataloader)

        training_complete = False
        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                new_batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                num_gen_batches += 1
                gen_batch = self._get_gen_batch(new_batch)
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps
                permanent_checkpoint_requested = False

                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, "red"):
                        gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, "red"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)

                            new_batch = new_batch.union(gen_baseline_output)
                            # compute reward model score on new_batch
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in new_batch.batch.keys():
                                rm_scores = self._compute_reward_colocate(new_batch)
                                new_batch = new_batch.union(rm_scores)
                            reward_baseline_tensor, _ = extract_reward(new_batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            new_batch.pop(batch_keys=list(keys_to_pop))

                            new_batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output

                    new_batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
                    )
                    # repeat to align with repeated responses in rollout
                    new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    new_batch = new_batch.union(gen_batch_output)

                    # On-policy distillation: ask colocated teacher for per-token logprobs
                    # of the student-sampled tokens. No-op unless distillation.enabled=true.
                    if self._should_compute_teacher_colocate(new_batch):
                        with marked_timer("teacher", timing_raw, color="cyan"):
                            batch_teacher = self._compute_teacher_colocate(new_batch)
                            new_batch = new_batch.union(batch_teacher)

                    if self.config.algorithm.use_kl_in_reward:
                        # We need these metrics for apply_kl_penalty if using kl in reward
                        new_batch = self.compute_kl_related_metrics(new_batch, metrics, timing_raw)
                        # otherwise, we will compute those after dynamic sampling

                    with marked_timer("reward", timing_raw, "yellow"):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm and "rm_scores" not in new_batch.batch.keys():
                            # we first compute reward model score
                            batch_reward = self._compute_reward_colocate(new_batch)
                            new_batch = new_batch.union(batch_reward)

                        # we combine with rule-based rm
                        reward_tensor, reward_extra_infos_dict = extract_reward(new_batch)

                        new_batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            new_batch.non_tensor_batch.update(
                                {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                            )

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            new_batch, kl_metrics = apply_kl_penalty(
                                new_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(
                                kl_metrics
                            )  # TODO: This will be cleared if we use multiple genenration batches
                        else:
                            new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]

                    # Collect accuracy on ALL generated responses (before filtering)
                    if "acc" in new_batch.non_tensor_batch:
                        generated_acc = new_batch.non_tensor_batch["acc"]
                        all_gen_acc.extend(generated_acc.tolist())
                        reward_group_counts = count_binary_reward_groups(
                            new_batch.non_tensor_batch["uid"], generated_acc
                        )
                        for group_kind, count in reward_group_counts.items():
                            all_gen_reward_group_counts[group_kind] += count

                    if not self.config.algorithm.filter_groups.enable:
                        batch = new_batch
                    else:  # NOTE: When prompts after filtering is less than train batch size,
                        # we skip to the next generation batch
                        metric_name = self.config.algorithm.filter_groups.metric
                        if metric_name == "seq_final_reward":
                            # Turn to numpy for easier filtering
                            new_batch.non_tensor_batch["seq_final_reward"] = (
                                new_batch.batch["token_level_rewards"].sum(dim=-1).numpy()
                            )
                        elif metric_name == "seq_reward":
                            new_batch.non_tensor_batch["seq_reward"] = (
                                new_batch.batch["token_level_scores"].sum(dim=-1).numpy()
                            )

                        # Collect the sequence reward for each trajectory
                        prompt_uid2metric_vals = defaultdict(list)
                        for uid, metric_val in zip(
                            new_batch.non_tensor_batch["uid"], new_batch.non_tensor_batch[metric_name], strict=True
                        ):
                            prompt_uid2metric_vals[uid].append(metric_val)

                        prompt_uid2metric_std = {}
                        for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                            prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)

                        kept_prompt_uids = [
                            uid
                            for uid, std in prompt_uid2metric_std.items()
                            if std > 0 or len(prompt_uid2metric_vals[uid]) == 1
                        ]
                        num_prompt_in_batch += len(kept_prompt_uids)

                        kept_traj_idxs = []
                        for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch["uid"]):
                            if traj_from_prompt_uid in kept_prompt_uids:
                                kept_traj_idxs.append(idx)

                        new_batch = new_batch[kept_traj_idxs]
                        batch = new_batch if batch is None else DataProto.concat([batch, new_batch])

                        prompt_bsz = self.config.data.train_batch_size
                        if num_prompt_in_batch < prompt_bsz:
                            print(f"{num_prompt_in_batch=} < {prompt_bsz=}")
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                print(f"{num_gen_batches=}. Keep generating...")
                                self.gen_steps += 1
                                is_last_step = self.global_steps >= self.total_training_steps
                                continue
                            else:
                                raise ValueError(
                                    f"{num_gen_batches=} >= {max_num_gen_batches=}."
                                    + " Generated too many. Please check if your data are too difficult."
                                    + " You could also try set max_num_gen_batches=0 to enable endless trials."
                                )
                        else:
                            # Align the batch
                            traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                            batch = batch[:traj_bsz]

                    # Save a random sample of trajectories for debugging
                    self._save_trajectories(batch, step=self.global_steps, sample_prob=0.001)

                    self.checkpoint_manager.sleep_replicas()

                    # === Updating ===
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    if not self.config.algorithm.use_kl_in_reward:
                        batch = self.compute_kl_related_metrics(batch, metrics, timing_raw)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, "cyan"):
                            values = self._compute_values(batch)
                            batch = batch.union(values)

                    # Compute rollout correction weights and off-policy metrics (inherited from RayPPOTrainer)
                    from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    if rollout_corr_config is not None and "rollout_log_probs" in batch.batch:
                        batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                        # IS and off-policy metrics already have rollout_corr/ prefix
                        metrics.update(is_metrics)

                    with marked_timer("adv", timing_raw, "brown"):
                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, "pink"):
                            critic_output = self._update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, "red"):
                            actor_output = self._update_actor(batch)

                        # Check if ESI/training plan is close to expiration
                        esi_close_to_expiration = should_save_ckpt_esi(
                            max_steps_duration=self.max_steps_duration,
                            redundant_time=self.config.trainer.esi_redundant_time,
                        )
                        if self.config.trainer.save_freq > 0 and (
                            is_last_step
                            or self.global_steps % self.config.trainer.save_freq == 0
                            or esi_close_to_expiration
                        ):
                            if esi_close_to_expiration:
                                print("Force saving checkpoint: ESI instance expiration approaching.")
                            permanent_checkpoint_requested = True

                        with marked_timer("update_weights", timing_raw, "red"):
                            self.checkpoint_manager.update_weights()
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                    # Upload a random sample of this step's train rollouts to wandb (fork addition)
                    self._log_train_generations_to_wandb(batch)

                # validate
                early_stop_triggered = False
                if self.config.trainer.test_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.test_freq == 0
                ):
                    with marked_timer("testing", timing_raw, "green"):
                        val_metrics: dict = self._validate()
                        last_val_metrics = val_metrics
                        if early_stopping is not None:
                            early_stop_triggered, early_metrics = early_stopping.observe(
                                val_metrics, step=self.global_steps
                            )
                            val_metrics.update(early_metrics)
                    metrics.update(val_metrics)

                # Persist validation-driven state before publishing this step's
                # checkpoint. This keeps the all-time best and patience counter
                # transactionally aligned with the model/optimizer/data cursor.
                if early_stop_triggered:
                    permanent_checkpoint_requested = True
                rolling_checkpoint_requested = self._should_save_rolling_checkpoint()
                if permanent_checkpoint_requested or rolling_checkpoint_requested:
                    with marked_timer("save_checkpoint", timing_raw, "green"):
                        self._save_checkpoint(permanent=permanent_checkpoint_requested)
                    if permanent_checkpoint_requested:
                        self._maybe_push_to_hf(self.global_steps)

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw.get("step", 0)
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                timing_raw = defaultdict(float)  # clear timing

                # Log raw accuracy and overlong penalty separately
                if "acc" in batch.non_tensor_batch:
                    acc_vals = batch.non_tensor_batch["acc"]
                    metrics["train/acc_mean"] = float(np.mean(acc_vals))
                if "score" in batch.non_tensor_batch:
                    score_vals = batch.non_tensor_batch["score"]
                    metrics["train/score_mean"] = float(np.mean(score_vals))
                if "overlong_reward" in batch.non_tensor_batch:
                    overlong_vals = batch.non_tensor_batch["overlong_reward"]
                    metrics["train/overlong_penalty_mean"] = float(np.mean(overlong_vals))
                    metrics["train/overlong_ratio"] = float(np.mean(np.array(overlong_vals) < 0))

                # Log accuracy on ALL generated responses (before dynamic sampling filter)
                if all_gen_acc:
                    metrics["train/acc_all_generated"] = float(np.mean(all_gen_acc))
                    metrics["train/num_responses_all_generated"] = len(all_gen_acc)
                    metrics.update(binary_reward_group_metrics(all_gen_reward_group_counts))

                metrics["train/num_gen_batches"] = num_gen_batches
                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0
                all_gen_acc = []
                all_gen_reward_group_counts = defaultdict(int)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if early_stop_triggered:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    print(
                        "EARLY_STOP_TRIGGERED "
                        f"step={self.global_steps} best={early_stopping.best_score} "
                        f"best_step={early_stopping.best_step} "
                        f"misses={early_stopping.non_improving_rounds} "
                        f"metric={early_stopping.metric}",
                        flush=True,
                    )
                    pprint(f"Early-stop validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    stop_reason = "early_stopping"
                    training_complete = True
                    break

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    training_complete = True
                    break

                progress_bar.update(1)
                self.global_steps += 1
                self.gen_steps += 1
            if training_complete:
                break
        # check if last step checkpint exists
        checkpoint_dir = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")
        if not os.path.exists(checkpoint_dir):
            # save last step checkpoint
            timing_raw = defaultdict(float)
            with marked_timer("save_checkpoint", timing_raw, "green"):
                self._save_checkpoint()
            self._maybe_push_to_hf(self.global_steps)
            metrics = {f"timing/{k}": v for k, v in timing_raw.items()}
            logger.log(data=metrics, step=self.global_steps)
        self._write_run_outcome(early_stopping=early_stopping, stop_reason=stop_reason)
