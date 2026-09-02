#!/usr/bin/env python3
"""Resolve or launch the resumable Gemma 4 difficulty continuation sweep."""

from __future__ import annotations

import argparse
import base64
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class JobSpec:
    key: str
    job_name: str
    run_file: str
    logical_runs: tuple[str, ...]
    env: dict[str, str]
    gpus: int = 8


MODEL_REVISIONS = {
    "e2b": "d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f",
    "e4b": "411aa17b749aa952df1359d2dcea73917a544d9a",
    "12b": "023679ed352de9bb66cc873c9009ce3482585c08",
    "26b-a4b": "24548b62aa021d562695c04aaf7758a1ea47990b",
}
DATASET_REPO = "JWei05/DeepScaleR-Easy-Medium-Hard-Gemma-26B-PT-10k"
DATASET_REVISION = "a0ba3c3dc07c7bc27e901670ceb1a0b0ceeaa8db"
RUN_ARTIFACT_S3_BASE = "s3://scale-ml/genai/rl-distill/gemma4-difficulty-s42-20260819"
FULL_CHECKPOINT_S3_BASE = f"{RUN_ARTIFACT_S3_BASE}-full-checkpoints"
# E2B bands are rerun from scratch under the new rolling-checkpoint infra. A fresh
# S3 base (and W&B suffix) keeps them from resuming the original runs' final
# checkpoints and from colliding with the original W&B curves.
E2B_RERUN_ARTIFACT_S3_BASE = "s3://scale-ml/genai/rl-distill/gemma4-difficulty-s42-e2b-rerun-20260902"
E2B_RERUN_FULL_CHECKPOINT_S3_BASE = f"{E2B_RERUN_ARTIFACT_S3_BASE}-full-checkpoints"
E2B_RERUN_WANDB_SUFFIX = "rerun-v2"
BANDS = ("easy", "medium", "hard")


def legacy_early_stopping_migration(
    *,
    model: str,
    difficulty: str,
    checkpoint_step: int,
    best_score: float,
    best_step: int,
    patience: int,
) -> str:
    """Encode an explicit, identity-bound migration for a legacy checkpoint."""
    state = {
        "protocol": "validation_early_stopping_v1",
        "config": {
            "metric": "val-core/math/acc/mean@16",
            "patience": patience,
            "mode": "max",
            "min_delta": 0.0,
            "include_initial_validation": True,
        },
        "state": {
            "best_score": best_score,
            "best_step": best_step,
            "non_improving_rounds": 0,
            "last_observed_step": checkpoint_step,
            "last_observed_score": best_score,
            "last_observed_improved": True,
            "last_observed_triggered": False,
        },
    }
    payload = {
        "protocol": "legacy_early_stopping_migration_v1",
        "checkpoint_step": checkpoint_step,
        "model": model,
        "difficulty": difficulty,
        "early_stopping_state": state,
    }
    return base64.b64encode(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).decode()


def build_job_specs(*, seed: int, total_steps: int, phase: str) -> list[JobSpec]:
    common = {
        "DATA_SEED": str(seed),
        "DIFFICULTY_DATASET_SOURCE": "gemma4_26b_bands",
        "DIFFICULTY_DATASET_REPO": DATASET_REPO,
        "DIFFICULTY_DATASET_REVISION": DATASET_REVISION,
        "MAX_PROMPT_LENGTH": "4096",
        "MAX_RESPONSE_LENGTH": "8192",
        "MAX_MODEL_LEN": "12288",
        "OVERLONG_BUFFER_LEN": "2048",
        "ENABLE_OVERLONG_BUFFER": "True",
        "OVERLONG_PENALTY_FACTOR": "1.0",
        "FSDP_CPU_OFFLOAD_POLICY": "True",
        "OFFLOAD": "False",
        "TRAIN_PROMPT_BSZ": "64",
        "GEN_PROMPT_BSZ": "64",
        "N_RESP_PER_PROMPT": "16",
        "TRAIN_PROMPT_MINI_BSZ": "32",
        "ACTOR_LR": "1e-6",
        "ACTOR_LR_WARMUP_STEPS": "20",
        "TEST_FREQ": "10",
        "SAVE_FREQ": "10",
        "ROLLING_CHECKPOINT_ENABLED": "True",
        "ROLLING_CHECKPOINT_FREQ": "1",
        "HF_PUSH_FREQ": "10",
        "HF_PUSH_ENABLE": "False",
        "HF_PUSH_REQUIRED": "False",
        # The remote protocol preserves every permanent 10-step snapshot and
        # one rolling step. Locally, keep only the previous actor checkpoint
        # until the next save commits so preemptible pods do not fill /tmp.
        "MAX_ACTOR_CKPT_TO_KEEP": "1",
        "HF_PUSH_MAX_TO_KEEP": "8",
        "RESUME_MODE": "auto",
        "RUN_ARTIFACT_S3_BASE": RUN_ARTIFACT_S3_BASE,
        "FULL_CHECKPOINT_S3_BASE": FULL_CHECKPOINT_S3_BASE,
        "TOTAL_TRAINING_STEPS": str(total_steps),
        "RUN_NAME_SUFFIX": "26b-bands-es5",
        "VAL_BEFORE_TRAIN": "True",
        "VAL_N": "1",
        "LOG_VAL_GENERATIONS": "100",
        "LOG_TRAIN_GENERATIONS": "100",
        "EARLY_STOPPING_ENABLED": "True",
        "EARLY_STOPPING_METRIC": "val-core/math/acc/mean@16",
        "EARLY_STOPPING_MODE": "max",
        "EARLY_STOPPING_MIN_DELTA": "0.0",
        "EARLY_STOPPING_INCLUDE_INITIAL_VALIDATION": "True",
        "SP_SIZE": "1",
        "GEN_TP": "1",
        "ACTOR_FSDP_SIZE": "-1",
        "ROUTER_Z_LOSS_COEF": "0.0",
    }
    if phase == "smoke":
        common.update(
            {
                "TOTAL_TRAINING_STEPS": "1",
                "TEST_FREQ": "1",
                "SAVE_FREQ": "1",
                "HF_PUSH_ENABLE": "False",
                "EARLY_STOPPING_ENABLED": "False",
                "RUN_NAME_SUFFIX": "26b-bands-smoke",
                "LOG_TRAIN_GENERATIONS": "16",
            }
        )

    e4b_hard_env = {
        **common,
        "EARLY_STOPPING_PATIENCE": "5",
        "GEMMA4_MODEL": "google/gemma-4-E4B",
        "GEMMA4_MODEL_REVISION": MODEL_REVISIONS["e4b"],
        "DIFFICULTY_SEQUENCE": "hard",
        "CUDA_VISIBLE_DEVICES": "0,1,2,3",
        "FSDP_CPU_OFFLOAD_POLICY": "False",
        "MICRO_BATCH_SIZE_PER_GPU": "1",
        "MAX_PADDED_TOKENS_PER_MICROBATCH": "4096",
        "ROLLOUT_GPU_MEMORY_UTILIZATION": "0.25",
        "VLLM_KV_CACHE_MEMORY_BYTES": "1073741824",
        "ROLLOUT_ENFORCE_EAGER": "False",
        "VLLM_DISABLE_COMPILE_CACHE": "0",
        "ROUTER_REPLAY_MODE": "disabled",
    }

    twelve_env = {
        **common,
        "EARLY_STOPPING_PATIENCE": "2",
        "EARLY_STOPPING_MIGRATE_PATIENCE_FROM": "1",
        "GEMMA4_MODEL": "google/gemma-4-12B",
        "GEMMA4_MODEL_REVISION": MODEL_REVISIONS["12b"],
        "DIFFICULTY_SEQUENCE": "easy medium hard",
        "EARLY_STOPPING_LEGACY_RUN_KEY": "12b-medium",
        "EARLY_STOPPING_LEGACY_STATE_B64": legacy_early_stopping_migration(
            model="google/gemma-4-12B",
            difficulty="medium",
            checkpoint_step=20,
            best_score=0.236875,
            best_step=20,
            patience=2,
        ),
        "MICRO_BATCH_SIZE_PER_GPU": "1",
        "MAX_PADDED_TOKENS_PER_MICROBATCH": "4096",
        "ROLLOUT_GPU_MEMORY_UTILIZATION": "0.45",
        "VLLM_KV_CACHE_MEMORY_BYTES": "5368709120",
        "ROLLOUT_ENFORCE_EAGER": "False",
        "VLLM_DISABLE_COMPILE_CACHE": "0",
        "VERL_SKIP_VLLM_MM_WEIGHT_RELOAD": "1",
        "ROUTER_REPLAY_MODE": "disabled",
        "SEQUENTIAL_COOLDOWN_SECONDS": "30",
    }
    moe_common = {
        **common,
        "EARLY_STOPPING_PATIENCE": "2",
        "EARLY_STOPPING_MIGRATE_PATIENCE_FROM": "1",
        "GEMMA4_MODEL": "google/gemma-4-26B-A4B",
        "GEMMA4_MODEL_REVISION": MODEL_REVISIONS["26b-a4b"],
        "MICRO_BATCH_SIZE_PER_GPU": "1",
        "MAX_PADDED_TOKENS_PER_MICROBATCH": "4096",
        "ROLLOUT_GPU_MEMORY_UTILIZATION": "0.10",
        "VLLM_KV_CACHE_MEMORY_BYTES": "3221225472",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "ROLLOUT_ENFORCE_EAGER": "True",
        "VERL_SKIP_VLLM_MM_WEIGHT_RELOAD": "1",
        "ROUTER_REPLAY_MODE": "R3",
        "SEQUENTIAL_COOLDOWN_SECONDS": "30",
    }
    # E2B rerun (dense model, 4 GPUs, one ScaleTrain job per band) under the new
    # rolling-checkpoint infra, from scratch on a fresh S3 base and W&B suffix.
    e2b_common = {
        **common,
        "EARLY_STOPPING_PATIENCE": "5",
        "GEMMA4_MODEL": "google/gemma-4-E2B",
        "GEMMA4_MODEL_REVISION": MODEL_REVISIONS["e2b"],
        "MICRO_BATCH_SIZE_PER_GPU": "8",
        "MAX_PADDED_TOKENS_PER_MICROBATCH": "12288",
        "ROLLOUT_GPU_MEMORY_UTILIZATION": "0.25",
        "VLLM_KV_CACHE_MEMORY_BYTES": "536870912",
        "ROLLOUT_ENFORCE_EAGER": "False",
        "VLLM_DISABLE_COMPILE_CACHE": "0",
        "ROUTER_REPLAY_MODE": "disabled",
        "RUN_ARTIFACT_S3_BASE": E2B_RERUN_ARTIFACT_S3_BASE,
        "FULL_CHECKPOINT_S3_BASE": E2B_RERUN_FULL_CHECKPOINT_S3_BASE,
        "WANDB_RUN_SUFFIX": E2B_RERUN_WANDB_SUFFIX,
        "RUN_NAME_SUFFIX": "26b-bands-es5-rerun",
    }

    specs = [
        JobSpec(
            key=f"26b-a4b-{band}",
            job_name=f"g4-26b-{band}-s{seed}-{phase}",
            run_file="run_gemma4_difficulty_sequential.sh",
            logical_runs=(f"26b-a4b-{band}",),
            env={
                **moe_common,
                "DIFFICULTY_SEQUENCE": band,
                **(
                    {
                        "EARLY_STOPPING_LEGACY_RUN_KEY": "26b-a4b-easy",
                        "EARLY_STOPPING_LEGACY_STATE_B64": legacy_early_stopping_migration(
                            model="google/gemma-4-26B-A4B",
                            difficulty="easy",
                            checkpoint_step=60,
                            best_score=0.9302083333333333,
                            best_step=60,
                            patience=2,
                        ),
                    }
                    if band == "easy"
                    else (
                        {
                            "EARLY_STOPPING_LEGACY_RUN_KEY": "26b-a4b-hard",
                            "EARLY_STOPPING_LEGACY_STATE_B64": legacy_early_stopping_migration(
                                model="google/gemma-4-26B-A4B",
                                difficulty="hard",
                                checkpoint_step=0,
                                best_score=0.10625,
                                best_step=0,
                                patience=2,
                            ),
                        }
                        if band == "hard"
                        else {}
                    )
                ),
            },
        )
        for band in BANDS
    ]
    specs.extend(
        [
            JobSpec(
                key="12b-sequential",
                job_name=f"g4-ds26b-12b-dp8-s{seed}-{phase}",
                run_file="run_gemma4_difficulty_sequential.sh",
                logical_runs=tuple(f"12b-{band}" for band in BANDS),
                env=twelve_env,
            ),
            JobSpec(
                key="e4b-hard",
                job_name=f"g4-e4b-hard-s{seed}-{phase}",
                run_file="run_gemma4_difficulty_sequential.sh",
                logical_runs=("e4b-hard",),
                env=e4b_hard_env,
            ),
        ]
    )
    specs.extend(
        [
            JobSpec(
                key=f"e2b-{band}",
                job_name=f"g4-e2b-{band}-s{seed}-{phase}",
                run_file="run_gemma4_difficulty_sequential.sh",
                logical_runs=(f"e2b-{band}",),
                env={**e2b_common, "DIFFICULTY_SEQUENCE": band},
                gpus=4,
            )
            for band in BANDS
        ]
    )
    return specs


def launch_command(
    spec: JobSpec,
    *,
    cluster: str,
    allow_borrowing: bool,
    dry_run: bool,
    priority: str = "high",
    image: str | None = None,
) -> list[str]:
    launcher = Path(__file__).with_name("launch_st_job.py")
    env_vars = ",".join(f"{key}={value}" for key, value in sorted(spec.env.items()))
    command = [
        sys.executable,
        str(launcher),
        "--cluster",
        cluster,
        "--n-instances",
        "1",
        "--gpus-per-instance",
        str(spec.gpus),
        "--priority",
        priority,
        "--team",
        "egp",
        "--product",
        "train.enterprise_rlvr",
        "--build-config-key",
        "train-rl-distill-verl",
        "--active-deadline-hours",
        "240",
        "--job-name",
        spec.job_name,
        "--run-file",
        spec.run_file,
        "--env-vars",
        env_vars,
    ]
    if allow_borrowing:
        command.append("--allow-borrowing")
    if image:
        command.extend(["--image", image])
    if dry_run:
        command.append("--dry-run")
    return command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("smoke", "full"), default="full")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-steps", type=int, default=400)
    parser.add_argument("--cluster", choices=("eks", "gke"), default="eks")
    parser.add_argument("--allow-borrowing", action="store_true")
    parser.add_argument("--priority", choices=("normal", "high"), default="high")
    parser.add_argument(
        "--image",
        default=None,
        help="Reuse an existing immutable image URI instead of rebuilding it for each selected job.",
    )
    parser.add_argument(
        "--select",
        action="append",
        default=[],
        help="Select a job key (26b-a4b-easy/medium/hard, 12b-sequential, e4b-hard).",
    )
    parser.add_argument(
        "--launch",
        action="store_true",
        help="Submit immediately. Without this flag, only resolved dry-run commands are emitted.",
    )
    args = parser.parse_args()
    if args.total_steps < 1:
        raise SystemExit("--total-steps must be positive")

    specs = build_job_specs(seed=args.seed, total_steps=args.total_steps, phase=args.phase)
    available = {spec.key for spec in specs}
    selected = set(args.select) if args.select else available
    unknown = selected - available
    if unknown:
        raise SystemExit(f"unknown --select values: {sorted(unknown)}; available: {sorted(available)}")

    chosen = [spec for spec in specs if spec.key in selected]
    logical_count = sum(len(spec.logical_runs) for spec in chosen)
    print(
        f"Resolved {logical_count} logical runs into {len(chosen)} full-node ScaleTrain jobs "
        f"(phase={args.phase}, cluster={args.cluster}, borrowing={args.allow_borrowing}, launch={args.launch})."
    )
    for spec in chosen:
        command = launch_command(
            spec,
            cluster=args.cluster,
            allow_borrowing=args.allow_borrowing,
            dry_run=not args.launch,
            priority=args.priority,
            image=args.image,
        )
        print(f"\n[{spec.key}] logical_runs={','.join(spec.logical_runs)}")
        print(shlex.join(command))
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
