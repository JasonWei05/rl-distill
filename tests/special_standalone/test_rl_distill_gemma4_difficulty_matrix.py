from __future__ import annotations

import base64
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_validation_expansion_is_exactly_16_per_stable_uid():
    data = _load(
        "prepare_gemma26b_difficulty_data",
        ROOT / "rl-distill-scripts/data/prepare_deepscaler_gemma4_26b_difficulty_rl_data.py",
    )
    base = pd.DataFrame(
        {
            "uid": ["a", "b", "c"],
            "data_source": ["math"] * 3,
            "prompt": [[{"role": "user", "content": str(index)}] for index in range(3)],
        }
    )
    repeated = data.repeat_validation(base, repeats=16)

    assert len(repeated) == 48
    assert repeated["uid"].nunique() == 3
    assert set(repeated.groupby("uid").size()) == {16}


def test_matrix_resolves_requested_continuations_into_five_full_node_jobs():
    matrix = _load(
        "difficulty_matrix",
        ROOT / "rl-distill-scripts/scale_train/launch_gemma4_difficulty_matrix.py",
    )
    specs = matrix.build_job_specs(seed=42, total_steps=400, phase="full")

    assert len(specs) == 8
    assert sum(len(spec.logical_runs) for spec in specs) == 10
    assert [spec.key for spec in specs] == [
        "26b-a4b-easy",
        "26b-a4b-medium",
        "26b-a4b-hard",
        "12b-sequential",
        "e4b-hard",
        "e2b-easy",
        "e2b-medium",
        "e2b-hard",
    ]
    by_key = {spec.key: spec for spec in specs}
    assert by_key["12b-sequential"].logical_runs == ("12b-easy", "12b-medium", "12b-hard")
    assert by_key["e4b-hard"].logical_runs == ("e4b-hard",)
    for band in ("easy", "medium", "hard"):
        assert by_key[f"26b-a4b-{band}"].logical_runs == (f"26b-a4b-{band}",)
        assert by_key[f"e2b-{band}"].logical_runs == (f"e2b-{band}",)

    for spec in specs:
        command = matrix.launch_command(spec, cluster="eks", allow_borrowing=False, dry_run=True)
        expected_gpus = "4" if spec.key.startswith("e2b-") else "8"
        assert command[command.index("--gpus-per-instance") + 1] == expected_gpus
        assert command[command.index("--priority") + 1] == "high"
        assert "--allow-borrowing" not in command

    # priority is overridable; E2B rerun uses 4 GPUs per job.
    normal_cmd = matrix.launch_command(by_key["e2b-easy"], cluster="eks", allow_borrowing=True, dry_run=True, priority="normal")
    assert normal_cmd[normal_cmd.index("--priority") + 1] == "normal"
    assert "--allow-borrowing" in normal_cmd

    image = "example.invalid/rl-distill:test"
    reuse_command = matrix.launch_command(specs[0], cluster="eks", allow_borrowing=False, dry_run=False, image=image)
    assert reuse_command[reuse_command.index("--image") + 1] == image


def test_matrix_preserves_training_validation_and_early_stop_contract():
    matrix = _load(
        "difficulty_matrix_contract",
        ROOT / "rl-distill-scripts/scale_train/launch_gemma4_difficulty_matrix.py",
    )
    specs = matrix.build_job_specs(seed=42, total_steps=400, phase="full")

    for spec in specs:
        assert spec.env["DIFFICULTY_DATASET_SOURCE"] == "gemma4_26b_bands"
        assert spec.env["DIFFICULTY_DATASET_REPO"] == matrix.DATASET_REPO
        assert spec.env["DIFFICULTY_DATASET_REVISION"] == matrix.DATASET_REVISION
        assert spec.env["MAX_RESPONSE_LENGTH"] == "8192"
        assert spec.env["OVERLONG_BUFFER_LEN"] == "2048"
        assert spec.env["TEST_FREQ"] == "10"
        assert spec.env["SAVE_FREQ"] == "10"
        assert spec.env["TOTAL_TRAINING_STEPS"] == "400"
        assert spec.env["N_RESP_PER_PROMPT"] == "16"
        assert spec.env["VAL_N"] == "1"
        assert spec.env["EARLY_STOPPING_ENABLED"] == "True"
        assert spec.env["EARLY_STOPPING_METRIC"] == "val-core/math/acc/mean@16"
        assert spec.env["EARLY_STOPPING_MIN_DELTA"] == "0.0"
        assert spec.env["MAX_ACTOR_CKPT_TO_KEEP"] == "1"
        assert spec.env["ROLLING_CHECKPOINT_ENABLED"] == "True"
        assert spec.env["ROLLING_CHECKPOINT_FREQ"] == "1"
        assert spec.env["HF_PUSH_ENABLE"] == "False"
        assert spec.env["HF_PUSH_REQUIRED"] == "False"
        assert spec.env["RESUME_MODE"] == "auto"
        if spec.key.startswith("e2b-"):
            assert spec.env["RUN_ARTIFACT_S3_BASE"] == matrix.E2B_RERUN_ARTIFACT_S3_BASE
            assert spec.env["FULL_CHECKPOINT_S3_BASE"] == matrix.E2B_RERUN_FULL_CHECKPOINT_S3_BASE
        else:
            assert spec.env["RUN_ARTIFACT_S3_BASE"] == matrix.RUN_ARTIFACT_S3_BASE
            assert spec.env["FULL_CHECKPOINT_S3_BASE"] == matrix.FULL_CHECKPOINT_S3_BASE
        assert spec.env["SP_SIZE"] == "1"
        assert spec.env["GEN_TP"] == "1"
        assert spec.env["ACTOR_FSDP_SIZE"] == "-1"
        assert spec.env["ROUTER_Z_LOSS_COEF"] == "0.0"

    by_key = {spec.key: spec for spec in specs}
    assert by_key["e4b-hard"].env["EARLY_STOPPING_PATIENCE"] == "5"
    assert by_key["e4b-hard"].env["FSDP_CPU_OFFLOAD_POLICY"] == "False"
    assert by_key["e4b-hard"].env["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
    assert by_key["e4b-hard"].env["DIFFICULTY_SEQUENCE"] == "hard"
    assert by_key["12b-sequential"].env["EARLY_STOPPING_PATIENCE"] == "2"
    assert by_key["12b-sequential"].env["EARLY_STOPPING_MIGRATE_PATIENCE_FROM"] == "1"
    assert by_key["12b-sequential"].env["MICRO_BATCH_SIZE_PER_GPU"] == "1"
    assert by_key["12b-sequential"].env["FSDP_CPU_OFFLOAD_POLICY"] == "True"
    assert by_key["12b-sequential"].env["VLLM_KV_CACHE_MEMORY_BYTES"] == "5368709120"
    assert by_key["12b-sequential"].env["ROUTER_REPLAY_MODE"] == "disabled"
    for band in ("easy", "medium", "hard"):
        spec = by_key[f"26b-a4b-{band}"]
        assert spec.env["EARLY_STOPPING_PATIENCE"] == "2"
        assert spec.env["EARLY_STOPPING_MIGRATE_PATIENCE_FROM"] == "1"
        assert spec.env["DIFFICULTY_SEQUENCE"] == band
        assert spec.env["MICRO_BATCH_SIZE_PER_GPU"] == "1"
        assert spec.env["FSDP_CPU_OFFLOAD_POLICY"] == "True"
        assert spec.env["VLLM_KV_CACHE_MEMORY_BYTES"] == "3221225472"
        assert spec.env["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
        assert spec.env["ROLLOUT_ENFORCE_EAGER"] == "True"
        assert spec.env["ROUTER_REPLAY_MODE"] == "R3"

    migration = json.loads(base64.b64decode(by_key["26b-a4b-easy"].env["EARLY_STOPPING_LEGACY_STATE_B64"]))
    assert migration["checkpoint_step"] == 60
    assert migration["early_stopping_state"]["state"]["best_step"] == 60
    assert migration["early_stopping_state"]["config"]["patience"] == 2
    assert "EARLY_STOPPING_LEGACY_STATE_B64" not in by_key["26b-a4b-medium"].env

    hard_migration = json.loads(base64.b64decode(by_key["26b-a4b-hard"].env["EARLY_STOPPING_LEGACY_STATE_B64"]))
    assert hard_migration["checkpoint_step"] == 0

    # E2B bands: dense model, 4 GPUs, rerun from scratch (fresh W&B suffix, no legacy migration).
    for band in ("easy", "medium", "hard"):
        spec = by_key[f"e2b-{band}"]
        assert spec.gpus == 4
        assert spec.env["GEMMA4_MODEL"] == "google/gemma-4-E2B"
        assert spec.env["EARLY_STOPPING_PATIENCE"] == "5"
        assert spec.env["MICRO_BATCH_SIZE_PER_GPU"] == "8"
        assert spec.env["MAX_PADDED_TOKENS_PER_MICROBATCH"] == "12288"
        assert spec.env["VLLM_KV_CACHE_MEMORY_BYTES"] == "536870912"
        assert spec.env["ROLLOUT_GPU_MEMORY_UTILIZATION"] == "0.25"
        assert spec.env["ROLLOUT_ENFORCE_EAGER"] == "False"
        assert spec.env["ROUTER_REPLAY_MODE"] == "disabled"
        assert spec.env["DIFFICULTY_SEQUENCE"] == band
        assert spec.env["WANDB_RUN_SUFFIX"] == matrix.E2B_RERUN_WANDB_SUFFIX
        assert "EARLY_STOPPING_LEGACY_STATE_B64" not in spec.env
    assert hard_migration["early_stopping_state"]["config"]["patience"] == 2
    assert hard_migration["early_stopping_state"]["state"]["best_score"] == 0.10625
    assert hard_migration["early_stopping_state"]["state"]["best_step"] == 0


def test_strict_high_early_stopping_counts_ties_and_stops_after_five_misses():
    trainer = _load(
        "difficulty_early_stopping",
        ROOT / "dapo/validation_early_stopping.py",
    )
    metric = "val-core/math/acc/mean@16"
    stopper = trainer.ValidationEarlyStopping(metric=metric, patience=5)

    triggered, values = stopper.observe({metric: 0.25}, step=0)
    assert not triggered
    assert values["trainer/early_stopping/best_step"] == 0
    for step, score in zip((10, 20, 30, 40), (0.25, 0.24, 0.23, 0.25), strict=True):
        triggered, _ = stopper.observe({metric: score}, step=step)
        assert not triggered
    triggered, values = stopper.observe({metric: 0.20}, step=50)
    assert triggered
    assert values["trainer/early_stopping/non_improving_rounds"] == 5
    assert stopper.best_score == 0.25
    assert stopper.best_step == 0


def test_strict_high_early_stopping_resets_patience_on_new_record():
    trainer = _load(
        "difficulty_early_stopping_reset",
        ROOT / "dapo/validation_early_stopping.py",
    )
    metric = "val-core/math/acc/mean@16"
    stopper = trainer.ValidationEarlyStopping(metric=metric, patience=5)
    stopper.observe({metric: 0.1}, step=0)
    stopper.observe({metric: 0.09}, step=10)
    triggered, values = stopper.observe({metric: 0.100001}, step=20)

    assert not triggered
    assert values["trainer/early_stopping/improved"] == 1
    assert stopper.non_improving_rounds == 0
    assert stopper.best_step == 20
    with pytest.raises(KeyError, match="was not emitted"):
        stopper.observe({"other": 1.0}, step=30)


def test_early_stopping_state_round_trip_preserves_history_and_deduplicates_resume_eval(tmp_path):
    trainer = _load(
        "difficulty_early_stopping_resume",
        ROOT / "dapo/validation_early_stopping.py",
    )
    metric = "val-core/math/acc/mean@16"
    original = trainer.ValidationEarlyStopping(metric=metric, patience=5)
    original.observe({metric: 0.5}, step=0)
    original.observe({metric: 0.4}, step=10)
    original.observe({metric: 0.45}, step=20)
    state_path = original.save(tmp_path / "global_step_20")

    restored = trainer.ValidationEarlyStopping(metric=metric, patience=5)
    restored.load(state_path.parent)
    assert restored.best_score == 0.5
    assert restored.best_step == 0
    assert restored.non_improving_rounds == 2
    assert restored.last_observed_step == 20
    assert restored.should_observe_initial_validation(step=20)
    assert not restored.should_observe_initial_validation(step=21)

    triggered, values = restored.observe({metric: 0.9}, step=20)
    assert not triggered
    assert values["trainer/early_stopping/duplicate_observation"] == 1
    assert values["trainer/early_stopping/current_score"] == 0.45
    assert restored.best_score == 0.5
    assert restored.non_improving_rounds == 2

    for step, expected_misses in ((30, 3), (40, 4)):
        triggered, _ = restored.observe({metric: 0.49}, step=step)
        assert not triggered
        assert restored.non_improving_rounds == expected_misses
    triggered, values = restored.observe({metric: 0.49}, step=50)
    assert triggered
    assert values["trainer/early_stopping/non_improving_rounds"] == 5


def test_early_stopping_state_rejects_changed_patience(tmp_path):
    trainer = _load(
        "difficulty_early_stopping_config_guard",
        ROOT / "dapo/validation_early_stopping.py",
    )
    metric = "val-core/math/acc/mean@16"
    original = trainer.ValidationEarlyStopping(metric=metric, patience=5)
    original.observe({metric: 0.5}, step=0)
    state_path = original.save(tmp_path)

    payload = json.loads(state_path.read_text())
    restored = trainer.ValidationEarlyStopping(metric=metric, patience=1)
    with pytest.raises(ValueError, match="configuration does not match"):
        restored.load_state_dict(payload)


def test_early_stopping_state_explicitly_migrates_only_patience_and_recomputes_trigger(tmp_path):
    trainer = _load(
        "difficulty_early_stopping_patience_migration",
        ROOT / "dapo/validation_early_stopping.py",
    )
    metric = "val-core/math/acc/mean@16"
    original = trainer.ValidationEarlyStopping(metric=metric, patience=2)
    original.observe({metric: 0.5}, step=0)
    original.observe({metric: 0.4}, step=10)
    original.save(tmp_path)

    restored = trainer.ValidationEarlyStopping(metric=metric, patience=1)
    state_path, migrated = restored.load(tmp_path, migrate_patience_from=2)

    assert migrated
    assert restored.non_improving_rounds == 1
    assert restored.last_observed_triggered
    persisted = json.loads(state_path.read_text())
    assert persisted["config"]["patience"] == 1
    assert persisted["state"]["last_observed_triggered"] is True


def test_early_stopping_patience_migration_rejects_other_policy_changes():
    trainer = _load(
        "difficulty_early_stopping_patience_migration_guard",
        ROOT / "dapo/validation_early_stopping.py",
    )
    original = trainer.ValidationEarlyStopping(metric="val/old", patience=2)
    original.observe({"val/old": 0.5}, step=0)
    restored = trainer.ValidationEarlyStopping(metric="val/new", patience=1)

    with pytest.raises(ValueError, match="configuration does not match"):
        restored.load_state_dict(original.state_dict(), migrate_patience_from=2)


def test_patience_two_stops_on_second_non_record():
    trainer = _load(
        "difficulty_early_stopping_patience_two",
        ROOT / "dapo/validation_early_stopping.py",
    )
    metric = "val-core/math/acc/mean@16"
    stopper = trainer.ValidationEarlyStopping(metric=metric, patience=2)
    stopper.observe({metric: 0.25}, step=0)
    triggered, values = stopper.observe({metric: 0.25}, step=10)

    assert not triggered
    assert values["trainer/early_stopping/non_improving_rounds"] == 1
    triggered, values = stopper.observe({metric: 0.24}, step=20)
    assert triggered
    assert values["trainer/early_stopping/non_improving_rounds"] == 2


def test_legacy_early_stopping_migration_is_identity_bound_and_persisted(tmp_path):
    trainer = _load(
        "difficulty_early_stopping_legacy_migration",
        ROOT / "dapo/validation_early_stopping.py",
    )
    matrix = _load(
        "difficulty_matrix_legacy_migration",
        ROOT / "rl-distill-scripts/scale_train/launch_gemma4_difficulty_matrix.py",
    )
    encoded = matrix.legacy_early_stopping_migration(
        model="google/gemma-4-12B",
        difficulty="medium",
        checkpoint_step=20,
        best_score=0.236875,
        best_step=20,
        patience=2,
    )
    stopper = trainer.ValidationEarlyStopping(metric="val-core/math/acc/mean@16", patience=2)
    state_path = stopper.migrate_legacy_state(
        encoded_payload=encoded,
        checkpoint_dir=tmp_path / "global_step_20",
        checkpoint_step=20,
        model="google/gemma-4-12B",
        difficulty="medium",
    )

    assert state_path.is_file()
    assert stopper.best_score == 0.236875
    assert stopper.best_step == 20
    assert stopper.non_improving_rounds == 0
    with pytest.raises(ValueError, match="does not match"):
        stopper.migrate_legacy_state(
            encoded_payload=encoded,
            checkpoint_dir=tmp_path / "wrong",
            checkpoint_step=30,
            model="google/gemma-4-12B",
            difficulty="medium",
        )


def test_trainer_checkpoints_the_observed_validation_state_before_upload():
    source = (ROOT / "rl-distill-scripts/dapo_ray_trainer.py").read_text()
    save_method = source.index("    def _save_checkpoint(self, *, permanent: bool = True):")
    state_save = source.index("early_stopping.save(checkpoint_dir)", save_method)
    base_save = source.index("super()._save_checkpoint(save_hf_model=permanent)", state_save)
    remote_upload = source.index("helper.upload_checkpoint", base_save)
    assert state_save < base_save < remote_upload

    validation = source.index("                # validate")
    observation = source.index("early_stop_triggered, early_metrics", validation)
    deferred_save = source.index('with marked_timer("save_checkpoint"', observation)
    assert validation < observation < deferred_save

    assert "rolling_checkpoint_requested = self._should_save_rolling_checkpoint()" in source
    assert "self._save_checkpoint(permanent=permanent_checkpoint_requested)" in source


def test_runtime_launcher_supports_26b_r3_id_only_validation_and_early_stopping():
    launcher = (ROOT / "rl-distill-scripts/scale_train/run_gemma4_pt_deepscaler_4of4strict_rl.sh").read_text()

    assert "gemma4-26b-a4b" in launcher
    assert "deepscaler_gemma4_26b_${DIFFICULTY_DATASET}_val300_x16.parquet" in launcher
    assert "DEFAULT_VAL_FILES=\"['${DATA_DIR}/${IN_DIST_VAL_BASENAME}']\"" in launcher
    assert 'actor_rollout_ref.actor.router_replay.mode="${ROUTER_REPLAY_MODE}"' in launcher
    assert 'actor_rollout_ref.actor.fsdp_config.router_replay.mode="${ROUTER_REPLAY_MODE}"' in launcher
    assert "verl.workers.config.EngineRouterReplayConfig" in (ROOT / "verl/trainer/config/engine/fsdp.yaml").read_text()
    assert 'actor_rollout_ref.rollout.enable_rollout_routing_replay="${ENABLE_ROLLOUT_ROUTING_REPLAY}"' in launcher
    assert "runtime_env.env_vars.PYTORCH_CUDA_ALLOC_CONF" in launcher
    assert '++trainer.early_stopping.enabled="${EARLY_STOPPING_ENABLED:-False}"' in launcher
    assert "ROUTER_Z_LOSS_COEF=0.0" in launcher
    assert "EARLY_STOP_TRIGGERED" in (ROOT / "rl-distill-scripts/dapo_ray_trainer.py").read_text()


def test_sequential_wrapper_supports_selected_bands_and_fails_closed():
    wrapper = (ROOT / "rl-distill-scripts/scale_train/run_gemma4_difficulty_sequential.sh").read_text()

    assert "DIFFICULTY_SEQUENCE:-easy medium hard" in wrapper
    assert 'run_child "${difficulty}" "${port_base}"' in wrapper
    assert "SEQUENTIAL_CHILD_FAILED" in wrapper
    assert "RUN_DONE rc=0" in wrapper
    assert "ROUTER_REPLAY_MODE_DEFAULT=R3" in wrapper
    assert "check-best-hf" in wrapper
    assert "check-completion-max" in wrapper
    assert 'RUN_ARTIFACT_S3_URI="${artifact_uri}"' in wrapper
    assert 'FULL_CHECKPOINT_S3_URI="${checkpoint_uri}"' in wrapper
    assert 'WANDB_RUN_ID="${wandb_run_id}"' in wrapper
    assert 'EARLY_STOPPING_LEGACY_STATE_B64="${legacy_early_stopping_state_b64}"' in wrapper


def test_terminal_resume_finalizes_without_an_extra_training_step():
    trainer = (ROOT / "rl-distill-scripts/dapo_ray_trainer.py").read_text()
    launcher = (ROOT / "rl-distill-scripts/scale_train/run_gemma4_pt_deepscaler_4of4strict_rl.sh").read_text()

    assert "TRAINING_ALREADY_AT_MAX_STEPS_ON_RESUME" in trainer
    assert trainer.index("TRAINING_ALREADY_AT_MAX_STEPS_ON_RESUME") < trainer.index("# add tqdm")
    assert launcher.index("publish-best-hf") < launcher.rindex("complete-run")


def test_borrowing_supervisor_setup_is_launch_gated_and_checks_all_terminal_receipts():
    starter = (ROOT / "rl-distill-scripts/scale_train/start_gemma4_difficulty_supervisor.sh").read_text()
    orchestrator = (ROOT / "rl-distill-scripts/scale_train/start_gemma4_difficulty_resume_supervisors.sh").read_text()

    assert "--allow-borrowing" in starter
    assert "--max-completion-step 400" in starter
    assert "expected_world_size=8" in starter
    assert "expected_world_size=4" in starter
    assert '--expected-completion-world-size "${expected_world_size}"' in starter
    assert "--poll-seconds 15" in starter
    assert "--retry-seconds 5" in starter
    assert "--pod-log-dir" in starter
    assert "--completion-best-hf-s3-uri" in starter
    assert 'best_hf_completion_keys+=("gemma4-${completion_key}")' in starter
    for key in ("e4b-hard",):
        assert key in starter
    for key in (
        "12b-easy",
        "12b-medium",
        "12b-hard",
        "26b-a4b-easy",
        "26b-a4b-medium",
        "26b-a4b-hard",
    ):
        assert key in starter

    assert "launch=false" in orchestrator
    assert 'if [ "${launch}" != true ]' in orchestrator
    assert "DRY RUN ONLY" in orchestrator
    assert "--allow-borrowing" in orchestrator
    assert "start_gemma4_difficulty_supervisor_after_image.sh" in orchestrator
    assert "QUEUE_KEYS=(26b-a4b-easy 26b-a4b-medium 26b-a4b-hard 12b-sequential e4b-hard)" in orchestrator


def test_packed_wrapper_starts_requested_pair_then_refills_first_free_slot():
    wrapper = (ROOT / "rl-distill-scripts/scale_train/run_gemma4_e2b_e4b_difficulty_queue.sh").read_text()

    assert "RUN_MODELS=(e2b e2b e2b e4b e4b e4b)" in wrapper
    assert "RUN_BANDS=(easy medium hard easy medium hard)" in wrapper
    assert wrapper.index("launch_next_for_slot 0") < wrapper.index("launch_next_for_slot 1")
    assert 'wait -n -p finished_pid "${!PID_RUN[@]}"' in wrapper
    assert 'launch_next_for_slot "${slot}"' in wrapper
    assert "PACK_QUEUE_CHILD_FAILED" in wrapper
    assert '"${child_rc}" -ne 0' in wrapper
    assert "RUN_DONE rc=0" in wrapper
    assert 'SLOT_GPUS=("0,1,2,3" "4,5,6,7")' in wrapper
    assert "PACK_QUEUE_SKIP_COMPLETE" in wrapper
    assert 'RUN_ARTIFACT_S3_URI="${artifact_uri}"' in wrapper
    assert 'FULL_CHECKPOINT_S3_URI="${checkpoint_uri}"' in wrapper
    assert 'WANDB_RUN_ID="${wandb_run_id}"' in wrapper


def test_live_checkpoint_watcher_only_maps_the_seed42_difficulty_sweep():
    watcher = _load(
        "difficulty_checkpoint_watcher",
        ROOT / "rl-distill-scripts/watch_gemma4_difficulty_checkpoints.py",
    )

    assert watcher.run_key(Path("gemma4-e4b-deepscaler-gemma26b-medium-seed42-26b-bands-es5")) == "e4b-medium"
    assert watcher.run_key(Path("gemma4-26b-a4b-deepscaler-gemma26b-hard-seed42-26b-bands-es5")) == "26b-a4b-hard"
    assert watcher.run_key(Path("unrelated-checkpoint")) is None


def test_checkpoint_supervisor_does_not_duplicate_native_uploads():
    supervisor = (ROOT / "rl-distill-scripts/scale_train/supervise_gemma4_difficulty_checkpoints.sh").read_text()

    assert "pod_uses_native_checkpoint_upload" in supervisor
    assert "FULL_CHECKPOINT_S3_(URI|BASE)=" in supervisor
    assert "detach_legacy_watcher" in supervisor
    assert "SKIP_NATIVE" in supervisor


def test_verl_image_keeps_rollout_third_party_adapter():
    dockerignore = (ROOT / "rl-distill-scripts/scale_train/st_config/.dockerignore.verl").read_text().splitlines()
    rules = {line.strip() for line in dockerignore if line.strip() and not line.lstrip().startswith("#")}
    dockerfile = (ROOT / "rl-distill-scripts/scale_train/st_config/Dockerfile").read_text()

    assert "third_party" not in rules
    assert "third_party/**" not in rules
    assert {"nemo-rl", "Megatron-Bridge"} <= rules
    assert (ROOT / "verl/third_party/vllm/__init__.py").is_file()
    assert "test -f /workspace/rl-distill/verl/third_party/vllm/__init__.py" in dockerfile
