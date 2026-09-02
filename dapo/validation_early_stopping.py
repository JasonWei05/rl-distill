"""Validation-driven early stopping helpers for DAPO training."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

EARLY_STOPPING_STATE_FILENAME = "validation_early_stopping.json"
_STATE_PROTOCOL = "validation_early_stopping_v1"


@dataclass
class ValidationEarlyStopping:
    """Track strict all-time validation improvements."""

    metric: str
    patience: int
    mode: str = "max"
    min_delta: float = 0.0
    include_initial_validation: bool = True
    best_score: float | None = None
    best_step: int | None = None
    non_improving_rounds: int = 0
    last_observed_step: int | None = None
    last_observed_score: float | None = None
    last_observed_improved: bool | None = None
    last_observed_triggered: bool = False

    def __post_init__(self) -> None:
        if not self.metric:
            raise ValueError("early-stopping metric must be non-empty")
        if self.patience < 1:
            raise ValueError("early-stopping patience must be at least 1")
        if self.mode not in {"max", "min"}:
            raise ValueError(f"early-stopping mode must be 'max' or 'min', got {self.mode!r}")
        if self.min_delta < 0:
            raise ValueError("early-stopping min_delta must be non-negative")
        self._validate_runtime_state()

    @classmethod
    def from_trainer_config(cls, trainer_config):
        config = trainer_config.get("early_stopping", None)
        if config is None or not bool(config.get("enabled", False)):
            return None
        return cls(
            metric=str(config.get("metric", "")),
            patience=int(config.get("patience", 5)),
            mode=str(config.get("mode", "max")),
            min_delta=float(config.get("min_delta", 0.0)),
            include_initial_validation=bool(config.get("include_initial_validation", True)),
        )

    def _config_state(self) -> dict[str, str | int | float | bool]:
        return {
            "metric": self.metric,
            "patience": self.patience,
            "mode": self.mode,
            "min_delta": self.min_delta,
            "include_initial_validation": self.include_initial_validation,
        }

    def _validate_runtime_state(self) -> None:
        if self.non_improving_rounds < 0:
            raise ValueError("early-stopping non_improving_rounds must be non-negative")
        if (self.best_score is None) != (self.best_step is None):
            raise ValueError("early-stopping best_score and best_step must either both be set or both be None")
        if (self.last_observed_step is None) != (self.last_observed_score is None):
            raise ValueError(
                "early-stopping last_observed_step and last_observed_score must either both be set or both be None"
            )
        if self.best_score is not None and not isfinite(float(self.best_score)):
            raise ValueError(f"early-stopping best_score must be finite, got {self.best_score}")
        if self.last_observed_score is not None and not isfinite(float(self.last_observed_score)):
            raise ValueError(f"early-stopping last_observed_score must be finite, got {self.last_observed_score}")
        if self.best_step is not None and self.best_step < 0:
            raise ValueError("early-stopping best_step must be non-negative")
        if self.last_observed_step is not None and self.last_observed_step < 0:
            raise ValueError("early-stopping last_observed_step must be non-negative")
        if (
            self.best_step is not None
            and self.last_observed_step is not None
            and self.best_step > self.last_observed_step
        ):
            raise ValueError("early-stopping best_step cannot be later than last_observed_step")
        if self.last_observed_step is None:
            if self.last_observed_improved is not None or self.last_observed_triggered:
                raise ValueError("early-stopping last-observation flags require an observed step")
            if self.best_score is not None or self.non_improving_rounds:
                raise ValueError("early-stopping history requires an observed step")
        else:
            if self.last_observed_improved is None:
                raise ValueError("early-stopping last_observed_improved is required with an observed step")
            if self.last_observed_improved and self.non_improving_rounds != 0:
                raise ValueError("an improving early-stopping observation must reset non_improving_rounds")
            if self.last_observed_triggered != (self.non_improving_rounds >= self.patience):
                raise ValueError("early-stopping trigger flag does not match the patience counter")

    def state_dict(self) -> dict[str, object]:
        """Return the complete, configuration-bound stopping state."""

        self._validate_runtime_state()
        return {
            "protocol": _STATE_PROTOCOL,
            "config": self._config_state(),
            "state": {
                "best_score": self.best_score,
                "best_step": self.best_step,
                "non_improving_rounds": self.non_improving_rounds,
                "last_observed_step": self.last_observed_step,
                "last_observed_score": self.last_observed_score,
                "last_observed_improved": self.last_observed_improved,
                "last_observed_triggered": self.last_observed_triggered,
            },
        }

    def load_state_dict(
        self,
        payload: dict[str, object],
        *,
        migrate_patience_from: int | None = None,
    ) -> bool:
        """Restore state, optionally allowing one explicit patience migration.

        The migration is deliberately narrow: every saved policy field except
        ``patience`` must match the active policy, and the old patience must be
        named by the caller. Runtime history is preserved, while the terminal
        flag is recomputed under the active patience.
        """

        if payload.get("protocol") != _STATE_PROTOCOL:
            raise ValueError(f"unsupported early-stopping state protocol: {payload.get('protocol')!r}")
        saved_config = payload.get("config")
        active_config = self._config_state()
        patience_migrated = False
        if saved_config != active_config:
            expected_saved_config = dict(active_config)
            if migrate_patience_from is not None:
                if migrate_patience_from < 1:
                    raise ValueError("migrated early-stopping patience must be at least 1")
                expected_saved_config["patience"] = migrate_patience_from
            if migrate_patience_from is None or saved_config != expected_saved_config:
                raise ValueError(
                    "early-stopping checkpoint configuration does not match the active configuration: "
                    f"saved={saved_config!r} active={active_config!r} "
                    f"migrate_patience_from={migrate_patience_from!r}"
                )
            patience_migrated = True
        state = payload.get("state")
        if not isinstance(state, dict):
            raise ValueError("early-stopping checkpoint state must be a JSON object")

        self.best_score = None if state.get("best_score") is None else float(state["best_score"])
        self.best_step = None if state.get("best_step") is None else int(state["best_step"])
        self.non_improving_rounds = int(state.get("non_improving_rounds", -1))
        self.last_observed_step = None if state.get("last_observed_step") is None else int(state["last_observed_step"])
        self.last_observed_score = (
            None if state.get("last_observed_score") is None else float(state["last_observed_score"])
        )
        last_improved = state.get("last_observed_improved")
        if last_improved is not None and not isinstance(last_improved, bool):
            raise ValueError("early-stopping last_observed_improved must be a boolean or null")
        self.last_observed_improved = last_improved
        last_triggered = state.get("last_observed_triggered", False)
        if not isinstance(last_triggered, bool):
            raise ValueError("early-stopping last_observed_triggered must be a boolean")
        self.last_observed_triggered = (
            self.non_improving_rounds >= self.patience if patience_migrated else last_triggered
        )
        self._validate_runtime_state()
        return patience_migrated

    def save(self, checkpoint_dir: Path) -> Path:
        """Atomically persist state inside a trainer checkpoint directory."""

        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        state_path = checkpoint_dir / EARLY_STOPPING_STATE_FILENAME
        temporary = state_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.state_dict(), sort_keys=True, indent=2) + "\n", encoding="utf-8")
        temporary.replace(state_path)
        return state_path

    def load(
        self,
        checkpoint_dir: Path,
        *,
        migrate_patience_from: int | None = None,
    ) -> tuple[Path, bool]:
        state_path = checkpoint_dir / EARLY_STOPPING_STATE_FILENAME
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"early-stopping checkpoint must be a JSON object: {state_path}")
        patience_migrated = self.load_state_dict(
            payload,
            migrate_patience_from=migrate_patience_from,
        )
        if patience_migrated:
            self.save(checkpoint_dir)
        return state_path, patience_migrated

    def migrate_legacy_state(
        self,
        *,
        encoded_payload: str,
        checkpoint_dir: Path,
        checkpoint_step: int,
        model: str,
        difficulty: str,
    ) -> Path:
        """Materialize an explicit, identity-bound state for an old checkpoint."""
        try:
            migration = json.loads(base64.b64decode(encoded_payload, validate=True))
        except Exception as error:
            raise ValueError("invalid EARLY_STOPPING_LEGACY_STATE_B64 payload") from error
        if not isinstance(migration, dict) or migration.get("protocol") != "legacy_early_stopping_migration_v1":
            raise ValueError("unsupported legacy early-stopping migration payload")
        expected_identity = {
            "checkpoint_step": checkpoint_step,
            "model": model,
            "difficulty": difficulty,
        }
        mismatches = [key for key, value in expected_identity.items() if migration.get(key) != value]
        if mismatches:
            raise ValueError(
                "legacy early-stopping migration does not match the restored run: "
                f"mismatches={mismatches} expected={expected_identity!r}"
            )
        migrated_state = migration.get("early_stopping_state")
        if not isinstance(migrated_state, dict):
            raise ValueError("legacy early-stopping migration is missing early_stopping_state")
        self.load_state_dict(migrated_state)
        return self.save(checkpoint_dir)

    def load_legacy_history(
        self,
        *,
        encoded_payload: str,
        current_step: int,
        model: str,
        difficulty: str,
    ) -> int:
        """Load identity-bound history that predates the current checkpoint.

        Unlike :meth:`migrate_legacy_state`, this accepts a source observation
        earlier than ``current_step``.  It is used when an evaluation completed
        before the first checkpoint existed, or when a later checkpoint was
        created by an older trainer that did not know about that observation.
        """

        try:
            migration = json.loads(base64.b64decode(encoded_payload, validate=True))
        except Exception as error:
            raise ValueError("invalid EARLY_STOPPING_LEGACY_STATE_B64 payload") from error
        if not isinstance(migration, dict) or migration.get("protocol") != "legacy_early_stopping_migration_v1":
            raise ValueError("unsupported legacy early-stopping migration payload")

        source_step = migration.get("checkpoint_step")
        if not isinstance(source_step, int) or isinstance(source_step, bool) or source_step < 0:
            raise ValueError(f"legacy early-stopping checkpoint_step must be a non-negative integer: {source_step!r}")
        expected_identity = {"model": model, "difficulty": difficulty}
        mismatches = [key for key, value in expected_identity.items() if migration.get(key) != value]
        if mismatches:
            raise ValueError(
                "legacy early-stopping migration does not match the restored run: "
                f"mismatches={mismatches} expected={expected_identity!r}"
            )
        if source_step > current_step:
            raise ValueError(
                "legacy early-stopping history is newer than the restored trainer state: "
                f"source_step={source_step} current_step={current_step}"
            )

        migrated_state = migration.get("early_stopping_state")
        if not isinstance(migrated_state, dict):
            raise ValueError("legacy early-stopping migration is missing early_stopping_state")
        self.load_state_dict(migrated_state)
        if self.last_observed_step is not None and self.last_observed_step > current_step:
            raise ValueError(
                "legacy early-stopping observation is newer than the restored trainer state: "
                f"last_observed_step={self.last_observed_step} current_step={current_step}"
            )
        return source_step

    def merge_historical_state(self, historical: ValidationEarlyStopping) -> bool:
        """Merge an omitted older all-time best without guessing patience state.

        A better historical best can be merged safely only when it belongs to
        the same latest validation observation as the current state.  Once a
        newer validation has happened, the compact checkpoint state is not
        enough to reconstruct how many misses that better baseline would have
        accumulated, so fail closed instead of silently changing semantics.
        """

        if historical._config_state() != self._config_state():
            raise ValueError(
                "historical early-stopping configuration does not match the active configuration: "
                f"historical={historical._config_state()!r} active={self._config_state()!r}"
            )
        historical._validate_runtime_state()
        self._validate_runtime_state()
        if historical.best_score is None:
            return False
        if self.best_score is None:
            self.load_state_dict(historical.state_dict())
            return True

        historical_is_better = (
            historical.best_score > self.best_score + self.min_delta
            if self.mode == "max"
            else historical.best_score < self.best_score - self.min_delta
        )
        if not historical_is_better:
            return False
        if historical.last_observed_step != self.last_observed_step:
            raise ValueError(
                "cannot safely merge a better historical best after a newer validation observation: "
                f"historical_last_step={historical.last_observed_step} "
                f"current_last_step={self.last_observed_step}"
            )
        self.load_state_dict(historical.state_dict())
        return True

    def should_observe_initial_validation(self, *, step: int) -> bool:
        """Return whether startup validation belongs to the stopping sequence.

        Step zero establishes the initial baseline. On resume, a checkpoint from
        a validation step is observed again only through the idempotent path in
        :meth:`observe`. A checkpoint saved between scheduled validations does
        not turn its startup diagnostic validation into an extra patience miss.
        """

        if not self.include_initial_validation:
            return False
        if self.last_observed_step is None:
            return step == 0
        return step == self.last_observed_step

    def _metrics(
        self,
        *,
        score: float,
        improved: bool,
        triggered: bool,
        duplicate: bool,
    ) -> dict[str, int | float]:
        assert self.best_score is not None
        assert self.best_step is not None
        return {
            "trainer/early_stopping/current_score": score,
            "trainer/early_stopping/best_score": float(self.best_score),
            "trainer/early_stopping/best_step": int(self.best_step),
            "trainer/early_stopping/non_improving_rounds": self.non_improving_rounds,
            "trainer/early_stopping/improved": int(improved),
            "trainer/early_stopping/triggered": int(triggered),
            "trainer/early_stopping/duplicate_observation": int(duplicate),
        }

    def observe(self, metrics: dict, *, step: int) -> tuple[bool, dict[str, int | float]]:
        if self.metric not in metrics:
            available = ", ".join(sorted(metrics))
            raise KeyError(
                f"early-stopping metric {self.metric!r} was not emitted at step {step}; available metrics: {available}"
            )
        score = float(metrics[self.metric])
        if not isfinite(score):
            raise ValueError(f"early-stopping metric {self.metric!r} is non-finite at step {step}: {score}")
        if self.last_observed_step is not None:
            if step < self.last_observed_step:
                raise ValueError(
                    "early-stopping observations must be monotonic: "
                    f"last_step={self.last_observed_step} new_step={step}"
                )
            if step == self.last_observed_step:
                assert self.last_observed_score is not None
                assert self.last_observed_improved is not None
                return self.last_observed_triggered, self._metrics(
                    score=self.last_observed_score,
                    improved=self.last_observed_improved,
                    triggered=self.last_observed_triggered,
                    duplicate=True,
                )

        if self.best_score is None:
            improved = True
        elif self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.best_step = step
            self.non_improving_rounds = 0
        else:
            self.non_improving_rounds += 1

        triggered = self.non_improving_rounds >= self.patience
        self.last_observed_step = step
        self.last_observed_score = score
        self.last_observed_improved = improved
        self.last_observed_triggered = triggered
        return triggered, self._metrics(
            score=score,
            improved=improved,
            triggered=triggered,
            duplicate=False,
        )
