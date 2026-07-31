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

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPOSITORY_ROOT / "rl-distill-scripts" / "gemma4_topk_distill_fsdp2.sh"
TEACHER_IDENTITY = "a" * 64
STUDENT_IDENTITY = "b" * 64
OVERLAY_SCHEMA_VERSION = "gemma4-hf-bf16-sdpa-topk-overlay-v1"


def _write_executable(path: Path, contents: str) -> None:
    path.write_text(contents, encoding="utf-8")
    path.chmod(0o755)


def _base_environment(tmp_path: Path) -> tuple[dict[str, str], Path]:
    student = tmp_path / "student"
    student.mkdir()
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(fake_bin / "torchrun", "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\"\n")
    environment = dict(os.environ)
    environment.update(
        {
            "ALLOW_QUESTION_OVERLAP": "false",
            "DATASET_INDEX": str(tmp_path / "dataset-index.json"),
            "DISTILL_DIRECTION": "e4b_rl100_to_e2b",
            "EXPECTED_STUDENT_IDENTITY_SHA256": STUDENT_IDENTITY,
            "EXPECTED_TEACHER_IDENTITY_SHA256": TEACHER_IDENTITY,
            "HF_PUSH_ENABLE": "false",
            "LOAD_DOTENV": "false",
            "MODEL_PATH": str(student),
            "NPROC_PER_NODE": "1",
            "PATH": f"{fake_bin}:{environment['PATH']}",
            "PREFLIGHT_LOCAL_FILES_ONLY": "true",
            "TRAIN_LOGGER": '["console"]',
            "VENV": str(tmp_path / "missing-venv"),
        }
    )
    return environment, fake_bin


def _preflight_output(train_files: list[str], validation_files: list[str], **overrides: str) -> str:
    values = {
        "TRAIN_FILES_HYDRA": json.dumps(train_files, separators=(",", ":")),
        "VAL_FILES_HYDRA": json.dumps(validation_files, separators=(",", ":")),
        "TOPK_WIDTH": "128",
        "TOPK_VALIDATION_TOLERANCE": "0.0025",
        "DATASET_INDEX_SHA256": "c" * 64,
        "GENERATION_EXPERIMENT_SHA256": "d" * 64,
        "DIRECTION": "e4b_rl100_to_e2b",
        "TEACHER_IDENTITY_SHA256": TEACHER_IDENTITY,
        "STUDENT_IDENTITY_SHA256": STUDENT_IDENTITY,
        "STUDENT_TOKENIZER_SHA256": "e" * 64,
    }
    values.update(overrides)
    return "".join(f"{key}={value}\n" for key, value in values.items())


def _write_fake_preflight_python(
    path: Path,
    output: str,
    *,
    schema_version: str = "gemma4-distill-topk-v1",
    argv_record: Path | None = None,
    verifier_argv_record: Path | None = None,
) -> None:
    preflight_record = (
        f"    Path({str(argv_record)!r}).write_text(json.dumps(sys.argv[1:]), encoding='utf-8')\n"
        if argv_record is not None
        else ""
    )
    verifier_record = (
        f"    Path({str(verifier_argv_record)!r}).write_text(json.dumps(sys.argv[1:]), encoding='utf-8')\n"
        if verifier_argv_record is not None
        else "    pass\n"
    )
    _write_executable(
        path,
        "#!/usr/bin/env python3\n"
        "import json\n"
        "import sys\n"
        "from pathlib import Path\n"
        f"schema_version = {schema_version!r}\n"
        "if len(sys.argv) > 1 and sys.argv[1] == '-c':\n"
        "    print(schema_version)\n"
        "elif len(sys.argv) > 1 and sys.argv[1].endswith('verify_gemma4_fsdp2_training_audit.py'):\n"
        + verifier_record
        + "else:\n"
        + preflight_record
        + f"    print({output!r}, end='')\n",
    )


def _run_launcher(environment: dict[str, str], *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(LAUNCHER), *arguments],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_launcher_uses_preflight_hydra_lists_without_eval(tmp_path: Path) -> None:
    environment, fake_bin = _base_environment(tmp_path)
    sentinel = tmp_path / "must-not-be-created"
    train_files = [f"$(touch {sentinel})", "/tmp/train with spaces.parquet"]
    validation_files = ["/tmp/validation.parquet"]
    fake_python = fake_bin / "preflight-python"
    _write_fake_preflight_python(fake_python, _preflight_output(train_files, validation_files))
    environment["PYTHON_BIN"] = str(fake_python)

    result = _run_launcher(environment)

    assert result.returncode == 0, result.stderr
    assert not sentinel.exists()
    argv = result.stdout.splitlines()
    assert f"data.train_files={json.dumps(train_files, separators=(',', ':'))}" in argv
    assert f"data.val_files={json.dumps(validation_files, separators=(',', ':'))}" in argv
    assert "teacher_model.top_k=128" in argv
    assert "teacher_model.chunk_size=4096" in argv
    assert "data.teacher_topk_validation_tolerance=0.0025" in argv
    assert "data.micro_batch_size_per_gpu=1" in argv
    assert "data.max_padded_tokens_per_microbatch=4096" in argv
    assert (
        "+engine.mixed_precision={param_dtype:bf16,reduce_dtype:fp32,buffer_dtype:fp32,"
        "cast_forward_inputs:true}" in argv
    )


def test_launcher_rejects_non_bf16_gemma4_parameter_views_without_explicit_opt_in(tmp_path: Path) -> None:
    environment, fake_bin = _base_environment(tmp_path)
    fake_python = fake_bin / "preflight-python"
    _write_fake_preflight_python(
        fake_python,
        _preflight_output(["/tmp/train.parquet"], ["/tmp/validation.parquet"]),
    )
    environment.update({"PYTHON_BIN": str(fake_python), "FSDP_PARAM_DTYPE": "fp32"})

    result = _run_launcher(environment)

    assert result.returncode == 2
    assert "Gemma 4 production requires FSDP_PARAM_DTYPE=bf16" in result.stderr

    environment["ALLOW_UNSAFE_GEMMA4_FSDP_PARAM_DTYPE"] = "true"
    result = _run_launcher(environment)

    assert result.returncode == 0, result.stderr
    assert (
        "+engine.mixed_precision={param_dtype:fp32,reduce_dtype:fp32,buffer_dtype:fp32,"
        "cast_forward_inputs:true}" in result.stdout.splitlines()
    )


def test_launcher_rejects_unverified_gemma4_batching_contract(tmp_path: Path) -> None:
    environment, fake_bin = _base_environment(tmp_path)
    environment["SOURCE_DATASET_INDEX"] = str(tmp_path / "source-dataset-index.json")
    environment["TRAINING_ENGINE_AUDIT_RECEIPT"] = str(tmp_path / "training-engine-audit.json")
    environment["NPROC_PER_NODE"] = "8"
    fake_python = fake_bin / "preflight-python"
    _write_fake_preflight_python(
        fake_python,
        _preflight_output(["/tmp/train.parquet"], ["/tmp/validation.parquet"]),
        schema_version=OVERLAY_SCHEMA_VERSION,
    )
    environment.update({"PYTHON_BIN": str(fake_python), "MAX_PADDED_TOKENS_PER_MICROBATCH": "0"})

    result = _run_launcher(environment)

    assert result.returncode == 2
    assert "must be a positive integer" in result.stderr

    invalid_contracts = {
        "MAX_PADDED_TOKENS_PER_MICROBATCH": "5120",
        "MICRO_BATCH_SIZE_PER_GPU": "2",
        "TRAIN_BATCH_SIZE": "64",
        "FULL_VOCAB_KL_CHUNK_SIZE": "1024",
        "MAX_LENGTH": "8192",
        "MAX_TOKEN_LEN_PER_GPU": "8192",
        "CLAMP_MIN_TOPK_KL": "true",
        "CHECKPOINT_DISTILL_CHUNKS": "false",
        "NPROC_PER_NODE": "4",
    }
    for name, value in invalid_contracts.items():
        invalid_environment = dict(environment)
        invalid_environment["MAX_PADDED_TOKENS_PER_MICROBATCH"] = "4096"
        invalid_environment[name] = value
        result = _run_launcher(invalid_environment)
        assert result.returncode == 2, (name, result.stderr)
        assert "audited overlay contract requires 8 GPUs" in result.stderr


def test_launcher_rejects_invalid_gradient_gate(tmp_path: Path) -> None:
    environment, fake_bin = _base_environment(tmp_path)
    fake_python = fake_bin / "preflight-python"
    _write_fake_preflight_python(
        fake_python,
        _preflight_output(["/tmp/train.parquet"], ["/tmp/validation.parquet"]),
    )
    environment.update({"PYTHON_BIN": str(fake_python), "VERL_MAX_PRECLIP_GRAD_NORM": "0"})

    result = _run_launcher(environment)

    assert result.returncode == 2
    assert "VERL_MAX_PRECLIP_GRAD_NORM must be a positive decimal number" in result.stderr


def test_launcher_forwards_explicit_schedule_and_epoch_contract(tmp_path: Path) -> None:
    environment, fake_bin = _base_environment(tmp_path)
    fake_python = fake_bin / "preflight-python"
    _write_fake_preflight_python(
        fake_python,
        _preflight_output(["/tmp/train.parquet"], ["/tmp/validation.parquet"]),
    )
    environment.update(
        {
            "PYTHON_BIN": str(fake_python),
            "LR": "2e-6",
            "LR_SCHEDULER_TYPE": "linear",
            "MIN_LR_RATIO": "0.1",
            "TOTAL_EPOCHS": "2",
            "TOTAL_TRAINING_STEPS": "750",
            "TRAIN_BATCH_SIZE": "128",
        }
    )

    result = _run_launcher(environment)

    assert result.returncode == 0, result.stderr
    argv = result.stdout.splitlines()
    assert "data.train_batch_size=128" in argv
    assert "optim.lr=2e-6" in argv
    assert "optim.lr_scheduler_type=linear" in argv
    assert "optim.min_lr_ratio=0.1" in argv
    assert "optim.total_training_steps=750" in argv
    assert "trainer.total_epochs=2" in argv
    assert "trainer.total_training_steps=750" in argv


def test_launcher_rejects_invalid_schedule_or_epoch(tmp_path: Path) -> None:
    environment, fake_bin = _base_environment(tmp_path)
    fake_python = fake_bin / "preflight-python"
    _write_fake_preflight_python(
        fake_python,
        _preflight_output(["/tmp/train.parquet"], ["/tmp/validation.parquet"]),
    )
    environment["PYTHON_BIN"] = str(fake_python)
    environment["LR_SCHEDULER_TYPE"] = "polynomial"

    result = _run_launcher(environment)

    assert result.returncode == 2
    assert "LR_SCHEDULER_TYPE must be constant, cosine, or linear" in result.stderr

    environment["LR_SCHEDULER_TYPE"] = "linear"
    environment["TOTAL_EPOCHS"] = "0"
    result = _run_launcher(environment)

    assert result.returncode == 2
    assert "TOTAL_EPOCHS must be a positive integer" in result.stderr


def test_launcher_rejects_identity_changed_after_preflight(tmp_path: Path) -> None:
    environment, fake_bin = _base_environment(tmp_path)
    fake_python = fake_bin / "preflight-python"
    _write_fake_preflight_python(
        fake_python,
        _preflight_output(
            ["/tmp/train.parquet"],
            ["/tmp/validation.parquet"],
            STUDENT_IDENTITY_SHA256="f" * 64,
        ),
    )
    environment["PYTHON_BIN"] = str(fake_python)

    result = _run_launcher(environment)

    assert result.returncode == 2
    assert "unexpected student identity" in result.stderr


def test_direct_file_bypass_is_limited_to_non_uploading_smoke(tmp_path: Path) -> None:
    environment, _ = _base_environment(tmp_path)
    environment.pop("DATASET_INDEX")
    environment.update(
        {
            "SMOKE_ONLY_ALLOW_DIRECT_FILES": "true",
            "TRAIN_FILE": "/tmp/train.parquet",
            "VAL_FILE": "/tmp/validation.parquet",
            "TOTAL_TRAINING_STEPS": "3",
        }
    )

    result = _run_launcher(environment)

    assert result.returncode == 2
    assert "between 1 and 2" in result.stderr


def test_launcher_rejects_post_preflight_hydra_overrides(tmp_path: Path) -> None:
    environment, fake_bin = _base_environment(tmp_path)
    marker = tmp_path / "torchrun-was-called"
    _write_executable(fake_bin / "torchrun", f"#!/usr/bin/env bash\ntouch {marker}\n")

    result = _run_launcher(environment, "data.train_files=/tmp/unvalidated.parquet")

    assert result.returncode == 2
    assert "Hydra overrides are disabled" in result.stderr
    assert not marker.exists()


def test_launcher_routes_overlay_schema_to_strict_overlay_preflight(tmp_path: Path) -> None:
    environment, fake_bin = _base_environment(tmp_path)
    source_index = tmp_path / "source-dataset-index.json"
    environment["SOURCE_DATASET_INDEX"] = str(source_index)
    environment["TRAINING_ENGINE_AUDIT_RECEIPT"] = str(tmp_path / "training-engine-audit.json")
    environment["NPROC_PER_NODE"] = "8"
    fake_python = fake_bin / "preflight-python"
    argv_record = tmp_path / "preflight-argv.json"
    verifier_argv_record = tmp_path / "verifier-argv.json"
    _write_fake_preflight_python(
        fake_python,
        _preflight_output(["/tmp/overlay-train.parquet"], ["/tmp/overlay-validation.parquet"]),
        schema_version=OVERLAY_SCHEMA_VERSION,
        argv_record=argv_record,
        verifier_argv_record=verifier_argv_record,
    )
    environment["PYTHON_BIN"] = str(fake_python)

    result = _run_launcher(environment)

    assert result.returncode == 0, result.stderr
    argv = json.loads(argv_record.read_text(encoding="utf-8"))
    assert argv[0].endswith("/data/preflight_gemma4_training_topk_overlay.py")
    source_flag = argv.index("--source-dataset-index")
    assert argv[source_flag + 1] == str(source_index)
    receipt_flag = argv.index("--receipt-cache")
    assert argv[receipt_flag + 1] == str(tmp_path / "training_preflight_receipt.json")
    verifier_argv = json.loads(verifier_argv_record.read_text(encoding="utf-8"))
    expected_verifier_arguments = {
        "--expected-world-size": "8",
        "--expected-train-batch-size": "128",
        "--expected-train-batches": "3",
        "--expected-micro-batch-size-per-gpu": "1",
        "--expected-max-padded-tokens-per-microbatch": "4096",
        "--expected-kl-chunk-size": "4096",
        "--expected-max-length": "12288",
    }
    for flag, expected in expected_verifier_arguments.items():
        flag_index = verifier_argv.index(flag)
        assert verifier_argv[flag_index + 1] == expected
    assert 'data.train_files=["/tmp/overlay-train.parquet"]' in result.stdout.splitlines()


def test_launcher_can_force_overlay_receipt_refresh(tmp_path: Path) -> None:
    environment, fake_bin = _base_environment(tmp_path)
    environment["SOURCE_DATASET_INDEX"] = str(tmp_path / "source-dataset-index.json")
    environment["TRAINING_ENGINE_AUDIT_RECEIPT"] = str(tmp_path / "training-engine-audit.json")
    environment["NPROC_PER_NODE"] = "8"
    environment["REFRESH_PREFLIGHT_RECEIPT"] = "true"
    fake_python = fake_bin / "preflight-python"
    argv_record = tmp_path / "preflight-argv.json"
    _write_fake_preflight_python(
        fake_python,
        _preflight_output(["/tmp/overlay-train.parquet"], ["/tmp/overlay-validation.parquet"]),
        schema_version=OVERLAY_SCHEMA_VERSION,
        argv_record=argv_record,
    )
    environment["PYTHON_BIN"] = str(fake_python)

    result = _run_launcher(environment)

    assert result.returncode == 0, result.stderr
    assert "--refresh-receipt" in json.loads(argv_record.read_text(encoding="utf-8"))


def test_launcher_requires_explicit_source_index_for_overlay(tmp_path: Path) -> None:
    environment, fake_bin = _base_environment(tmp_path)
    fake_python = fake_bin / "preflight-python"
    _write_fake_preflight_python(
        fake_python,
        _preflight_output(["/tmp/train.parquet"], ["/tmp/validation.parquet"]),
        schema_version=OVERLAY_SCHEMA_VERSION,
    )
    environment["PYTHON_BIN"] = str(fake_python)

    result = _run_launcher(environment)

    assert result.returncode == 2
    assert "SOURCE_DATASET_INDEX" in result.stderr


def test_launcher_requires_training_engine_audit_for_overlay(tmp_path: Path) -> None:
    environment, fake_bin = _base_environment(tmp_path)
    environment["SOURCE_DATASET_INDEX"] = str(tmp_path / "source-dataset-index.json")
    fake_python = fake_bin / "preflight-python"
    _write_fake_preflight_python(
        fake_python,
        _preflight_output(["/tmp/train.parquet"], ["/tmp/validation.parquet"]),
        schema_version=OVERLAY_SCHEMA_VERSION,
    )
    environment["PYTHON_BIN"] = str(fake_python)

    result = _run_launcher(environment)

    assert result.returncode != 0
    assert "TRAINING_ENGINE_AUDIT_RECEIPT" in result.stderr


def test_launcher_rejects_source_index_for_vllm_bundle(tmp_path: Path) -> None:
    environment, fake_bin = _base_environment(tmp_path)
    environment["SOURCE_DATASET_INDEX"] = str(tmp_path / "unexpected-source.json")
    fake_python = fake_bin / "preflight-python"
    _write_fake_preflight_python(
        fake_python,
        _preflight_output(["/tmp/train.parquet"], ["/tmp/validation.parquet"]),
    )
    environment["PYTHON_BIN"] = str(fake_python)

    result = _run_launcher(environment)

    assert result.returncode == 2
    assert "valid only for an unsharded-HF overlay" in result.stderr
