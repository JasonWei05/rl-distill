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

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPOSITORY_ROOT / "rl-distill-scripts" / "data" / "run_gemma4_trace_dataset.sh"


def test_trace_dataset_launcher_pins_registered_generation_contract() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")

    for required_argument in (
        "--samples-per-question 5",
        "--temperature 1.0",
        "--top-p 1.0",
        "--sampling-top-k -1",
        "--max-prompt-tokens 4096",
        "--max-response-tokens 8192",
        "--max-model-len 12288",
        "--tensor-parallel-size 1",
    ):
        assert required_argument in source


def test_trace_dataset_launcher_validates_both_complete_splits_before_upload() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")

    assert "run_split train" in source
    assert "run_split validation" in source
    assert '--split-dir "train=${output_root}/train"' in source
    assert '--split-dir "validation=${output_root}/validation"' in source
    assert "--expected-train-questions 9723" in source
    assert "--expected-validation-questions 200" in source
    assert '"$uploader"' in source
    assert '--dataset-path "$output_root"' in source


def test_trace_dataset_launcher_requires_clean_code_and_environment_only_token() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")

    assert 'git -C "$REPO_ROOT" status --porcelain' in source
    assert "HF_TOKEN must be exported" in source
    assert "--token" not in source
    assert "--allow-question-overlap" in source
    assert "GPU identifiers must be unique" in source
    assert "--output-root must be outside the source repository" in source
    assert "skip_upload != true && -z $hf_repo_id" in source


def test_trace_dataset_launcher_isolates_and_cleans_worker_process_groups() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")

    assert "setsid env CUDA_VISIBLE_DEVICES" in source
    assert 'kill -TERM -- "-${worker_pid}"' in source
    assert "trap terminate_worker INT TERM" in source


def test_trace_dataset_launcher_does_not_retry_deterministic_validation_failures() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")

    assert "worker_status == 3" in source
    assert "refusing identical-seed retries" in source
