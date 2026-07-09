#!/usr/bin/env bash
# Ensure Gemma3 processor/tokenizer metadata exists alongside a converted MoE
# checkpoint (preprocessor_config.json, processor_config.json, ...) so
# AutoProcessor/vLLM consumers can load it. Downloads any missing root-level
# non-weight file from GEMMA3_PROCESSOR_BASE_REPO into MODEL_PATH ($1).
# Idempotent; never overwrites files already present in MODEL_PATH.
set -euo pipefail

MODEL_PATH="${1:?usage: ensure_gemma3_processor_files.sh <model_path>}"
BASE_REPO="${GEMMA3_PROCESSOR_BASE_REPO:-google/gemma-3-4b-pt}"

python3 - "${MODEL_PATH}" "${BASE_REPO}" <<'PY'
import os
import sys

from huggingface_hub import HfApi, hf_hub_download

dst, repo = sys.argv[1], sys.argv[2]
weight_suffixes = (".safetensors", ".bin", ".gguf", ".pt", ".onnx")
weight_prefixes = ("model-", "pytorch_model-", "model.safetensors.index")
# The converted checkpoint owns its architecture and sampling settings.
keep_local = {"config.json", "generation_config.json", "README.md"}

present = set(os.listdir(dst))
for name in HfApi().list_repo_files(repo_id=repo, repo_type="model"):
    if "/" in name:  # root-level metadata only
        continue
    if name.startswith(weight_prefixes) or name.endswith(weight_suffixes):
        continue
    if name in keep_local or name in present or name.startswith("."):
        continue
    path = hf_hub_download(repo_id=repo, filename=name, local_dir=dst)
    print(f"ensure_gemma3_processor_files: downloaded {name} -> {path}")
PY
