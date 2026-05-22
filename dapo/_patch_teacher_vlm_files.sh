#!/usr/bin/env bash
# Download the VLM-specific metadata files that verl's checkpoint save omitted
# (preprocessor_config.json, processor_config.json, etc.) and place them alongside
# our trained safetensors so vLLM can load the model as Gemma3ForConditionalGeneration.
set -euo pipefail
cd /mlx_devbox/users/jason.wei/playground/rl-distill
source .venv/bin/activate
set -a; source .env; set +a

DST=/tmp/teacher_27b_step40

python3 - <<'PY'
import os
from huggingface_hub import HfApi, hf_hub_download

repo = "google/gemma-3-27b-it"
dst = "/tmp/teacher_27b_step40"
api = HfApi()

# Enumerate non-model, non-safetensors files in the source repo
files = api.list_repo_files(repo_id=repo, repo_type="model")
skip_prefixes = ("model-", "pytorch_model-", "model.safetensors.index")
skip_exact = {"config.json"}    # we already have our trained config
dl = []
for f in files:
    base = os.path.basename(f)
    if any(base.startswith(p) for p in skip_prefixes) or base in skip_exact:
        continue
    # skip weight files and model indexes; keep configs, processor json, chat templates, etc.
    if f.endswith(".safetensors") or f.endswith(".bin"):
        continue
    dl.append(f)

present = set(os.listdir(dst))
for f in dl:
    base = os.path.basename(f)
    if base in present:
        continue
    try:
        p = hf_hub_download(repo_id=repo, filename=f, local_dir=dst)
        print(f"downloaded {f} -> {p}")
    except Exception as e:
        print(f"SKIP {f}: {e}")

print("")
print("now contains:")
for n in sorted(os.listdir(dst)):
    print("  ", n)
PY
