#!/usr/bin/env bash
# Generalized teacher snapshot download: pulls <TEACHER_REPO>/<TEACHER_REV>/*
# into a flat local dir suitable for loading as a normal HF model path.
# For VLM bases (Gemma 3 4B+), also fetches preprocessor/processor configs from
# the corresponding base PT repo if missing — vLLM init fails without them.
# Idempotent — safe to re-run.
#
# Required env:
#   TEACHER_REPO       HF repo id, e.g. JWei05/dapo-gemma3-4b-pt
#   TEACHER_REV        subfolder inside the repo, e.g. step_000060
# Optional env:
#   DEST               destination dir (default /tmp/verl/models/<basename(repo)>-<rev>)
#   BASE_HF_REPO       base repo for missing preprocessor files
#                      (default derived from `dapo-gemma3-<tag>-<pt|it>` -> `google/gemma-3-<tag>-<pt|it>`)
#   CHAT_TEMPLATE_REPO repo to copy a missing Gemma chat template from, e.g.
#                      google/gemma-3-27b-it for PT checkpoints saved without one.
#
# Prints the destination dir as the last line of stdout.
set -euo pipefail

: "${TEACHER_REPO:?TEACHER_REPO is required}"
: "${TEACHER_REV:?TEACHER_REV is required}"

REPO_BASENAME="$(basename "${TEACHER_REPO}")"
DEST="${DEST:-/tmp/verl/models/${REPO_BASENAME}-${TEACHER_REV}}"

# Derive base repo from common naming pattern if not provided.
if [ -z "${BASE_HF_REPO:-}" ]; then
    if [[ "${REPO_BASENAME}" =~ ^dapo-gemma3-([0-9]+b)-(pt|it)$ ]]; then
        BASE_HF_REPO="google/gemma-3-${BASH_REMATCH[1]}-${BASH_REMATCH[2]}"
    fi
fi

if [ ! -f "${DEST}/config.json" ]; then
    # Wipe any stale partial state from a prior failed launch (broken symlinks,
    # half-downloaded files) before re-downloading.
    rm -rf "${DEST}"
    mkdir -p "${DEST}"
    echo "[prep] downloading ${TEACHER_REPO}/${TEACHER_REV}/* -> ${DEST}" >&2
    python3 - <<PY
import os, glob, shutil
from huggingface_hub import snapshot_download
# Download to HF cache first, then move/copy the subfolder contents into DEST
# so we end up with a flat dir of real files (no symlinks back into the cache).
src = snapshot_download(
    repo_id="${TEACHER_REPO}",
    allow_patterns=["${TEACHER_REV}/*"],
)
sub = os.path.join(src, "${TEACHER_REV}")
for f in glob.glob(os.path.join(sub, "*")):
    dst = os.path.join("${DEST}", os.path.basename(f))
    # Resolve symlinks so we copy actual files (snapshot_download links cache
    # blobs into the snapshot dir). Use copy2 to preserve mtime; cheap because
    # both src cache and DEST live on /tmp.
    real = os.path.realpath(f)
    if os.path.isdir(real):
        shutil.copytree(real, dst, symlinks=False)
    else:
        shutil.copy2(real, dst)
print("[prep] ok ->", "${DEST}")
PY
else
    echo "[prep] already present at ${DEST}" >&2
fi

# Backfill VLM preprocessor / processor configs from base repo if absent.
if [ -n "${BASE_HF_REPO:-}" ]; then
    NEED_PREP=0
    [ ! -f "${DEST}/preprocessor_config.json" ] && NEED_PREP=1
    [ ! -f "${DEST}/processor_config.json" ] && NEED_PREP=1
    if [ "${NEED_PREP}" = "1" ]; then
        echo "[prep] backfilling preprocessor configs from ${BASE_HF_REPO}" >&2
        python3 - <<PY
import os
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
dest = "${DEST}"
base = "${BASE_HF_REPO}"
for fname in ("preprocessor_config.json", "processor_config.json"):
    out = os.path.join(dest, fname)
    if os.path.exists(out):
        continue
    try:
        src = hf_hub_download(repo_id=base, filename=fname)
        os.symlink(src, out)
        print(f"[prep] linked {base}/{fname} -> {out}")
    except EntryNotFoundError:
        print(f"[prep] base {base} has no {fname} (text-only model — ok)")
PY
    fi
fi

if [ -n "${CHAT_TEMPLATE_REPO:-}" ]; then
    echo "[prep] ensuring chat_template from ${CHAT_TEMPLATE_REPO} if missing" >&2
    python3 - <<PY
import json
import os
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError

dest = "${DEST}"
src_repo = "${CHAT_TEMPLATE_REPO}"
tcfg_path = os.path.join(dest, "tokenizer_config.json")
if not os.path.exists(tcfg_path):
    raise SystemExit(f"missing tokenizer_config.json in {dest}")

tcfg = json.load(open(tcfg_path))
if tcfg.get("chat_template"):
    print("[prep] chat_template already present")
else:
    chat_template = None
    try:
        src = hf_hub_download(repo_id=src_repo, filename="chat_template.json")
        chat_template = json.load(open(src)).get("chat_template")
        print(f"[prep] copied chat_template from {src_repo}/chat_template.json")
    except EntryNotFoundError:
        src = hf_hub_download(repo_id=src_repo, filename="tokenizer_config.json")
        chat_template = json.load(open(src)).get("chat_template")
        print(f"[prep] copied chat_template from {src_repo}/tokenizer_config.json")
    if not chat_template:
        raise SystemExit(f"could not find chat_template in {src_repo}")
    patched = chat_template.replace(
        '{{ raise_exception("Conversation roles must alternate user/assistant/user/assistant/...") }}',
        '{# alternation check disabled for verl/vLLM probing #}',
    )
    tcfg["chat_template"] = patched
    json.dump(tcfg, open(tcfg_path, "w"), indent=2)
    print(f"[prep] inlined chat_template into {tcfg_path} ({len(patched)} chars)")
PY
fi

echo "${DEST}"
