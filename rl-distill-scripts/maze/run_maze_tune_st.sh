#!/usr/bin/env bash
# ScaleTrain pod entry (1 GPU): tune_pass1.py, i.e. hit a target pass@1 by interpolating two checkpoints of one run, screened
# on the 1000 dev mazes at 2048 samples per maze, with a fresh-seed confirmation of the chosen model. Endpoints are S3
# prefixes (s3://...) or Hub paths (hub:<dataset repo>:<path in repo>). tune.json + the chosen model (best/) go to ${OUT_S3}
# (re-uploaded after the run; a relaunch resumes from ${OUT_S3}/tune.json). Writes ${OUT_S3}/COMPLETE when done.
#   env: A_SRC, B_SRC, TARGET, OUT_S3, optional PA / PB (endpoint pass@1 already measured at 2048 x 1000), MAX_ITERS (3)
source "$(dirname "${BASH_SOURCE[0]}")/st_setup.sh"
A_SRC="${A_SRC:?}"; B_SRC="${B_SRC:?}"; TARGET="${TARGET:?}"; OUT_S3="${OUT_S3:?}"
echo "MAZE_TUNE_START $(date -u +%FT%TZ) target=${TARGET} a=${A_SRC} b=${B_SRC}"
fetch() {  # fetch <src> <local dir>
  case "$1" in
    s3://*) aws s3 sync --only-show-errors "$1" "$2" ;;
    hub:*) "${PY}" - "$1" "$2" <<'EOF'
import os, shutil, sys
from huggingface_hub import snapshot_download
_, repo, path = sys.argv[1].split(":", 2)
root = snapshot_download(repo, repo_type="dataset", allow_patterns=[f"{path}/*"], local_dir="/tmp/maze/hub")
shutil.copytree(os.path.join(root, path), sys.argv[2], dirs_exist_ok=True)
EOF
    ;;
    *) echo "FATAL: source must be s3://... or hub:<repo>:<path>: $1" >&2; exit 2 ;;
  esac
}
fetch "${A_SRC}" /tmp/maze/src/a; fetch "${B_SRC}" /tmp/maze/src/b
OUT=/tmp/maze/tune; mkdir -p "${OUT}"
aws s3 cp --only-show-errors "${OUT_S3}/tune.json" "${OUT}/tune.json" 2>/dev/null || true
extra=(); [ -n "${PA:-}" ] && extra+=(--pa "${PA}"); [ -n "${PB:-}" ] && extra+=(--pb "${PB}")
( while sleep 300; do aws s3 cp --only-show-errors "${OUT}/tune.json" "${OUT_S3}/tune.json" 2>/dev/null || true; done ) & sidecar=$!
(cd /tmp/maze && HOME=/tmp/maze "${PY}" "${M}/tune_pass1.py" --tailrl-root "${T}" --a /tmp/maze/src/a --b /tmp/maze/src/b \
   --target "${TARGET}" --out "${OUT}" --max-iters "${MAX_ITERS:-3}" "${extra[@]}")
kill "${sidecar}" 2>/dev/null || true
aws s3 cp --only-show-errors "${OUT}/tune.json" "${OUT_S3}/tune.json"
aws s3 sync --only-show-errors "${OUT}/best" "${OUT_S3}/best"
echo "{\"done\": \"$(date -u +%FT%TZ)\", \"target\": ${TARGET}, \"a\": \"${A_SRC}\", \"b\": \"${B_SRC}\"}" | aws s3 cp - "${OUT_S3}/COMPLETE"
echo "MAZE_TUNE_DONE ${OUT_S3}"
