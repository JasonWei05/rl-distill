#!/usr/bin/env bash
# ScaleTrain pod entry (1 GPU): pass@k screening of text-maze checkpoints with screen_sft_ckpts.py, results mirrored to S3.
# Builds TailRL @ 5682c6a + maze/tailrl_text_maze.patch and its venv in /tmp (BUDGET_STUDY_RUNBOOK "Setting up on another
# machine"), downloads the requested matched bases from Jiahao's public Hub archive, screens them on the dev set (TailRL
# data/eval1000) and copies the results JSON to ${OUT_S3} after every checkpoint; labels already in ${OUT_S3} are skipped,
# so a relaunch resumes. Writes ${OUT_S3%.json}.COMPLETE when done.
#   env: BASES ('+'-separated <size>_<level>, since --env-vars splits on commas; size medium|large, level p01|p03|p1|p3|p5|p10), OUT_S3 (s3://.../<name>.json),
#        K (2048), N_PROMPTS (1000), SEED (0)
#   launched by: CODE_REF=<jiahaoz commit> bash launch_st_with_code.sh --gpus-per-instance 1 --run-file ../maze/run_maze_screen_st.sh ...
source "$(dirname "${BASH_SOURCE[0]}")/st_setup.sh"
echo "MAZE_START $(date -u +%FT%TZ) host=$(hostname) nproc=$(nproc) bases=${BASES:-unset}"
BASES="${BASES:?BASES=<size>_<level>+...}"; OUT_S3="${OUT_S3:?OUT_S3=s3://.../<name>.json}"
K="${K:-2048}"; N_PROMPTS="${N_PROMPTS:-1000}"; SEED="${SEED:-0}"
mkdir -p /tmp/maze/bases

# bases: Hub path per level (budget study: used/<level>; extra levels: extra/used/<level>), S0 = medium, L0 = large
paths="$("${PY}" - "${BASES}" <<'EOF'
import os, sys
from huggingface_hub import snapshot_download
repo = "hi-todayis-jh/tailrl-text-maze-budget-study"
out = []
for lab in sys.argv[1].replace(",", "+").split("+"):
    size, level = lab.split("_")
    sub = {"medium": "S0_base", "large": "L0_base"}[size]
    hub = f"{'extra/' if level in ('p03', 'p3', 'p5') else ''}used/{level}/{sub}"
    root = snapshot_download(repo, repo_type="dataset", allow_patterns=[f"{hub}/*"], local_dir="/tmp/maze/hub")
    link = f"/tmp/maze/bases/{lab}"
    if not os.path.islink(link):
        os.symlink(os.path.join(root, hub), link)
    out.append(link)
print(" ".join(out))
EOF
)"
echo "MAZE_BASES ${paths}"

OUT=/tmp/maze/screen.json
aws s3 cp --only-show-errors "${OUT_S3}" "${OUT}" 2>/dev/null || echo '{}' > "${OUT}"
todo="$("${PY}" - "${OUT}" ${paths} <<'EOF'
import json, os, sys
done = json.load(open(sys.argv[1]))
print(" ".join(p for p in sys.argv[2:] if os.path.basename(p) not in done))
EOF
)"
echo "MAZE_TODO ${todo:-none}"
( last=""; while sleep 60; do h="$(sha256sum "${OUT}" | cut -c1-16)"; [ "${h}" != "${last}" ] && aws s3 cp --only-show-errors "${OUT}" "${OUT_S3}" && last="${h}"; done ) &
sidecar=$!
if [ -n "${todo}" ]; then
  (cd /tmp/maze && HOME=/tmp/maze "${PY}" "${M}/screen_sft_ckpts.py" --tailrl-root "${T}" --paths ${todo} --k "${K}" --n-prompts "${N_PROMPTS}" \
     --seed "${SEED}" --out "${OUT}")
fi
kill "${sidecar}" 2>/dev/null || true
aws s3 cp --only-show-errors "${OUT}" "${OUT_S3}"
echo "{\"done\": \"$(date -u +%FT%TZ)\", \"bases\": \"${BASES}\", \"k\": ${K}, \"n_prompts\": ${N_PROMPTS}}" | aws s3 cp - "${OUT_S3%.json}.COMPLETE"
echo "MAZE_DONE ${OUT_S3}"
