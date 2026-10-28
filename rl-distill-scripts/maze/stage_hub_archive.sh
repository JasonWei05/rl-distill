#!/usr/bin/env bash
# Stage the text-maze budget study checkpoints for the Hub (dataset repo), HF exports only (no optimizer / FSDP shards):
#   used/<pair>/<label>/        every checkpoint the study evaluates (Arm B points, teachers, student finals, bases)
#   archive/<run>.tar           one tar per RL run: global_step_*/actor/huggingface + flops_ledger.json + the console log
#   meta/                       plans, results, manifest, dashboard datasets, sha256 of the tars, used_manifest.json
# Usage: bash stage_hub_archive.sh [staging-dir]   (default ~/tailrl_maze_hub); idempotent, safe to re-run.
set -euo pipefail
R=/home/jiahaoz4/tailrl_maze_runs; C=$R/checkpoints/tailrl-text-maze; H=${1:-/home/jiahaoz4/tailrl_maze_hub}
DASH=/tmp/claude-25721/-home-jiahaoz4-rl-distill/48ed4138-f6ae-485e-8229-c366ed3e4e6e/scratchpad/dash
mkdir -p "$H/used" "$H/archive" "$H/meta"
echo "$(date -Is) staging into $H"

# 1. used subset: every hf path in the results files, plus the matched base checkpoints
python3 - "$R" "$H" <<'EOF'
import json, os, shutil, sys, glob
R, H = sys.argv[1], sys.argv[2]
rows = []
for pair in ["p10", "p1", "p01", "p10_s1"]:
    f = f"{R}/budget/{pair}/results_dev_k64.json"
    if not os.path.exists(f):
        continue
    for label, v in json.load(open(f)).items():
        src = v["hf"]
        dst = f"{H}/used/{pair}/{label}"
        if not os.path.isdir(dst):
            shutil.copytree(src, dst)
        rows.append(dict(pair=pair, label=label, hub_path=f"used/{pair}/{label}", local_path=src,
                         total_tflops=v["total_tflops"], note=v.get("note")))
for pair in ["p10", "p1", "p01"]:
    for s in ["S0", "L0"]:
        src = os.path.realpath(f"{R}/matched/{pair}_{s}")
        dst = f"{H}/used/{pair}/{s}_base"
        if not os.path.isdir(dst):
            shutil.copytree(src, dst)
        rows.append(dict(pair=pair, label=f"{s}_base", hub_path=f"used/{pair}/{s}_base", local_path=src,
                         total_tflops=0.0, note=f"matched starting checkpoint ({os.path.basename(src)})"))
json.dump(rows, open(f"{H}/meta/used_manifest.json", "w"), indent=1)
print(f"used/: {len(rows)} checkpoint dirs")
EOF

# 2. metadata
for p in p10 p1 p01 p10_s1; do
  mkdir -p "$H/meta/$p"
  for f in plan.json plan_ext_r25.json results_dev_k64.json results_test_k256.json teacher_curve_dev_k64.json; do
    [ -f "$R/budget/$p/$f" ] && cp "$R/budget/$p/$f" "$H/meta/$p/$f"
  done
done
cp "$R/matched/manifest.json" "$H/meta/matched_manifest.json"
mkdir -p "$H/meta/dashboard"; for f in results.json curves.json pairs.json; do [ -f "$DASH/$f" ] && cp "$DASH/$f" "$H/meta/dashboard/$f"; done

# 3. one tar per RL run (HF exports + ledgers + log), skipping any tar already complete
tar_run() {  # tar_run <run-tag> <tar-name>
  local run=$1 name=$2 d log
  d=$(ls -d "$C"/textmaze_"${run}"_grpo_binary_* 2>/dev/null | head -1) || true
  [ -n "$d" ] || { echo "no run dir for $run"; return 0; }
  [ -f "$H/archive/$name.tar.done" ] && { echo "archive/$name.tar already done"; return 0; }
  log=$R/rl/$(echo "$run" | sed 's/-s0$/_S0/; s/-l0$/_L0/')_rl.log
  echo "$(date -Is) tar $name <- $(basename "$d") ($(ls -d "$d"/global_step_* | wc -l) ckpts)"
  tar -cf "$H/archive/$name.tar" --exclude='*.pt' -C "$(dirname "$d")" "$(basename "$d")"
  [ -f "$log" ] && tar -rf "$H/archive/$name.tar" -C "$R/rl" "$(basename "$log")"
  touch "$H/archive/$name.tar.done"
}
for run in p10-s0 p10-l0 p10s1-s0 p10s1-l0 p1-s0 p1-l0 p1s1-s0 p01-s0 p01-l0 p01s1-s0 p01n64-s0; do tar_run "$run" "$run"; done
# small leftovers: the primary-pair trial runs, smoke tests, the earlier main runs, and the OPD student run dirs
if [ ! -f "$H/archive/misc.tar.done" ]; then
  echo "$(date -Is) tar misc"
  ( cd "$C" && ls -d textmaze_primary-* textmaze_smoke-* textmaze_main_* 2>/dev/null | tar -cf "$H/archive/misc.tar" --exclude='*.pt' -T - )
  touch "$H/archive/misc.tar.done"
fi
if [ ! -f "$H/archive/opd-runs.tar.done" ]; then
  echo "$(date -Is) tar opd-runs"
  ( cd "$C" && ls -d textmaze_p*-opd_* 2>/dev/null | tar -cf "$H/archive/opd-runs.tar" --exclude='*.pt' -T - )
  touch "$H/archive/opd-runs.tar.done"
fi
rm -f "$H"/archive/*.done.tmp
( cd "$H/archive" && sha256sum *.tar > "$H/meta/archive_sha256.txt" )
du -sh "$H"/used "$H"/archive "$H"/meta
echo "$(date -Is) STAGING_DONE"
