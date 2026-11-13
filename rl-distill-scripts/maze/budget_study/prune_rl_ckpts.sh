#!/usr/bin/env bash
# After the Hub archive is verified (verify_hub_archive.py prints VERIFY OK): free the local RL checkpoint space.
#   1. delete every optimizer / FSDP shard (*.pt) under checkpoints/tailrl-text-maze (resume state only; the runs are complete)
#   2. delete every global_step_* directory the study does not use (their HF exports are in archive/<run>.tar on the Hub)
# Keeps every checkpoint named in meta/used_manifest.json (minus *.pt, so all recorded paths stay valid), the run-level
# files (logs, latest_checkpointed_iteration.txt) and everything outside checkpoints/ (budget/, matched/, rl/).
#   DRY_RUN=1 bash prune_rl_ckpts.sh   # list only (default)
#   DRY_RUN=0 bash prune_rl_ckpts.sh   # delete
set -euo pipefail
R=/home/jiahaoz4/tailrl_maze_runs; C=$R/checkpoints/tailrl-text-maze; H=${HUB_DIR:-/home/jiahaoz4/tailrl_maze_hub}; DRY_RUN=${DRY_RUN:-1}
[ -f "$H/meta/used_manifest.json" ] || { echo "no $H/meta/used_manifest.json"; exit 1; }
declare -A KEEPSTEP
while IFS= read -r k; do
  case $k in "$C"/*/global_step_*) s=${k#"$C"/}; s=${s%%/actor/*}; KEEPSTEP["$C/$s"]=1 ;; esac
done < <(python3 -c "import json,sys; [print(r['local_path']) for r in json.load(open(sys.argv[1]))]" "$H/meta/used_manifest.json")
for k in "${!KEEPSTEP[@]}"; do [ -d "$k" ] || { echo "keep target missing: $k"; exit 1; }; done
echo "keep: ${#KEEPSTEP[@]} checkpoint dirs under $C (minus *.pt)"
n_pt=$(find "$C" -name '*.pt' | wc -l)
s_pt=$(find "$C" -name '*.pt' -printf '%s\n' | awk '{s+=$1} END{printf "%.0f", s/1e9}')
echo "1. delete $n_pt *.pt files, $s_pt GB"
if [ "$DRY_RUN" = 0 ]; then find "$C" -name '*.pt' -delete; fi
del=(); for d in "$C"/*/global_step_*; do [[ -v KEEPSTEP["$d"] ]] || del+=("$d"); done
s_del=$(printf '%s\0' "${del[@]}" | xargs -0 -r du -sBM | awk '{s+=$1} END{printf "%.0f", s/1000}')
echo "2. delete ${#del[@]} unused checkpoint dirs, ~$s_del GB (before step 1: includes their *.pt)"
if [ "$DRY_RUN" = 0 ]; then printf '%s\0' "${del[@]}" | xargs -0 -r rm -rf; fi
echo "$([ "$DRY_RUN" = 0 ] && echo DELETED || echo DRY RUN)"
if [ "$DRY_RUN" = 0 ]; then du -sh "$C"; df -h /home | tail -1; fi
exit 0
