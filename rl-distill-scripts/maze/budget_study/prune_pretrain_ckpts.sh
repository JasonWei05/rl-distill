#!/usr/bin/env bash
# Prune the pretraining checkpoints (text-maze study) down to what the three matched pairs need.
# Keeps, per pair (p10 / p1 / p01) and the alt1 / alt2 pairs:
#   - the averaged candidate dirs the matched/ symlinks point to (avg-<center>-w60)
#   - the raw checkpoints inside each primary candidate's averaging window (center +- 60, stride 20: 7 per candidate)
#   - "same-data" raw pairs: S and L checkpoints at the SAME step (data_seed 0 => identical batches seen), at both the
#     S0 center and the L0 center of each pair, in S_ds0_is4 and L_ds0_is0_rep
# and every metrics / screening / selection file. Everything else under pretrain20k/*/ (ckpt-*, avg-*, interp-*), and the
# superseded pretrain/ (6k-step runs) and pretrain_probe/ directories, is deleted.
#   DRY_RUN=1 bash prune_pretrain_ckpts.sh   # list only (default)
#   DRY_RUN=0 bash prune_pretrain_ckpts.sh   # delete
set -euo pipefail
R=/home/jiahaoz4/tailrl_maze_runs; P=$R/pretrain20k; DRY_RUN=${DRY_RUN:-1}
S_RUN=S_ds0_is4; L_RUN=L_ds0_is0_rep
S_CENTERS=(10040 9160 8080); L_CENTERS=(4020 3240 2680)
declare -A KEEP
# averaged candidates behind every matched/ symlink (primary, p10, p1, p01, alt1, alt2)
for l in "$R"/matched/*_S0 "$R"/matched/*_L0; do t=$(readlink -f "$l"); [[ $t == $P/* ]] && KEEP["$t"]=1; done
# averaging windows of the primary candidates, and same-step raw pairs at both centers
for c in "${S_CENTERS[@]}" "${L_CENTERS[@]}"; do
  for run in $S_RUN $L_RUN; do KEEP["$P/$run/ckpt-$c"]=1; done
done
for c in "${S_CENTERS[@]}"; do for s in $(seq $((c-60)) 20 $((c+60))); do KEEP["$P/$S_RUN/ckpt-$s"]=1; done; done
for c in "${L_CENTERS[@]}"; do for s in $(seq $((c-60)) 20 $((c+60))); do KEEP["$P/$L_RUN/ckpt-$s"]=1; done; done
missing=0; for k in "${!KEEP[@]}"; do [ -d "$k" ] || { echo "KEEP target missing: $k"; missing=1; }; done
[ $missing = 0 ] || { echo "abort: a keep target is missing"; exit 1; }
echo "keep set: ${#KEEP[@]} dirs"; printf '%s\n' "${!KEEP[@]}" | sed "s|$P/||" | sort | tr '\n' ' '; echo
total_del=0; n_del=0
for run in "$P"/*/; do
  del=(); for d in "$run"ckpt-* "$run"avg-* "$run"interp-*; do [ -d "$d" ] || continue; [[ -v KEEP["${d%/}"] ]] || del+=("${d%/}"); done
  kept=$(ls "$run" | grep -cE '^(ckpt|avg|interp)-' ); kept=$((kept - ${#del[@]}))
  echo "$(basename "$run"): delete ${#del[@]} dirs, keep $kept checkpoint dirs"
  n_del=$((n_del + ${#del[@]}))
  if [ "$DRY_RUN" = 0 ]; then printf '%s\0' "${del[@]}" | xargs -0 -r rm -rf; fi
done
for extra in "$R/pretrain" "$R/pretrain_probe"; do [ -d "$extra" ] && { echo "delete whole dir: $extra ($(du -sh "$extra" | cut -f1))"; [ "$DRY_RUN" = 0 ] && rm -rf "$extra"; }; done
echo "$([ "$DRY_RUN" = 0 ] && echo DELETED || echo DRY RUN): $n_del checkpoint dirs under pretrain20k + pretrain/ + pretrain_probe/"
[ "$DRY_RUN" = 0 ] && { du -sh "$P"; df -h /home | awk 'NR==2{print "free now: "$4}'; }
exit 0
