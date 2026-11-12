#!/usr/bin/env bash
# Small-model maze RL grid: 5 pass@1 levels x prompts per step {4,16,64,256} x RL seeds {0,1,2} = 60 GRPO+TIS runs,
# packed on one-GPU ScaleTrain pods (every pod gets a mix of batch sizes; largest batches start first).
# Bases: best-matching small seed per level (s3://.../maze/rl_small_v1/bases.json). One supervisor per pod.
#   GRID=v1: lr 1e-4, patience 20 (run names sml-*, pods maze-rl-g0..g3, s3://.../maze/rl_small_v1)
#   GRID=v2: lr 2.5e-5, patience 50 (run names sml2-*, pods maze-rl-v2-g0..g3, s3://.../maze/rl_small_v2)
#   GRID=sp1: the v2 recipe with the shortest-path reward (RL_REWARD=shortest) from the shortest-path-pretrained small
#     bases (s3://.../maze/rl_small_shortest_v1/bases/<level>_s<seed>, chosen by pick_shortest_bases.py, which prints
#     BASES_SPEC); run names sps-*, pods maze-rl-sp-*, W&B project tailrl-text-maze-shortest
#   GRID=sp2: sp1 at lr 1e-5 (small batches at low starts collapsed at 2.5e-5), same bases (rl_small_shortest_v1/bases);
#     run names sps2-*, pods maze-rl-sp2-*, s3://.../maze/rl_small_shortest_v2
#   GRID=eps1: sp2 (lr 1e-5, patience 50, shortest-path reward) on the depth-first eps=0.5 noisy-demonstration study: its
#     small bases (s3://.../maze/rl_small_dfs_eps05_v1/bases, pick_shortest_bases.py --pt .../pt_dfs_eps05_v1), RL prompts and
#     validation mazes from s3://.../maze/data/dfs_eps05; run names spe-*, pods maze-rl-eps-*, W&B tailrl-text-maze-dfs-eps05
#   GRID=e25: eps1 on the 25x25 study (bases s3://.../maze/rl_small_dfs25_eps05_v1/bases from pt_dfs25_eps05_v1, prompts and
#     validation from s3://.../maze/data/dfs25_eps05, prompt cap 660, response cap 300); run names spe25-*, pods maze-rl-e25-*,
#     W&B tailrl-text-maze-dfs25-eps05
#   MAX_STEPS (default per grid, else 5000) caps every run on top of the early-stop rule (patience PAT).
#   GRID=epi / e25i: eps1 / e25 with RL_LOSS=icepop (IcePop train-inference masking [0.5, 5.0]); GRID=mdi: the 17x17 medium sweep
#     (IcePop, prompts per step 2 8 32 128, RL seed 0). BATCHES / RL_SEEDS override the per-grid lists.
#   GRID=eri / mdr / e25r: REINFORCE (RL_ADV=reinforce_batch) versions of epi / mdi / e25i at the same samples per step.
#   GRID=t3s / t3m: 25x25 GRPO + TIS (min(ratio, 2)) from the per-seed tune3 bases (run seed i starts from base seed i:
#     BASES_SPEC lists levels only, bases ${OUT_S3}/bases/<level>_s<seed>), small prompts per step 16 64 256 / medium 8 32 128,
#     lr 1e-5, validation every 20 steps, patience 50 validations (1,000 steps), MAX_STEPS 10000, update micro-batch capped at
#     MICRO samples (512 small / 256 medium), 4 reward workers, shared-prompt decode (PREFIX_DECODE=1, needs FAST_DECODE=1) with
#     one generate call per batch (ROLLOUT_MICRO_BATCH_SIZE=8000); run names st3-* / mt3-*, S3 .../maze/rl_{small,medium}_dfs25_tis3s_v1
#   CODE_REF=<jiahaoz commit> GRID=v2 SLOTS=10 MPS=1 bash rl-distill-scripts/maze/launch_rl_small_grid.sh [g0 g1 g2 g3]
#   GRID=t3sx / t3mx: the next rung of t3s / t3m (small 1024 / medium 512 prompts per step), same S3 grid, run names and W&B
#     suffix, own pod names (one run per pod).
# Overrides: BATCHES / RL_SEEDS / BASES_SPEC (grid lists), MAX_STEPS, PATIENCE_OVERRIDE (validations), MICRO (update
#   micro-batch cap), TAG_PREFIX_OVERRIDE (own pod names for a subset), EXTRA_ENVS_APPEND (",K=V,..." more pod env; the last
#   value of a key wins), WSFX_OVERRIDE (W&B run-id suffix, e.g. a fresh id for restarted runs), COST_DIV (REINFORCE grids).
# Resume onto more pods: NPODS=16 spreads the runs without a COMPLETE.json on S3 over pods <prefix>-r<GPUS>-g0..g<NPODS-1>
# (longest-processing-time first by per-step cost, SLOTS = runs on that pod), each resuming from its S3 checkpoint. Stop
# the old pods first (no run may be trained by two pods). FAST_DECODE=1 turns on the plain-copy + CUDA-graph rollouts.
# DRY=1 prints the plan only. GPUS=4 makes each pod a 4-GPU job (runs spread over its GPUs; idle GPUs take runs over).
set -euo pipefail
: "${CODE_REF:?}"
MAX_STEPS_IN="${MAX_STEPS:-}"   # step cap on top of early stopping (default per grid, else 5000)
# per-grid outputs of the case below: never inherited from the caller's environment
EXTRA_ENVS=""; WSFX=""; BATCHES_DEFAULT=""; RL_SEEDS_DEFAULT=""; MICRO_DEFAULT=""; MAX_STEPS_DEFAULT=""; PER_SEED_BASE=0; COST_DIV=1
case "${GRID:-v1}" in
  v1) OUT_S3=s3://scale-ml/genai/rl-distill/maze/rl_small_v1; TAG_PREFIX=maze-rl; SP=sml; LR=1e-4; PAT=20 ;;
  v2) OUT_S3=s3://scale-ml/genai/rl-distill/maze/rl_small_v2; TAG_PREFIX=maze-rl-v2; SP=sml2; LR=2.5e-5; PAT=50 ;;
  sp1) OUT_S3=s3://scale-ml/genai/rl-distill/maze/rl_small_shortest_v1; TAG_PREFIX=maze-rl-sp; SP=sps; LR=2.5e-5; PAT=50
       : "${BASES_SPEC:?GRID=sp1 needs BASES_SPEC (pick_shortest_bases.py)}"
       EXTRA_ENVS=",RL_REWARD=shortest,BASES_S3=${OUT_S3}/bases,WANDB_PROJECT_NAME=tailrl-text-maze-shortest" ;;
  sp2) OUT_S3=s3://scale-ml/genai/rl-distill/maze/rl_small_shortest_v2; TAG_PREFIX=maze-rl-sp2; SP=sps2; LR=1e-5; PAT=50
       : "${BASES_SPEC:?GRID=sp2 needs BASES_SPEC (pick_shortest_bases.py)}"
       EXTRA_ENVS=",RL_REWARD=shortest,BASES_S3=s3://scale-ml/genai/rl-distill/maze/rl_small_shortest_v1/bases,WANDB_PROJECT_NAME=tailrl-text-maze-shortest" ;;
  eps1) OUT_S3=s3://scale-ml/genai/rl-distill/maze/rl_small_dfs_eps05_v1; TAG_PREFIX=maze-rl-eps; SP=spe; LR=1e-5; PAT=50
       : "${BASES_SPEC:?GRID=eps1 needs BASES_SPEC (pick_shortest_bases.py)}"
       EXTRA_ENVS=",RL_REWARD=shortest,BASES_S3=${OUT_S3}/bases,DATA_S3=s3://scale-ml/genai/rl-distill/maze/data/dfs_eps05,WANDB_PROJECT_NAME=tailrl-text-maze-dfs-eps05" ;;
  e25) OUT_S3=s3://scale-ml/genai/rl-distill/maze/rl_small_dfs25_eps05_v1; TAG_PREFIX=maze-rl-e25; SP=spe25; LR=1e-5; PAT=50
       : "${BASES_SPEC:?GRID=e25 needs BASES_SPEC (pick_shortest_bases.py)}"
       EXTRA_ENVS=",RL_REWARD=shortest,BASES_S3=${OUT_S3}/bases,DATA_S3=s3://scale-ml/genai/rl-distill/maze/data/dfs25_eps05,WANDB_PROJECT_NAME=tailrl-text-maze-dfs25-eps05,MAX_PROMPT_LENGTH=660,MAX_RESPONSE_LENGTH=300" ;;
  # IcePop (RL_LOSS=icepop: train-inference token mask [0.5, 5.0]) reruns of eps1 / e25, plus the 17x17 medium sweep
  epi) OUT_S3=s3://scale-ml/genai/rl-distill/maze/rl_small_dfs_eps05_ice_v1; TAG_PREFIX=maze-rl-epi; SP=spi; LR=1e-5; PAT=50
       : "${BASES_SPEC:?GRID=epi needs BASES_SPEC}"
       EXTRA_ENVS=",RL_REWARD=shortest,RL_LOSS=icepop,BASES_S3=${OUT_S3}/bases,DATA_S3=s3://scale-ml/genai/rl-distill/maze/data/dfs_eps05,WANDB_PROJECT_NAME=tailrl-text-maze-dfs-eps05" ;;
  mdi) OUT_S3=s3://scale-ml/genai/rl-distill/maze/rl_medium_dfs_eps05_ice_v1; TAG_PREFIX=maze-rl-mdi; SP=smi; LR=1e-5; PAT=50
       : "${BASES_SPEC:?GRID=mdi needs BASES_SPEC (select_aligned_bases.py)}"
       BATCHES_DEFAULT="2 8 32 128"; RL_SEEDS_DEFAULT="0"
       EXTRA_ENVS=",RL_REWARD=shortest,RL_LOSS=icepop,BASES_S3=${OUT_S3}/bases,DATA_S3=s3://scale-ml/genai/rl-distill/maze/data/dfs_eps05,WANDB_PROJECT_NAME=tailrl-text-maze-dfs-eps05" ;;
  e25i) OUT_S3=s3://scale-ml/genai/rl-distill/maze/rl_small_dfs25_eps05_ice_v1; TAG_PREFIX=maze-rl-e25i; SP=spi25; LR=1e-5; PAT=50
       : "${BASES_SPEC:?GRID=e25i needs BASES_SPEC}"
       EXTRA_ENVS=",RL_REWARD=shortest,RL_LOSS=icepop,BASES_S3=${OUT_S3}/bases,DATA_S3=s3://scale-ml/genai/rl-distill/maze/data/dfs25_eps05,WANDB_PROJECT_NAME=tailrl-text-maze-dfs25-eps05,MAX_PROMPT_LENGTH=660,MAX_RESPONSE_LENGTH=300" ;;
  # REINFORCE variants of epi / mdi / e25i (RL_ADV=reinforce_batch: 1 sample per prompt, advantage normalized across the
  # batch), same samples per step as the GRPO grid (prompts per step = 16 x GRPO's); COST_DIV maps them onto GRPO pack costs
  eri) OUT_S3=s3://scale-ml/genai/rl-distill/maze/rl_small_dfs_eps05_ice_reinf_v2; TAG_PREFIX=maze-rl-eri; SP=sri; LR=1e-5; PAT=50
       : "${BASES_SPEC:?GRID=eri needs BASES_SPEC}"; BATCHES_DEFAULT="64 256 1024 4096"; COST_DIV=16
       EXTRA_ENVS=",RL_REWARD=shortest,RL_LOSS=icepop,RL_ADV=reinforce_batch,BASES_S3=${OUT_S3}/bases,DATA_S3=s3://scale-ml/genai/rl-distill/maze/data/dfs_eps05,WANDB_PROJECT_NAME=tailrl-text-maze-dfs-eps05" ;;
  mdr) OUT_S3=s3://scale-ml/genai/rl-distill/maze/rl_medium_dfs_eps05_ice_reinf_v2; TAG_PREFIX=maze-rl-mdr; SP=smr; LR=1e-5; PAT=50
       : "${BASES_SPEC:?GRID=mdr needs BASES_SPEC}"; BATCHES_DEFAULT="32 128 512 2048"; RL_SEEDS_DEFAULT="0"; COST_DIV=16
       EXTRA_ENVS=",RL_REWARD=shortest,RL_LOSS=icepop,RL_ADV=reinforce_batch,BASES_S3=${OUT_S3}/bases,DATA_S3=s3://scale-ml/genai/rl-distill/maze/data/dfs_eps05,WANDB_PROJECT_NAME=tailrl-text-maze-dfs-eps05" ;;
  e25r) OUT_S3=s3://scale-ml/genai/rl-distill/maze/rl_small_dfs25_eps05_ice_reinf_v2; TAG_PREFIX=maze-rl-e25r; SP=sri25; LR=1e-5; PAT=50
       : "${BASES_SPEC:?GRID=e25r needs BASES_SPEC}"; BATCHES_DEFAULT="64 256 1024 4096"; COST_DIV=16
       EXTRA_ENVS=",RL_REWARD=shortest,RL_LOSS=icepop,RL_ADV=reinforce_batch,BASES_S3=${OUT_S3}/bases,DATA_S3=s3://scale-ml/genai/rl-distill/maze/data/dfs25_eps05,WANDB_PROJECT_NAME=tailrl-text-maze-dfs25-eps05,MAX_PROMPT_LENGTH=660,MAX_RESPONSE_LENGTH=300" ;;
  # TIS on the per-seed 25x25 tune3 bases (run seed i from base seed i), longer runs, cheaper steps
  t3s|t3m)
       [ "${GRID}" = t3s ] && { OUT_S3=s3://scale-ml/genai/rl-distill/maze/rl_small_dfs25_tis3s_v1; TAG_PREFIX=maze-rl-t3s; SP=st3; BATCHES_DEFAULT="16 64 256"; }
       [ "${GRID}" = t3m ] && { OUT_S3=s3://scale-ml/genai/rl-distill/maze/rl_medium_dfs25_tis3s_v1; TAG_PREFIX=maze-rl-t3m; SP=mt3; BATCHES_DEFAULT="8 32 128"; }
       LR=1e-5; PAT=50; MAX_STEPS_DEFAULT=10000; PER_SEED_BASE=1; BASES_SPEC="${BASES_SPEC:-p1 p3 p10}"
       [ "${GRID}" = t3s ] && MICRO_DEFAULT=512 || MICRO_DEFAULT=256   # update micro-batch cap (1024 OOMed with 4 heavy runs per GPU)
       EXTRA_ENVS=",RL_REWARD=shortest,RL_LOSS=tis,TEST_FREQ=20,PPO_MICRO_SAMPLES=${MICRO:-${MICRO_DEFAULT}},REWARD_NUM_PROCESSES=4,PREFIX_DECODE=1,ROLLOUT_MICRO_BATCH_SIZE=8000,BASES_S3=${OUT_S3}/bases,DATA_S3=s3://scale-ml/genai/rl-distill/maze/data/dfs25_eps05,WANDB_PROJECT_NAME=tailrl-text-maze-dfs25-eps05,MAX_PROMPT_LENGTH=660,MAX_RESPONSE_LENGTH=300" ;;
  # next batch rung of t3s / t3m (small 1024, medium 512 prompts per step), one run per 1-GPU pod: same S3 grid, run names and
  # W&B id suffix as t3s / t3m, own pod names; larger update micro-batch (a run has the GPU to itself), 12 reward workers,
  # one generate call per batch
  t3sx|t3mx)
       [ "${GRID}" = t3sx ] && { OUT_S3=s3://scale-ml/genai/rl-distill/maze/rl_small_dfs25_tis3s_v1; TAG_PREFIX=maze-rl-t3sx; SP=st3; BATCHES_DEFAULT="1024"; MICRO_DEFAULT=1024; WSFX=t3s; }
       [ "${GRID}" = t3mx ] && { OUT_S3=s3://scale-ml/genai/rl-distill/maze/rl_medium_dfs25_tis3s_v1; TAG_PREFIX=maze-rl-t3mx; SP=mt3; BATCHES_DEFAULT="512"; MICRO_DEFAULT=512; WSFX=t3m; }
       LR=1e-5; PAT=2000; MAX_STEPS_DEFAULT=20000; PER_SEED_BASE=1; BASES_SPEC="${BASES_SPEC:-p1 p3 p10}"
       EXTRA_ENVS=",RL_REWARD=shortest,RL_LOSS=tis,TEST_FREQ=20,PPO_MICRO_SAMPLES=${MICRO:-${MICRO_DEFAULT}},REWARD_NUM_PROCESSES=12,PREFIX_DECODE=1,ROLLOUT_MICRO_BATCH_SIZE=16384,BASES_S3=${OUT_S3}/bases,DATA_S3=s3://scale-ml/genai/rl-distill/maze/data/dfs25_eps05,WANDB_PROJECT_NAME=tailrl-text-maze-dfs25-eps05,MAX_PROMPT_LENGTH=660,MAX_RESPONSE_LENGTH=300" ;;
  *) echo "unknown GRID ${GRID}" >&2; exit 1 ;;
esac
export MAX_STEPS="${MAX_STEPS_IN:-${MAX_STEPS_DEFAULT:-5000}}"
PAT="${PATIENCE_OVERRIDE:-${PAT}}"   # e.g. 1000 validations: never early-stop (run to MAX_STEPS)
# TAG_PREFIX_OVERRIDE: own pod names for a subset of a grid (with BATCHES); EXTRA_ENVS_APPEND: more pod env (",K=V,..."),
# e.g. ",PREFIX_FORWARD=1,PPO_MICRO_SAMPLES=1024"; WSFX_OVERRIDE: W&B run-id suffix (a fresh id for restarted runs)
TAG_PREFIX="${TAG_PREFIX_OVERRIDE:-${TAG_PREFIX}}"; EXTRA_ENVS="${EXTRA_ENVS:-}${EXTRA_ENVS_APPEND:-}"; WSFX="${WSFX_OVERRIDE:-${WSFX:-}}"
# run spec <level>.<base seed>.<prompts>.<RL seed>; PER_SEED_BASE=1: LEVELS are bare levels and RL seed i uses base seed i
spec_of() { if [ "${PER_SEED_BASE:-0}" = 1 ]; then echo "$1.s$3.$2.$3"; else echo "$1.$2.$3"; fi; }
LEVELS=(p01.s2 p03.s1 p1.s0 p3.s1 p10.s0); [ -n "${BASES_SPEC:-}" ] && read -r -a LEVELS <<< "${BASES_SPEC}"
read -r -a BATCHES <<< "${BATCHES:-${BATCHES_DEFAULT:-4 16 64 256}}"; read -r -a RL_SEEDS <<< "${RL_SEEDS:-${RL_SEEDS_DEFAULT:-0 1 2}}"
declare -A RUNS_OF
for li in "${!LEVELS[@]}"; do for bi in "${!BATCHES[@]}"; do for s in "${RL_SEEDS[@]}"; do
  g=$(( (li + bi + s) % 4 )); RUNS_OF[$g]+="${BATCHES[$bi]} $(spec_of "${LEVELS[$li]}" "${BATCHES[$bi]}" "${s}")"$'\n'
done; done; done
ENVS="MPS=${MPS:-1},WANDB_ID_SUFFIX=${WSFX:-${GRID:-v1}},SUITE_PREFIX=${SP},ACTOR_LR=${LR},FAST_DECODE=${FAST_DECODE:-0}${EXTRA_ENVS:-}"
if [ -n "${NPODS:-}" ]; then
  done_runs="$({ AWS_PROFILE=ml-worker aws s3 ls --recursive "${OUT_S3}/runs/" || true; } | awk '/\/COMPLETE.json$/ && !/\/state\// {split($4, a, "/"); print a[length(a) - 1]}')"
  plan="$(for li in "${!LEVELS[@]}"; do for b in "${BATCHES[@]}"; do for s in "${RL_SEEDS[@]}"; do
      lvl="${LEVELS[$li]%%.*}"; echo "${SP}-${lvl}-b${b}-s${s} $(spec_of "${LEVELS[$li]}" "${b}" "${s}") ${b}"; done; done; done \
    | awk -v done="$(printf '%s ' ${done_runs})" -v n="${NPODS}" -v div="${COST_DIV:-1}" '
        BEGIN {split(done, d, " "); for (i in d) isdone[d[i]] = 1; cost[1024] = 20; cost[512] = 10; cost[256] = 5.4; cost[128] = 2.8; cost[64] = 1.64; cost[32] = 1.35; cost[16] = 1.2; cost[8] = 1.15; cost[4] = 1.12; cost[2] = 1.1}
        !($1 in isdone) {
          if (!(($3 / div) in cost)) {print "no packing cost for " $3 " prompts per step" > "/dev/stderr"; exit 1}
          spec[++m] = $2; c[m] = cost[$3 / div]}
        END {  # LPT: heaviest run first onto the least-loaded pod
          for (i = 1; i <= m; i++) for (j = i + 1; j <= m; j++) if (c[j] > c[i]) {t = c[i]; c[i] = c[j]; c[j] = t; t = spec[i]; spec[i] = spec[j]; spec[j] = t}
          for (i = 1; i <= m; i++) {best = 0; for (k = 1; k < n; k++) if (load[k] < load[best]) best = k; load[best] += c[i]; runs[best] = runs[best] (runs[best] ? "+" : "") spec[i]; cnt[best]++}
          for (k = 0; k < n; k++) if (cnt[k]) print "g" k, cnt[k], runs[k]}')"
  [ -n "${plan}" ] || { echo "nothing to launch: every run of GRID=${GRID:-v1} has a COMPLETE.json"; exit 0; }
  echo "${plan}" | awk '{print $1, $2, "runs:", $3}'
  [ "${DRY:-0}" = 1 ] && exit 0
  while read -r p cnt runs; do
    CODE_REF="${CODE_REF}" TAG="${TAG_PREFIX}-r${GPUS:-1}-${p}" RUNS="${runs}" SLOTS="${cnt}" OUT_S3="${OUT_S3}" PATIENCE="${PAT}" GPUS="${GPUS:-1}" MAX_STEPS="${MAX_STEPS}" \
      EXTRA_ENV_VARS="${ENVS}" bash "$(dirname "${BASH_SOURCE[0]}")/start_rl_pack_supervisor.sh"
  done <<< "${plan}"
  exit 0
fi
pods=("$@"); [ "${#pods[@]}" -gt 0 ] || pods=(g0 g1 g2 g3)
for p in "${pods[@]}"; do
  runs="$(printf '%s' "${RUNS_OF[${p#g}]}" | sort -k1,1nr -s | awk '{print $2}' | paste -sd+)"
  echo "${TAG_PREFIX}-${p}: ${runs}"
  CODE_REF="${CODE_REF}" TAG="${TAG_PREFIX}-${p}" RUNS="${runs}" SLOTS="${SLOTS:-10}" OUT_S3="${OUT_S3}" PATIENCE="${PAT}" MAX_STEPS="${MAX_STEPS}" \
    EXTRA_ENV_VARS="${ENVS}" \
    bash "$(dirname "${BASH_SOURCE[0]}")/start_rl_pack_supervisor.sh"
done
