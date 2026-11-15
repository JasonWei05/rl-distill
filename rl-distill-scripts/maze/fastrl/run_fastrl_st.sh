#!/usr/bin/env bash
# ScaleTrain run-file for fastrl (launch_st_with_code.sh --run-file ../maze/fastrl/run_fastrl_st.sh).
#   MODE=runs: trains RUNS on the pod's GPUs (round robin) and mirrors each run dir to ${OUT_S3}/runs/<name>/ every
#     SYNC_SECONDS and at the end; a relaunched pod resumes each run from there and skips finished ones (COMPLETE.json).
#     Writes ${OUT_S3}/pods/${TAG}.COMPLETE once every run is complete.
#   MODE=bench: on one GPU, the test suite, then bench.py for each of BENCH_BASES, then verl (TailRL train.sh with the t3
#     recipe of rl/rl_tis_run.sh: shortest-path reward, TIS, FAST_DECODE + PREFIX_DECODE + PREFIX_FORWARD) for VERL_STEPS
#     steps per VERL_BATCHES entry (W&B project WANDB_PROJECT_NAME, run names verlbench-*); logs to ${OUT_S3}/bench/${TAG}/,
#     then ${OUT_S3}/pods/${TAG}.COMPLETE.
#   env: MODE, TAG, OUT_S3, DATA_S3 (dir with main_parquet/, eval500/ and fastrl_tokens/<split>.tokens.npy),
#        WANDB_PROJECT_NAME (fastrl-maze); bases are paths under s3://scale-ml/genai/rl-distill/maze/
#        runs: RUNS ("<name>:<base>:<prompts per step>:<seed>:<micro>" joined by +), GPUS (1), STEPS (10000), SYNC_SECONDS (300)
#        bench: BENCH_BASES ("<base>" joined by +), BENCH_BATCHES ("16:512 64:512 256:1024 1024:1024"),
#               VERL_BATCHES ("16:512 64:512 256:1024"), VERL_STEPS (40)
source "$(dirname "${BASH_SOURCE[0]}")/../st_setup.sh"
: "${MODE:?}" "${TAG:?}" "${OUT_S3:?}" "${DATA_S3:?}"
MAZE_S3=s3://scale-ml/genai/rl-distill/maze; D=/tmp/fastrl/data; R=/tmp/fastrl/runs; B=/tmp/fastrl/bases; F="${M}/fastrl"
mkdir -p "${D}" "${R}" "${B}"
aws s3 sync --only-show-errors "${DATA_S3}/main_parquet" "${D}/main_parquet"; aws s3 sync --only-show-errors "${DATA_S3}/eval500" "${D}/eval500"
aws s3 cp --only-show-errors "${DATA_S3}/fastrl_tokens/train.tokens.npy" "${D}/main_parquet/train.parquet.tokens.npy"
aws s3 cp --only-show-errors "${DATA_S3}/fastrl_tokens/eval500.tokens.npy" "${D}/eval500/test.parquet.tokens.npy"
export WANDB_ENTITY=rl-distill WANDB_MODE=online TORCHINDUCTOR_CACHE_DIR=/tmp/fastrl/inductor PYTHONPATH="${F}"
WP="${WANDB_PROJECT_NAME:-fastrl-maze}"
base() { aws s3 sync --only-show-errors "${MAZE_S3}/$1" "${B}/$1"; echo "${B}/$1"; }

if [ "${MODE}" = runs ]; then
  IFS=+ read -r -a specs <<< "${RUNS:?}"
  # a transient S3 error must not end the sidecar (set -e): the final sync below is the one that has to succeed
  ( while sleep "${SYNC_SECONDS:-300}"; do aws s3 sync --only-show-errors "${R}" "${OUT_S3}/runs" --exclude "*/wandb/*" || echo "SYNC_RETRY"; done ) & sync=$!
  pids=()
  for spec in "${specs[@]}"; do
    IFS=: read -r name bpath batch seed micro <<< "${spec}"
    aws s3 sync --only-show-errors "${OUT_S3}/runs/${name}" "${R}/${name}"
    if [ -f "${R}/${name}/COMPLETE.json" ]; then echo "SKIP_COMPLETE ${name}"; continue; fi
    bdir="$(base "${bpath}")"; gpu=$(( ${#pids[@]} % ${GPUS:-1} ))
    CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" "${F}/fastrl.py" --base "${bdir}" --train "${D}/main_parquet/train.parquet" \
      --val "${D}/eval500/test.parquet" --out "${R}/${name}" --batch "${batch}" --seed "${seed}" --micro "${micro}" \
      --steps "${STEPS:-10000}" --wandb-project "${WP}" --wandb-id "${name}" > "${R}/${name}.log" 2>&1 &
    pids+=($!); echo "RUN_START ${name} gpu ${gpu}"
  done
  rc=0; for p in "${pids[@]}"; do wait "${p}" || rc=1; done
  kill "${sync}"; aws s3 sync --only-show-errors "${R}" "${OUT_S3}/runs" --exclude "*/wandb/*"
  for spec in "${specs[@]}"; do
    name="${spec%%:*}"; aws s3 ls "${OUT_S3}/runs/${name}/COMPLETE.json" >/dev/null || { echo "NOT_COMPLETE ${name}"; exit 1; }
  done
  echo "{\"done\": \"$(date -u +%FT%TZ)\"}" | aws s3 cp - "${OUT_S3}/pods/${TAG}.COMPLETE"; echo "FASTRL_POD_DONE ${TAG} rc=${rc}"
  exit "${rc}"
fi

[ "${MODE}" = bench ] || { echo "unknown MODE ${MODE}" >&2; exit 1; }
L=/tmp/fastrl/bench_logs; mkdir -p "${L}"
IFS=+ read -r -a bases <<< "${BENCH_BASES:?}"
[[ "${bases[0]}" == */p3_* ]] || { echo "the first bench base must be a p3_ level (the tests also use its p10_ counterpart)" >&2; exit 1; }
uv pip install --python "${PY}" -q pytest
(cd "${F}" && FASTRL_DATA="${D}" FASTRL_BASE="$(base "${bases[0]}")" FASTRL_BASE_HI="$(base "${bases[0]/p3_/p10_}")" \
   PYTHONPATH="${F}:${T}/experiments/text_maze:${M}/rl" "${PY}" -m pytest -q -p no:cacheprovider test_fastrl.py) > "${L}/tests.log" 2>&1 \
  || { aws s3 cp --only-show-errors "${L}/tests.log" "${OUT_S3}/bench/${TAG}/tests.log"; tail -n 30 "${L}/tests.log"; exit 1; }
grep -E "passed" "${L}/tests.log"
for b in "${bases[@]}"; do
  read -r -a batches <<< "${BENCH_BATCHES:-16:512 64:512 256:1024 1024:1024}"
  WANDB_MODE=disabled "${PY}" "${F}/bench.py" --base "$(base "${b}")" --data "${D}" --batches "${batches[@]}" 2>&1 | tee -a "${L}/fastrl_full.log" | grep BENCH | tee -a "${L}/fastrl.log"
done
aws s3 sync --only-show-errors "${L}" "${OUT_S3}/bench/${TAG}"
cat > "${T}/.env" <<EOF
TAILRL_MAZE_DATA_DIR=${D}
TAILRL_MAZE_RUN_DIR=/tmp/fastrl/verl_runs
WANDB_MODE=online
WANDB_ENTITY=rl-distill
WANDB_PROJECT=${WP}
EOF
# the t3 recipe as rl/rl_tis_run.sh runs it (keep the two in sync)
mkdir -p /tmp/fastrl/verl_base && ln -sfn "$(base "${bases[0]}")" /tmp/fastrl/verl_base/ckpt-0
cd "${T}/experiments/text_maze"
export ATTN_IMPL=sdpa ACTOR_LR=1e-5 MAX_ACTOR_CKPT_TO_KEEP=2 RAY_PORT_INDEX=0 VAL_PARQUET_DIR="${D}/eval500" REWARD_NUM_PROCESSES=4 \
  WANDB_RESUME=allow ROLLOUT_MICRO_BATCH_SIZE=8000 MAX_PROMPT_LENGTH=660 VERL_VAL_SKIP_BOOTSTRAP=1 PHYS_GPU_IDS=0
export EXTRA_HYDRA_ARGS="actor_rollout_ref.actor.optim.lr_warmup_steps=20 +actor_rollout_ref.rollout.return_log_probs=True \
+algorithm.skip_old_log_prob=True actor_rollout_ref.actor.policy_loss.loss_mode=tis +actor_rollout_ref.actor.policy_loss.tis_cap=2.0 \
+data.dataloader_num_workers=2 +trainer.early_stop_patience=2000 +trainer.early_stop_metric=val-aux/maze_17/is_shortest/mean@16 \
custom_reward_function.path=${M}/rl/maze_shortest_reward.py custom_reward_function.name=compute_score \
+actor_rollout_ref.rollout.plain_generate=True +actor_rollout_ref.rollout.graph_generate=True +actor_rollout_ref.rollout.prefix_decode=True \
+actor_rollout_ref.actor.prefix_shared_forward=True actor_rollout_ref.model.enable_gradient_checkpointing=False"
rc=0
for spec in ${VERL_BATCHES:-16:512 64:512 256:1024}; do
  batch="${spec%%:*}"; micro="${spec##*:}"
  PPO_MICRO_BATCH=$(( batch * 16 < micro ? batch * 16 : micro )) bash scripts/train.sh --suite "verlbench-${TAG}-b${batch}" --method grpo \
    --model-base-dir /tmp/fastrl/verl_base --ckpt-step 0 --seed 3 --gpu-ids 0 --n-rollouts 16 --batch-size "${batch}" \
    --total-steps "${VERL_STEPS:-40}" --reward binary --reward-transform raw --val-n 16 --val-prompts 500 --test-freq 20 \
    --save-freq 1000 --max-response-length 300 --train-parquet-dir "${D}/main_parquet" --val-parquet-dir "${D}/eval500" \
    > "${L}/verl_b${batch}.log" 2>&1 || { echo "VERL_FAILED b${batch}"; rc=1; }
  aws s3 sync --only-show-errors "${L}" "${OUT_S3}/bench/${TAG}"
done
[ "${rc}" = 0 ] || exit 1
echo "{\"done\": \"$(date -u +%FT%TZ)\"}" | aws s3 cp - "${OUT_S3}/pods/${TAG}.COMPLETE"; echo "FASTRL_BENCH_DONE ${TAG}"
