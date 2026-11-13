#!/usr/bin/env bash
# One text-maze RL run with the small-model recipe (2026-11-07): GRPO (std-normalized, group 16), binary goal_reached
# reward, one update per rollout batch with token-level truncated importance sampling against the rollout engine's
# log-probs (w = min(pi_train / pi_rollout, 2); no old-log-prob recompute), constant lr (ACTOR_LR, 1e-4) after a 20-step warmup,
# validation every TEST_FREQ (10) steps on 500 dev mazes x 16 samples, checkpoint every validation (last 2 kept) + best_hf on every
# new best, early stop after PATIENCE validations without a new best, else MAX_STEPS. Needs TailRL with tailrl_text_maze.patch
# and tailrl_rl_tis.patch applied.
#   env: BASE_DIR (dir containing ckpt-0), SUITE (unique run name), BATCH (prompts per step), SEED, GPU (id),
#        SLOT (Ray port band 0..59, unique per concurrent run on a machine), VAL_DIR (dir with the 500-maze test.parquet),
#        TAILRL_ROOT, ACTOR_LR (1e-4), MAX_STEPS (10000), PATIENCE (20), FAST_DECODE (0; 1 = rollouts from a plain copy of
#        the policy with CUDA-graph decoding: same sampling distribution and loss, ~2-4x faster generation), WANDB_* (forwarded), WANDB_RUN_ID (stable id -> resumes in place),
#        RL_REWARD (goal | shortest: reward 1 only for a shortest path, maze_shortest_reward.py, early stop on is_shortest),
#        MAX_PROMPT_LENGTH (320; 25x25 prompts are 654 tokens), MAX_RESPONSE_LENGTH (180; 25x25 paths reach ~250 moves),
#        RL_LOSS (tis: token weight min(ratio, 2) | icepop: IcePop train-inference masking, weight = ratio inside
#        [ICEPOP_LOW, ICEPOP_HIGH] = [0.5, 5.0], else 0),
#        RL_ADV (grpo: 16 samples per prompt, group-normalized advantage | reinforce_batch: 1 sample per prompt, advantage
#        normalized across the whole batch; BATCH is then prompts = samples per step, and the PPO micro-batch is BATCH / 16
#        samples so memory per micro-batch matches GRPO at the same samples per step),
#        PREFIX_DECODE (0; 1 with FAST_DECODE=1: shared-prompt decode, see below), PREFIX_FORWARD (0; 1: shared-prompt update),
#        RUN_GPU_IDS (0; e.g. 0,1,2,3 = data-parallel on 4 GPUs, set by the pack script's RUN_ALL_GPUS),
#        ROLLOUT_MICRO_BATCH_SIZE (1000 samples per generate call per GPU), REWARD_NUM_PROCESSES (4), WANDB_RESUME (allow),
#        TEST_FREQ (10; validation + checkpoint interval in steps; PATIENCE counts validations),
#        PPO_MICRO_SAMPLES (grpo: cap on samples per update micro-batch per GPU; see below), PPO_MICRO_BATCH (reinforce_batch:
#        explicit micro-batch, default BATCH / 16), EXTRA_RL_ARGS (more Hydra overrides),
#        DATALOADER_WORKERS (2; verl's default 8 x ~8 packed runs per pod = ~64 workers whose copy-on-read dataset pages
#        grew until the 192 GiB pod limit OOM-killed a data-loader worker on REINFORCE pods, 2026-11-09)
set -euo pipefail
: "${BASE_DIR:?}" "${SUITE:?}" "${BATCH:?}" "${SEED:?}" "${GPU:?}" "${SLOT:?}" "${VAL_DIR:?}" "${TAILRL_ROOT:?}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
case "${RL_REWARD:-goal}" in
  goal) REWARD_HYDRA=""; ES_METRIC=goal_reached ;;
  shortest) REWARD_HYDRA="custom_reward_function.path=${HERE}/maze_shortest_reward.py custom_reward_function.name=compute_score"; ES_METRIC=is_shortest ;;
  *) echo "unknown RL_REWARD ${RL_REWARD}" >&2; exit 1 ;;
esac
cd "${TAILRL_ROOT}/experiments/text_maze"
source "${TAILRL_ROOT}/.venv/bin/activate"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-320}"
export ATTN_IMPL=sdpa ACTOR_LR="${ACTOR_LR:-1e-4}" MAX_ACTOR_CKPT_TO_KEEP=2 RAY_PORT_INDEX="${SLOT}" VAL_PARQUET_DIR="${VAL_DIR}" \
       REWARD_NUM_PROCESSES="${REWARD_NUM_PROCESSES:-4}" WANDB_RESUME="${WANDB_RESUME:-allow}" \
       ROLLOUT_MICRO_BATCH_SIZE="${ROLLOUT_MICRO_BATCH_SIZE:-1000}"  # samples per generate call (batches are split into ~equal chunks)
# skip verl's best@n / worst@n validation bootstrap (~40 s of CPU per validation); mean@16, the early-stop metric, is unchanged
export VERL_VAL_SKIP_BOOTSTRAP=1
case "${RL_LOSS:-tis}" in
  tis) LOSS_HYDRA="actor_rollout_ref.actor.policy_loss.loss_mode=tis +actor_rollout_ref.actor.policy_loss.tis_cap=2.0" ;;
  icepop) LOSS_HYDRA="actor_rollout_ref.actor.policy_loss.loss_mode=icepop +actor_rollout_ref.actor.policy_loss.icepop_low=${ICEPOP_LOW:-0.5} +actor_rollout_ref.actor.policy_loss.icepop_high=${ICEPOP_HIGH:-5.0}" ;;
  *) echo "unknown RL_LOSS ${RL_LOSS}" >&2; exit 1 ;;
esac
export EXTRA_HYDRA_ARGS="actor_rollout_ref.actor.optim.lr_warmup_steps=20 \
+actor_rollout_ref.rollout.return_log_probs=True +algorithm.skip_old_log_prob=True \
${LOSS_HYDRA} \
+data.dataloader_num_workers=${DATALOADER_WORKERS:-2} \
+trainer.early_stop_patience=${PATIENCE:-20} +trainer.early_stop_metric=val-aux/maze_17/${ES_METRIC}/mean@16 \
trainer.resume_mode=auto ${REWARD_HYDRA} ${EXTRA_RL_ARGS:-}"
if [ "${PREFIX_DECODE:-0}" = 1 ] && [ "${FAST_DECODE:-0}" != 1 ]; then
  echo "PREFIX_DECODE=1 needs FAST_DECODE=1 (it replaces the plain-copy CUDA-graph decoder)" >&2; exit 1
fi
if [ "${FAST_DECODE:-0}" = 1 ]; then
  EXTRA_HYDRA_ARGS+=" +actor_rollout_ref.rollout.plain_generate=True +actor_rollout_ref.rollout.graph_generate=True"
  # PREFIX_DECODE=1: the graph decode keeps one prompt KV cache per group of samples and drops finished rows as it goes
  # (rollout/prefix_decode.py); same sampling distribution, far less decode work
  [ "${PREFIX_DECODE:-0}" = 1 ] && EXTRA_HYDRA_ARGS+=" +actor_rollout_ref.rollout.prefix_decode=True"
fi
# PREFIX_FORWARD=1: the update runs each distinct prompt once and only the response tokens per sample (actor/prefix_forward.py);
# same log-probs, gradients equal up to fp16 rounding
# (needs the KV cache in training, so it turns gradient checkpointing off: HF drops use_cache under checkpointing; the shared
# forward needs ~1/3 of the activation memory of the standard one, and skips checkpointing's recompute)
[ "${PREFIX_FORWARD:-0}" = 1 ] && EXTRA_HYDRA_ARGS+=" +actor_rollout_ref.actor.prefix_shared_forward=True actor_rollout_ref.model.enable_gradient_checkpointing=False"
NGPU=$(awk -F, '{print NF}' <<< "${RUN_GPU_IDS:-0}")
case "${RL_ADV:-grpo}" in
  grpo) METHOD=grpo; NROLL=16
        # PPO_MICRO_SAMPLES: cap on samples per update micro-batch on each GPU (default: BATCH samples, so the update
        # runs 16 / NGPU micro-batches); e.g. 1024 runs a 256-sample update in one pass and a 4096-sample one in four
        if [ -n "${PPO_MICRO_SAMPLES:-}" ]; then
          PER_GPU=$(( BATCH * NROLL / NGPU ))
          export PPO_MICRO_BATCH=$(( PER_GPU < PPO_MICRO_SAMPLES ? PER_GPU : PPO_MICRO_SAMPLES ))
        fi ;;
  reinforce_batch) METHOD=reinforce_batch; NROLL=1; export PPO_MICRO_BATCH="${PPO_MICRO_BATCH:-$(( BATCH / 16 > 0 ? BATCH / 16 : 1 ))}" ;;
  *) echo "unknown RL_ADV ${RL_ADV}" >&2; exit 1 ;;
esac
# verl splits each update's BATCH * NROLL samples over NGPU GPUs and asserts the per-GPU share divides into micro-batches:
# check it here, before Ray and the model start
PER_GPU=$(( BATCH * NROLL / NGPU ))
if (( BATCH * NROLL % NGPU != 0 || PER_GPU % ${PPO_MICRO_BATCH:-${BATCH}} != 0 )); then
  echo "update of ${BATCH} x ${NROLL} samples on ${NGPU} GPU(s) does not split into micro-batches of ${PPO_MICRO_BATCH:-${BATCH}}" >&2
  exit 1
fi
# logical GPU ids RUN_GPU_IDS (default 0; experiment name / checkpoint dir independent of the GPU), physical GPU(s) via PHYS_GPU_IDS
export PHYS_GPU_IDS="${GPU}"
exec bash scripts/train.sh --suite "${SUITE}" --method "${METHOD}" --model-base-dir "${BASE_DIR}" --ckpt-step 0 --seed "${SEED}" \
  --gpu-ids "${RUN_GPU_IDS:-0}" --n-rollouts "${NROLL}" --batch-size "${BATCH}" --total-steps "${MAX_STEPS:-10000}" --reward binary \
  --reward-transform raw --val-n 16 --val-prompts 500 --test-freq "${TEST_FREQ:-10}" --save-freq "${TEST_FREQ:-10}" --max-response-length "${MAX_RESPONSE_LENGTH:-180}"
