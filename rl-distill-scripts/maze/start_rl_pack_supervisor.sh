#!/usr/bin/env bash
# Supervised launch of one packed maze RL pod (maze/run_maze_rl_pack_st.sh, GPUS GPUs, default 1) from the main rl-distill checkout:
# supervise_borrowing_job.py relaunches it after preemption / failure / cancel until ${OUT_S3}/pods/${TAG}.COMPLETE exists;
# a relaunched pod skips finished runs and resumes the others from S3. tmux session rl-<TAG>.
#   CODE_REF=<jiahaoz commit> TAG=maze-rl-g0 RUNS="p1.s0.16.0+..." SLOTS=12 OUT_S3=s3://.../maze/rl_small_v1 \
#     bash rl-distill-scripts/maze/start_rl_pack_supervisor.sh
#   env: GPUS (1), MAX_STEPS (10000), PATIENCE (20), SYNC_SECONDS (300), EXTRA_ENV_VARS (",K=V,..." more pod env),
#        ALLOW_BORROWING (true), SUP_TAG (suffix of the state dir), MAX_RELAUNCHES (30), DEADLINE_HOURS (72), ST_IMAGE
#   Refuses to start when ${OUT_S3}/pods/${TAG}.COMPLETE already exists (the supervisor would see it and exit at once):
#   remove it or use another TAG to rerun that pod.
#   stop: touch <state>/STOP; tmux kill-session -t "=rl-<TAG>"
set -euo pipefail
MAIN=/mnt/efs/jasonwei/rl-distill; cd "${MAIN}"
IMG="${ST_IMAGE:-692474966980.dkr.ecr.us-west-2.amazonaws.com/scale_train/shared/training/tmp:20260829-002920.cd13c11a-e0f3-4f65-a74b-0e0c5d72d58a}"
: "${CODE_REF:?}" "${TAG:?}" "${RUNS:?}" "${SLOTS:?}" "${OUT_S3:?}"
if env -u AWS_PROFILE aws s3 ls "${OUT_S3}/pods/${TAG}.COMPLETE" >/dev/null 2>&1; then
  echo "${OUT_S3}/pods/${TAG}.COMPLETE exists: the supervisor would exit at once; remove it or use another TAG" >&2; exit 1
fi
ALLOW_BORROWING="${ALLOW_BORROWING:-true}"; BORROW=""; [ "${ALLOW_BORROWING}" = true ] && BORROW="--allow-borrowing"
sup=".scale_train_supervisors/${TAG}-$(date -u +%Y%m%d)${SUP_TAG:-}"; mkdir -p "${sup}/pod-logs"; rm -f "${sup}/STOP"
GPUS="${GPUS:-1}"
envs="GPUS=${GPUS},RUNS=${RUNS},SLOTS=${SLOTS},TAG=${TAG},OUT_S3=${OUT_S3},MAX_STEPS=${MAX_STEPS:-10000},PATIENCE=${PATIENCE:-20},SYNC_SECONDS=${SYNC_SECONDS:-300}${EXTRA_ENV_VARS:+,${EXTRA_ENV_VARS}}"
tmux kill-session -t "=rl-${TAG}" 2>/dev/null || true
tmux new-session -d -s "rl-${TAG}" "env -u AWS_PROFILE python3 rl-distill-scripts/scale_train/supervise_borrowing_job.py --name ${TAG} \
  --relaunch-on-cancel --relaunch-on-failure --max-relaunches ${MAX_RELAUNCHES:-30} --quick-cancel-seconds 300 --max-quick-cancels 3 \
  --failure-backoff-seconds 300 --max-failure-backoff-seconds 1800 \
  --launch-log ${sup}/launch.log --monitor-log ${sup}/monitor.log --state-file ${sup}/state.json --stop-file ${sup}/STOP --pod-log-dir ${sup}/pod-logs \
  --completion-marker-s3-uri ${OUT_S3}/pods/${TAG}.COMPLETE \
  -- env CODE_REF=${CODE_REF} bash rl-distill-scripts/scale_train/launch_st_with_code.sh --image ${IMG} --gpus-per-instance ${GPUS} --priority high ${BORROW} \
     --active-deadline-hours ${DEADLINE_HOURS:-72} --run-file ../maze/run_maze_rl_pack_st.sh --job-name ${TAG} --env-vars ${envs} \
  2>&1 | tee -a ${sup}/supervisor.out"
echo "started supervisor rl-${TAG} (state ${sup}, done when ${OUT_S3}/pods/${TAG}.COMPLETE exists, code ${CODE_REF})"
