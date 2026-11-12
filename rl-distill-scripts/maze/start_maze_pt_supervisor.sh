#!/usr/bin/env bash
# Supervised launch of one size of the shortest-path maze study (maze/run_maze_pretrain_st.sh on a GPUS-GPU pod) from the
# main rl-distill checkout: supervise_borrowing_job.py relaunches it after preemption / failure until
# ${OUT_S3}/${SIZE}/COMPLETE exists (a relaunch keeps finished runs and levels, see the run-file). tmux session rl-<TAG>.
#   CODE_REF=<jiahaoz commit> SIZE=XS OUT_S3=s3://scale-ml/genai/rl-distill/maze/pt_shortest_v1 \
#     bash rl-distill-scripts/maze/start_maze_pt_supervisor.sh
#   env: GPUS (2), ALLOW_BORROWING (false), MAX_STEPS (10000), STOP_AT (0.20), DEADLINE_HOURS (72), TAG (maze-pt-<size>),
#        EXTRA_ENVS (more run-file env, e.g. DATA_S3=...,TRAIN_JSON=...,EVAL_S3=...,WANDB_PROJECT_NAME=...,SEEDS=0-1-2-3-4: env
#        values cannot hold spaces; the run-file default is seeds 0 1 2), SUP_TAG (suffix of the state dir), MAX_RELAUNCHES (20)
#   stop: touch <state>/STOP; tmux kill-session -t "=rl-<TAG>"
set -euo pipefail
MAIN=/mnt/efs/jasonwei/rl-distill; cd "${MAIN}"
IMG="${ST_IMAGE:-692474966980.dkr.ecr.us-west-2.amazonaws.com/scale_train/shared/training/tmp:20260829-002920.cd13c11a-e0f3-4f65-a74b-0e0c5d72d58a}"
: "${CODE_REF:?}" "${SIZE:?}" "${OUT_S3:?}"
GPUS="${GPUS:-2}"; TAG="${TAG:-maze-pt-${SIZE,,}}"
BORROW=""; [ "${ALLOW_BORROWING:-false}" = true ] && BORROW="--allow-borrowing"
sup=".scale_train_supervisors/${TAG}-$(date -u +%Y%m%d)${SUP_TAG:-}"; mkdir -p "${sup}/pod-logs"; rm -f "${sup}/STOP"
envs="SIZE=${SIZE},OUT_S3=${OUT_S3},GPUS=${GPUS},MAX_STEPS=${MAX_STEPS:-10000},STOP_AT=${STOP_AT:-0.20}${EXTRA_ENVS:+,${EXTRA_ENVS}}"
tmux kill-session -t "=rl-${TAG}" 2>/dev/null || true
tmux new-session -d -s "rl-${TAG}" "env -u AWS_PROFILE python3 rl-distill-scripts/scale_train/supervise_borrowing_job.py --name ${TAG} \
  --relaunch-on-cancel --relaunch-on-failure --max-relaunches ${MAX_RELAUNCHES:-20} --quick-cancel-seconds 300 --max-quick-cancels 3 \
  --failure-backoff-seconds 300 --max-failure-backoff-seconds 1800 \
  --launch-log ${sup}/launch.log --monitor-log ${sup}/monitor.log --state-file ${sup}/state.json --stop-file ${sup}/STOP --pod-log-dir ${sup}/pod-logs \
  --completion-marker-s3-uri ${OUT_S3}/${SIZE}/COMPLETE \
  -- env CODE_REF=${CODE_REF} bash rl-distill-scripts/scale_train/launch_st_with_code.sh --image ${IMG} --gpus-per-instance ${GPUS} --priority high ${BORROW} \
     --active-deadline-hours ${DEADLINE_HOURS:-72} --run-file ../maze/run_maze_pretrain_st.sh --job-name ${TAG} --env-vars ${envs} \
  2>&1 | tee -a ${sup}/supervisor.out"
echo "started supervisor rl-${TAG} (state ${sup}, done when ${OUT_S3}/${SIZE}/COMPLETE exists, code ${CODE_REF})"
