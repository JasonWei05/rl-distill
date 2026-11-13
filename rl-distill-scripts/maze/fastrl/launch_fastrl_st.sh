#!/usr/bin/env bash
# Supervised fastrl pod (run_fastrl_st.sh) from the main rl-distill checkout: supervise_borrowing_job.py relaunches it after
# preemption / failure until the pod writes ${OUT_S3}/pods/${TAG}.COMPLETE. tmux session rl-<TAG>.
#   CODE_REF=<jiahaoz commit> MODE=runs TAG=fastrl-b256 GPUS=4 RUNS="<name>:<base>:256:2:1024+..." bash launch_fastrl_st.sh
#   CODE_REF=<jiahaoz commit> MODE=bench TAG=fastrl-bench BENCH_BASES="rl_2xs_dfs25_tis3s_v1/bases/p3_s3+..." bash launch_fastrl_st.sh
#   env: OUT_S3 (s3://.../maze/fastrl_v1), DATA_S3 (s3://.../maze/data/dfs25_eps05), ALLOW_BORROWING (true), GPUS (1),
#        EXTRA_ENV_VARS (",K=V,..." more pod env: STEPS, SYNC_SECONDS, BENCH_BATCHES, VERL_BATCHES, VERL_STEPS)
#   stop: touch <state>/STOP; tmux kill-session -t "=rl-<TAG>"
set -euo pipefail
MAIN=/mnt/efs/jasonwei/rl-distill; cd "${MAIN}"
IMG="${ST_IMAGE:-692474966980.dkr.ecr.us-west-2.amazonaws.com/scale_train/shared/training/tmp:20260829-002920.cd13c11a-e0f3-4f65-a74b-0e0c5d72d58a}"
: "${CODE_REF:?}" "${MODE:?}" "${TAG:?}"
OUT_S3="${OUT_S3:-s3://scale-ml/genai/rl-distill/maze/fastrl_v1}"; DATA_S3="${DATA_S3:-s3://scale-ml/genai/rl-distill/maze/data/dfs25_eps05}"
GPUS="${GPUS:-1}"; BORROW=""; [ "${ALLOW_BORROWING:-true}" = true ] && BORROW="--allow-borrowing"
case "${MODE}" in
  runs) envs="RUNS=${RUNS:?}" ;;
  bench) envs="BENCH_BASES=${BENCH_BASES:?}" ;;
  *) echo "unknown MODE ${MODE}" >&2; exit 1 ;;
esac
envs="MODE=${MODE},TAG=${TAG},OUT_S3=${OUT_S3},DATA_S3=${DATA_S3},GPUS=${GPUS},${envs}${EXTRA_ENV_VARS:-}"
sup=".scale_train_supervisors/${TAG}-$(date -u +%Y%m%d)"; mkdir -p "${sup}/pod-logs"; rm -f "${sup}/STOP"
tmux kill-session -t "=rl-${TAG}" 2>/dev/null || true
tmux new-session -d -s "rl-${TAG}" "env -u AWS_PROFILE python3 rl-distill-scripts/scale_train/supervise_borrowing_job.py --name ${TAG} \
  --relaunch-on-cancel --relaunch-on-failure --max-relaunches 10 --quick-cancel-seconds 300 --max-quick-cancels 3 \
  --failure-backoff-seconds 300 --max-failure-backoff-seconds 1800 --completion-marker-s3-uri ${OUT_S3}/pods/${TAG}.COMPLETE \
  --launch-log ${sup}/launch.log --monitor-log ${sup}/monitor.log --state-file ${sup}/state.json --stop-file ${sup}/STOP --pod-log-dir ${sup}/pod-logs \
  -- env CODE_REF=${CODE_REF} bash rl-distill-scripts/scale_train/launch_st_with_code.sh --image ${IMG} --gpus-per-instance ${GPUS} --priority high ${BORROW} \
     --active-deadline-hours ${DEADLINE_HOURS:-24} --run-file ../maze/fastrl/run_fastrl_st.sh --job-name ${TAG} --env-vars ${envs} \
  2>&1 | tee -a ${sup}/supervisor.out"
echo "started supervisor rl-${TAG} (state ${sup}, code ${CODE_REF})"
