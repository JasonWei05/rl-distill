#!/usr/bin/env bash
# Upload a plan (plan.py) as ${S3}/plans/<TAG>.json and run it on one supervised ScaleTrain pod (run_synth_st.sh):
# supervise_borrowing_job.py relaunches the pod after preemption / failure (it resumes from S3) until the pod writes
# ${S3}/pods/<TAG>.COMPLETE. tmux session rl-<TAG>.
#   CODE_REF=<jiahaoz commit> TAG=synth-sft-maze33 GPUS=4 PLAN=/tmp/synth/plans/sft-maze33.json bash launch_synth_st.sh
#   env: ALLOW_BORROWING (true), DEADLINE_HOURS (48), ST_IMAGE (the maze grids' image), S3 (s3://scale-ml/genai/rl-distill/synth)
#   a TAG whose COMPLETE marker exists is refused; stop: touch <state>/STOP; tmux kill-session -t "=rl-<TAG>"
set -euo pipefail
MAIN=/mnt/efs/jasonwei/rl-distill; cd "${MAIN}"
IMG="${ST_IMAGE:-692474966980.dkr.ecr.us-west-2.amazonaws.com/scale_train/shared/training/tmp:20260829-002920.cd13c11a-e0f3-4f65-a74b-0e0c5d72d58a}"
: "${CODE_REF:?}" "${TAG:?}" "${GPUS:?}" "${PLAN:?}"
S3="${S3:-s3://scale-ml/genai/rl-distill/synth}"
export AWS_PROFILE=ml-worker   # writes s3://scale-ml (the plan); the supervisor below runs with the instance role
if aws s3 ls "${S3}/pods/${TAG}.COMPLETE" >/dev/null 2>&1; then
  echo "${S3}/pods/${TAG}.COMPLETE exists: the supervisor would exit at once; remove it or use another TAG" >&2; exit 1
fi
aws s3 cp --only-show-errors "${PLAN}" "${S3}/plans/${TAG}.json"
BORROW=""; [ "${ALLOW_BORROWING:-true}" = true ] && BORROW="--allow-borrowing"
envs="TAG=${TAG},GPUS=${GPUS},S3=${S3}"
sup=".scale_train_supervisors/${TAG}-$(date -u +%Y%m%d)"; mkdir -p "${sup}/pod-logs"; rm -f "${sup}/STOP"
tmux kill-session -t "=rl-${TAG}" 2>/dev/null || true
tmux new-session -d -s "rl-${TAG}" "env -u AWS_PROFILE python3 rl-distill-scripts/scale_train/supervise_borrowing_job.py --name ${TAG} \
  --relaunch-on-cancel --relaunch-on-failure --max-relaunches 10 --quick-cancel-seconds 300 --max-quick-cancels 3 \
  --failure-backoff-seconds 300 --max-failure-backoff-seconds 1800 --completion-marker-s3-uri ${S3}/pods/${TAG}.COMPLETE \
  --launch-log ${sup}/launch.log --monitor-log ${sup}/monitor.log --state-file ${sup}/state.json --stop-file ${sup}/STOP --pod-log-dir ${sup}/pod-logs \
  -- env CODE_REF=${CODE_REF} bash rl-distill-scripts/scale_train/launch_st_with_code.sh --image ${IMG} --gpus-per-instance ${GPUS} --priority high ${BORROW} \
     --active-deadline-hours ${DEADLINE_HOURS:-48} --run-file ../synth/run_synth_st.sh --job-name ${TAG} --env-vars '${envs}' \
  2>&1 | tee -a ${sup}/supervisor.out"
echo "started supervisor rl-${TAG} (state ${sup}, code ${CODE_REF})"
