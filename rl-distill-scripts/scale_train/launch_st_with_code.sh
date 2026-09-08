#!/usr/bin/env bash
# Launch a ScaleTrain job on the known-good pre-built rl-distill image, running the *current HEAD* of this repo.
# The remote image builds never materialize, so: pack `git archive HEAD` into a tarball, upload it (ml-worker profile)
# and pass it as --code-s3-uri; the job command unpacks it over the baked /workspace/rl-distill before the run-file starts.
# Uncommitted changes are NOT shipped -- commit first.
#
#   bash launch_st_with_code.sh --gpus-per-instance 2 --priority high --active-deadline-hours 72 \
#     --run-file run_gemma4_student_ckpt_passk_st.sh --job-name gemma4-e4bbase-passk-12b --env-vars "STUDENT=12b"
#   (add --allow-borrowing for preemptible capacity; every other flag is passed through to launch_st_job.py)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
REPO_ROOT="$(git rev-parse --show-toplevel)"
IMAGE="${ST_IMAGE:-692474966980.dkr.ecr.us-west-2.amazonaws.com/scale_train/shared/training/tmp:20260829-002920.cd13c11a-e0f3-4f65-a74b-0e0c5d72d58a}"
CODE_S3_PREFIX="${CODE_S3_PREFIX:-s3://scale-ml/genai/rl-distill/code}"
export AWS_PROFILE="${AWS_PROFILE:-ml-worker}"   # the instance role cannot write the bucket
sha="$(git -C "${REPO_ROOT}" rev-parse --short=8 HEAD)"
if [ -n "$(git -C "${REPO_ROOT}" status --porcelain -- rl-distill-scripts/scale_train | head -1)" ]; then
  echo "WARNING: uncommitted changes under rl-distill-scripts/scale_train are not in the tarball (git archive HEAD)" >&2
fi
tarball="/tmp/rl-distill-code-${sha}.tar.gz"
uri="${CODE_S3_PREFIX}/rl-distill-code-${sha}.tar.gz"
if aws s3 ls "${uri}" >/dev/null 2>&1; then
  echo "CODE_TARBALL ${uri} commit=${sha} (already uploaded)"
else
  git -C "${REPO_ROOT}" archive --format=tar.gz -o "${tarball}" HEAD
  aws s3 cp --only-show-errors "${tarball}" "${uri}"
  echo "CODE_TARBALL ${uri} commit=${sha} size=$(du -h "${tarball}" | cut -f1)"
fi
exec python3 launch_st_job.py --n-instances 1 --image "${IMAGE}" --code-s3-uri "${uri}" "$@"
