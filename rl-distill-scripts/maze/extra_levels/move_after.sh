#!/usr/bin/env bash
# When the run <suite> ends, move another run onto its card (move_run.sh checks the card first), so the card is not
# left empty for others to take and two runs sharing a card get one each.
#   setsid nohup bash move_after.sh <suite to wait for> <move_run.sh args...> > log 2>&1 &
set -uo pipefail
SUITE=$1; shift
echo "$(date '+%m-%d %H:%M:%S') waiting for $SUITE to end -> then move_run.sh $*"
while ps -u "$(id -un)" -o args= | grep -q -- "[s]cripts/train.sh --suite $SUITE "; do sleep 10; done
echo "$(date '+%m-%d %H:%M:%S') $SUITE ended"
bash /work2/jiahaoz4/maze_extra/move_run.sh "$@"
echo "$(date '+%m-%d %H:%M:%S') move_run exit=$?"
