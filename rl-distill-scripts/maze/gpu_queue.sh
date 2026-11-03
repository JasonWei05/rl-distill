#!/usr/bin/env bash
# Run a list of GPU jobs on a set of GPUs: one detached worker per GPU, one job per GPU at a time, and every worker pulls
# the next ready job the moment its current one ends, so no GPU idles while work is left and nobody has to watch.
#
#   bash gpu_queue.sh <queue-file> <gpu> [<gpu> ...]   start a worker per GPU (a GPU that already has a live worker for this
#                                                     queue is skipped, so running it again only adds GPUs)
#   bash gpu_queue.sh --status <queue-file>            one line per job (DONE / FAILED / RUNNING / READY / WAITING) + workers
#
# Queue file: one job per line; blank lines and lines starting with '#' are ignored.
#   <name> [after=<name>[,<name>...]] <command>
#   <command> runs under `bash -c` from the rl-distill root with FOREGROUND=1, QUEUE_GPU=<gpu>, M=<this directory> and
#   PY=${PY:-~/TailRL/.venv/bin/python} exported; the literal {gpu} is replaced by the worker's GPU index. The command must
#   block until its work is finished (launch_single_rl.sh and launch_pretrain_matrix.sh do under FOREGROUND=1) and exit 0
#   on success. Workers take jobs in file order among those whose `after` jobs are DONE; a job whose dependency FAILED is
#   marked FAILED too. Workers re-read the file on every pass, so jobs appended later are picked up by live workers.
#
# State in $QUEUE_DIR (default ~/tailrl_maze_runs/queues/<queue-file name without extension>): <name>.lock/ while running
# (owner = "gpu worker-pid start"), <name>.log (the job's output), <name>.DONE / <name>.FAILED ("gpu rc start end"),
# worker_gpu<g>.{log,pid}. Re-run a failed job: rm <name>.FAILED, then start a worker again. Locks left by a dead worker
# (machine reboot) are cleared when workers are started. Env: POLL (seconds between passes when waiting, 60).
set -uo pipefail
SELF=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")
M_DIR=$(dirname "$SELF")
ROOT=$(cd "$M_DIR/../.." && pwd)
PY=${PY:-$HOME/TailRL/.venv/bin/python}
POLL=${POLL:-60}

usage() { sed -n '2,24p' "$SELF" | sed 's/^# \{0,1\}//'; exit 1; }

qdir_of() {  # qdir_of <queue-file>
  local b; b=$(basename "$1"); b=${b%.*}
  echo "${QUEUE_DIR:-$HOME/tailrl_maze_runs/queues/$b}"
}

# parse one job line into NAME / DEPS / CMD; returns 1 for a malformed line
parse() {
  NAME=""; DEPS=""; CMD=""
  if [[ $1 =~ ^[[:space:]]*([^[:space:]]+)[[:space:]]+(after=([^[:space:]]+)[[:space:]]+)?(.+)$ ]]; then
    NAME=${BASH_REMATCH[1]}; DEPS=${BASH_REMATCH[3]}; CMD=${BASH_REMATCH[4]}
    return 0
  fi
  return 1
}

read_jobs() {  # read_jobs <queue-file> -> JOB_LINES array
  mapfile -t JOB_LINES < <(grep -vE '^[[:space:]]*(#|$)' "$1")
}

validate() {  # validate <queue-file>: unique names, known dependencies
  local qf=$1 line bad=0
  declare -A seen=()
  read_jobs "$qf"
  for line in "${JOB_LINES[@]}"; do
    parse "$line" || { echo "malformed line: $line" >&2; bad=1; continue; }
    [ -z "${seen[$NAME]:-}" ] || { echo "duplicate job name: $NAME" >&2; bad=1; }
    seen[$NAME]=1
  done
  for line in "${JOB_LINES[@]}"; do
    parse "$line" || continue
    local d; IFS=',' read -r -a _deps <<< "$DEPS"
    for d in "${_deps[@]}"; do
      [ -z "$d" ] || [ -n "${seen[$d]:-}" ] || { echo "job $NAME depends on unknown job $d" >&2; bad=1; }
    done
  done
  return $bad
}

state_of() {  # state_of <qdir> <name> <deps> -> DONE FAILED RUNNING STALE READY WAITING
  local q=$1 n=$2 deps=$3 d ready=1
  [ -f "$q/$n.DONE" ] && { echo DONE; return; }
  [ -f "$q/$n.FAILED" ] && { echo FAILED; return; }
  if [ -d "$q/$n.lock" ]; then
    local pid; pid=$(awk '{print $2}' "$q/$n.lock/owner" 2>/dev/null)
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then echo RUNNING; else echo STALE; fi
    return
  fi
  IFS=',' read -r -a _deps <<< "$deps"
  for d in "${_deps[@]}"; do
    [ -n "$d" ] || continue
    [ -f "$q/$d.FAILED" ] && { echo DEPFAILED; return; }
    [ -f "$q/$d.DONE" ] || ready=0
  done
  [ $ready = 1 ] && echo READY || echo WAITING
}

worker() {  # worker <queue-file> <gpu>
  local qf=$1 gpu=$2 q line claimed waiting st start rc
  q=$(qdir_of "$qf")
  echo $$ > "$q/worker_gpu$gpu.pid"
  echo "$(date -Is) worker on GPU $gpu (pid $$), queue $qf"
  while :; do
    read_jobs "$qf"
    claimed=""; waiting=0
    for line in "${JOB_LINES[@]}"; do
      parse "$line" || continue
      st=$(state_of "$q" "$NAME" "$DEPS")
      case $st in
        DONE|FAILED) continue ;;
        RUNNING|STALE|WAITING) waiting=1; continue ;;
        DEPFAILED)
          if mkdir "$q/$NAME.lock" 2>/dev/null; then
            echo "$(date -Is) not run: a dependency in [$DEPS] FAILED" >> "$q/$NAME.log"
            echo "$gpu dep-failed $(date -Is) $(date -Is)" > "$q/$NAME.FAILED"
            rm -rf "$q/$NAME.lock"
            echo "$(date -Is) $NAME FAILED (dependency failed)"
          fi
          continue ;;
      esac
      mkdir "$q/$NAME.lock" 2>/dev/null || { waiting=1; continue; }   # another worker claimed it first
      start=$(date -Is)
      echo "$gpu $$ $start" > "$q/$NAME.lock/owner"
      claimed=$NAME
      local cmd=${CMD//\{gpu\}/$gpu}
      echo "$start start $NAME on GPU $gpu: $cmd"
      ( cd "$ROOT" && export FOREGROUND=1 QUEUE_GPU="$gpu" M="$M_DIR" PY="$PY" && bash -c "$cmd" ) >> "$q/$NAME.log" 2>&1 < /dev/null
      rc=$?
      if [ $rc = 0 ]; then echo "$gpu $rc $start $(date -Is)" > "$q/$NAME.DONE"
      else echo "$gpu $rc $start $(date -Is)" > "$q/$NAME.FAILED"; fi
      rm -rf "$q/$NAME.lock"
      echo "$(date -Is) end $NAME on GPU $gpu: rc=$rc"
      break
    done
    if [ -z "$claimed" ]; then
      if [ $waiting = 1 ]; then sleep "$POLL"; continue; fi
      echo "$(date -Is) worker on GPU $gpu: nothing left to run"
      break
    fi
  done
}

status() {  # status <queue-file>
  local qf=$1 q line st f
  q=$(qdir_of "$qf")
  echo "queue $qf (state in $q)"
  read_jobs "$qf"
  for line in "${JOB_LINES[@]}"; do
    parse "$line" || continue
    st=$(state_of "$q" "$NAME" "$DEPS")
    case $st in
      DONE|FAILED) f=$(cat "$q/$NAME.$st" 2>/dev/null); printf '  %-18s %-8s gpu %s rc %s  %s -> %s\n' "$NAME" "$st" $f ;;
      RUNNING|STALE) printf '  %-18s %-8s gpu %s since %s\n' "$NAME" "$st" "$(awk '{print $1}' "$q/$NAME.lock/owner")" "$(awk '{print $3}' "$q/$NAME.lock/owner")" ;;
      WAITING) printf '  %-18s %-8s after %s\n' "$NAME" "$st" "$DEPS" ;;
      *) printf '  %-18s %s\n' "$NAME" "$st" ;;
    esac
  done
  local pf g
  for pf in "$q"/worker_gpu*.pid; do
    [ -f "$pf" ] || continue
    g=$(basename "$pf" .pid); g=${g#worker_gpu}
    if kill -0 "$(cat "$pf")" 2>/dev/null; then echo "  worker GPU $g: alive (pid $(cat "$pf"))"; else echo "  worker GPU $g: exited"; fi
  done
}

case "${1:-}" in
  --worker) shift; worker "$1" "$2"; exit 0 ;;
  --status) shift; [ -f "${1:-}" ] || usage; status "$1"; exit 0 ;;
  ""|-h|--help) usage ;;
esac

QF=$(cd "$(dirname "$1")" && pwd)/$(basename "$1"); shift
[ -f "$QF" ] || { echo "no queue file $QF" >&2; exit 1; }
[ $# -ge 1 ] || usage
validate "$QF" || { echo "queue file has errors; nothing started" >&2; exit 1; }
Q=$(qdir_of "$QF"); mkdir -p "$Q"
# clear locks left by dead workers (the job goes back to the queue)
for l in "$Q"/*.lock; do
  [ -d "$l" ] || continue
  pid=$(awk '{print $2}' "$l/owner" 2>/dev/null)
  if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then echo "clearing stale lock $(basename "$l") (worker ${pid:-?} is gone)"; rm -rf "$l"; fi
done
read_jobs "$QF"
echo "${#JOB_LINES[@]} jobs in $QF; state in $Q"
for g in "$@"; do
  pf="$Q/worker_gpu$g.pid"
  if [ -f "$pf" ] && kill -0 "$(cat "$pf")" 2>/dev/null; then echo "GPU $g: worker already running (pid $(cat "$pf")), not starting another"; continue; fi
  setsid nohup bash "$SELF" --worker "$QF" "$g" >> "$Q/worker_gpu$g.log" 2>&1 < /dev/null &
  echo "GPU $g: worker pid $! -> $Q/worker_gpu$g.log"
done
echo "status: bash $SELF --status $QF"
