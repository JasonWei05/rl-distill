#!/usr/bin/env bash
# Relaunch RL runs that die before their STEPS (another user's job OOMs the card, a Ray port band is still held, ...).
# Every POLL seconds, for each runs/<suite>.env (written by rl_launch.sh at every launch):
#   * skip if runs/<suite>.done, locks/<suite>.moving, its train.sh is running, or it was launched < 3 min ago;
#   * newest checkpoint >= STEPS -> runs/<suite>.done (finished);
#   * otherwise relaunch it through rl_launch.sh (resumes from its newest checkpoint): on its card unless another user
#     is on it (then my least-loaded card without other users), on its port band unless its last log refused the band
#     (then a free one). At most MAX_RESTARTS per run; after that it is left for a human (logged as GIVE UP).
#   setsid nohup bash watchdog.sh >> logs/watchdog.log 2>&1 &
set -uo pipefail
D=/work2/jiahaoz4/maze_extra
CK=$HOME/tailrl_maze_runs/checkpoints/tailrl-text-maze
RL=$HOME/tailrl_maze_runs/rl
POLL=${POLL:-120}; MAX_RESTARTS=${MAX_RESTARTS:-5}
ME=$(id -un)
log() { echo "$(date -u '+%m-%d %H:%M:%S') $*"; }
# ps is captured before the grep: piped into grep -q under pipefail, a SIGPIPE to ps would read as "not running"
running() { local out; out=$(ps -u "$ME" -o args=); grep -q -- "[s]cripts/train.sh --suite $1 " <<< "$out"; }
latest_ckpt() {
  local d
  d=$(find "$CK" -maxdepth 1 -type d -name "textmaze_$1_grpo_binary_*" | head -1)   # -type d skips the GPU aliases
  [ -n "$d" ] && cat "$d/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0
}
others_on() {  # other users' pids on GPU index $1
  local bus
  bus=$(nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader | awk -F', ' -v g="$1" '$1 == g {print $2}')
  nvidia-smi --query-compute-apps=gpu_bus_id,pid --format=csv,noheader | awk -F', ' -v b="$bus" '$1 == b {print $2}' |
    while read -r p; do u=$(ps -o user= -p "$p" 2>/dev/null); [ -n "$u" ] && [ "$u" != "$ME" ] && echo "$p"; done
}
running_field() {  # values of field $1 over the env files of runs whose train.sh is running
  for e in "$D"/runs/*.env; do s=$(basename "$e" .env); running "$s" && grep "^$1=" "$e" | cut -d= -f2; done
}
pick_gpu() {  # my least-loaded card with no other user's process; ties -> the larger card
  local best="" bestkey=""
  while IFS=', ' read -r g mem; do
    [ -n "$(others_on "$g")" ] && continue
    n=$(running_field GPU | grep -cx "$g")
    { [ "$mem" -lt 70000 ] && [ "$n" -ge 2 ]; } && continue   # at most 2 runs on a 40 GB card
    [ "$n" -ge 3 ] && continue
    key=$(printf "%d%06d" "$n" $((100000 - mem / 1000)))
    if [ -z "$bestkey" ] || [ "$key" \< "$bestkey" ]; then best=$g; bestkey=$key; fi
  done < <(nvidia-smi --query-gpu=index,memory.total --format=csv,noheader,nounits)
  echo "$best"
}
free_port() {
  local used; used=$(running_field PORT)
  for i in $(seq 0 11); do
    echo "$used" | grep -qx "$i" && continue
    ss -ltnH 2>/dev/null | awk '{print $4}' | grep -qE ":(70|71|72|73|75|76|77|78|79)$(printf %02d "$i")$" && continue
    echo "$i"; return
  done
}
if [ "${WATCHDOG_TEST:-0}" = 1 ]; then   # dry check of the helpers, no action
  echo "pick_gpu=$(pick_gpu) free_port=$(free_port) ports in use: $(running_field PORT | sort -n | tr '\n' ' ')"
  for e in "$D"/runs/*.env; do s=$(basename "$e" .env); echo "$s running=$(running "$s" && echo y || echo n) ckpt=$(latest_ckpt "$s") steps=$(grep ^STEPS= "$e" | cut -d= -f2)"; done
  exit 0
fi
log "watchdog start (poll ${POLL}s, max ${MAX_RESTARTS} restarts per run)"
while true; do
  for e in "$D"/runs/*.env; do
    [ -e "$e" ] || continue
    s=$(basename "$e" .env)
    [ -e "$D/runs/$s.done" ] && continue
    [ -e "$D/locks/$s.moving" ] && continue
    running "$s" && continue
    # shellcheck disable=SC1090
    unset SIDE PAIR SEED STEPS GPU PORT ROLLOUT LOGPROB LAUNCHED; . "$e"
    [ $(( $(date +%s) - $(date -d "$LAUNCHED" +%s) )) -lt 180 ] && continue
    last=$(latest_ckpt "$s")
    if [ "$last" -ge "$STEPS" ]; then log "$s finished (checkpoint $last >= $STEPS)"; touch "$D/runs/$s.done"; continue; fi
    n=$(cat "$D/runs/$s.restarts" 2>/dev/null || echo 0)
    if [ "$n" -ge "$MAX_RESTARTS" ]; then
      [ -e "$D/runs/$s.gaveup" ] || { log "GIVE UP $s: dead at checkpoint $last after $n restarts"; touch "$D/runs/$s.gaveup"; }
      continue
    fi
    tag=${s%-*}; logf="$RL/${tag}_${SIDE}_rl.log"
    why=$(grep -hoE 'OutOfMemoryError|refusing to start Ray|Error executing job' "$logf" 2>/dev/null | tail -1)
    g=$GPU; [ -n "$(others_on "$g")" ] && g=$(pick_gpu)
    p=$PORT; { grep -q 'refusing to start Ray' "$logf" 2>/dev/null || running_field PORT | grep -qx "$p"; } && p=$(free_port)
    if [ -z "$g" ] || [ -z "$p" ]; then log "$s dead at checkpoint $last (${why:-no error line}); no card/port free now"; continue; fi
    log "$s dead at checkpoint $last (${why:-no error line}); relaunch $((n + 1))/$MAX_RESTARTS on GPU $g port $p"
    echo $((n + 1)) > "$D/runs/$s.restarts"
    SIDE=$SIDE RAY_PORT_INDEX=$p ROLLOUT_MICRO_BATCH_SIZE=$ROLLOUT LOG_PROB_MICRO_BATCH=$LOGPROB \
      bash "$D/rl_launch.sh" "$PAIR" "$SEED" "$STEPS" "$g" 2>&1 | sed 's/^/    /'
  done
  sleep "$POLL"
done
