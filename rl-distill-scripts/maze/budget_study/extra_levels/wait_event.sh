#!/usr/bin/env bash
# Block until something needs attention, then print what and exit (run it in the background; its exit is the wake-up):
#   * one of my selection / make_pair / RL-trainer / screening processes present at start has exited (done or crashed);
#   * a new OOM / Traceback line appears in an RL or selection log;
#   * /work2 drops below MIN_FREE_GB (default 5);
#   * TIMEOUT_MIN (default 40) elapses.
TIMEOUT_MIN=${1:-40}; MIN_FREE_GB=${MIN_FREE_GB:-5}
R=$HOME/tailrl_maze_runs/pretrain20k_rerun
watched() { ps -u jiahaoz4 -o pid=,args= | awk '/[s]elect_matched_init|[m]ake_pair.py|[v]erl.trainer.main_ppo|[s]creen_watch|[s]creen_sft_ckpts/ && !/awk/ {print $1}' | sort; }
errcount() { cat "$HOME"/tailrl_maze_runs/rl/p*_S0_rl.log "$HOME"/tailrl_maze_runs/rl/p*_L0_rl.log "$R"/*/select_goal_reached_*.log 2>/dev/null | grep -cE 'OutOfMemoryError|Traceback'; }
start=$(watched); e0=$(errcount); t0=$(date +%s)
echo "watching $(echo "$start" | grep -c .) processes; errors so far $e0"
while true; do
  sleep 30
  now=$(watched)
  gone=$(comm -23 <(echo "$start") <(echo "$now"))
  if [ -n "$gone" ]; then echo "EXITED: $(echo $gone)"; break; fi
  e=$(errcount); if [ "$e" -gt "$e0" ]; then echo "NEW ERRORS: $e0 -> $e"; break; fi
  free=$(df -BG /work2 | tail -1 | awk '{print $4}' | tr -d G); if [ "$free" -lt "$MIN_FREE_GB" ]; then echo "DISK LOW: ${free}G"; break; fi
  if [ $(( $(date +%s) - t0 )) -ge $(( TIMEOUT_MIN * 60 )) ]; then echo "TIMEOUT ${TIMEOUT_MIN} min"; break; fi
done
date
