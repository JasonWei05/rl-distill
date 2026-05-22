#!/usr/bin/env bash
set -euo pipefail

mapfile -t pids < <(ps -eo pid,args | awk '/generate_teacher_data[.]py/ {print $1}')
if ((${#pids[@]})); then
  printf 'killing generate_teacher_data pids: %s\n' "${pids[*]}"
  kill -9 "${pids[@]}" || true
else
  echo "no generate_teacher_data.py processes"
fi

mapfile -t launch_pids < <(ps -eo pid,args | awk '/launch_teacher_gen[.]sh/ {print $1}')
if ((${#launch_pids[@]})); then
  printf 'killing launch_teacher_gen pids: %s\n' "${launch_pids[*]}"
  kill -9 "${launch_pids[@]}" || true
else
  echo "no launch_teacher_gen.sh processes"
fi
