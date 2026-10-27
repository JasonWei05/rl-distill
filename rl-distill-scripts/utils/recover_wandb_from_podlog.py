"""Recover wandb metrics from verl pod logs (single-line `step:N - key:value - ...` records)
and re-log them to a recovery wandb run. Idempotent: re-running overwrites the same run id.

Usage: WANDB_API_KEY=... python recover_wandb_from_podlog.py <pod_log> <run_name> <run_id>
"""

import re
import sys
from collections import defaultdict

import wandb

log_path, run_name, run_id = sys.argv[1], sys.argv[2], sys.argv[3]

NUM = re.compile(
    r"(?:np\.(?:float|int)\d*\()?(-?[0-9]+(?:\.[0-9]+)?(?:e-?[0-9]+)?)\)?$"
)
LINE = re.compile(r"step:(\d+) - (.*)")

steps = defaultdict(dict)
for raw in open(log_path, errors="replace"):
    # strip ansi + ray prefix
    line = re.sub(r"\x1b\[[0-9;]*m", "", raw)
    m = LINE.search(line)
    if not m:
        continue
    step = int(m.group(1))
    for pair in m.group(2).split(" - "):
        if ":" not in pair:
            continue
        key, _, val = pair.partition(":")
        key = key.strip()
        v = NUM.match(val.strip())
        if v and re.match(r"^[\w@/.\-]+$", key):
            steps[step][key] = float(v.group(1))

print(f"parsed {len(steps)} steps: {min(steps)}..{max(steps)}, "
      f"{sum(len(v) for v in steps.values())} datapoints")

run = wandb.init(
    project="DAPO", entity="rl-distill", name=run_name, id=run_id, resume="allow"
)
for step in sorted(steps):
    run.log(steps[step], step=step)
run.finish()
print(f"RECOVERY_SYNCED {run_name} ({run_id})")
