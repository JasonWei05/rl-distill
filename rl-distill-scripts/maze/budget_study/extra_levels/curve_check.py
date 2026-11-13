#!/usr/bin/env python3
"""The runbook's stop/extend rule for one RL run's dev curve (all log parts concatenated, later lines win):
flat = the rolling mean of five dev points stays within 1 pt of its plateau (max rolling mean) over the last 500 steps;
accelerating = the rise over the last 500 steps exceeds the rise over the 500 before.
    curve_check.py <tag> [<tag> ...] [--side S0|L0]       e.g. curve_check.py p5 p03 p03s1
"""
import argparse
import glob
import os
import re

RL = os.path.expanduser("~/tailrl_maze_runs/rl")
KEY = re.compile(r"val-aux/maze_17/goal_reached/mean@16:([0-9.]+)")
STEP = re.compile(r"step:(\d+) - ")
PF = re.compile(r"tflops/cumulative_total:([0-9.eE+-]+)")


def curve(tag, side):
    log = os.path.join(RL, f"{tag}_{side}_rl.log")
    parts = sorted(glob.glob(log + ".part*"), key=lambda p: int(p.rsplit("part", 1)[1])) + [log]
    dev, pf = {}, {}
    for p in parts:
        if not os.path.exists(p):
            continue
        for line in open(p, errors="replace"):
            s = STEP.search(line)
            if not s:
                continue
            st = int(s.group(1))
            v = KEY.search(line)
            if v:
                dev[st] = float(v.group(1))
            c = PF.search(line)
            if c:
                pf[st] = float(c.group(1)) / 1000
    return dev, pf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--side", default="S0")
    args = ap.parse_args()
    for tag in args.tags:
        dev, pf = curve(tag, args.side)
        steps = sorted(dev)
        if len(steps) < 15:
            print(f"{tag}: only {len(steps)} dev points")
            continue
        vals = [dev[s] for s in steps]
        roll = {steps[i]: sum(vals[i - 4:i + 1]) / 5 for i in range(4, len(steps))}
        last = steps[-1]
        plateau = max(roll.values())
        recent = [roll[s] for s in roll if s >= last - 500]
        flat = all(plateau - r <= 0.01 for r in recent)
        at = lambda s: roll[min(roll, key=lambda x: abs(x - s))]
        rise_last, rise_prev = at(last) - at(last - 500), at(last - 500) - at(last - 1000)
        print(f"{tag} {args.side}: step {last} ({pf.get(last, max(pf.values()) if pf else 0):.1f} PF), dev {vals[-1]:.3f}, "
              f"rolling-5 {roll[last]:.3f}, plateau {plateau:.3f} | flat(1 pt, 500 steps) {flat} | rise last 500 "
              f"{rise_last:+.3f} vs previous 500 {rise_prev:+.3f} -> {'ACCELERATING' if rise_last > rise_prev else 'decelerating'}")


if __name__ == "__main__":
    main()
