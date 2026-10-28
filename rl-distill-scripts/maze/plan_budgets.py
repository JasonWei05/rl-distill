#!/usr/bin/env python3
"""Budget-study planner (text-maze, design doc section "Budget study"): from one matched pair's S0 RL run and L0 RL run,
derive the five total-FLOPs budgets and every Arm A cell (budget x detour split), and emit the job list.

Budgets. B5 = cumulative FLOPs of S0 RL at saturation: the first dev point (goal_reached mean@N in the RL log) within
--plateau-tol of the plateau, the plateau being the median of the last quarter of the dev curve. B_j = B5 / 2^(5-j).
Arm B at B_j = the S0 checkpoint with the largest cumulative FLOPs <= B_j (constant lr, one run serves every budget).
Arm A at (B_j, rho) = the L0 checkpoint with the largest cumulative FLOPs <= rho * B_j (the teacher), then distillation
into S0 with the remainder D = B_j - C_L:
  on-policy   MAX_TOTAL_TFLOPS = D (the trainer's ledger counts student gen / old-logprob / update and the teacher's
              reference forwards, and stops the run at D)
  off-policy  n_traces = D / (c_teacher_forward + c_student_update) teacher samples (1 per prompt), one epoch of exact
              forward KL, --tflops-budget D as the hard cap (teacher generation FLOPs + student update FLOPs)
Per-sample costs come from the RL ledgers themselves (cumulative phase TFLOPs / samples seen), so they carry the real
sequence lengths. All FLOPs in TFLOPs.

    ~/TailRL/.venv/bin/python rl-distill-scripts/maze/plan_budgets.py --pair p10 \
        --s-root ~/tailrl_maze_runs/checkpoints/tailrl-text-maze/textmaze_p10-s0_grpo_binary_ckpt0_bs256_G16_1gpu_0_seed0 \
        --l-root ~/tailrl_maze_runs/checkpoints/tailrl-text-maze/textmaze_p10-l0_grpo_binary_ckpt0_bs256_G16_1gpu_1_seed0 \
        --s-log ~/tailrl_maze_runs/rl/p10_S0_rl.log --l-log ~/tailrl_maze_runs/rl/p10_L0_rl.log \
        --out ~/tailrl_maze_runs/budget/p10/plan.json
"""

import argparse
import glob
import json
import math
import os
import re
import statistics


def read_ledgers(root):
    """{step: {"total": TFLOPs, phase: TFLOPs, "gen_tokens": n}} for every saved checkpoint."""
    out = {}
    for p in glob.glob(os.path.join(root, "global_step_*", "flops_ledger.json")):
        step = int(p.split("global_step_")[1].split(os.sep)[0])
        d = json.load(open(p))
        out[step] = dict(d["cumulative_tflops"], gen_tokens=d.get("cumulative_gen_tokens", 0))
    return dict(sorted(out.items()))


def read_val_curve(log, key):
    """[(step, value)] of the dev metric in a verl console log, including the step-0 validation."""
    pts = []
    init = re.compile(r"'" + re.escape(key) + r"': ([0-9.eE+-]+)")
    line_re = re.compile(r"step:(\d+) - ")
    val_re = re.compile(re.escape(key) + r":([0-9.eE+-]+)")
    in_init = False  # the step-0 dict is pprint'ed over many lines
    for line in open(log, errors="replace"):
        if not pts and "Initial validation metrics" in line:
            in_init = True
        if in_init:
            m = init.search(line)
            if m:
                pts.append((0, float(m.group(1))))
                in_init = False
        m = line_re.search(line)
        v = val_re.search(line)
        if m and v:
            pts.append((int(m.group(1)), float(v.group(1))))
        elif v and not pts:  # the step-0 validation is printed in key:value form without a step prefix
            pts.append((0, float(v.group(1))))
    return pts


def at_or_below(table, budget):
    """Largest step whose cumulative total is <= budget (None if even the first checkpoint exceeds it)."""
    best = None
    for step, d in table.items():
        if d["total"] <= budget:
            best = step
    return best


def nearest_val(curve, step):
    if not curve:
        return None
    return min(curve, key=lambda p: abs(p[0] - step))[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True, help="pair tag, e.g. p10 / p1 / p01")
    ap.add_argument("--s-root", required=True, help="S0 RL checkpoint root (global_step_N dirs)")
    ap.add_argument("--l-root", required=True, help="L0 RL checkpoint root")
    ap.add_argument("--s-log", required=True, help="S0 RL console log (dev curve)")
    ap.add_argument("--l-log", default=None, help="L0 RL console log (teacher dev accuracy per cell)")
    ap.add_argument("--val-key", default="val-aux/maze_17/goal_reached/mean@16")
    ap.add_argument("--ratios", type=float, nargs="+", default=[0.5, 0.75, 0.9], help="fraction of the budget given to L0 RL")
    ap.add_argument("--n-budgets", type=int, default=5)
    ap.add_argument("--plateau-tol", type=float, default=0.01, help="saturation = first dev point within this of the plateau")
    ap.add_argument("--smooth", type=int, default=5, help="rolling-mean window (dev points) applied before the saturation test; the raw mean@16 curve jitters by 2-5 pt")
    ap.add_argument("--t-sat", type=int, default=None, help="override the saturation step")
    ap.add_argument("--b5-tflops", type=float, default=None, help="override B5 directly (TFLOPs)")
    ap.add_argument("--rl-samples-per-step", type=int, default=4096, help="256 prompts x 16 rollouts")
    ap.add_argument("--opd-batch", type=int, default=128, help="student samples per on-policy distillation step")
    ap.add_argument("--off-batch", type=int, default=128, help="traces per off-policy optimizer step")
    ap.add_argument("--min-traces", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    S, L = read_ledgers(args.s_root), read_ledgers(args.l_root)
    if not S or not L:
        raise SystemExit(f"no flops_ledger.json under {args.s_root} / {args.l_root}")
    s_curve = read_val_curve(args.s_log, args.val_key)
    l_curve = read_val_curve(args.l_log, args.val_key) if args.l_log else []

    # --- per-sample costs from the ledgers (last checkpoint = most samples) ------------------------------------------
    def per_sample(table, phase):
        step = max(table)
        return table[step][phase] / (step * args.rl_samples_per_step)
    c_s_fwd, c_s_upd = per_sample(S, "old_log_prob"), per_sample(S, "update_actor")
    c_l_fwd = per_sample(L, "old_log_prob")
    opd_per_sample = 2 * c_s_fwd + c_l_fwd + c_s_upd  # student gen + old_log_prob + teacher ref + student update
    off_per_trace = c_l_fwd + c_s_upd  # teacher generation + one student epoch

    # --- saturation and budgets ---------------------------------------------------------------------------------------
    if args.b5_tflops is not None:
        b5, t_sat, plateau = args.b5_tflops, None, None
    else:
        if args.t_sat is None:
            if len(s_curve) < 8:
                raise SystemExit(f"only {len(s_curve)} dev points in {args.s_log}; too early to call saturation")
            w = max(1, args.smooth)
            smooth = [(s_curve[i][0], sum(v for _, v in s_curve[max(0, i - w + 1): i + 1]) / len(s_curve[max(0, i - w + 1): i + 1]))
                      for i in range(len(s_curve))]  # trailing rolling mean, so a point never borrows from later steps
            tail = [v for _, v in smooth[-max(2, len(smooth) // 4):]]
            plateau = statistics.median(tail)
            t_sat = next(s for s, v in smooth if v >= plateau - args.plateau_tol)
        else:
            t_sat = args.t_sat
            plateau = nearest_val(s_curve, t_sat)
        later = [s for s in S if s >= t_sat]  # the first checkpoint at or after t_sat
        s_step = later[0] if later else max(S)
        b5 = S[s_step]["total"]
    budgets = [b5 / 2 ** (args.n_budgets - 1 - j) for j in range(args.n_budgets)]

    plan = {
        "pair": args.pair, "s_root": args.s_root, "l_root": args.l_root, "s_log": args.s_log, "l_log": args.l_log,
        "val_key": args.val_key, "saturation": {"t_sat": t_sat, "plateau": plateau, "b5_tflops": b5, "tol": args.plateau_tol},
        "per_sample_tflops": {"s_forward": c_s_fwd, "s_update": c_s_upd, "l_forward": c_l_fwd,
                              "opd_per_sample": opd_per_sample, "off_per_trace": off_per_trace},
        "ratios": args.ratios, "budgets": [], "arm_b": [], "cells": [],
    }
    print(f"{args.pair}: S0 saturation step {t_sat} (plateau {plateau}), B5 = {b5:.1f} TFLOPs; "
          f"per-sample TFLOPs: S fwd {c_s_fwd:.5f}, S upd {c_s_upd:.5f}, L fwd {c_l_fwd:.5f}; "
          f"OPD {opd_per_sample:.5f}/sample, off-policy {off_per_trace:.5f}/trace")
    print(f"{'budget':>7} {'TFLOPs':>9} | {'ArmB S step':>11} {'dev':>6} | {'rho':>5} {'L step':>6} {'C_L':>8} {'teacher dev':>11} {'D':>8} {'OPD steps':>9} {'traces':>8} {'off steps':>9}")
    for j, b in enumerate(budgets, 1):
        s_step = at_or_below(S, b)
        plan["budgets"].append({"j": j, "tflops": b})
        plan["arm_b"].append({"j": j, "tflops": b, "s_step": s_step, "s_tflops": S[s_step]["total"] if s_step else None,
                              "dev": nearest_val(s_curve, s_step) if s_step else None,
                              "hf": os.path.join(args.s_root, f"global_step_{s_step}", "actor", "huggingface") if s_step else None})
        first = True
        seen_l_steps = {}  # checkpoint granularity can map two ratios of one budget onto the same teacher: run that cell once
        for rho in args.ratios:
            l_step = at_or_below(L, rho * b)
            cell = {"id": f"{args.pair}-b{j}-r{int(round(rho * 100)):02d}", "j": j, "budget_tflops": b, "rho": rho, "l_step": l_step}
            if l_step is None:
                cell["feasible"] = False
                cell["why"] = f"first L0 checkpoint ({min(L)} steps, {L[min(L)]['total']:.1f} TFLOPs) exceeds rho*B = {rho * b:.1f}"
            elif l_step in seen_l_steps:
                cell["feasible"] = False
                cell["dup_of"] = seen_l_steps[l_step]
                cell["why"] = f"same teacher checkpoint and budget as {seen_l_steps[l_step]} (L0 checkpoint granularity)"
            else:
                seen_l_steps[l_step] = cell["id"]
                d = b - L[l_step]["total"]
                n_traces = int(d / off_per_trace)
                cell.update({
                    "feasible": n_traces >= args.min_traces, "c_l_tflops": L[l_step]["total"], "distill_tflops": d,
                    "teacher_hf": os.path.join(args.l_root, f"global_step_{l_step}", "actor", "huggingface"),
                    "teacher_dev": nearest_val(l_curve, l_step),
                    "opd": {"max_total_tflops": d, "batch": args.opd_batch, "est_steps": int(d / (opd_per_sample * args.opd_batch))},
                    "off": {"n_traces": n_traces, "batch": min(args.off_batch, max(1, n_traces)),
                            "steps": max(1, math.ceil(n_traces / min(args.off_batch, max(1, n_traces)))), "tflops_budget": d},
                })
                if n_traces < args.min_traces:
                    cell["why"] = f"only {n_traces} traces fit in D = {d:.2f} TFLOPs"
            plan["cells"].append(cell)
            lead = f"{'B' + str(j):>7} {b:9.1f} | {str(s_step):>11} {nearest_val(s_curve, s_step) if s_step else float('nan'):6.3f} |" if first else f"{'':>7} {'':>9} | {'':>11} {'':>6} |"
            first = False
            if cell["feasible"]:
                print(f"{lead} {rho:5.2f} {l_step:>6} {cell['c_l_tflops']:8.1f} {str(cell['teacher_dev']):>11} {cell['distill_tflops']:8.1f} "
                      f"{cell['opd']['est_steps']:>9} {cell['off']['n_traces']:>8} {cell['off']['steps']:>9}")
            else:
                print(f"{lead} {rho:5.2f} {str(l_step):>6}  infeasible: {cell['why']}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(plan, f, indent=1)
    n_ok = sum(c["feasible"] for c in plan["cells"])
    print(f"wrote {args.out}: {n_ok} feasible cells of {len(plan['cells'])} (x2 variants)")


if __name__ == "__main__":
    main()
