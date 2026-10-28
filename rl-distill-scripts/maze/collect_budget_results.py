#!/usr/bin/env python3
"""Evaluate every point of a budget plan (plan_budgets.py) that has finished and print the pass@1-vs-budget table.

Points: Arm B (the S0 RL checkpoint at each budget), the teachers (L0 RL checkpoint of each cell), and each cell's final
student from on-policy (train.sh experiment "<cell>-opd", its last checkpoint = the budget stop) and off-policy
(budget/<pair>/off/<cell>/ckpt-*, DONE marker) distillation. Each is screened once per (k, parquet) on the dev mazes
(default) or on --parquet test1000 and cached in the results file; the actual FLOPs spent are read back from the
ledgers / train_metrics so the table shows real totals, not the plan.

    CUDA_VISIBLE_DEVICES=7 ~/TailRL/.venv/bin/python rl-distill-scripts/maze/collect_budget_results.py \
        --plan ~/tailrl_maze_runs/budget/p10/plan.json --k 64
"""

import argparse
import glob
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import screen_sft_ckpts as ss  # noqa: E402

import torch  # noqa: E402

CKPT_ROOT = os.path.expanduser("~/tailrl_maze_runs/checkpoints/tailrl-text-maze")


def opd_final(cell, seed=0):
    """(hf dir, actual TFLOPs, step) of the finished on-policy run for a cell, or None."""
    for exp in glob.glob(os.path.join(CKPT_ROOT, f"textmaze_{cell}-opd_*_seed{seed}")):
        latest = os.path.join(exp, "latest_checkpointed_iteration.txt")
        if not os.path.isfile(latest):
            continue
        step = int(open(latest).read().strip())
        hf = os.path.join(exp, f"global_step_{step}", "actor", "huggingface")
        ledger = os.path.join(exp, f"global_step_{step}", "flops_ledger.json")
        if os.path.isfile(os.path.join(hf, "config.json")):
            tf = json.load(open(ledger))["cumulative_tflops"]["total"] if os.path.isfile(ledger) else None
            return hf, tf, step
    return None


def off_final(budget_dir, cell):
    out = os.path.join(budget_dir, "off", cell)
    if not os.path.isfile(os.path.join(out, "DONE")):
        return None
    ck = sorted((int(d.split("-")[1]), d) for d in os.listdir(out) if d.startswith("ckpt-"))
    if not ck:
        return None
    last = None
    for line in open(os.path.join(out, "train_metrics.jsonl")):
        last = json.loads(line)
    tf = (last.get("cumulative_student_tflops", 0.0) + last.get("teacher_trace_tflops", 0.0)) if last else None
    return os.path.join(out, ck[-1][1]), tf, ck[-1][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--parquet", default=None, help="default: TailRL's dev eval1000; pass ~/tailrl_maze_data/test1000/test.parquet for final numbers")
    ap.add_argument("--budget-dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tailrl-root", default=ss.DEFAULT_TAILRL)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-screen", action="store_true", help="only print what is finished, without evaluating")
    ap.add_argument("--labels-regex", default=None, help="evaluate only points whose label matches (to shard the screening over GPUs; results merge by label)")
    ap.add_argument("--extra-s-roots", nargs="*", default=[], metavar="NAME=CKPT_ROOT",
                    help="more S0 RL runs (e.g. seed replicates) read off at the same budgets, labelled armB-<NAME>-b<j>")
    ap.add_argument("--opd-seed", type=int, default=0, help="train.sh seed of the on-policy runs to collect (the budget dir of a replicate matrix holds its off-policy runs)")
    args = ap.parse_args()
    plan = json.load(open(args.plan))
    pair = plan["pair"]
    budget_dir = args.budget_dir or os.path.expanduser(f"~/tailrl_maze_runs/budget/{pair}")
    parquet = args.parquet or os.path.join(args.tailrl_root, "experiments/text_maze/data/eval1000/test.parquet")
    tag = "test" if args.parquet else "dev"
    out = args.out or os.path.join(budget_dir, f"results_{tag}_k{args.k}.json")
    results = json.load(open(out)) if os.path.exists(out) else {}

    # ---- the points -------------------------------------------------------------------------------------------------
    points = []  # (label, hf, total_tflops, note)
    for b in plan["arm_b"]:
        if b["hf"] and os.path.isfile(os.path.join(b["hf"], "config.json")):
            points.append((f"armB-b{b['j']}", b["hf"], b["s_tflops"], f"S0 RL step {b['s_step']}"))
    if args.extra_s_roots:
        from plan_budgets import at_or_below, read_ledgers
        for spec in args.extra_s_roots:
            name, root = spec.split("=", 1)
            S2 = read_ledgers(root)
            for b in plan["arm_b"]:
                st = at_or_below(S2, b["tflops"])
                hf = os.path.join(root, f"global_step_{st}", "actor", "huggingface") if st else None
                if hf and os.path.isfile(os.path.join(hf, "config.json")):
                    points.append((f"armB-{name}-b{b['j']}", hf, S2[st]["total"], f"{name} S0 RL step {st}"))
    seen_teachers = set()
    for c in plan["cells"]:
        if not c["feasible"]:
            continue
        if c["l_step"] not in seen_teachers and os.path.isfile(os.path.join(c["teacher_hf"], "config.json")):
            seen_teachers.add(c["l_step"])
            points.append((f"teacher-L{c['l_step']}", c["teacher_hf"], c["c_l_tflops"], f"L0 RL step {c['l_step']}"))
        o = opd_final(c["id"], args.opd_seed)
        if o:
            points.append((f"{c['id']}-opd", o[0], c["c_l_tflops"] + (o[1] or 0.0), f"opd step {o[2]}, distill {o[1]}"))
        f = off_final(budget_dir, c["id"])
        if f:
            points.append((f"{c['id']}-off", f[0], c["c_l_tflops"] + (f[1] or 0.0), f"off step {f[2]}, distill {f[1]}"))

    if not args.no_screen:
        judge = ss.load_judge(args.tailrl_root)
        prompts, gts = ss.load_prompts(parquet, 1000)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        import re
        for label, hf, total, note in points:
            if args.labels_regex and not re.search(args.labels_regex, label):
                continue
            results = json.load(open(out)) if os.path.exists(out) else {}  # re-read: other shards may have written
            if label in results and results[label]["hf"] == hf:
                continue
            torch.manual_seed(args.seed)
            r = ss.screen_one(hf, prompts, gts, judge, SimpleNamespace(k=args.k, batch_prompts=128, max_new_tokens=180), device)
            results = json.load(open(out)) if os.path.exists(out) else {}
            results[label] = {"hf": hf, "total_tflops": total, "note": note, "goal_reached": r["goal_reached"], "is_shortest": r["is_shortest"],
                              "mean_response_tokens": r["mean_response_tokens"]}
            tmp = out + f".tmp{os.getpid()}"
            with open(tmp, "w") as fh:
                json.dump(results, fh)
            os.replace(tmp, out)
            print(f"{label:>22} {hf.split('/')[-3] if 'global_step' in hf else os.path.basename(hf):>40} goal p@1 {r['goal_reached']['pass@1']:.4f} "
                  f"p@16 {r['goal_reached'].get('pass@16', float('nan')):.4f} short p@1 {r['is_shortest']['pass@1']:.4f} total {total:.1f} TFLOPs", flush=True)

    # ---- the table ----------------------------------------------------------------------------------------------------
    g = lambda label: results[label]["goal_reached"]["pass@1"] if label in results else None
    fmt = lambda v: f"{v:7.4f}" if v is not None else "      -"
    rhos = plan["ratios"]
    print(f"\n{pair} ({tag}, k={args.k}): goal_reached pass@1 by total budget; Arm A cells as teacher / on-policy / off-policy")
    head = f"{'budget':>7} {'TFLOPs':>9} {'ArmB':>7} |" + " |".join(f"{'rho=' + str(r):>8} {'OPD':>7} {'off':>7}" for r in rhos)
    print(head)
    for b in plan["arm_b"]:
        row = "%7s %9.1f %s |" % ("B%d" % b["j"], b["tflops"], fmt(g("armB-b%d" % b["j"])))
        for r in rhos:
            c = next((c for c in plan["cells"] if c["j"] == b["j"] and c["rho"] == r), None)
            if c is None or (not c["feasible"] and "dup_of" not in c):
                row += "%24s |" % "infeasible"
            else:
                cid = c.get("dup_of", c["id"])  # a duplicate shows the cell it shares its teacher and budget with
                row += " %s %s %s%s|" % (fmt(g("teacher-L%d" % c["l_step"])), fmt(g(cid + "-opd")), fmt(g(cid + "-off")), "=" if "dup_of" in c else " ")
        print(row)
    print(f"({len(results)} points evaluated; results in {out})")


if __name__ == "__main__":
    main()
