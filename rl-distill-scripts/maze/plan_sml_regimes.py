#!/usr/bin/env python3
"""Job plan for the {S, M, L} x six-start regime study (text-maze, SML_REGIME_PLAN.md): every teacher -> student
distillation at the three absolute budgets C = 10 / 32 / 128 PFLOPs (shared by all pairs), with the teacher checkpoint
each job starts from.

Pairs. A student (S or M) at start level i; a teacher of the same or a larger size at level i + 1, i or i - 1 of the
ladder 0.1 / 0.3 / 1 / 3 / 5.5 / 10%. Priority 1 = a same-size teacher one level down (expected to lose), else 0.
Phase 1 = S and L only (every RL run exists); phase 2 = pairs with M (not trained yet: its fields are TBD).
Teacher at budget C = the last saved checkpoint of its RL run with cumulative FLOPs <= rho * C (rho = 0.75); a run that
ends earlier gives its final checkpoint ("capped": only the 10% runs, which have plateaued). On-policy distillation gets
D = C - C_T. Direct-RL arm = the student's own run at its last saved checkpoint <= C. Spot checks at C_large on the 1%
students: rho 0.5 / 0.9 for every cell type, off-policy rho 0.75 for the same-level types. Replicates: the 1% students
again with seed-1 teacher, student and distillation runs.
Saved steps: every 5 for the budget study's seed-0 runs (launch_pair_rl.sh), every 10 otherwise (launch_single_rl.sh).
FLOPs at a saved step are interpolated from the dev curve (a point every 10 steps): recompute them from each
checkpoint's flops_ledger.json (the Hub archive tars) before launching.
    plan_sml_regimes.py [--extra /work2/jiahaoz4/maze_extra/extra/rl_curves_extra.json] [--out-dir <maze>/data]
"""
import argparse
import csv
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
LEVELS = [("p01", "0.1%"), ("p03", "0.3%"), ("p1", "1%"), ("p3", "3%"), ("p5", "5.5%"), ("p10", "10%")]
N_PARAMS = {"S": 3.9, "M": 7.2, "L": 13.3}  # millions; M = geometric mean of S and L (proposed)
REGIMES = [("small", 10.0), ("mid", 32.0), ("large", 128.0)]
RHO = 0.75
PF_PER_STEP = {"S": 0.066, "M": 0.12, "L": 0.21}  # M: scaled from S by parameter count
PF_PER_GPU_HOUR = 25.0  # federer RL logs: S ~10 s/step at 0.066 PF, L 17-30 s/step at 0.21 PF


def load(v1, extra):
    """starts[pair][size] = start profile; runs[(pair, size, seed)] = {src, pts: [(step, pflops, dev)]}. A later file
    replaces a whole (pair, model, seed) curve, as rl_curve_family.py does."""
    starts, runs = {}, {}
    for src, path in (("v1", v1), ("extra", extra)):
        d = json.load(open(path))
        for pair, v in d["starts"].items():
            starts.setdefault(pair, {}).update({m[0]: prof for m, prof in v.items()})
        new = {}
        for c in d["curves"]:
            new.setdefault((c["pair"], c["model"][0], c["seed"]), []).append((c["step"], c["pflops"], c["dev"]))
        runs.update({k: {"src": src, "pts": sorted(p)} for k, p in new.items()})
    return starts, runs


def interp(pts, step, col):
    if step >= pts[-1][0]:
        return pts[-1][col]
    for a, b in zip(pts, pts[1:]):
        if a[0] <= step <= b[0]:
            return a[col] + (b[col] - a[col]) * (step - a[0]) / max(1, b[0] - a[0])
    return pts[0][col]


def ckpt_at(run, seed, budget):
    """(step, pflops, capped) of the last saved checkpoint with cumulative FLOPs <= budget."""
    pts, every = run["pts"], 5 if (run["src"] == "v1" and seed == 0) else 10
    if pts[-1][1] <= budget:
        return pts[-1][0], pts[-1][1], True
    best = (0, 0.0)
    for s in range(every, pts[-1][0] + 1, every):
        p = interp(pts, s, 1)
        if p > budget:
            break
        best = (s, p)
    return best[0], best[1], False


def plateau_note(run, budget):
    """A run that ends before the budget: capped if its dev moved <= 0.01 over the last 20% of steps, else extend it."""
    pts = run["pts"]
    d20 = pts[-1][2] - interp(pts, 0.8 * pts[-1][0], 2)
    head = "capped" if abs(d20) <= 0.01 else f"still rising: extend the run to {budget:.0f} PF"
    return f"{head} (ends at {pts[-1][1]:.0f} PF, step {pts[-1][0]}, dev {d20:+.3f} over its last 20% of steps)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", default=os.path.join(HERE, "data", "rl_curves_v1.json"))
    ap.add_argument("--extra", default="/work2/jiahaoz4/maze_extra/extra/rl_curves_extra.json")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "data"))
    args = ap.parse_args()
    starts, runs = load(args.v1, args.extra)

    def cap(size, i):
        return None if size == "M" else math.log10(starts[LEVELS[i][0]][size]["pass@16"])

    rows = []

    def add(kind, prio, s, i, t, j, regime, C, rho=RHO, variant="opd", seed=0):
        cs, ct = cap(s, i), cap(t, j)
        row = dict(phase=2 if "M" in (s, t) else 1, priority=prio, kind=kind, student=f"{s}@{LEVELS[i][1]}",
                   teacher=f"{t}@{LEVELS[j][1]}", size_ratio=round(N_PARAMS[t] / N_PARAMS[s], 2),
                   dcap=None if None in (cs, ct) else round(ct - cs, 3), regime=regime, C_PF=C, rho=rho,
                   variant=variant, seed=seed, teacher_run=f"{LEVELS[j][0]} {t}0 s{seed}")
        run = runs.get((LEVELS[j][0], t, seed))
        if run is None:  # an M teacher: its RL run does not exist yet
            row.update(teacher_step="TBD", C_T_PF="TBD", rho_eff="TBD", D_PF=round((1 - rho) * C, 1),
                       note="teacher run to train" + ("; student to train" if s == "M" else ""))
        else:
            step, ct_pf, capped = ckpt_at(run, seed, rho * C)
            notes = [plateau_note(run, rho * C)] if capped else []
            if not capped and ct_pf / C < rho - 0.1:
                notes.append("coarse saves: rerun the first 40 steps saving every step")
            if s == "M":
                notes.append("student to train")
            row.update(teacher_step=step, C_T_PF=round(ct_pf, 2), rho_eff=round(ct_pf / C, 2),
                       D_PF=round(C - ct_pf, 2), note="; ".join(notes))
        rows.append(row)

    for s in ("S", "M"):
        for i in range(len(LEVELS)):
            for t in ("S", "M", "L"):
                if N_PARAMS[t] < N_PARAMS[s]:
                    continue
                for j in (i + 1, i, i - 1):
                    if not 0 <= j < len(LEVELS):
                        continue
                    prio = 1 if (j == i - 1 and t == s) else 0
                    for regime, C in REGIMES:
                        add("main", prio, s, i, t, j, regime, C)
                    if LEVELS[i][0] == "p1":  # spot checks and replicates on the 1% students
                        add("spot", prio, s, i, t, j, "large", 128.0, rho=0.5)
                        add("spot", prio, s, i, t, j, "large", 128.0, rho=0.9)
                        if j == i:
                            add("spot", prio, s, i, t, j, "large", 128.0, variant="off")
                        for regime, C in REGIMES:
                            add("replicate", prio, s, i, t, j, regime, C, seed=1)

    direct = []
    for s in ("S", "M"):
        for i, (pair, lv) in enumerate(LEVELS):
            for seed in ((0, 1) if pair == "p1" else (0,)):
                for regime, C in REGIMES:
                    run = runs.get((pair, s, seed))
                    if run is None:
                        direct.append(dict(student=f"{s}@{lv}", seed=seed, regime=regime, C_PF=C, step="TBD",
                                           PF="TBD", dev_mean16="TBD", note="student run to train"))
                        continue
                    step, pf, capped = ckpt_at(run, seed, C)
                    direct.append(dict(student=f"{s}@{lv}", seed=seed, regime=regime, C_PF=C, step=step,
                                       PF=round(pf, 2), dev_mean16=round(interp(run["pts"], step, 2), 3),
                                       note=plateau_note(run, C) if capped else ""))

    os.makedirs(args.out_dir, exist_ok=True)
    for name, data in (("sml_regime_plan.csv", rows), ("sml_regime_direct_rl.csv", direct)):
        with open(os.path.join(args.out_dir, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
        print(f"{name}: {len(data)} rows")

    # summary: jobs and distillation PFLOPs per phase and kind, then the phase-1 pair table
    print("\n| phase | kind | pairs | runs | distillation PF |\n|---|---|---|---|---|")
    for ph in (1, 2):
        for kind in ("main", "spot", "replicate"):
            sel = [r for r in rows if r["phase"] == ph and r["kind"] == kind]
            pairs = {(r["student"], r["teacher"]) for r in sel}
            print(f"| {ph} | {kind} | {len(pairs)} | {len(sel)} | {sum(r['D_PF'] for r in sel):.0f} |")
    print("\nphase-1 main pairs (teacher step / C_T PF at C = 10 / 32 / 128):")
    print("| student | teacher | N_T/N_S | dcap | prio | C=10 | C=32 | C=128 | note |\n|---|---|---|---|---|---|---|---|---|")
    main_rows = [r for r in rows if r["phase"] == 1 and r["kind"] == "main"]
    for k in range(0, len(main_rows), 3):
        r3 = main_rows[k:k + 3]
        cells = [f"{r['teacher_step']} / {r['C_T_PF']}" for r in r3]
        notes = sorted({n for r in r3 for n in r["note"].split("; ") if n})
        print(f"| {r3[0]['student']} | {r3[0]['teacher']} | {r3[0]['size_ratio']} | {r3[0]['dcap']:+.2f} | "
              f"{r3[0]['priority']} | " + " | ".join(cells) + f" | {'; '.join(notes)} |")
    print("\ndirect-RL arm (S students, seed 0: step / PF / dev mean@16):")
    for r in direct:
        if r["student"].startswith("S") and r["seed"] == 0:
            print(f"  {r['student']:7s} {r['regime']:5s} C={r['C_PF']:5.0f}: {r['step']} / {r['PF']} / {r['dev_mean16']}"
                  f"  {r['note']}")


if __name__ == "__main__":
    main()
