#!/usr/bin/env python3
"""CSV datasets for the "Maze RL-curve fits" dashboard, from the final step4_L.sh outputs in maze_extra/extra/ and
rl_curves_v1.json (a later file replaces a whole (pair, model, seed) curve, as rl_curve_family.py fit does).
    make_csvs.py   -> artifact/{curves,starts,family,by_start,cmid_curves}.csv"""
import csv
import json
import math
import os

X = "/work2/jiahaoz4/maze_extra/extra"
V1 = "/work2/jiahaoz4/rl-distill/rl-distill-scripts/maze/data/rl_curves_v1.json"
OUT = "/work2/jiahaoz4/maze_extra/artifact"
PAIRS = ["p10", "p5", "p3", "p1", "p03", "p01"]
LEVEL = {"p10": "10%", "p5": "5.5%", "p3": "3%", "p1": "1%", "p03": "0.3%", "p01": "0.1%"}
SIZE = {"S0": ("S", "3.9M"), "L0": ("L", "13.3M")}
ND = 6  # decimals kept in the CSVs, so the page rounds each value only once


def write(name, cols, rows):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows([[r[c] for c in cols] for r in rows])
    print(f"{name}: {len(rows)} rows, {os.path.getsize(os.path.join(OUT, name))} bytes")


starts, curves = {}, {}
for p in (V1, f"{X}/rl_curves_extra.json"):
    d = json.load(open(p))
    for pair, v in d["starts"].items():
        starts.setdefault(pair, {}).update(v)
    new = {}
    for c in d["curves"]:
        new.setdefault((c["pair"], c["model"], c["seed"]), []).append(c)
    curves.update(new)

rows = []
for model in ("S0", "L0"):
    for pair in PAIRS:
        for seed in (0, 1):
            for c in sorted(curves.get((pair, model, seed), []), key=lambda c: (c["step"], c["pflops"])):
                rows.append(dict(model=SIZE[model][0], pair=pair, seed=seed, step=c["step"], pflops=c["pflops"], dev=c["dev"]))
write("curves.csv", ["model", "pair", "seed", "step", "pflops", "dev"], rows)

rows = [dict(model=SIZE[m][0], pair=p, level=LEVEL[p], pass1=round(starts[p][m]["pass@1"], ND),
             pass16=round(starts[p][m]["pass@16"], ND), pass64=round(starts[p][m]["pass@64"], ND))
        for m in ("S0", "L0") for p in PAIRS]
write("starts.csv", ["model", "pair", "level", "pass1", "pass16", "pass64"], rows)

fam = {m: json.load(open(f"{X}/rl_curve_family_{m}.json"))[m] for m in ("S0", "L0")}
sig = {m: json.load(open(f"{X}/rl_curve_family_{m}_SIG.json"))[m] for m in ("S0", "L0")}
per = {m: {(c["pair"], c["seed"]): c for c in json.load(open(f"{X}/rl_cmid_per_curve_{m}.json"))["curves"]} for m in ("S0", "L0")}
rows = [dict(model=SIZE[m][0], size=SIZE[m][1], form="PAPER", **{k: round(v, ND) for k, v in fam[m]["params"].items()})
        for m in ("S0", "L0")]
write("family.csv", ["model", "size", "form", "A", "B", "m0", "m1"], rows)

rows = []
for p in PAIRS:
    r = dict(pair=p, level=LEVEL[p])
    for m in ("S0", "L0"):
        t = SIZE[m][0].lower()
        p16 = starts[p][m]["pass@16"]
        loo = fam[m]["loo"][p]["params"]
        r.update({f"{t}_rmse_s0": round(fam[m]["in_sample_rmse"][f"{p}:s0"], ND),
                  f"{t}_rmse_s1": round(fam[m]["in_sample_rmse"][f"{p}:s1"], ND),
                  f"{t}_loo": round(fam[m]["loo"][p]["rmse"], ND),
                  f"{t}_loo_sig": round(sig[m]["loo"][p]["rmse"], ND),
                  f"{t}_cmid_fit": round(fam[m]["per_start"][p]["Cmid_pf"], ND),
                  f"{t}_cmid_loo": round(10 ** (loo["m0"] + loo["m1"] * math.log10(p16)), ND),
                  f"{t}_cmid_s0": per[m][(p, 0)]["Cmid_pf"], f"{t}_cmid_s1": per[m][(p, 1)]["Cmid_pf"]})
    rows.append(r)
cols = ["pair", "level"] + [f"{t}_{k}" for t in "sl" for k in
                            ("rmse_s0", "rmse_s1", "loo", "loo_sig", "cmid_fit", "cmid_loo", "cmid_s0", "cmid_s1")]
write("by_start.csv", cols, rows)

rows = []
for m in ("S0", "L0"):
    for p in PAIRS:
        for seed in (0, 1):
            c = per[m][(p, seed)]
            pts = sorted(curves[(p, m, seed)], key=lambda c: (c["step"], c["pflops"]))
            kept = [q["pflops"] for q in pts if q["step"] >= 30 and q["pflops"] > 0]   # the fit's own point set
            rows.append(dict(model=SIZE[m][0], pair=p, level=LEVEL[p], seed=seed, pass16=round(c["p16"], ND),
                             r_ref=round(fam[m]["per_start"][p][f"R_ref_s{seed}"], ND), cmid_pf=c["Cmid_pf"],
                             se_log10=c["log10_Cmid_se"], rmse=c["rmse"], first_pf=round(min(kept), ND),
                             last_pf=round(max(kept), ND)))
write("cmid_curves.csv", ["model", "pair", "level", "seed", "pass16", "r_ref", "cmid_pf", "se_log10", "rmse", "first_pf",
                          "last_pf"], rows)
