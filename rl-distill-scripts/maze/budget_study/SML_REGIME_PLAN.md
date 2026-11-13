# {S, M, L} × six-start regime study: plan (2026-11-06)

**Question** (2026-11-05 meeting). When the detour (RL a teacher, then distil it into the student) beats direct RL on the
student, is that the teacher's better starting point (pre-training → starting capability) or its size (optimization →
size)? At each of three compute budgets, fit one regression over all pairs and compare the two weights across budgets.

## Design

- **Models.**
  - S = 3.9M and L = 13.3M exist (same pretraining data order).
  - M ≈ 7.2M is to be trained. It is the geometric mean of S and L, so M/S ≈ L/M ≈ 1.85, and M→S against L→M separates
    the size ratio from the student's own size.
  - Each size has six starts, goal_reached pass@1 ≈ 0.1 / 0.3 / 1 / 3 / 5.5 / 10%, matched across sizes (S and L within
    0.0004 at k = 1024).
- **Pairs.**
  - The student is S or M at start level i. The teacher has the same or a larger size and sits at level i + 1, i or
    i − 1.
  - That gives 80 pairs: 48 with an S student (32 of them S/L only) and 32 with an M student.
  - Priority 1 (optional) is a same-size teacher one level down. It is expected to lose, and it anchors the negative Δcap
    side: 10 pairs.
- **Budgets.** C = 10 / 32 / 128 PFLOPs, the same for every pair.
  - 10 is the smallest budget at which every RL run is past step 30, where the RL-curve fits start. L reaches step 30 at
    6.3 PF ≤ 0.75 C.
  - 32 is where the budget study's Δcap = 0 L→S pairs turn from losing to winning: ≈ 25 PF at 1%, ≈ 33 PF at 0.1%.
  - 128 is the most the existing runs cover without extension, except S@10%.
- **Protocol for each (pair, C).**
  - Teacher: the last saved checkpoint of its RL run with cumulative FLOPs ≤ 0.75 C. If a run plateaued before that, use
    its final checkpoint.
  - Detour: on-policy PG-k1 distillation into the student's base with D = C − C_T (`MAX_TOTAL_TFLOPS`, as
    `run_budget_matrix.sh` runs Arm A).
  - Direct RL: the student's own run at its last saved checkpoint ≤ C.
  - Evaluation: both arms as in the budget study. Dev = TailRL eval1000 at k = 64; test = 1000 fresh mazes at k = 256.
  - Outcome: y(C) = dev(detour) − dev(direct).
  - Why ρ = 0.75, on-policy: over the budget study's 15 budgets it was the best of the six configurations
    (ρ 0.5 / 0.75 / 0.9 × on- / off-policy) in 8, and within 3.0 pt of the best in all of them. Distillation cells differ
    by 1–3 pt between seeds.
- **Analysis.**
  - At each C, fit y ~ Δcap + log(N_T/N_S) + cap_S + log N_S, with Δcap = log10 pass@16 of the teacher base − log10
    pass@16 of the student base. A logistic fit on sign(y) is the check.
  - Report each arm's progress at C from the RL-curve fits as a diagnostic.
- **Checks.**
  - Spot checks run at C = 128 on the 1% students.
    - ρ 0.5 and ρ 0.9 for every (teacher size, student size, level offset) type.
    - Off-policy at ρ 0.75 for the same-level types.
    - If any of these beats ρ 0.75 by more than 3 pt, sweep ρ for that type.
  - Replicates repeat the 1% students' pairs with seed-1 teacher, student and distillation runs.

## Jobs

| Phase | Kind | Pairs | Distillation runs | Distillation PF | ≈ GPU-hours |
|---|---|---|---|---|---|
| 1 (S, L) | main | 32 | 96 | 1,604 | 64 |
| 1 | spot | 6 types | 14 | 534 | 21 |
| 1 | replicate | 6 | 18 | 270 | 11 |
| 2 (with M) | main | 48 | 144 | 2,135 | 85 |
| 2 | spot | 9 types | 21 | 794 | 32 |
| 2 | replicate | 9 | 27 | 393 | 16 |

- **GPU-hours** assume ≈ 25 PF per GPU-hour, the RL throughput on federer (S ≈ 10 s per step at 0.066 PF; L 17–30 s per
  step at 0.21 PF). On-policy distillation is assumed to be similar.
- **Phase 2 M numbers** use D = 0.25 C until M's runs exist.
- **Priority 1** is 5 pairs in each phase: 20 runs and ≈ 335 PF each.

RL work before the distillations:

- **Phase 1.**
  - Extend `p10 S0 s0` from 27 PF to 128 PF, ≈ 1,500 steps and ≈ 100 PF. Its dev still rises 0.03 over its last 20% of
    steps, so it cannot stand in for 32 or 128 PF, either as the direct-RL arm or as a teacher.
  - Rerun the first 40 steps saving every step, ≈ 8 PF each, for `p03 L0 s0`, `p3 L0 s0` and `p5 L0 s0`, plus the
    replicate teachers `p03 L0 s1`, `p1 L0 s1` and `p3 L0 s1`. These runs saved only every 10 steps, so at C = 10 the
    teacher would be step 30 (ρ ≈ 0.63) instead of step 35 (0.75). The alternative is to accept 0.63 for these cells.
  - `p10 L0 s0` plateaus at 63 PF (dev −0.002 over its last 20%), so at C = 128 it is capped there.
- **Phase 2.** Train M, then run its RL:
  1. Pretrain M with `launch_pretrain_matrix.sh` (an M config and the same data order, init seeds as for S and L).
  2. SWA candidates, screening, then selection matched to S and L at the six levels (pass@1 within 0.0004 at k = 1024;
     2048 at 0.1%).
  3. RL at the six levels × 2 seeds to 128 PF: ≈ 1,070 steps at ≈ 0.12 PF per step. Save every 5 steps, and every step
     for the first 40.
  4. Rerun the planner, which fills in M's teacher steps and Δcap, then run the distillations.
  - M's RL is ≈ 1,540 PF, ≈ 61 GPU-hours, plus pretraining.

## Phase-1 pairs

Each cell gives the teacher step / its cumulative PF at C = 10, 32 and 128. Δcap is in log10 pass@16.

| Student | Teacher | N_T/N_S | Δcap | Prio | C = 10 | C = 32 | C = 128 | Note |
|---|---|---|---|---|---|---|---|---|
| S@0.1% | S@0.3% | 1.0 | +0.47 | 0 | 110 / 7.13 | 370 / 23.93 | 1480 / 95.83 |  |
| S@0.1% | S@0.1% | 1.0 | +0.00 | 0 | 110 / 7.45 | 350 / 23.67 | 1435 / 95.89 |  |
| S@0.1% | L@0.3% | 3.41 | +0.45 | 0 | 30 / 6.24 | 110 / 23.0 | 450 / 95.39 | dense-save rerun (C = 10) |
| S@0.1% | L@0.1% | 3.41 | +0.00 | 0 | 35 / 7.3 | 115 / 23.92 | 445 / 95.68 |  |
| S@0.3% | S@1% | 1.0 | +0.47 | 0 | 115 / 7.46 | 370 / 23.97 | 1480 / 95.87 |  |
| S@0.3% | S@0.3% | 1.0 | +0.00 | 0 | 110 / 7.13 | 370 / 23.93 | 1480 / 95.83 |  |
| S@0.3% | S@0.1% | 1.0 | −0.47 | 1 | 110 / 7.45 | 350 / 23.67 | 1435 / 95.89 |  |
| S@0.3% | L@1% | 3.41 | +0.45 | 0 | 35 / 7.27 | 115 / 23.82 | 460 / 95.24 |  |
| S@0.3% | L@0.3% | 3.41 | −0.01 | 0 | 30 / 6.24 | 110 / 23.0 | 450 / 95.39 | dense-save rerun (C = 10) |
| S@0.3% | L@0.1% | 3.41 | −0.46 | 0 | 35 / 7.3 | 115 / 23.92 | 445 / 95.68 |  |
| S@1% | S@3% | 1.0 | +0.39 | 0 | 110 / 7.17 | 360 / 23.44 | 1460 / 95.68 |  |
| S@1% | S@1% | 1.0 | +0.00 | 0 | 115 / 7.46 | 370 / 23.97 | 1480 / 95.87 |  |
| S@1% | S@0.3% | 1.0 | −0.47 | 1 | 110 / 7.13 | 370 / 23.93 | 1480 / 95.83 |  |
| S@1% | L@3% | 3.41 | +0.40 | 0 | 30 / 6.3 | 110 / 23.28 | 440 / 95.58 | dense-save rerun (C = 10) |
| S@1% | L@1% | 3.41 | −0.02 | 0 | 35 / 7.27 | 115 / 23.82 | 460 / 95.24 |  |
| S@1% | L@0.3% | 3.41 | −0.48 | 0 | 30 / 6.24 | 110 / 23.0 | 450 / 95.39 | dense-save rerun (C = 10) |
| S@3% | S@5.5% | 1.0 | +0.18 | 0 | 110 / 7.2 | 360 / 23.62 | 1460 / 95.87 |  |
| S@3% | S@3% | 1.0 | +0.00 | 0 | 110 / 7.17 | 360 / 23.44 | 1460 / 95.68 |  |
| S@3% | S@1% | 1.0 | −0.39 | 1 | 115 / 7.46 | 370 / 23.97 | 1480 / 95.87 |  |
| S@3% | L@5.5% | 3.41 | +0.17 | 0 | 30 / 6.32 | 110 / 23.35 | 440 / 95.73 | dense-save rerun (C = 10) |
| S@3% | L@3% | 3.41 | +0.00 | 0 | 30 / 6.3 | 110 / 23.28 | 440 / 95.58 | dense-save rerun (C = 10) |
| S@3% | L@1% | 3.41 | −0.41 | 0 | 35 / 7.27 | 115 / 23.82 | 460 / 95.24 |  |
| S@5.5% | S@10% | 1.0 | +0.14 | 0 | 110 / 7.3 | 355 / 23.78 | 400 / 26.81 | extend S@10% RL (C = 128) |
| S@5.5% | S@5.5% | 1.0 | +0.00 | 0 | 110 / 7.2 | 360 / 23.62 | 1460 / 95.87 |  |
| S@5.5% | S@3% | 1.0 | −0.18 | 1 | 110 / 7.17 | 360 / 23.44 | 1460 / 95.68 |  |
| S@5.5% | L@10% | 3.41 | +0.14 | 0 | 35 / 7.34 | 110 / 23.03 | 300 / 62.73 | capped at plateau (C = 128) |
| S@5.5% | L@5.5% | 3.41 | −0.01 | 0 | 30 / 6.32 | 110 / 23.35 | 440 / 95.73 | dense-save rerun (C = 10) |
| S@5.5% | L@3% | 3.41 | −0.18 | 0 | 30 / 6.3 | 110 / 23.28 | 440 / 95.58 | dense-save rerun (C = 10) |
| S@10% | S@10% | 1.0 | +0.00 | 0 | 110 / 7.3 | 355 / 23.78 | 400 / 26.81 | extend S@10% RL (C = 128) |
| S@10% | S@5.5% | 1.0 | −0.14 | 1 | 110 / 7.2 | 360 / 23.62 | 1460 / 95.87 |  |
| S@10% | L@10% | 3.41 | +0.00 | 0 | 35 / 7.34 | 110 / 23.03 | 300 / 62.73 | capped at plateau (C = 128) |
| S@10% | L@5.5% | 3.41 | −0.15 | 0 | 30 / 6.32 | 110 / 23.35 | 440 / 95.73 | dense-save rerun (C = 10) |

The direct-RL arm of the S students at C = 10 / 32 / 128 has these dev mean@16 values (preview; the plan evaluates at
k = 64):

| Student | C = 10 | C = 32 | C = 128 |
|---|---|---|---|
| S@0.1% | 0.049 | 0.062 | 0.080 |
| S@0.3% | 0.106 | 0.170 | 0.291 |
| S@1% | 0.405 | 0.485 | 0.587 |
| S@3% | 0.523 | 0.741 | 0.846 |
| S@5.5% | 0.677 | 0.804 | 0.912 |
| S@10% | 0.823 | 0.882 (run ends at 27 PF) | to extend |

Phase 2 adds the same pattern with M:
- M teachers for S students: 16 pairs.
- M and L teachers for M students: 32 pairs.

## Files

- `plan_sml_regimes.py` regenerates the job list. It interpolates FLOPs at a saved step from the dev curves, which have a
  point every 10 steps. It assumes the budget study's seed-0 runs saved every 5 steps and all others every 10. Before
  launching, recompute both from each checkpoint's `flops_ledger.json` in the Hub archive tars.
- `data/sml_regime_plan.csv` has one row per distillation run, 320 in all. Columns: phase, priority, kind, student,
  teacher, size ratio, Δcap, C, ρ, variant, seed, teacher run and step, C_T, ρ_eff, D, note.
- `data/sml_regime_direct_rl.csv` has the direct-RL checkpoint per student and budget, 42 rows, with its dev mean@16 as a
  preview.

## Open decisions

- **M's size.** 7.2M is proposed.
- **Priority-1 pairs.** Keep them or not (40 runs, ≈ 670 PF over both phases).
- **C_large: 128 or 256 PF.** 256 shows the hard-start size effect more clearly: the 0.1% Δcap = 0 pair goes from +15 pt
  at 133 PF to +32 pt at 266 PF. The cost is extending most S runs and running M to 256 PF.
- **Dense-save reruns at C = 10.** Run them for the six L teacher runs, or accept ρ ≈ 0.63 there.
