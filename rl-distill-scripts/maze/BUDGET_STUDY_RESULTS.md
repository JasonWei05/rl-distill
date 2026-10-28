# Budget study results (text-maze RL vs distillation), 2026-10-27

Three matched S/L pairs (S = 3.9M, L = 13.3M, same pretraining data order) at goal_reached pass@1 ≈ 10% / 1% / 0.1%;
per pair five total-FLOPs budgets B1..B5 = B5/16 .. B5 (B5 = S0 RL saturation, or the whole run when it never saturates);
Arm B = S0 direct RL (GRPO, 256 × 16, lr 1e-4); Arm A = L0 RL for a fraction ρ ∈ {0.5, 0.75, 0.9} of the budget, then
distillation into S0 with the rest, on-policy (PG-k1) and off-policy (exact forward KL on teacher samples, one epoch).
Dev = TailRL eval1000 at k = 64; test = 1000 fresh mazes at k = 256. All numbers: `~/tailrl_maze_runs/budget/<pair>/results_*.json`.

## Across the three starting levels

| Start | B1 | B3 | B5 | Verdict (dev; test agrees) |
| --- | --- | --- | --- | --- |
| 10% (B5 = 19.4 PF) | 0.525 vs 0.294 | 0.750 vs 0.529 | 0.862 vs 0.767 | direct RL by 7–23 pt at every budget |
| 1% (B5 = 139 PF) | 0.401 vs 0.298 | 0.483 vs 0.521 | 0.653 vs 0.729 | crossover at B3; detour +8 (dev) / +4 (test) at B5 |
| 0.1% (B5 = 266 PF) | 0.071 vs 0.063 | 0.083 vs 0.132 | 0.120 vs 0.441 | crossover at B3; detour +32 pt at B5 |

(Arm B vs the best detour cell, goal_reached pass@1, dev k = 64.)

1. The deciding quantity is how much of the budget the small model's own RL can use. From 10% S0 learns about as much
   per step as L0 while an L0 step costs 3.2× more, so the affordable teacher is always behind S0. From 1% L0 pulls ahead
   per FLOP from 17 PFLOPs on. From 0.1% S0's RL barely moves (0.08 → 0.13 over 266 PFLOPs; same in a seed-1 replicate
   and a 64-rollout variant) while L0 takes off (0.48 at 1085 steps), so nearly everything the detour delivers comes from
   the teacher.
2. Distillation recovers the teacher when the student starts strong relative to it (10%: on-policy matches its teacher
   at ρ = 0.5) and most of it when the gap is large (0.1%: 85–93% at ρ = 0.9); the lost fraction grows with the gap.
3. The best split follows: half the budget to the teacher at 10% (distillation compute is the binding constraint), nine
   tenths at 0.1% (every teacher RL step is worth more than more distillation). ρ = 0.25 never helps.
4. On-policy distillation wins on pass@1 when it gets enough updates (10% pair) and collapses when it does not (1%,
   ρ = 0.9 at B5: 4,500 on-policy steps against a 0.78 teacher end at 0.56, one off-policy epoch at 0.69). Off-policy is
   never far behind on pass@1 and keeps the widest coverage.
5. Coverage: off-policy students keep pass@64 20–45 pt above S0 RL at every budget of every pair; RL from a weak start
   narrows the tail without raising pass@1 (0.1%: S0 RL pass@64 0.15 → 0.11 while pass@1 stays near 0.08).

## Per-pair tables (dev k = 64; cells = teacher / on-policy student / off-policy student)

### 10% pair (B = 1.2 / 2.4 / 4.8 / 9.7 / 19.4 PFLOPs)
| Budget | Arm B | ρ 0.5 | ρ 0.75 | ρ 0.9 |
| --- | --- | --- | --- | --- |
| B1 | 0.525 | 0.266 / 0.242 / 0.147 | 0.344 / 0.294 / 0.184 | = ρ 0.75 (same teacher ckpt) |
| B2 | 0.590 | 0.344 / 0.329 / 0.235 | 0.441 / 0.419 / 0.282 | 0.491 / 0.409 / 0.281 |
| B3 | 0.750 | 0.491 / 0.480 / 0.392 | 0.592 / 0.529 / 0.439 | 0.641 / 0.513 / 0.413 |
| B4 | 0.820 | 0.656 / 0.631 / 0.582 | 0.718 / 0.661 / 0.602 | 0.742 / 0.654 / 0.549 |
| B5 | 0.862 | 0.762 / 0.767 / 0.721 | 0.784 / 0.740 / 0.681 | 0.812 / 0.685 / 0.643 |
Test (k = 256): Arm B 0.528 / 0.585 / 0.746 / 0.845 / 0.887; best detour 0.302 / 0.435 / 0.536 / 0.676 / 0.789.
Seed-1 replicates: S0 RL within 1–4 pt; all 28 distillation cells within 1–3 pt. ρ = 0.25 extension: student tracks its
weaker teacher (B5: 0.653 / 0.655 / 0.623 test).

### 1% pair (B = 8.7 / 17.4 / 34.8 / 69.6 / 139 PFLOPs; S0 saturates at step 2150, dev 0.68)
| Budget | Arm B | ρ 0.5 | ρ 0.75 | ρ 0.9 |
| --- | --- | --- | --- | --- |
| B1 | 0.401 | 0.271 / 0.282 / 0.243 | 0.330 / 0.292 / 0.277 | 0.362 / 0.298 / 0.255 |
| B2 | 0.414 | 0.377 / 0.355 / 0.340 | 0.402 / 0.377 / 0.356 | 0.442 / 0.327 / 0.323 |
| B3 | 0.483 | 0.487 / 0.456 / 0.461 | 0.569 / 0.521 / 0.514 | 0.555 / 0.452 / 0.437 |
| B4 | 0.600 | 0.540 / 0.563 / 0.521 | 0.652 / 0.595 / 0.613 | 0.680 / 0.547 / 0.557 |
| B5 | 0.653 | 0.699 / 0.716 / 0.701 | 0.750 / 0.699 / 0.729 | 0.778 / 0.561 / 0.693 |
Test: Arm B 0.411 / 0.452 / 0.529 / 0.605 / 0.692; best detour 0.303 / 0.401 / 0.534 / 0.606 / 0.734. Seed-1 S0 RL:
0.391 / 0.424 / 0.485 / 0.611 / 0.663. Off-policy pass@64 0.80–0.94 vs S0 RL 0.55–0.73.

### 0.1% pair (B = 16.6 / 33 / 66.5 / 133 / 266 PFLOPs; B5 = whole 4000-step S0 run, never saturated)
| Budget | Arm B | ρ 0.5 | ρ 0.75 | ρ 0.9 |
| --- | --- | --- | --- | --- |
| B1 | 0.071 | 0.060 / 0.062 / 0.060 | 0.067 / 0.061 / 0.063 | 0.061 / 0.060 / 0.049 |
| B2 | 0.088 | 0.078 / 0.083 / 0.077 | 0.090 / 0.089 / 0.089 | 0.103 / 0.090 / 0.091 |
| B3 | 0.083 | 0.100 / 0.111 / 0.101 | 0.125 / 0.132 / 0.121 | 0.140 / 0.123 / 0.111 |
| B4 | 0.086 | 0.130 / 0.135 / 0.126 | 0.225 / 0.234 / 0.202 | 0.254 / 0.167 / 0.214 |
| B5 | 0.120 | 0.319 / 0.352 / 0.321 | 0.442 / 0.424 / 0.432 | 0.476 / 0.404 / 0.441 |
Test: Arm B 0.067 / 0.085 / 0.077 / 0.086 / 0.144; best detour 0.060 / 0.092 / 0.116 / 0.215 / 0.431. Seed-1 S0 RL:
0.059 / 0.084 / 0.085 / 0.076 / 0.160. Off-policy pass@64 0.49–0.62 at B4–B5 vs S0 RL 0.11–0.15.

Caveat: the largest off-policy cells of the 1% and 0.1% pairs asked for more teacher samples than the 1.3M training
mazes, so they ran 1.4–5 epochs over 1.3M unique traces within their budget instead of one epoch over more.
