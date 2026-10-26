# Progress Log

> Gemma 4 E2B/E4B distillation-vs.-RL planning, current artifact status, and the production runbook
> now live in [`GEMMA4_DISTILL_VS_RL_EXPERIMENTS.md`](GEMMA4_DISTILL_VS_RL_EXPERIMENTS.md). That
> canonical document supersedes older Gemma 4 status statements in this chronological log.

> All Qwen3 math-suite scores (base / RL / off-policy / on-policy distillation, 1.7B and 4B students) in one place:
> [`QWEN3_MATH_SUITE_RESULTS.md`](QWEN3_MATH_SUITE_RESULTS.md), regenerated from S3 by `summarize_qwen3_math_suite.py`.

A running log of experiments/work in this repo. Newest entries on top. Each entry records the goal,
what was run (config + exact scripts/data), results, and status, so work is resumable and auditable.

---

## 2026-10-23 — FLOP-matched sweep: on-policy distillation from intermediate 4B medium RL checkpoints into the 1.7B BASE LAUNCHED

Goal (10-22 meeting): FLOP-vs-performance curves for distillation vs RL at 1.7B on medium. Direct-RL curve = the existing 1.7B medium
RL run; distillation points = 4B medium RL checkpoint at step k (teacher) -> on-policy distillation into the untrained 1.7B base, with
x = 4B RL FLOPs to step k + distillation FLOPs (essential FLOPs: no checkpoint recompute, padding, old-log-prob recompute, LM head on
prompts, validation; see the FLOP notes in the 10-23 discussion).
- Teachers: qwen3-4b-base-rl-full-checkpoints/qwen30b-medium-es5/global_step_{20,50,100,150}/actor/huggingface (val 30.5 / 43.2 / 51.2 / ~57).
- Student: Qwen3-1.7B-Base @ea980cb0 (no off-policy init). Same PG-k1 recipe as the 10-20 batch (128 x 1, one update / step, lr 2e-6,
  20 warmup, advantage clip [-5, 5], IcePop [0.5, 2]); 250 steps, NO early stopping, validation every 5 steps, permanent checkpoint
  (model + Adam + HF) every 10, rolling every 5; 2 GPUs; borrowing false, priority high. Code = HEAD (includes the distillation
  global_batch_info fix 07e9460f, unlike the pinned pgk1-v1 batch).
- SPEC=q4bm-s<STEP>-to-q1p7b-base, RUN_TAG=pgk1-base-v1 (launcher f1112b4d). Jobs: s20 job_dbdpc08cdkag08afn8f0, s50
  job_dbdpc08cdkag08afn8eg, s100 job_dbdpc08cdkag07kobajg, s150 job_dbdpc0degg5g08bj72bg; W&B q3-opd-<SPEC>-pgk1-base-v1-s42.
- Teacher steps 200 and 250 (checkpoints verified on S3) start automatically once 2 of the first 4 runs complete:
  `scale_train/run_qwen3_opd_sweep_followup.sh` in tmux `q3-opd-sweep-followup` (b3f3f7b8), same settings.
- 20:59Z (student steps 119/114/89/69): val ~18 / ~24 / ~28 / ~32 for teacher steps 20 / 50 / 100 / 150; k1 0.125/0.146/0.151/0.197 at
  step 1 -> 0.057/0.059/0.070/0.079; no divergence. Stronger teacher checkpoint -> stronger student.
- 2026-10-23 ~20:16Z -> 2026-10-24 ~07:00Z: ScaleTrain / k8s API outage (status polls `status_unavailable`, every pod-log stream
  ended 22:28Z with `http2: server sent GOAWAY`, kubectl hung). Jobs kept training; progress was tracked from S3 checkpoint pointers
  (rolling + permanent). The follow-up trigger was made to accept the S3 `run_complete.json` receipt (17ac6582) and launched teacher
  steps 200 / 250 at 01:24Z (jobs job_dbe10etegg5g08bj73h0 / job_dbe10eocdkag07kobbag, submitted 01:56Z; code tarball verified).
- Completed (supervisor-verified receipts, attempt 1): 4B easy RL, 1.7B easy RL (600 steps each), OPD teacher steps 20, 50, 100.
  Teacher-200 / 250 students run ~10-14 steps/h (long responses + validation every 5 steps).
- Best val (W&B) vs teacher step: s20 18.7@50, s50 25.5@225, s100 30.2@210, s150 34.0@180, s200 34.9@175 / s250 35.9@180 (both
  still running at ~180); 1.7B direct RL 33.8@520; two-stage (off-policy 1000 + OPD from the 4B @510 teacher) 38.2@120.
- 2026-10-24: anchor run with the best teacher, 4B medium RL @510 -> 1.7B base, same settings (job_dbecqqocdkag08afna50), to test
  whether the off-policy stage adds anything beyond speed at a fixed teacher.
  DONE 2026-10-25 06:57Z (250 steps): best **37.3 @210** (34.9 by step 120; final 36.3; flat ~35.5-37 over 180-250) vs two-stage
  38.2 @120 and 1.7B RL 34.7 @680 -> the off-policy stage is worth ~1 pt at this teacher (and ~2x faster in OPD steps).
- 2026-10-24: same OPD-from-base recipe with the best 4B RL teacher of the other bands: easy (4B easy RL @500, val 89.1;
  job_dbed39gcdkag08afna5g) and hard (4B hard RL @560, val 41.2; job_dbed39legg5g08bj74p0); launcher tag q4be added (5cfea86f).
  Compare against the two-stage (off-policy + OPD) students and direct 1.7B RL per band.
- TODO: best_hf of the finished OPD runs is the final step (no early stopping); re-publish the best-val step from W&B (the pod
  logs miss 22:28Z-07:00Z) before any math suite.
- 2026-10-24: FLOP ledger + plot (`analysis/qwen3_flops_matched_opd_vs_rl.py`, `figures/qwen3_flops_matched_opd_vs_rl_medium.{png,json}`).
  Accounting: essential FLOPs incl. attention scores (2*L*n_q*d_h*S^2 per forward); RL trains only mixed-correctness groups (all-0 /
  all-1 groups still pay generation); per-sample lengths from the per-step 100-trace W&B tables, rescaled to the logged batch means.
  Distill point: x = 4B RL to step k + OPD to its peak step, y = peak val in OPD steps 1-200; faint line = the whole OPD val
  curve offset by the teacher FLOPs (4B@510 dropped from the plot). Attention is only 5-12% of FLOPs (~1-2k-token
  responses); the zero-advantage exclusion cuts RL by 20% (1.7B: 9.4 -> 7.5 EFLOP over 600 steps).
  | 4B teacher step | 20 | 50 | 100 | 150 | 200 | 250 |
  |---|---|---|---|---|---|---|
  | EFLOP (teacher + OPD to peak) | 0.7+0.16 | 1.7+0.41 | 3.5+0.58 | 5.8+1.16 | 8.8+1.35 | 11.8+1.38 |
  | peak val (OPD step) | 18.7 (50) | 25.1 (120) | 30.0 (135) | 34.0 (180) | 34.9 (175) | 35.9 (180) |
  | 1.7B RL running best at same EFLOP | 20.2 | 26.8 | 30.9 | 33.8 | 33.8 | 33.8 |
  2026-10-25 01Z refresh (1.7B RL continuation at step 720 = 9.2 EFLOP): RL new best 34.7 @680 (8.6 EFLOP); 4B@200 (34.9, 10.1
  EFLOP) and 4B@250 (35.9, 13.2 EFLOP) are now past the end of the RL curve -- compare again when the run reaches 1000.
  2026-10-25 03:23Z: 1.7B medium continuation EARLY-STOPPED @760 (5 misses vs best **34.7 @680**, +0.9 over the first run; best_hf =
  step 680; first-run step-520 copy in archive_first_run/). Final RL curve = 9.8 EFLOP, plateau drawn dashed past it. Final matched
  comparison: RL leads by 0.9-1.7 below ~6 EFLOP, tie at 4B@150 (34.0 vs 33.8), OPD +0.2 (4B@200) / +1.2 (4B@250) over the RL plateau.
  1.7B RL best 33.8 @520 = 6.3 EFLOP. Below ~6 EFLOP direct RL is ahead; above it, teacher+OPD passes the RL plateau (+0.2 to +2.1).
- 2026-10-24 20:30Z: 1.7B medium RL **continued 600 -> 1000** (user request): it stopped on the 600-step cap, not early stopping
  (2 misses carried; val slope still ~+1 pt/100 steps over 300-600, 75% mixed groups), so the plot's flat RL line past 6.3 EFLOP was an
  extrapolation. Exact resume from global_step_600 (Adam + data cursor + ES state), same S3 prefixes / W&B id; first-run markers +
  a copy of best_hf (step 520, used by the math suite) in `<ckpt>/archive_first_run/{run_complete.json,run_outcome.json,
  best_hf_REMOTE_COMPLETE.json,best_hf_step520/}`. `QWEN_SIZE=1.7b BAND=medium TOTAL_TRAINING_STEPS=1000 SUP_TAG=-to1000
  ALLOW_BORROWING=false MAX_RELAUNCHES=200 FAILURE_BACKOFF_SECONDS=3600 MAX_FAILURE_BACKOFF_SECONDS=3600 bash
  start_qwen3_30b_a3b_medium_rl_supervisor.sh`; job_dbehaltegg5g08bj74tg (priority high, 2 GPUs); state
  `.scale_train_supervisors/q3-1p7b-rl-med-20261024-to1000`. Re-run the FLOP script when it finishes.
- 2026-10-24 ~21:00Z: plateau audit of the 9 Qwen RL runs (1.7B/4B/8B x easy/medium/hard). Plateaued / declining: 1.7B easy, 4B easy,
  4B medium (ES @590), 4B hard (peak 41.2 @560, then 36.4/38.7/36.2), 8B easy. Stopped while improving: 8B medium (ES @330, best 64.7
  @270) and 8B hard (ES @220, best 36.9 @170, 89% mixed groups) -- 8B led 4B at every common step but ended below 4B's final best
  (65.8 / 41.2) because a 5-eval dip inside the 1-pt tolerance tripped patience 5. 1.7B hard: slow climb (+0.5-0.8 pt/100 steps).
  Continued (user request; borrowing false, priority high, exact resume, same S3 prefixes / W&B ids; first-run markers + best_hf copy in
  `<ckpt>/archive_first_run/`): 8B medium 330 -> 500 (job_dbehjq8cdkag07kobcag) and 8B hard 220 -> 500 (job_dbehjq8cdkag07kobcb0) with
  the miss streak reset once via the new `EARLY_STOPPING_RESET_PATIENCE_AT_STEP=330|220` (fdfa6053; keeps the all-time best, patience 5
  afterwards); 1.7B hard 600 -> 1000 (job_dbehjqdegg5g08bj74u0, 0 misses carried, no reset). Supervisors
  `.scale_train_supervisors/q3-8b-rl-{med,hard}-20261024-ext`, `q3-1p7b-rl-hard-20261024-to1000`.
- 2026-10-25 03:51Z: **dense-ladder OPD-from-base grid** (user decision: Qwen sizes 1.7B / 4B / 8B only, bands easy / medium / hard;
  9 teacher->student pairs = 4B->1.7B, 8B->1.7B, 8B->4B; 4B->1.7B easy/medium/hard already done / running). Same recipe as the
  from-base sweep: PG-k1, 128 x 1, lr 2e-6, 250 steps, NO early stopping, val every 5, permanent ckpt every 10; 2 GPUs for a 1.7B
  student, 4 for a 4B student; borrowing false, priority high; RUN_TAG=pgk1-base-v1; teacher pinned to its RL best step.
  Launcher gained 8B teachers (dd5396c9: q8be/q8bm/q8bh, vLLM TP1, 0.32 util). Launched now: 8B easy @460 -> 1.7B base
  (job_dbenpjocdkag08afnb40) and -> 4B base (job_dbenpjocdkag07kobctg). Medium / hard start automatically when the 8B RL
  continuations finish (`scale_train/run_qwen3_opd_8b_teacher_followup.sh`, tmux q3-opd-8b-followup, log
  `.scale_train_supervisors/q3-opd-8b-followup.log`): it waits for the fresh run_complete.json, takes best_step from run_outcome.json
  once its final_step matches the receipt, then launches q8b<band>-s<best>-to-{q1p7b,q4b}-base.
  2026-10-25 09:25Z: 8B easy -> 1.7B base DONE: best 73.2 @240 (final 71.3) ~= 4B easy teacher's 73.9 and above 1.7B RL easy 72.6.
  8B easy -> 4B base DONE 10:03Z: best 88.3 @240 (87.8 by step 70, ~87 plateau after) vs 4B RL easy 89.1 @500 -> OPD -0.8.
  2026-10-25 10:25Z: 8B medium RL continuation DONE at the 500 cap: best **68.3 @480** (was 64.7 @270; 4B medium 65.8). Follow-up
  launched 8B medium @480 -> 1.7B base (job_dbetjalegg5g07kc3jqg) and -> 4B base (job_dbetjalegg5g08bj76b0) at 10:27Z.
  2026-10-25 11:41Z: 1.7B hard RL continuation DONE at the 1000 cap: best **14.1 @910** (was 12.6 @550; +1.5 over 400 steps).
  2026-10-25 15:50Z: 4B hard @560 -> 1.7B base OPD DONE: best **19.5 @215** (final 19.1) = two-stage 19.7 @110 (-0.2), vs 1.7B RL hard 14.1.
  2026-10-25 16:04Z: 8B hard RL continuation EARLY-STOPPED @390: best **40.8 @310** (was 36.9 @170; 4B hard 41.2). Follow-up launched
  8B hard @310 -> 1.7B base (job_dbf2jk0cdkag07kobe0g) and -> 4B base (job_dbf2jk5egg5g08bj7730) at 16:09Z; all 9 grid pairs now done or running.
- 2026-10-25 ~17:30Z: **continuations of the non-saturated finished runs** (user request; borrowing false, priority high, exact resume,
  same S3 prefixes / W&B ids; markers + best_hf copy archived): 8B medium RL 500 -> 800 (job_dbf2qmtegg5g07kc3kb0, normal ES: patience 5,
  1-pt tolerance; its two students keep the step-480 teacher); 4B -> 1.7B best-teacher OPD easy / medium / hard 250 -> 500
  (job_dbf2qmtegg5g08bj773g / job_dbf2qmocdkag08afncl0 / job_dbf2qmocdkag07kobe20) with early stopping ON = no +0.5 pt gain in 100 steps
  (patience 10, min_delta 0.005, val every 10 so every best has an HF export). Their checkpoints predate ES state, so the history is
  seeded from W&B via EARLY_STOPPING_LEGACY_STATE_B64 (`scale_train/build_es_migration_from_wandb.py`, 8e56b858; step-10 evals only):
  easy best 73.5 @240 (1 miss), medium 36.9 @190 (6 misses), hard 19.1 @250 (0 misses).

## 2026-10-25 — Pre-training vs RL ablation for the 4B -> 1.7B distillation gain (LAUNCHED stage 1)

Question: 4B-RL -> 1.7B-base OPD beats 1.7B RL (+1.3 / +2.6 / +5.4 easy / medium / hard). Is that from the 4B's better
pre-training behavior (P) or its larger capacity / better RL optimization (C)? 2x2 over teacher {P source} x {capacity}, every
teacher RL-trained on the band then on-policy distilled into the 1.7B base; baseline = direct 1.7B RL:
- (P 1.7B, C 1.7B) control: 1.7B RL -> 1.7B base (same-size self-distillation)
- (P 4B,   C 4B)   main result: 4B RL -> 1.7B base
- (P 1.7B, C 4B)   Exp 1: 4B' = 4B base on-policy distilled FROM the 1.7B base, then RL, then -> 1.7B base
- (P 4B,   C 1.7B) Exp 2: 1.7B' = 1.7B base on-policy distilled FROM the 4B base, then RL, then -> 1.7B base
All distillation = on-policy (PG-k1, user decision 10-25). Budgets: pre-distill 250 steps fixed (k1 closes 95% by step 37-111 in
every finished OPD run), per band; ablation RL cap 1000 with patience 10 evals + 1-pt tolerance (replay: never stops any of the 9
main RL curves before its best; patience 5 cut 8B med/hard 3.6-3.9 pts early); final OPD cap 500 with "+0.5 in 100 steps" ES.
**Stage 1 launched 2026-10-25 ~17:45Z** (launcher 2245bdd2: base teachers qb1p7b*/qb4b* from pinned Hub revisions, 1.7B RL
teachers q1p7b*), borrowing false, priority high:
- pre-distill (RUN_TAG pgk1-pre-v1, 250 steps, val + HF export every 10, no ES): qb1p7b{e,m,h}-to-q4b-base (4 GPUs) and
  qb4b{e,m,h}-to-q1p7b-base (2 GPUs).
- control (RUN_TAG pgk1-base-v1, cap 500, val every 10, ES patience 10 / min_delta 0.005): q1p7be-s470 / q1p7bm-s680 /
  q1p7bh-s910 -to-q1p7b-base (teacher = 1.7B RL best step).
**USER GATE: after pre-distillation, STOP and let the user inspect** (val by step vs the source base, k1, entropy, length) and pick
the checkpoint per band/direction; only then launch the ablation RL (needs an S3-init option in the Qwen RL launcher) and the
final OPD stage.
- ~18:30Z (user request: 2/3 of the queued borrowing-false jobs -> borrowing true, auto-resume on preemption): queued jobs cancelled by
  ID and relaunched by fresh supervisors (SUP_TAG -borrow, ALLOW_BORROWING=true; PREEMPTED -> immediate relaunch from the newest
  S3 checkpoint, MAX_RELAUNCHES=200): qb1p7b{e,m,h}-to-q4b-base (job_dbf49p8cdkag08afncug / job_dbf49pdegg5g07kc3kfg /
  job_dbf49p8cdkag08afncv0) and controls q1p7bm-s680 / q1p7bh-s910 (job_dbf49qocdkag08afncvg / job_dbf4a1ocdkag08afnd00) -- all
  IN_PROGRESS within minutes. qb4b{e,m,h}-to-q1p7b-base stay queued on borrowing false.
- 2026-10-26: **on-policy pre-distillation abandoned** (user): reverse KL to a high-entropy base teacher collapses the student's entropy
  (own-sample NLL/token ~0.9 -> ~0.5 in both directions within 50 steps, k1 plateau 0.05-0.07, best@16 well below the source base; on
  medium/hard 4B' and 1.7B' converge to the SAME mid-level accuracy ~14 / ~5.5) -> not a copy of the source base. Report:
  figures/qwen3_predistill_inspection.{md,png} (584619d2). The 2 unfinished runs were cancelled by ID; supervisors stopped.
- 2026-10-26: **off-policy forward-KL pre-distillation instead** (user): ONE run per direction over the union of the easy + medium + hard
  train splits (9000 q x 8 teacher traces = 72k, top-128, temp 1, 8192 tok; val 900 q x 1), batch 128, lr 2e-6 -> 2e-7 (100 warmup),
  1000 steps (= 1.78 epochs; the per-band runs were 3000 x 16 = 2.7 epochs), val KL on 300 teacher val generations every 10, permanent
  checkpoint + HF export every 250. Evals (stage 3): steps 0 (= student base) / 250 / 500 / 750 / 1000 on each band's 300-q val at n=32,
  pass@k for all k <= 32 + own-sample NLL (entropy estimate). Specs q1p7b-base-all -> q4b (4B') and q4b-base-all -> q1p7b (1.7B')
  (eb8b4e12); one 8-GPU borrowing job each (traces -> distill -> evals in one job, every stage resumable), supervisors
  rl-q3-1p7bbase-dist-4b / rl-q3-4bbase-dist-1p7b. USER GATE unchanged: report vs the teacher base, the user picks checkpoints.
- 2026-10-26: reference = the earlier 4B base -> 30B base top-128 forward-KL students (report figures/qwen3_distill_vs_teacher.md,
  34c2abd3): pass@1..32 converge to the teacher's (pass@32 90.3 vs 92.7 medium, 73.7 vs 72.0 hard @1000), fwd KL 0.062 -> 0.039 /
  0.098 -> 0.032, BUT own-sample entropy stays at the 30B base's (0.43-0.46 vs teacher 0.96-0.97). Cause: the loss is the top-128 sum
  only, minimized by the renormalized teacher head with ZERO tail mass (student top-128 mass 0.986 vs teacher 0.970); 46% of the 4B
  base's own-sample NLL (0.52 of 1.12 nats/tok) comes from the 5.3% of sampled tokens outside its top-128. Known bias (Sparse Logit
  Sampling, ACL'25 "ghost token"; TA-OPD; TAD). Fix (user): tail bucket p_tail log(p_tail/q_tail) (b9f594c7, TOPK_KL_TAIL_BUCKET,
  default on for the *-all specs). Jobs moved 2 GPUs -> 4 GPUs (user), then restarted on the tail-bucket code during stage 1 (trace
  shards resume from S3): supervisors *-20261026-dp4b, 4 GPUs, borrowing false.
- 2026-10-26 ~14:00Z: **forward-KL (tail-bucket) pre-distillations DONE** (traces 9000 q x 8, 1000 steps, evals n=32 on each band;
  figures/qwen3_distill_vs_teacher.md section B). Tail bucket worked: student mass on the teacher top-128 stays at / just below the
  teacher's (0.961 vs 0.964; 0.952 vs 0.957), unlike the 30B runs without it (0.986 vs 0.970).
  * 4B' (1.7B base -> 4B base): MATCHES the 1.7B-base teacher in pass@1..32 AND entropy on all bands (e.g. @750 easy 29.4 / pass@32 92.3
    / NLL 0.986 vs teacher 31.2 / 92.7 / 0.953; medium 8.7 / 63.7 / 1.033 vs 9.3 / 67.3 / 0.988; hard 3.9 / 44.7 / 1.049 vs 3.7 / 44.0
    / 1.012); fwd KL (incl. tail) 0.079 -> 0.048. Closest overall: step 750 (500 similar).
  * 1.7B' (4B base -> 1.7B base): copies the style (entropy / length) but NOT the ability -- accuracy stays at (slightly below) the
    1.7B base on every band (medium mean@32 8.7 vs base 9.3 vs teacher 20.6; hard pass@32 39.7 vs 44.0 vs 70.3); fwd KL only
    0.088 -> 0.073. Exp 2 therefore cannot separate pre-training from capacity (1.7B' ~ the 1.7B base). GATE: user picks checkpoints.
- 2026-10-26: user picked **4B' step 1000** (final checkpoint, lowest KL). **Exp 1 RL launched** on easy / medium / hard: init =
  s3://.../qwen3-4b-base-distill-ckpts-v1/q1p7b-base-all-to-q4b-bs128-s1000-lr2e-6/global_step_1000/huggingface (INIT_MODEL_S3_URI,
  e9f1158d), the 4B RL recipe (GRPO n=16, bsz 64 / mini 32, lr 1e-6), cap 1000 steps, early stopping patience 10 evals (100 steps) +
  1-pt tolerance, RUN_TAG p4b1000-es10 (own S3 prefixes / W&B ids q3-4b-<band>-p4b1000-es10-s42-v1), 4 GPUs, borrowing false, priority
  high; supervisors rl-q3-4b-rl-{easy,med,hard} (state *-p4b). Next: distill each 4B'-RL best into the 1.7B base (OPD, cap 500,
  +0.5/100-step ES) vs 1.7B RL. Exp 2 (1.7B') on hold pending the user's call.
- 2026-10-26: **8B RL ~= 4B RL after RL** (bases 57.0 / 27.6 / 12.6 vs 44.5 / 21.0 / 8.6, RL bests 91.3 / 68.3 / 40.8 vs 89.1 / 65.8 /
  41.2). No infra fault (trainer-vLLM log-ppl gap ~0.001). All runs collapse entropy to ~0.13-0.20 by step 50-150, then let it grow
  (no KL / entropy term, clip-higher 0.28); 8B hard is the outlier: entropy jumped 0.20 -> 0.48 at ~step 160-180 (training reward
  stalled ~60 steps, val 36.9 -> 33.9, per-seq ppl spikes up to 4.5k), recovered, then ran away 0.55 -> 0.93 at 340-400 with falling
  training accuracy. 8B easy fully stable; 8B medium no spike but overfits after ~450 (train acc up, val down). **8B hard RERUN**
  (user): same recipe + seed, RUN_TAG rerun-es10 (patience 10, 1-pt tolerance, cap 500), borrowing true, supervisor
  rl-q3-8b-rl-hard (state *-rerun). Recipe ablations (lr 5e-7 / symmetric clip) proposed, not launched.
- 2026-10-25 grid results: 8B medium @480 -> 1.7B base DONE 19:44Z best 33.5 @230 (4B-teacher student 37.3, 1.7B RL 34.7); 8B medium
  @480 -> 4B base DONE 20:39Z best 61.1 @245 (4B RL medium 65.8) -> on medium the 8B teacher is a worse teacher than the 4B for
  1.7B and distillation from base trails RL for the 4B student (like easy, -0.8). Controls (same-size 1.7B RL -> 1.7B base) reach
  their teacher's val within ~60-130 steps with k1 ~0.001-0.004.
  Controls early-stopped (no +0.5 in 100 steps): easy 1.7B RL @470 -> 1.7B base best 72.5 @130 (stop 230; teacher 72.6), hard @910
  best 13.7 @80 (stop 180; teacher 14.1) -> same-size distillation reproduces the RL teacher, no gain beyond it.
  Medium control @680 best 34.5 @90 (stop 190; teacher 34.7). 4B -> 1.7B extensions early-stopped: easy best 74.0 @340 (stop 440;
  was 73.9 @245), medium 37.5 @260 (stop 360; was 37.3 @210; two-stage 38.2) -> the step-250 slope was a tail, +0.1-0.2 only.
  Hard extension: no new best on the step-10 evals (19.1 @250; stop 350; best on any eval stays 19.5 @215) -> 250 steps was enough.
  8B medium RL 500 -> 800 continuation EARLY-STOPPED @610 with no new best (68.3 @480 stands) -> saturated; its two OPD students
  already used the final best teacher (step 480).
  8B hard @310 -> 1.7B base DONE 04:51Z: best 15.3 @225 (4B hard teacher's student 19.5; 1.7B RL hard 14.1) -> the 8B teacher
  is again the WORSE teacher for 1.7B (as on medium: 33.5 vs 37.3). 8B hard -> 4B base at step 210: 34.5 (4B RL hard 41.2).
  8B hard @310 -> 4B base DONE 07:10Z: best 34.5 @215 (4B RL hard 41.2). **Dense-ladder grid COMPLETE** (best val mean@16;
  RL diagonal / OPD-from-base below it):
  | band | RL 1.7B | RL 4B | RL 8B | 4B->1.7B | 8B->1.7B | 8B->4B |
  |---|---|---|---|---|---|---|
  | easy | 72.6 | 89.1 | 91.3 | 74.0 | 73.2 | 88.3 |
  | medium | 34.7 | 65.8 | 68.3 | 37.5 | 33.5 | 61.1 |
  | hard | 14.1 | 41.2 | 40.8 | 19.5 | 15.3 | 34.5 |
  4B -> 1.7B beats 1.7B RL everywhere (+1.4 / +2.8 / +5.4); 8B -> 1.7B ~ RL (+0.6 / -1.2 / +1.2); 8B -> 4B below 4B RL (-0.8 /
  -4.7 / -6.7). The bigger (8B) teacher is the worse teacher for both students.

## 2026-10-23 — Qwen3-8B RL (easy / medium / hard) + 4B / 1.7B RL on easy LAUNCHED (borrowing on, priority high)

From the 10-22 meeting (`distill-vs-rl` notes doc): add a dense 8B to the Qwen ladder (replaces 30B-A3B MoE as the large dense
teacher) and fill in the easy band for the boundary condition. Same DAPO recipe as the other Qwen RL runs (GRPO n=16, 64 prompts /
mini 32, lr 1e-6 + 20 warmup, clip 0.2/0.28, token-mean, 8k + 2k overlong buffer, token TIS 2, val mean@16 every 10 steps).
- Early stopping: 5 validations in a row more than 1 point (0.01) below the best (patience 5, miss tolerance 0.01).
- Checkpoints: permanent every 10 steps (model + Adam + LR/RNG + dataloader + HF export), rolling every 5; supervisors relaunch
  on preemption / failure and the run-file resumes from the newest complete S3 checkpoint.
- Qwen3-8B-Base @49e3418f: 4 GPUs, FSDP2 DP4, no offload (training state resident), vLLM TP1 x4 with 16 GiB KV, 4 GiB weight-sync
  buckets, micro-batch 4 seqs / 8192 padded tokens; max 500 steps. 4B / 1.7B easy: 4 / 2 GPUs, micro-batch doubled (8 / 16384);
  max 600 steps.
- Launcher `scale_train/launch_qwen3_30b_a3b_medium_rl.sh` gained QWEN_SIZE=8b, BAND=easy and MBS / PADDED_TOKENS (f21cdf42).
- Jobs: 8B easy job_dbddma8cdkag08afn7n0, 8B medium job_dbddma8cdkag07koba1g, 8B hard job_dbddmadegg5g08bj71ig, 4B easy
  job_dbddmadegg5g07kc3gq0, 1.7B easy job_dbddmadegg5g08bj71i0; supervisors tmux `rl-q3-{8b-rl-easy,8b-rl-med,8b-rl-hard,4b-rl-easy,1p7b-rl-easy}`.
  W&B q3-{8b,4b,1.7b}-<band>-es5-s42-v1; S3 qwen3-{8b,4b,1.7b}-base-rl{,-full-checkpoints}.

## 2026-10-22 — Mode-covering (forward KL, off-policy) vs mode-seeking (reverse KL, on-policy): entropy + diversity analysis

Scripts: `analysis/qwen3_opd_entropy_from_logs.py` (student per-token entropy = NLL of its own T=1 samples, from the OPD pod logs'
`rollout_corr/training_log_ppl`; steps 1-10 = the off-policy student), `analysis/qwen3_answer_diversity.py` (distinct last-box answers
per question over 16 samples + answer entropy, math-suite samples). Teacher entropy = mean NLL of the teacher's own sampled tokens in
one train shard of its trace bundle (same band's train prompts).

Per-token entropy (nats): teacher | off-policy (FKL) student | after OPD (RKL, steps 111-120)
- 4B m/h/vh -> 1.7B: 0.52/0.58/0.60 | 0.76/0.96/1.05 | 0.47/0.64/0.85
- 30B m/h/vh -> 1.7B: 0.29/0.42/0.59 | 0.54/0.86/0.99 | 0.39/0.63/0.85
- 30B m/h/vh -> 4B: 0.29/0.42/0.59 | 0.36/0.58/0.76 | 0.28/0.41/0.64
FKL students are 1.2-2.3x the teacher's entropy (mode covering); RKL pulls them back toward (twice slightly below) the teacher.
Answer diversity (distinct answers / 16, hard val): off-policy 9.6-11.3 (1.7B), 6.9-8.0 (4B) -> OPD 7.7-9.4, 5.6-6.3 (~ the RL level:
1.7B RL 7.4-9.1, 4B RL 5.2-5.7); pass@16 - pass@1 shrinks on medium / MATH500 (e.g. 30B medium->4B medium 41.2 -> 34.6). Yet pass@16
still rises at 1.7B because pass@1 rises much more; at 4B pass@16 is ~flat (the classic trade-off shows there).

## 2026-10-20 — On-policy distillation (policy-gradient k1) of the 9 off-policy-distilled Qwen students LAUNCHED

Goal: continue each off-policy-distilled student (step 1000) with on-policy distillation from its own RL teacher (RL best_hf) on its
band. Recipe chosen from a literature survey (sampled-token reverse KL as a per-token advantage is what most frontier reports use —
Thinking Machines, MiMo/MOPD, GLM-5, Nemotron-Cascade 2 / Nemotron 3 Ultra, Kimi K3; MOPD 2606.30406 finds it ≈ top-k logit
distillation and more robust): Nemotron-Cascade 2 + MOPD's advantage clip.
- Advantage A_t = sg[log π_teacher(y_t) − log π_train(y_t)] (verl `k1` + `use_policy_gradient=True`, `use_task_rewards=False`), no
  normalisation, clipped to [−5, 5] (`loss_max_clamp=5`); `distillation/k1` and `distillation/k3` both logged.
- IcePop token IS mask [0.5, 2] between trainer and vLLM (`ROLLOUT_IS_THRESHOLD=0.5_2.0`); trainer-recomputed old log-probs.
- 128 prompts × 1 sample per step, one update per step (PPO ratio clip inactive), lr 2e-6, 20-step warmup, constant, token-mean,
  temp 1 / top-p 1 / top-k off, 8k responses, 150 steps, early stopping off.
- Validation / checkpoints = the Qwen RL contract: band val300 ×16 mean@16 before training and every 10 steps; permanent S3 checkpoint
  + HF export every 10, rolling every 5; best_hf at the end.
- vLLM TP 1 for students and 4B teachers; 30B-A3B teacher at TP 2 (util 0.45) with actor offload. 1.7B students 2 GPUs, 4B students 4.
- Scripts: `scale_train/launch_qwen3_onpolicy_distill.sh` (SPEC=<teacher><band>-to-<student>), `start_qwen3_onpolicy_distill_supervisor.sh`,
  `run_qwen3_onpolicy_distill_queue.sh` (≤ 8 GPUs in flight, in order 4B→1.7B ×3, 30B→1.7B ×3, 30B→4B ×3; tmux `q3-opd-queue`);
  run-file gained an S3 teacher source, Qwen-compatible S3 init and `ROLLOUT_IS_THRESHOLD` (e851ecc6, 520adcaa).
- Borrowing false, priority high. Outputs: s3://scale-ml/genai/rl-distill/qwen3-opd{,-full-checkpoints}/<SPEC>-pgk1-v1; W&B q3-opd-<SPEC>-pgk1-v1-s42.
- 2026-10-20 17:29Z q4bm-to-q1p7b (job_dbbq9utegg5g08bj6qug); 17:43–17:44Z q4bh-, q4bvh-, q30bm-to-q1p7b via the queue.
- 2026-10-20 20:47Z user: stop every run at **120 steps**. The four runs launched with 150 are stopped at 120 by
  `scale_train/stop_qwen3_opd_at_step.sh` (waits for the step-120 validation + complete permanent checkpoint, stops the supervisor,
  cancels the job, publishes best_hf = best val step <= 120 + run_outcome.json); new runs default to 120. q4bvh-to-q1p7b stopped at
  step 124 (best val 12.96 @110). q30bh-to-q1p7b had just been submitted with 150 (no steps) -> cancelled and resubmitted with 120.
- Stopped at 120: q4bm-to-q1p7b best val 38.25 @120, q30bm-to-q1p7b 34.98 @110, q4bvh-to-q1p7b 12.96 @110, q4bh-to-q1p7b 19.69 @110
  (1.7B direct RL re-evals: medium 33.1, hard 12.1, very_hard 12.1). W&B lost step 120 for q4bm / q30bm (job cancelled before the
  async uploader flushed) -> backfilled from the pod log (`scale_train/wandb_backfill_step_from_pod_log.py`); the stop script now
  waits for W&B to have the step before cancelling (3a6a36e0).
- Math suite on every OPD best_hf: watcher `scale_train/run_qwen3_opd_math_suite_watcher.sh` (tmux `q3-opd-math-watch`) starts a
  1-GPU SINGLE_PASS suite job (8 sets, borrowing false, priority high) per finished run; registry `suite_models opd`
  (`<spec>-opd120`). 2026-10-21 04:22Z started for q4bm/q4bh/q4bvh/q30bm-to-q1p7b.
- **Math suite, first 4 OPD students (done ~05:0xZ), mean@n avg over the 8 sets** (pass@n avg): 4B medium->1.7B OPD **39.7** (66.2) vs its
  off-policy start 33.7 (62.9); 4B hard->1.7B OPD **39.8** (68.0) vs 32.8 (61.2); 4B very_hard->1.7B OPD 35.9 (63.7) vs 31.3 (62.2);
  30B medium->1.7B OPD 36.9 (64.9) vs 32.1 (62.0). Best direct 1.7B RL: medium 36.0 (63.8). OPD (<=120 steps) beats direct RL on
  both mean@n and pass@n; per set e.g. 4B hard->1.7B OPD: easy 74.7 / medium 37.9 / hard 19.4 / very_hard 14.1 / GSM8K 85.5 /
  MATH500 73.1 vs RL medium 70.7 / 33.1 / 14.6 / 12.1 / 83.1 / 66.2. Caveat: best_hf was picked on the band's own val set.
- 2026-10-21 14:1xZ fixes: (1) the math-suite watcher had died at 04:22Z (`set -e` on `aws s3 ls` of a not-yet-published
  prefix) -> fixed (a6d31615). (2) Runs that finish on their own (q30bh/q30bvh-to-q1p7b, q30bm-to-q4b so far) got best_hf = the
  final step (run-file behaviour with early stopping off, best_score null), unlike the 4 manually stopped runs (best val <= 120).
  `scale_train/publish_qwen3_opd_best_hf.sh` re-publishes the best-val step (cp, not sync — sync kept the newer same-size final
  weights; verified byte-identical to the chosen step); the 3 suites started on step-120 weights were cancelled before writing
  anything and restarted. Best val steps: q30bh-to-q1p7b 19.0 @100, q30bvh-to-q1p7b 11.04 @100, q30bm-to-q4b 61.19 @110.
- All 9 trained (q30bh-to-q4b best val 42.19 @120, q30bvh-to-q4b 24.63 @100). Math suite mean@n avg (pass@n avg), off-policy start -> OPD:
  30B hard->1.7B 31.8 -> **39.2** (63.5 -> 68.1); 30B very_hard->1.7B 28.7 -> 34.3 (59.9 -> 62.5); 30B medium->4B 46.7 -> 50.0
  (73.8 -> 74.8); 30B very_hard->4B 43.4 -> 47.8 (71.3 -> 73.1); 30B hard->4B suite still running. 4B direct RL: medium 53.1 (74.9),
  hard 54.1 (79.4), very_hard 47.9 (72.1). Last suite (19:32Z): **30B hard->4B OPD 54.9 (77.7)** vs its start 50.1 (77.9) and
  4B RL hard 54.1 (79.4): easy 88.8 / medium 65.6 / hard 41.8 / very_hard 30.6 / GSM8K 93.0 / MATH500 85.1 / AIME 18.6, 16.0.
  -> at 4B, OPD from the 30B teacher ~ties the best direct 4B RL on the hard band (+0.8 mean@n, -1.7 pass@n) and trails it by
  ~3 on medium; at 1.7B every OPD student beats same-band RL.

## 2026-10-19 — Qwen math suite on all remaining RL-best / distilled / base models (+ very_hard val) DONE

User request: the full math suite (the Qwen analogue of the Gemma §8 math evals) for every new best-RL and distilled Qwen model. Same
protocol as the 2026-10-09 4B suite below (0-shot ChatML + boxed instruction, temp 1, 8192 tokens, strict last-box grading; n=16, AIME 32)
plus the `very_hard` validation split (300 q, dataset rev 9bec69ca). Sets: easy / medium / hard / very_hard val, GSM8K, MATH500, AIME25/26.
Model registry `scale_train/qwen3_math_suite_models.sh` (b7336bb1): RL best = run_outcome best_step best_hf, distilled = step 1000.
- `SUITE=q30b-4b` (8 GPUs, job_dbb339817jf007n6q20g) CANCELED 15:2xZ at the user's request (only 1.7B / 4B matter; no 30B evals
  finished). Replaced by `SUITE=q4b`, 4 GPUs, job_dbb3a4g17jf007n6q21g: 4B RL very_hard best (@620), 30B very_hard -> 4B s1000, + very_hard
  for the five 2026-10-09 models. 21 tasks.
- Prompt check: rendered prompts are the Qwen 0-shot ChatML (`qwen3_base_native_chat.jinja`) + "Please reason step by step, and put
  your final answer within \boxed{}." (e.g. 135 prompt tokens), not the Gemma 12-shot template.
- `SUITE=q1p7b`, 4 GPUs, job_dbb32vo17jf007n6q200: 1.7B RL best medium/hard/very_hard (@520/550/810), 1.7B base, 30B -> 1.7B and
  4B -> 1.7B s1000 x3 bands. 73 tasks (q30bm-to-q1p7b-s1000 already had 7 sets).
- Both priority high, borrowing false (not on the fallback watcher, per user), supervisors tmux `rl-q3-ms-{q4b,q1p7b}`, state
  `.scale_train_supervisors/q3-ms-*-20261019/`. Results: s3://scale-ml/genai/rl-distill/qwen3-math-suite-v1/<model>/<set>/summary.json.

**DONE 2026-10-19 17:46Z** (q4b 15:22-16:28Z, q1p7b 15:07-17:46Z, 4 GPUs each). Rows include the 2026-10-09 models; avg = plain mean over
the 8 sets; n = 16, AIME 32.

**4B mean@n**
| model | easy | medium | hard | very_hard | gsm8k | math500 | aime2025 | aime2026 | avg |
|---|---|---|---|---|---|---|---|---|---|
| q4b-base | 44.8 | 21.0 | 8.3 | 6.9 | 42.2 | 39.9 | 2.8 | 2.7 | 21.1 |
| q4b-rl-medium-best | 89.1 | 64.5 | 37.4 | 25.5 | 93.3 | 83.7 | 17.3 | 14.0 | 53.1 |
| q4b-rl-hard-best | 87.6 | 65.2 | 39.4 | 29.2 | 92.8 | 84.0 | 19.1 | 15.8 | 54.1 |
| q4b-rl-very_hard-best | 84.3 | 55.5 | 30.6 | 23.8 | 92.5 | 77.7 | 8.9 | 10.0 | 47.9 |
| q30bm-to-q4b-s1000 | 85.2 | 52.1 | 27.1 | 20.6 | 91.8 | 77.4 | 10.7 | 9.1 | 46.7 |
| q30bh-to-q4b-s1000 | 85.2 | 57.3 | 34.4 | 23.2 | 91.6 | 81.4 | 16.1 | 11.2 | 50.1 |
| q30bvh-to-q4b-s1000 | 79.9 | 46.8 | 22.3 | 19.6 | 90.5 | 73.5 | 7.7 | 6.6 | 43.4 |

**1.7B mean@n**
| model | easy | medium | hard | very_hard | gsm8k | math500 | aime2025 | aime2026 | avg |
|---|---|---|---|---|---|---|---|---|---|
| q1p7b-base | 31.4 | 9.6 | 3.8 | 3.3 | 40.3 | 30.1 | 1.0 | 0.5 | 15.0 |
| q1p7b-rl-medium-best | 70.7 | 33.1 | 14.6 | 12.1 | 83.1 | 66.2 | 5.4 | 2.5 | 36.0 |
| q1p7b-rl-hard-best | 66.4 | 28.4 | 12.1 | 11.2 | 80.9 | 62.6 | 2.7 | 3.3 | 33.5 |
| q1p7b-rl-very_hard-best | 62.5 | 27.8 | 10.4 | 12.1 | 80.4 | 60.1 | 2.0 | 2.0 | 32.2 |
| q30bm-to-q1p7b-s1000 | 64.2 | 25.9 | 11.5 | 8.8 | 80.2 | 61.4 | 2.9 | 2.1 | 32.1 |
| q30bh-to-q1p7b-s1000 | 61.8 | 24.9 | 11.3 | 9.0 | 79.1 | 62.3 | 3.8 | 2.4 | 31.8 |
| q30bvh-to-q1p7b-s1000 | 56.8 | 20.9 | 9.2 | 7.4 | 76.9 | 54.7 | 2.2 | 1.5 | 28.7 |
| q4bm-to-q1p7b-s1000 | 67.2 | 27.8 | 12.1 | 9.4 | 82.6 | 64.9 | 3.5 | 2.5 | 33.7 |
| q4bh-to-q1p7b-s1000 | 64.2 | 26.4 | 11.9 | 9.7 | 81.0 | 63.4 | 4.2 | 2.0 | 32.8 |
| q4bvh-to-q1p7b-s1000 | 61.1 | 25.3 | 10.9 | 10.0 | 78.6 | 59.7 | 2.5 | 2.2 | 31.3 |

**4B pass@n**
| model | easy | medium | hard | very_hard | gsm8k | math500 | aime2025 | aime2026 | avg |
|---|---|---|---|---|---|---|---|---|---|
| q4b-base | 97.3 | 83.3 | 52.3 | 50.7 | 97.3 | 90.4 | 30.0 | 16.7 | 64.8 |
| q4b-rl-medium-best | 98.7 | 95.3 | 78.0 | 68.7 | 98.3 | 93.4 | 36.7 | 30.0 | 74.9 |
| q4b-rl-hard-best | 98.3 | 94.7 | 83.7 | 74.7 | 98.4 | 95.4 | 53.3 | 36.7 | 79.4 |
| q4b-rl-very_hard-best | 98.0 | 89.7 | 71.7 | 63.7 | 98.0 | 92.6 | 30.0 | 33.3 | 72.1 |
| q30bm-to-q4b-s1000 | 99.7 | 93.3 | 73.7 | 67.7 | 98.2 | 94.2 | 40.0 | 23.3 | 73.8 |
| q30bh-to-q4b-s1000 | 99.0 | 96.7 | 86.7 | 73.0 | 98.6 | 95.8 | 43.3 | 30.0 | 77.9 |
| q30bvh-to-q4b-s1000 | 99.7 | 90.0 | 72.3 | 62.0 | 97.9 | 92.2 | 23.3 | 33.3 | 71.3 |

**1.7B pass@n**
| model | easy | medium | hard | very_hard | gsm8k | math500 | aime2025 | aime2026 | avg |
|---|---|---|---|---|---|---|---|---|---|
| q1p7b-base | 85.7 | 55.7 | 32.3 | 30.3 | 94.6 | 81.0 | 13.3 | 16.7 | 51.2 |
| q1p7b-rl-medium-best | 93.7 | 79.7 | 54.0 | 50.0 | 97.0 | 89.4 | 33.3 | 13.3 | 63.8 |
| q1p7b-rl-hard-best | 94.7 | 75.0 | 51.0 | 48.7 | 97.0 | 88.4 | 23.3 | 26.7 | 63.1 |
| q1p7b-rl-very_hard-best | 89.7 | 67.3 | 40.0 | 42.7 | 96.6 | 84.2 | 20.0 | 16.7 | 57.1 |
| q30bm-to-q1p7b-s1000 | 95.0 | 75.7 | 49.3 | 46.7 | 96.9 | 89.4 | 20.0 | 23.3 | 62.0 |
| q30bh-to-q1p7b-s1000 | 95.3 | 77.7 | 52.7 | 49.3 | 97.1 | 89.4 | 26.7 | 20.0 | 63.5 |
| q30bvh-to-q1p7b-s1000 | 92.0 | 69.0 | 48.3 | 40.0 | 97.0 | 86.6 | 23.3 | 23.3 | 59.9 |
| q4bm-to-q1p7b-s1000 | 95.3 | 74.7 | 49.7 | 44.0 | 96.7 | 89.2 | 30.0 | 23.3 | 62.9 |
| q4bh-to-q1p7b-s1000 | 93.7 | 74.7 | 54.0 | 47.7 | 97.0 | 89.2 | 20.0 | 13.3 | 61.2 |
| q4bvh-to-q1p7b-s1000 | 92.0 | 72.3 | 49.0 | 45.7 | 96.9 | 88.4 | 23.3 | 30.0 | 62.2 |

Direct RL wins mean@n at both sizes (4B: RL hard 54.1 vs best distill 30B-hard->4B 50.1; 1.7B: RL medium 36.0 vs 4B-medium->1.7B 33.7);
distillation ~matches RL on pass@n (4B 77.9 vs 79.4; 1.7B 63.5 vs 63.8). very_hard training (RL or teacher) is worst at both sizes,
even on very_hard val. For 1.7B the 4B teacher beats the 30B teacher. Vs Gemma (shared sets only; Gemma 12-shot, Qwen 0-shot):
MATH500 mean@16 base Gemma E4B 10.9 vs Qwen 4B 39.9 / 1.7B 30.1 (Gemma 26B base 27.2); best trained E4B 38.2 vs Qwen 4B 84.0 / 1.7B 66.2.

## 2026-10-19 — Gemma-4-26B-PT vs Qwen3-30B-A3B-Base difficulty bands: overlap analysis

Both band datasets come from the same 38,796-question DeepScaleR `dedup_v2` pool graded at n=8 with the same strict-boxed grader
(Gemma bands: `JWei05/DeepScaleR-Easy-Medium-Hard-Gemma-26B-PT-10k`; Qwen: `...-Qwen3-30B-A3B-Base-10k` incl. `very_hard`).
Script: `data/compare_gemma_qwen_band_overlap.py`.

- Qwen solves far more: mean pass rate 0.339 vs Gemma 0.104; Qwen > Gemma on 68% of questions, Gemma > Qwen on 6%; Spearman 0.48.
- Released 3,300-q sets, shared questions (Jaccard %), rows Gemma / cols Qwen easy, medium, hard, very_hard:
  easy 723 (12.3) / 87 / 16 / 14; medium 514 (8.4) / 250 (3.9) / 120 / 112; hard 357 (5.7) / 311 (4.9) / 375 (6.0) / 242 (3.8).
  Largest overlaps: G-easy∩Q-easy, then G-medium∩Q-easy, G-hard∩Q-hard, G-hard∩Q-easy. Released sets are random subsamples of each
  pool band, so the full-pool crosstab is the cleaner comparison.
- Full pool, where the other model places each band (row %): Gemma easy (3,314) -> Qwen easy 91%; Gemma medium (3,498) -> Qwen easy
  68%, medium 19%; Gemma hard (5,342) -> Qwen easy 46%, medium 27%, hard 14%, 0/8 13%. Reverse: Qwen hard (6,440) is 84% Gemma 0/8;
  Qwen medium 75% Gemma 0/8; Qwen easy 43% Gemma 0/8.
- Pass rate of the other model on each released band: Gemma hard -> Qwen 0.41; Gemma medium -> Qwen 0.55; Qwen hard / very_hard -> Gemma 0.03.
- Cross-val contamination is small (≤43 of 300 val questions in any other-model train band; e.g. G-medium val ∩ Q-easy train = 43).

## 2026-10-08 — Qwen3-4B-Base medium RL FINISHED (400 -> 600 extension early-stopped @590)

`q3-4b-rl-med` (resumed from step 400 with Adam + data cursor) early-stopped at step 590 (5 consecutive misses after the best):
best val mean@16 **0.6581 @ step 510** (was 0.644 @350 / 0.649 @440); 590 scored 0.611. best_hf (step 510) + run receipts verified
on S3 (s3://scale-ml/genai/rl-distill/qwen3-4b-base-rl/qwen3-4b-medium-es5/; the step-400 best is kept under best_hf_at_step400/).
The trailing wandb teardown ConnectionResetError is harmless (run synced; RUN_DONE rc=0).

---

## 2026-10-13 — very_hard RL teachers -> students: 30B -> 4B, 30B -> 1.7B, 4B -> 1.7B (traces + distill + evals) DONE

User request: make trace data from the best 30B and best 4B very_hard RL checkpoints and distill 30B -> 4B / 1.7B and 4B -> 1.7B,
same grid as medium/hard. New specs (c4e5d4b6): q30bvh-very_hard (teacher .../qwen3-30b-a3b-base-rl-full-checkpoints/qwen30b-very_hard-es5/
global_step_460, val 0.3946) and q4bvh-very_hard (.../qwen3-4b-base-rl-full-checkpoints/qwen30b-very_hard-es5/global_step_620, val 0.2444);
very_hard data at dataset revision 9bec69ca (traces + stage-3 val download). Same trace contract (3000 train q x 16 + 300 val q x 1,
top-128, temp 1, 8192 tokens), same §9 distill recipe (bs 128, 2e-6 -> 2e-7, 100 warmup, 1000 steps, ckpt every 250), n=16 evals of
steps 250/500/750/1000 on the very_hard validation split. Borrowing off + 60-min fallback watcher (tmux borrowing-fallback).
- q3-30bvh-dist-4b: 8 GPUs, traces on 8 x TP-1 engines (MAX_NUM_SEQS=48, util 0.92, pinned engine identity, as q30bh-hard), then 4B distill.
- q3-4bvh-dist-1p7b: 8 GPUs, traces on 8 x TP-1 engines, then 1.7B distill.
- q3-30bvh-dist-1p7b: 2 GPUs, reuses the q30bvh bundle; started by tmux `wait-q30bvh-1p7b` once its COMPLETE.json is on S3.
Outputs: traces s3://scale-ml/genai/rl-distill/<qwen3-30b-rl|qwen3-4b-rl>-traces-topk128-v1/<spec>/, ckpts .../<family>-distill-ckpts-v1/
<spec>-to-<q4b|q1p7b>-bs128-s1000-lr2e-6/, evals .../<family>-distill-evals-v1/<same>/step_*/summary.json. References on very_hard val
(mean@16): 30B teacher 39.5, 4B teacher / 4B RL best 24.4, 1.7B RL best 12.4, bases 11.2 / 6.7 / 3.1 (30B / 4B / 1.7B).
First launch failed (generate_gemma4_distill_traces.py --direction allow-list lacked the new names; fixed d15fb6e3), relaunched
as -r2 dirs. Traces: q30bvh done 23:31Z, q4bvh done 00:06Z; 30B -> 1.7B auto-started 23:33Z and skipped stage 1 (bundle reuse).
Results (very_hard val, 300 q x 16; mean@16 / pass@16):

| Student | step 250 | step 500 | step 750 | step 1000 | ref: student RL best | ref: teacher |
|---|---|---|---|---|---|---|
| 30B -> 4B | 14.9 / 62.7 | 18.1 / 65.3 | 18.9 / 66.3 | **19.6** / 61.7 | 24.4 | 39.5 |
| 4B -> 1.7B | 8.8 / 47.0 | 9.5 / 46.3 | 9.2 / 43.0 | **10.0** / 46.0 | 12.4 | 24.4 |
| 30B -> 1.7B | 6.5 / 40.7 | 6.7 / 40.0 | **7.5** / 42.3 | 7.3 / 40.0 | 12.4 | 39.5 |

Both finished distills land below direct RL on the same student (4B: 19.6 vs 24.4; 1.7B: 10.0 vs 12.4), as on medium/hard; still
rising slowly at step 1000 (mean@16), pass@16 flat-to-down after step 750.
30B -> 1.7B (done 02:49Z) is the weakest: 7.5 best vs 10.0 from the 4B teacher -- the bigger teacher is the worse 1.7B
teacher, same ordering as medium/hard (25.9 vs 28.6, 11.3 vs 11.9) but a wider gap on very_hard. All 3 runs COMPLETED_RECEIPTS.
pass@16 comparison (10-14): same stage-3 protocol run locally on GPU 5 for the RL best checkpoints + bases
(`run_qwen3_very_hard_val_evals_local.sh`, s3://scale-ml/genai/rl-distill/qwen3-very_hard-val-evals-v1/<tag>/summary.json;
RL logs only have mean@16 + bootstrap best@16). mean@16 / pass@16:

| Model | 30B | 4B | 1.7B |
|---|---|---|---|
| base | 9.2 / 68.7 | 6.8 / 50.7 | 3.3 / 30.3 |
| RL best (460 / 620 / 810) | 38.4 / 76.3 | 23.7 / 63.3 | 12.1 / 42.7 |
| best distill (by pass@16) | - | 30B->4B s750 18.9 / 66.3 | 4B->1.7B s250 8.8 / 47.0; 30B->1.7B s750 7.5 / 42.3 |

pass@16: distill ~= RL for 4B (+3.0) and for 1.7B (+4.3 from the 4B teacher, -0.4 from 30B); 1 SE on 300 q is ~2.8 pt, so
"comparable, not lower" rather than a clear win. RL raises pass@16 over base here (+8 / +13 / +12), unlike the usual pass@k
collapse -- very_hard has many base-0/16 questions RL can unlock. mean@16 RL lead unchanged (RL eval 38.4/23.7/12.1 vs logged 39.5/24.4/12.4).
Medium + hard re-evals (10-14, same script generalized to `BAND=`, GPUs 1 / 5 locally; s3://.../qwen3-<band>-val-evals-v1/):
base 30B / 4B / 1.7B medium 29.3/95.7, 21.4/83.3, 9.6/55.7; hard 13.8/84.7, 9.0/54.0, 3.8/32.3. RL best medium 74.3/96.3 (@320),
65.5/93.7 (@510), 33.1/80.0 (@520); hard 61.4/89.0 (@410), 40.6/85.7 (@560), 12.1/51.0 (@550). Every RL re-eval is 0.3-1.5 pt below the
logged best (max-of-~50-evals selection bias; one eval's sampling SE ~0.4-0.5). pass@16: RL >= distill on medium (4B 93.7 vs 93.3,
1.7B 80.0 vs 78.3), distill >= RL on hard (4B 86.7 vs 85.7, 1.7B 54.0 vs 51.0), all within noise. Within-1-SE checkpoint re-evals for
4B / 1.7B (`run_qwen3_rl_ckpt_candidates_local.sh`) were started then stopped at the user's request (none completed).

---

## 2026-10-10 — Qwen3 30B-A3B / 4B / 1.7B DAPO RL on the new `very_hard` band LAUNCHED (borrowing on)

User request: run all three sizes on the `very_hard` config exactly as before, borrowing true. Same recipe as the medium/hard runs (GRPO
n=16, bsz 64 / mini 32, lr 1e-6, 8k responses + 2k overlong buffer, val mean@16 every 10 steps on the 300 very_hard validation q,
early stopping patience 5 / 1-point tolerance incl. step 0). Data prep now accepts `very_hard` (Hub revision 9bec69ca, pinned SHA-256s;
easy/medium/hard still read be4dcd8b) and the launch chain takes `BAND=very_hard` (job tag `vhard`) -- 6d498841.
- 30B-A3B: 8 GPUs, R3, max 500 steps, permanent ckpt every 10 + async S3 upload (no rolling), as the ckpt-fix runs.
- 4B: 4 GPUs, max 600 steps, rolling ckpt every 5. 1.7B: 2 GPUs, max 600 steps, rolling ckpt every 5.
`QWEN_SIZE=<30b-a3b|4b|1.7b> BAND=very_hard TOTAL_TRAINING_STEPS=<500|600> ALLOW_BORROWING=true MAX_RELAUNCHES=200 FAILURE_BACKOFF_SECONDS=3600
MAX_FAILURE_BACKOFF_SECONDS=3600 bash start_qwen3_30b_a3b_medium_rl_supervisor.sh` -> tmux rl-q3-{30b,4b,1p7b}-rl-vhard; checkpoints
s3://scale-ml/genai/rl-distill/qwen3-<size>-base-rl-full-checkpoints/qwen30b-very_hard-es5 (30B: qwen3-30b-a3b-base-rl-full-checkpoints/...),
W&B DAPO runs q3-{30b-a3b,4b,1.7b}-very_hard-es5-s42-v1.

**1.7B DONE 2026-10-11 12:19Z** (max steps 600, never early-stopped; job_db55td0c78j0089ltei0): very_hard-val mean@16 3.1 (step 0) ->
6.0 (90) -> 9.1 (250) -> 10.6 (490) -> **10.8 @550** (best), 10.0 @600. best_hf (step 550) at
s3://scale-ml/genai/rl-distill/qwen3-1.7b-base-rl/qwen3-1.7b-very_hard-es5/best_hf, run completion receipt verified.
**4B DONE 2026-10-11 18:51Z** (max steps 600, never early-stopped; job_db55td3mjdjg088cenjg): very_hard-val mean@16 6.7 (step 0) ->
15.2 (150) -> 19.7 (250) -> 23.75 (430) -> **24.3 @540** (best), 23.0 @600. best_hf (step 540) at
s3://scale-ml/genai/rl-distill/qwen3-4b-base-rl/qwen3-4b-very_hard-es5/best_hf, run completion receipt verified (the end-of-run
traceback is the usual harmless wandb teardown ConnectionResetError; RUN_DONE rc=0).
**30B-A3B DONE 2026-10-12 10:50Z** (max steps 500, never early-stopped; job_db55td3mjdjg088cenj0): very_hard-val mean@16 11.2 (step 0)
-> 20.4 (80) -> 27.8 (150) -> 33.5 (250) -> 38.5 (350) -> **39.5 @460** (best), 39.1 @500. best_hf (step 460) at
s3://scale-ml/genai/rl-distill/qwen3-30b-a3b-base-rl/qwen3-30b-a3b-very_hard-es5/best_hf, run completion receipt verified.
**very_hard summary (best val mean@16, step 0 -> best):** 30B 11.2 -> 39.5, 4B 6.7 -> 24.3, 1.7B 3.1 -> 10.8; on hard the same models
reached 62.5 / 41.2 / 12.6 from 13.6 / 8.6 / 3.8. All three still creeping up at the step cap (none early-stopped).
Progress vs hard at matched steps (own val sets): 30B +9.2 vs +16.2 by step 80, 4B +10.7 vs +22.4 by step 200, 1.7B +3.7 vs +5.1 by 200;
per-step wall clock equal or faster. 30B MoE router metrics (all bands): dead_expert_frac (per micro-batch, per layer) ~8% at step 1 ->
~11% by step 50-150 then flat; router entropy flat 4.21-4.24; no collapse (no balancing loss in RL).
Plateau check: val flat for the last 100-200 steps (30B 37.1-39.5 since 380, 4B 22.3-24.3 since 380, 1.7B 10.0-10.8 since 480) while
train acc kept rising (30B 57 -> 61, 4B 31 -> 42, 1.7B 10 -> 14 over the last ~250 steps) -- ~11-13 epochs over the 3,000 train q.
The 1-point tolerance resets the miss streak, so a flat plateau never trips early stopping.

**+100-step continuations LAUNCHED 2026-10-12 17:38Z** (user request: 100 more steps or until early stopping, borrowing true).
Same S3 prefixes / W&B ids, exact resume from the final full checkpoint (Adam, LR = constant 1e-6, dataloader cursor, early-stopping
history incl. the all-time best). The first-run `run_complete.json` (which makes the run-file exit RUN_ALREADY_COMPLETE) was moved,
with run_outcome.json and the best_hf marker, to `<ckpt prefix>/archive_first_run/`; best_hf is republished at the end (all-time best).
`QWEN_SIZE=<size> BAND=very_hard TOTAL_TRAINING_STEPS=<600|700|700> SUP_TAG=-ext100 ALLOW_BORROWING=true MAX_RELAUNCHES=200
FAILURE_BACKOFF_SECONDS=3600 MAX_FAILURE_BACKOFF_SECONDS=3600 [30B: ROLLING_CHECKPOINT_ENABLED=False ASYNC_CHECKPOINT_UPLOAD=True]
bash start_qwen3_30b_a3b_medium_rl_supervisor.sh`; state `.scale_train_supervisors/q3-*-rl-vhard-20261012-ext100`.
Jobs: 30B job_db6hm7oc78j0089lto80 (500 -> 600), 4B job_db6hm7rmjdjg088cf01g (600 -> 700, 2 misses carried), 1.7B
job_db6hm7oc78j007hgltsg (600 -> 700).
**Continuations DONE 2026-10-12/13** (all receipts verified; best_hf republished = all-time best):
- 30B: 38.2 / 37.4 / 37.7 / 37.9 / 38.3 @510-550 -> EARLY-STOPPED @550 (5 misses vs 39.5). Best unchanged **39.5 @460**.
- 4B: 23.8, **24.4 @620** (new best, +0.1), then 23.3 / 23.4 / 23.5 / 23.0 / 23.0 / 22.3 / 23.2 / 22.7 -> EARLY-STOPPED @700 (5 misses).
- 1.7B: 10.3 / 10.3 / 10.7 / 10.6 / 11.3 / 10.9 / 10.7 / **11.5 @680** (new best, +0.7) / 10.9 / 10.8 @700 (max steps, 0 misses).
Takeaway: confirmed plateau -- +0.0 / +0.1 / +0.7 for 30B / 4B / 1.7B; more gains need more (fresh) train questions, not steps.
(1.7B supervisor briefly logged RELAUNCHING_MISSING_COMPLETION -- it polled seconds before the receipt upload; it re-verified
COMPLETED_RECEIPTS 11 s later with no relaunch.)
**1.7B 700 -> 1000 DONE 2026-10-13** (user request; same procedure, step-700 receipt moved to `<ckpt>/archive_ext100/`; SUP_TAG=-to1000,
job via `rl-q3-1p7b-rl-vhard`, max steps 1000, never early-stopped -- longest streak 4 misses @870-900): best **12.4 @810** (best_hf),
11.3 @1000. Mean val per 100 steps: 10.8 (610-700) -> 11.0 (710-800) -> 11.3 (810-900) -> 11.5 (910-1000), i.e. ~+0.25/100 steps,
while train acc went 13.7 (551-600) -> 17.5 (701-750) -> 27.2 (951-1000) -- mostly memorizing the 3,000 train q (~21 epochs).

---

## 2026-10-10 — Resample of the 9,182 DeepScaleR questions Qwen3-30B-A3B-Base solved 0/8: 16 new samples each

User request: of the 38,796 deduplicated DeepScaleR questions, 9,182 (23.7%) were 0/8 in the 2026-10-05 30B n=8 run (their responses: mean 1,213 /
median 771 / std 1,458 / min 4 / max 8,192 tokens, 1.80% at the cap, vs 915 / 642 / 1,056 for the rest). Resampled each 16x with the identical
protocol (0-shot ChatML + boxed instruction, temp 1, top-p 1, 8192 tokens, same stops, vLLM 0.15.1) but seed 1042 (the default seed would
repeat the n=8 draws): 2 GPUs (DP 2 x TP 1, 64 seqs/engine), borrowing, job_db4ltvjmjdjg07lcrt20, 21:39 -> 08:16Z (~1,840 tok/s/engine).
`start_qwen3_30b_zero8_resample_supervisor.sh` + SEED/MAX_NUM_SEQS knobs in `run_deepscaler_samples_st.sh` (c70258af). The pod's final merge
asserted the full-set size (38,796) and failed after all chunks were done; merged locally (fix 66c01686).
Output: s3://scale-ml/genai/rl-distill/qwen3-deepscaler-samples/qwen3-30b-a3b-base-0shot-qwenmath-zero8-n16-seed1042-8k/ (`all.parquet`,
`grades_strict_n16.parquet`, `summary.json`). Strict boxed grading, k/16:

| k/16 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 13 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| questions | 5,548 | 1,785 | 833 | 471 | 250 | 142 | 71 | 38 | 29 | 6 | 6 | 2 | 1 |

**39.6% (3,634) are solved at least once in 16 new samples; 60.4% (5,548) stay 0/16**; mean per-sample accuracy 5.2%; half of the recovered
questions are 1/16. New responses: mean 967 / median 733 / std 981 tokens, 0.52% at the cap -- shorter than the same questions' original
0/8 samples (selection effect: "all 8 failed" over-selects long/truncated runaways).
**`very_hard` config added to the Hub dataset (2026-10-10):** user request -- from the 3,634 >=1/16 questions sample 3,300 and split
3,000 / 300, add to `JWei05/DeepScaleR-Easy-Medium-Hard-Qwen3-30B-A3B-Base-10k` like the other bands and document the methodology.
`data/build_deepscaler_qwen3_30b_very_hard_band.py` (f80144b6): same SHA-256 hash-rank selection (seed 42) / validation split (seed 43)
with band name `very_hard`, same verl schema, extra_info + `resample_{n_correct,n_samples,pass_rate,accs,seed}`; validated no overlap with
easy/medium/hard and the exact split sizes. Re-grade with per-trace flags matched the first grading exactly (grades:
`.../zero8-n16-seed1042-8k/grades_strict_n16_accs.parquet`). Train resample scores 1/16: 1,488, 2/16: 692, 3/16: 372, 4-13/16: 448;
validation 1/16: 138, 2/16: 69, 3/16: 49, 4-9/16: 44. Hub commit **9bec69ca** (parent be4dcd8b; README gains a "Very-hard configuration"
section, manifest gains `very_hard`). Local build: /mnt/efs/jasonwei/verl/data/deepscaler_easy_medium_hard_very_hard_qwen3_30b_a3b_base_20261010/.
Existing training/eval code pins revision be4dcd8b and is unaffected; use revision 9bec69ca for `very_hard`.

---

## 2026-10-09 — Gemma 4 medium-trained E4B models evaluated on a 300-question sample of the medium TRAIN split (local GPUs)

User request: for the E4B RL and distilled models trained on the medium band, eval on a random 300-question sample of the training
set (avg@16 / pass@16, normal Gemma 4 sampling). Sample: 300 text-deduplicated questions (seed 42) from
JWei05/DeepScaleR-Easy-Medium-Hard-Gemma-26B-PT-10k@a0ba3c3d medium/train.parquet (`data/prepare_gemma4_train_sample_eval_data.py`).
Protocol = the study's recorded eval config (eval_math_passk.py: 12-shot template, temp 1, top-p 1, 8192 tokens, strict boxed, monte-carlo
pass@k, no logprobs, 2 vLLM slots/GPU at util 0.4 + 16 GiB KV), local GPUs 0/1/5/6 (`run_gemma4_train_sample_eval_local.sh`, d90e6478),
~20-30 min. The registry URI for distill_26bw_lr2e6_medium_to_e4b_step1000 is empty; the reported 26bw student is the `-n8` run
(.../26bw-medium-to-e4b-bs128-s1000-lr2e-6-n8/global_step_1000), whose stored val scores match the doc. Results:
s3://scale-ml/genai/rl-distill/gemma4-train-sample-evals-v1/train_medium/<tag>/metrics.json.

| model | train sample avg@16 / pass@16 | medium val avg@16 / pass@16 | train - val |
|---|---|---|---|
| E4B RL seed 43 (step 120) | 32.4 / 80.7 | 31.1 / 78.3 | +1.3 / +2.4 |
| E4B RL 4x batch seed 46 (step 100) | 57.6 / 81.0 | 39.9 / 71.0 | +17.7 / +10.0 |
| Untrained 12B RL -> E4B (12br, step 1000) | 46.9 / 89.0 | 38.8 / 85.0 | +8.1 / +4.0 |
| Off-policy distilled 12B RL -> E4B (12bd, step 1000) | 53.5 / 87.7 | 36.6 / 82.3 | +16.9 / +5.4 |
| Untrained 26B RL -> E4B (26bw, step 1000) | 53.1 / **92.0** | 42.0 / 90.7 | +11.1 / +1.3 |
| Off-policy distilled 26B RL -> E4B (26bd, step 1000) | **65.4** / 89.7 | 42.6 / 85.3 | +22.8 / +4.4 |

The 64-prompt RL run barely separates train from val (~2.6 passes over the 3,000 train q); the 4x-batch run (~8.5 passes) and every
distilled student (16 teacher traces per train question) score much higher on train questions, and the gap is largest for the
distilled-lineage teachers (12bd, 26bd) whose traces are sharper. pass@16 gaps stay small (+1 to +5) except the 4x RL run.

---

## 2026-10-09 — Best 4B medium / hard RL ckpts -> Qwen3-1.7B-Base distillation (traces + distill + evals) LAUNCHED

User request: complete the teacher/student grid for both bands -- 30B RL -> 4B (done), 30B RL -> 1.7B (medium done, hard training),
and the new 4B RL -> 1.7B. New trace specs q4bm-medium (teacher = s3 .../qwen3-4b-base-rl-full-checkpoints/qwen30b-medium-es5/global_step_510,
val 0.6581) and q4bh-hard (.../qwen30b-hard-es5/global_step_560, val 0.4117), directions q4bm_medium_r510_to_1p7b / q4bh_hard_r560_to_1p7b,
same trace contract (3000 train q x 16 + 300 val q x 1, top-128, temp 1, 8192 tokens); pipeline family qwen3-4b-rl (d7c99a5c): traces
s3://scale-ml/genai/rl-distill/qwen3-4b-rl-traces-topk128-v1/<spec>/, ckpts .../qwen3-4b-rl-distill-ckpts-v1/<spec>-to-q1p7b-bs128-s1000-lr2e-6/,
evals .../qwen3-4b-rl-distill-evals-v1/<same>/step_*/summary.json, W&B qwen3-4b-rl-distill-v1. One 8-GPU job per band (borrowing off +
fallback watcher): 8 x TP-1 trace engines, then the same §9 1.7B distill recipe (bs 128, 2e-6 -> 2e-7, 100 warmup, 1000 steps, ckpt every
250), then n=16 evals of steps 250/500/750/1000 on the band's validation split. Supervisors rl-q3-4b{m,h}-dist-1p7b.
**Medium DONE 2026-10-10 03:38Z** (job_db4kik3mjdjg088ceklg): traces 48,300 rows / 111.1M tokens (TP1 x 8, ~1,050 tok/s/engine,
20:11-01:14Z); distill ~5.4 s/step on 8 GPUs, val KL 0.214 (10) -> 0.109 (100) -> 0.0997 (250) -> 0.0956 (500) -> 0.0940 (750) -> 0.0933 (1000).
Medium-val mean@16 / pass@4 / pass@16: step 250 25.9 / 49.7 / 73.7, 500 26.2 / 50.5 / 74.3, 750 **28.6** / 53.3 / **78.3**, 1000 27.7 / 51.1 / 74.0.
Same set: 1.7B base 9.1, 30B medium RL -> 1.7B 25.9 (@1000), 1.7B medium RL 29.6 (@280/310, running). The weaker 4B teacher (65.8) gives a
better 1.7B student than the 30B teacher (75.75) despite a higher KL to its teacher.
**Hard DONE 2026-10-10 05:40Z** (job_db4kik0c78j0089ltbl0): traces 48,300 rows / 150.1M tokens (20:08-02:51Z); distill ~6.4 s/step on 8 GPUs,
val KL 0.240 (10) -> 0.136 (100) -> 0.125 (250) -> 0.120 (500) -> 0.118 (750) -> 0.117 (1000). Hard-val mean@16 / pass@4 / pass@16:
step 250 10.3 / 27.0 / 52.3, 500 11.1 / 28.2 / 52.7, 750 11.5 / 28.8 / 53.0, 1000 **11.9** / 29.6 / **54.0**. Same set: 1.7B base 3.8,
30B hard RL -> 1.7B 11.3 / 53.0 (@1000), 1.7B hard RL 10.9 (@340, running).

**Teacher -> student grid complete (2026-10-10), in-distribution val mean@16 / pass@16 at the best distill checkpoint:**

| teacher -> student | medium | hard |
|---|---|---|
| 30B RL -> 4B | 52.2 / 93.3 | 34.4 / 86.7 |
| 30B RL -> 1.7B | 25.9 / 75.7 | 11.3 / 53.0 |
| 4B RL -> 1.7B | **28.6 / 78.3** | **11.9 / 54.0** |
| direct RL 4B (best) | 65.8 | 41.2 |
| direct RL 1.7B (running, 2026-10-10 05:40Z) | 30.9 | 10.9 |

The 4B RL teacher beats the 30B RL teacher for a 1.7B student on both bands, despite a weaker teacher (65.8 / 41.2 vs 75.75 / 62.5).

---

## 2026-10-09 — Qwen math suite (easy/medium/hard val, GSM8K, MATH500, AIME25/26) on 4B RL-best vs 30B->4B distilled LAUNCHED

User request: put the best 4B RL medium / hard checkpoints and the 30B medium / hard -> 4B distilled students through the extensive math
eval, AIME duplicated 32x. Models: q4b-rl-medium-best (step 510), q4b-rl-hard-best (step 560), q30bm-to-q4b-s1000, q30bh-to-q4b-s1000
(final distill step; there is no step-100 save), plus Qwen3-4B-Base as the reference. Sets: Qwen3-30B-band easy / medium / hard validation
(300 q each), GSM8K test (1,319), MATH500 (500), AIME 2025 / 2026 (30 each) at the Gemma 4 suite's pinned revisions; 16 samples/question,
AIME 32. Qwen RL prompt contract (0-shot ChatML + boxed instruction, temp 1, 8192 tokens, strict last-box math_verify) -> mean@n,
pass@k / pass^k. One 4-GPU ScaleTrain job, borrowing off (fallback watcher), one TP-1 engine per GPU on a shared (model, set) queue:
`run_qwen3_math_suite_st.sh` / `start_qwen3_math_suite_supervisor.sh` (192a3b9a). Results:
s3://scale-ml/genai/rl-distill/qwen3-math-suite-v1/<model>/<set>/summary.json.
**DONE 2026-10-09 19:19Z** (35/35; moved to borrowing at the user's request after 18 min queued, ran 16:06-19:19Z on 4 GPUs).
mean@n / pass@n (n = 16, AIME 32):

| model | easy | medium | hard | gsm8k | math500 | aime2025 | aime2026 | avg mean / pass |
|---|---|---|---|---|---|---|---|---|
| Qwen3-4B-Base | 44.8 / 97.3 | 21.0 / 83.3 | 8.3 / 52.3 | 42.2 / 97.3 | 39.9 / 90.4 | 2.8 / 30.0 | 2.7 / 16.7 | 23.1 / 66.8 |
| 4B RL medium best (@510) | **89.1** / 98.7 | 64.5 / 95.3 | 37.4 / 78.0 | **93.3** / 98.3 | 83.7 / 93.4 | 17.3 / 36.7 | 14.0 / 30.0 | 57.1 / 75.8 |
| 4B RL hard best (@560) | 87.6 / 98.3 | **65.2** / 94.7 | **39.4** / 83.7 | 92.8 / 98.4 | **84.0** / 95.4 | **19.1 / 53.3** | **15.8 / 36.7** | **57.7 / 80.1** |
| 30B medium RL -> 4B (s1000) | 85.2 / **99.7** | 52.1 / 93.3 | 27.1 / 73.7 | 91.8 / 98.2 | 77.4 / 94.2 | 10.7 / 40.0 | 9.1 / 23.3 | 50.5 / 74.6 |
| 30B hard RL -> 4B (s1000) | 85.2 / 99.0 | 57.3 / **96.7** | 34.4 / **86.7** | 91.6 / **98.6** | 81.4 / **95.8** | 16.1 / 43.3 | 11.2 / 30.0 | 53.9 / 78.6 |

pass^n (consistency): RL medium 62.3 / 23.0 / 5.3 / 81.1 / 63.0 / 3.3 / 0.0; RL hard 61.0 / 20.3 / 6.3 / 79.3 / 62.2 / 3.3 / 0.0;
distill medium 50.0 / 9.0 / 2.7 / 75.3 / 48.4 / 0 / 0; distill hard 50.3 / 13.3 / 2.3 / 71.6 / 53.0 / 3.3 / 0 (same set order). All trained
models box ~100% (base 70-81%). Direct 4B RL beats distillation from the 30B RL teacher on mean@n on every set (hard-RL best overall);
the hard-teacher distill has the best pass@16 on all five non-AIME sets but trails RL on AIME (30 q, 1 question = 3.3 pts of pass@32).
Hard-band training (RL or teacher) transfers down to medium / MATH500 / AIME better than medium-band training.
Grader note: a first-call math_verify pool cold start took 28 s locally (load ~70) vs the 30 s per-call timeout; re-grading the first
sample of all 59 finished eval sets with a warmed pool gave 0 mismatches, so pod results are unaffected; grade_band_eval_samples.py now
warms the pool first (c84c92fc).

---

## 2026-10-09 — Best 30B medium / hard RL ckpts -> Qwen3-1.7B-Base distillation LAUNCHED (2 GPUs each, borrowing off)

Same pipeline as the 4B students, student swapped (`STUDENT=q1p7b`, acbea94c): stage 1 skipped (reuses the q30bm-medium / q30bh-hard
trace bundles), stage 2 = §9 recipe (bs 128, 2e-6 -> 2e-7, 100 warmup, 1000 steps, 4 seqs/micro, KL chunk 4096, ckpt every 250) into
Qwen/Qwen3-1.7B-Base@ea980cb0 on FSDP2 DP2, stage 3 = n=16 evals of steps 250/500/750/1000 on the band's validation split.
`TEACHER_SPEC=<q30bm-medium|q30bh-hard> STUDENT=q1p7b GPUS=2 ALLOW_BORROWING=false MAX_RELAUNCHES=200 FAILURE_BACKOFF_SECONDS=3600
MAX_FAILURE_BACKOFF_SECONDS=3600 bash start_qwen3_30bm_to_4b_distill_supervisor.sh` -> tmux rl-q3-30b{m,h}-dist-1p7b; jobs
job_db4gaaoc78j007hglkgg (medium) / job_db4gaaoc78j0089ltagg (hard), p5.48xlarge:2. Ckpts .../qwen3-30b-rl-distill-ckpts-v1/<spec>-to-q1p7b-bs128-s1000-lr2e-6/,
evals .../qwen3-30b-rl-distill-evals-v1/<spec>-to-q1p7b-bs128-s1000-lr2e-6/step_*/summary.json.
**Medium DONE 2026-10-09 19:27Z** (job_db4gaaoc78j007hglkgg; ~8-9 s/step on 2 GPUs, val KL 0.109 (10) -> 0.0753 (250) -> 0.0731 (1000)).
Medium-val mean@16 / pass@4 / pass@16: step 250 22.8 / 46.5 / 71.3, 500 23.7 / 47.3 / 71.3, 750 25.3 / 49.7 / 74.7, 1000 **25.9** / 51.1 / 75.7
(boxed ~99.5%). Same set: 1.7B base 9.1, 1.7B medium RL 21.5 @100 (running), 30B teacher 75.75. Still rising at step 1000.
Step-1000 student also queued on the math suite (q3-ms-30bm-1p7b-s1000, 1 GPU, SINGLE_PASS=1 = all 7 sets in one vLLM generate call);
hard student step 500 on the math suite (q3-ms-30bh-1p7b-s500): hard 11.2 mean@16 / 50.0 pass@16.
Math suite, 30B medium RL -> 1.7B step 1000 (SINGLE_PASS, all 7 sets in one generate call, ~31 min wall on 1 GPU incl. prep/grading;
medium identical to the pipeline eval because each request carries the same fixed seed), mean@n / pass@n: easy 64.2 / 95.0,
medium 25.9 / 75.7, hard 11.5 / 49.3, GSM8K 80.2 / 96.9, MATH500 61.4 / 89.4, AIME25 2.9 / 20.0, AIME26 2.1 / 23.3.
**Hard DONE 2026-10-09 22:30Z** (job_db4gaaoc78j0089ltagg; ~20 s/step on 2 GPUs). Hard-val mean@16 / pass@4 / pass@16 (val KL):
step 250 9.8 / 25.7 / 49.3 (0.1238), 500 11.2 / 27.5 / 50.0 (0.1206), 750 11.1 / 28.2 / 52.0 (0.1190), 1000 **11.3** / 28.3 / **53.0** (0.1185).
Same set: 1.7B base 3.8, 1.7B hard RL 8.3 @180 (running), 30B teacher 62.5. KL to the 30B teacher stays ~1.6x the 4B student's (0.073).
Math suite, 30B hard RL -> 1.7B step 500 (sequential, ~1h40): easy 61.1 / 93.0, medium 24.1 / 77.7, hard 11.2 / 50.0, GSM8K 78.7 / 97.1,
MATH500 61.8 / 88.6, AIME25 4.8 / 26.7, AIME26 2.1 / 26.7 -- at half the steps it matches the medium-teacher step-1000 student (better
pass@16 on medium/hard/AIME, ~1-3 pts lower mean on easy/medium/GSM8K).

---

## 2026-10-09 — Qwen3-1.7B-Base medium + hard DAPO RL LAUNCHED (2 GPUs each, borrowing off, 600 steps)

Same recipe as the 4B / 30B Qwen runs (GRPO n=16, bsz 64 / mini 32, lr 1e-6, 8k responses, val mean@16 every 10 steps, early stopping
patience 5 with a 1-point miss tolerance incl. the initial validation), max 600 steps. New `QWEN_SIZE=1.7b` layout (738f26d4):
Qwen/Qwen3-1.7B-Base@ea980cb0, FSDP2 DP2 (no offload) + vLLM DP2 (TP 1, 24 GiB KV), Qwen3DecoderLayer wrap; launcher gained
`ALLOW_BORROWING=false`. `QWEN_SIZE=1.7b BAND=<b> ALLOW_BORROWING=false TOTAL_TRAINING_STEPS=600 MAX_RELAUNCHES=200 FAILURE_BACKOFF_SECONDS=3600
MAX_FAILURE_BACKOFF_SECONDS=3600 bash start_qwen3_30b_a3b_medium_rl_supervisor.sh` -> tmux rl-q3-1p7b-rl-{med,hard}; jobs
job_db4g91bmjdjg07lcrs2g (medium) / job_db4g91bmjdjg088cejmg (hard), p5.48xlarge:2. Checkpoints
s3://scale-ml/genai/rl-distill/qwen3-1.7b-base-rl-full-checkpoints/qwen30b-<band>-es5, best_hf under .../qwen3-1.7b-base-rl/qwen3-1.7b-<band>-es5/.
**Hard DONE 2026-10-10 13:08Z** (max steps 600, never early-stopped; ~2.2 min/step): hard-val mean@16 3.8 (step 0) -> 7.7 (110) -> 9.0 (250)
-> 10.9 (340) -> 12.3 (430) -> best **12.6 @ 550**, 12.2 at 600. best_hf (step 550) + run_outcome verified on S3; the trailing wandb teardown
ConnectionResetError is harmless (RUN_DONE rc=0). Beats both 1.7B hard distillations (30B RL -> 1.7B 11.3, 4B RL -> 1.7B 11.9).
**Medium DONE 2026-10-10 14:42Z** (max steps 600, never early-stopped): medium-val mean@16 9.1 (0) -> 14.0 (30) -> 20.25 (70) -> 25.1 (160)
-> 28.4 (250) -> 31.8 (380) -> best **33.8 @ 520**, 32.4 at 600. best_hf (step 520) + run_outcome verified on S3 (RUN_DONE rc=0).
Beats both 1.7B medium distillations (30B RL -> 1.7B 25.9, 4B RL -> 1.7B 28.6). Final Qwen3 direct-RL vs distillation (val mean@16):
4B: RL 65.8 / 41.2 vs best distill (30B teacher) 52.2 / 34.4; 1.7B: RL 33.8 / 12.6 vs best distill (4B teacher) 28.6 / 11.9 (medium / hard).

---

## 2026-10-09 — pass@32 evals: Qwen3-30B-A3B-Base vs Qwen3-4B-Base vs 4B->30B distilled students LAUNCHED

User request: eval the 4B->30B distilled 30B students on their own band's validation set, plus Qwen3-4B-Base (the teacher) and
Qwen3-30B-A3B-Base on both, pass@k up to 32, compare directly. One 8-GPU ScaleTrain job, borrowing off (`run_qwen3_passk32_evals_st.sh`,
supervisor `start_qwen3_passk32_evals_supervisor.sh` -> tmux rl-q3-passk32-evals, job_db4g7qrmjdjg07lcrs1g): 12 tasks pulled from a
shared queue by 4 TP-2 vLLM engines -- q30b-base + q4b-base on medium and hard, q4b30b-<band>-s{250,500,750,1000} on their own band
(bases + step 1000 first). Protocol = Qwen band-eval at n=32 (300 q, 0-shot ChatML + boxed instruction, temp 1, 8192 tokens, strict
boxed math_verify): mean@32, unbiased pass@k / pass^k for k = 1..32. Results: s3://scale-ml/genai/rl-distill/qwen3-passk32-evals-v1/<model>/<band>/summary.json.
**DONE 2026-10-09 16:34Z** (all 12, ~22 min per eval per TP-2 engine). 300 q x 32 per row; q4b30b-<band>-sN = 30B student at distill step N:

| band | model | mean@32 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 | pass^2 | pass^4 | boxed | trunc | mean len |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| medium | Qwen3-4B-Base (teacher) | 20.8 | 34.9 | 52.9 | 70.9 | 84.8 | 92.7 | 6.7 | 1.4 | 80.8% | 2.1% | 1163 |
| medium | Qwen3-30B-A3B-Base | 27.9 | 45.9 | 67.3 | 85.6 | 95.8 | 99.0 | 9.8 | 2.1 | 82.4% | 0.6% | 874 |
| medium | 4B->30B s250 | 18.0 | 30.8 | 47.9 | 66.2 | 81.5 | 91.0 | 5.1 | 0.8 | 86.6% | 1.4% | 1029 |
| medium | 4B->30B s500 | 17.8 | 30.8 | 48.2 | 66.9 | 82.1 | 91.7 | 4.9 | 0.7 | 85.7% | 1.3% | 1012 |
| medium | 4B->30B s750 | 16.8 | 29.0 | 45.5 | 63.5 | 78.9 | 89.3 | 4.6 | 0.7 | 84.1% | 1.1% | 1000 |
| medium | 4B->30B s1000 | 18.2 | 31.1 | 48.1 | 66.0 | 80.6 | 90.3 | 5.3 | 0.9 | 84.6% | 1.4% | 1017 |
| hard | Qwen3-4B-Base (teacher) | 8.6 | 15.7 | 26.7 | 41.5 | 57.7 | 72.0 | 1.5 | 0.1 | 79.7% | 2.3% | 1175 |
| hard | Qwen3-30B-A3B-Base | 13.0 | 23.2 | 38.1 | 56.2 | 73.1 | 86.0 | 2.8 | 0.3 | 81.8% | 0.6% | 912 |
| hard | 4B->30B s250 | 7.5 | 13.7 | 23.5 | 37.2 | 53.3 | 69.7 | 1.2 | 0.1 | 85.9% | 1.2% | 1045 |
| hard | 4B->30B s500 | 7.2 | 13.3 | 23.1 | 37.1 | 53.8 | 70.3 | 1.1 | 0.1 | 84.6% | 1.4% | 1031 |
| hard | 4B->30B s750 | 7.7 | 14.2 | 24.5 | 39.1 | 56.6 | 73.0 | 1.3 | 0.1 | 85.4% | 1.2% | 1036 |
| hard | 4B->30B s1000 | 7.9 | 14.5 | 25.0 | 39.6 | 56.8 | 73.7 | 1.3 | 0.1 | 84.5% | 1.4% | 1049 |

Distilling the untrained 4B into the 30B erases the 30B's advantage within 250 steps: the student sits at (hard) or 2-3 pts below
(medium) the 4B teacher across the whole pass@k curve and stays flat through step 1000. pass^32 = 0 for every model (best question 26/32).

---

## 2026-10-09 — Qwen3-30B-A3B hard RL FINISHED (early stopping @500 = max steps)

`q3-30b-rl-hard` (job_db38lnjmjdjg088cefb0, ckpt-fix relaunch from 195) stopped at step 500, which was both the step cap and the 5th
consecutive miss: best val mean@16 **0.6252 @ step 410**; 490 scored 0.603, 500 scored 0.611. run_outcome early_stopping / final 500 /
best 410; best_hf (step 410) + _REMOTE_COMPLETE verified on S3 (s3://scale-ml/genai/rl-distill/qwen3-30b-a3b-base-rl/qwen3-30b-a3b-hard-es5/).
The 30B-hard -> 4B distill below already uses this step-410 checkpoint as its teacher.

---

## 2026-10-09 — Best Qwen3-30B-A3B *hard* RL ckpt (step 410, still training) -> Qwen3-4B-Base: traces + distill + hard-val eval LAUNCHED

User request: same pipeline as the medium one, from the current best 30B hard checkpoint (val mean@16 0.6252 @ step 410; the RL run keeps
going to 500), borrowing OFF. Spec q30bh-hard (teacher s3 .../qwen30b-hard-es5/global_step_410/actor/huggingface, hard band 3000 q x 16 +
300 val q x 1, TP2 x 2 engines) -> 4B distill (same recipe, ckpt every 250) -> n=16 eval on the hard validation split (the RL hard val set).
Supervisor `TEACHER_SPEC=q30bh-hard ALLOW_BORROWING=false bash start_qwen3_30bm_to_4b_distill_supervisor.sh` (tmux rl-q3-30bh-dist-4b).
Outputs: traces s3://scale-ml/genai/rl-distill/qwen3-30b-rl-traces-topk128-v1/q30bh-hard/, ckpts .../qwen3-30b-rl-distill-ckpts-v1/q30bh-hard-to-q4b-bs128-s1000-lr2e-6/,
evals .../qwen3-30b-rl-distill-evals-v1/q30bh-hard-to-q4b-bs128-s1000-lr2e-6/step_*/summary.json. References on the hard val set: 4B base
(RL step 0) 8.6, 4B hard RL best 41.2 (@560, still running), 30B hard teacher 62.5 (@410).
**Stage 1 DONE 2026-10-09 11:38Z:** bundle VALID, 48,300 rows / 143.2M response tokens (~2,970 tokens/response, 2.5x medium).
Moved to 8 GPUs, then cut over mid-run (06:0xZ) from TP2 x 4 to TP1 x 8 engines with `--pin-engine-identity` so the 24 finished TP2 shards
were kept; TP1 ran ~1,000-1,100 tok/s per GPU (vs ~435 per GPU at TP2), train split 94 shards done 11:17Z, validation split 11:24Z,
parallel bundle validation (16 procs) 14 min. Only incident: vLLM engine-init race (status 2) on 4/8 workers at the cutover and 1/8 at the
validation split, each self-healed by the supervisor's retry (per-worker caches + staggered start fix it for future pods, e67c511d).
Stage 2 (4B distill, 8 GPUs) started 11:38Z.
**DONE 2026-10-09 16:16Z** (job COMPLETED). Distill: 1000 steps at ~11 s/step (8 GPUs), val KL 0.1415 (step 10) -> 0.0813 (100) -> 0.0765 (250)
-> 0.0741 (500) -> 0.0731 (750) -> 0.0727 (1000). Hard-val eval (300 q x 16, temp 1, 8192 tok, strict boxed):

| student step | mean@16 | pass@2 | pass@4 | pass@8 | pass@16 | pass^4 | pass^16 | boxed | trunc | mean len |
|---|---|---|---|---|---|---|---|---|---|---|
| 250 | 29.5 | 44.4 | 60.2 | 73.7 | 83.0 | 6.4 | 1.7 | 99.5% | 0.1% | 2958 |
| 500 | 32.7 | 48.0 | 63.1 | 75.4 | 84.0 | 7.9 | 1.7 | 99.5% | 0.1% | 3116 |
| 750 | 32.3 | 47.5 | 63.0 | 76.1 | 85.3 | 8.2 | 2.0 | 99.6% | 0.1% | 3127 |
| 1000 | **34.4** | 49.7 | 64.7 | 77.3 | 86.7 | 9.5 | 2.3 | 99.8% | 0.0% | 3033 |

Same set, same metric (mean@16): Qwen3-4B-Base 8.6, **4B distilled from 30B hard RL 34.4**, 4B hard RL best 41.2 (@560), 30B hard RL teacher
62.5 (@410). Recovers ~48% of the base->teacher gap (medium: ~57%), ~7 pts below direct 4B RL; unlike medium it still rises to step 1000.

---

## 2026-10-08 — Best Qwen3-30B-A3B medium RL ckpt (step 320) -> Qwen3-4B-Base: traces + distill + medium-val eval LAUNCHED

User request: distill the best 30B medium RL checkpoint into 4B and eval on the medium val set, normal protocol. One 4-GPU ScaleTrain job
(`run_qwen3_30bm_to_4b_distill_st.sh`, supervisor `start_qwen3_30bm_to_4b_distill_supervisor.sh` -> tmux rl-q3-30bm-dist-4b, EKS, borrowing):
1. traces: spec q30bm-medium (teacher = s3 .../qwen3-30b-a3b-base-rl-full-checkpoints/qwen30b-medium-es5/global_step_320/actor/huggingface),
   3000 train q x 16 + 300 val q x 1, top-128, 2 vLLM engines x TP 2, per-shard S3 mirror -> s3://scale-ml/genai/rl-distill/qwen3-30b-rl-traces-topk128-v1/q30bm-medium/
2. distill into Qwen3-4B-Base@906bfd4b (STUDENT=q4b, Qwen3DecoderLayer wrap): §9 recipe (bs 128, 2e-6 -> 2e-7, 100 warmup, 1000 steps, val KL
   every 10) with the 26bd/12br micro-batching (4 seqs / 8192 cap, KL chunk 4096); permanent S3 checkpoint + HF export every 250 steps ->
   s3://scale-ml/genai/rl-distill/qwen3-30b-rl-distill-ckpts-v1/q30bm-medium-to-q4b-bs128-s1000-lr2e-6/
3. eval of steps 250/500/750/1000 on the medium validation split (300 q = the RL val set), Qwen band-eval protocol (n=16, temp 1, 8192
   tokens, strict boxed math_verify = the RL reward) -> mean@16 / pass@k / pass^k at s3://scale-ml/genai/rl-distill/qwen3-30b-rl-distill-evals-v1/.../step_*/summary.json.
   References on the same set/metric: 30B teacher 75.75 (RL val @320), 4B RL best 65.8 (@510, still running), 4B base 21.0 (RL step 0).
All three stages resume after preemption (S3 shards / S3 checkpoints / per-step eval markers).
**Stage 1 DONE 2026-10-08 18:15Z:** bundle VALID, 48,300 rows / 57.0M response tokens (~1,180 tokens/response; trace gen 07:21-16:52Z
uninterrupted after a broken-node day: i-0dc001458af0d6a38 rejected every 4-GPU pod until ~05:40Z; one borrowing preemption was misreported
COMPLETED -> supervisor --completion-marker-s3-uri). Stage 2 (4B distill, 4 GPUs) started 18:15Z.
**DONE 2026-10-08 21:24Z** (receipts verified). Distill: 1000 steps at ~9 s/step, val KL 0.0808 (step 10) -> 0.0495 (100) -> 0.0469 (250)
-> 0.0460 (500) -> 0.0456 (750) -> 0.0455 (1000), i.e. flat after ~250. Medium-val eval (300 q x 16, temp 1, 8192 tok, strict boxed):

| student step | mean@16 | pass@2 | pass@4 | pass@8 | pass@16 | pass^4 | pass^16 | boxed | trunc | mean len |
|---|---|---|---|---|---|---|---|---|---|---|
| 250 | 48.8 | 65.0 | 78.5 | 87.3 | 92.7 | 20.2 | 9.0 | 99.6% | 0.1% | 1253 |
| 500 | 51.9 | 68.0 | 81.2 | 90.0 | 95.0 | 23.4 | 11.0 | 99.6% | 0.3% | 1252 |
| 750 | 51.5 | 67.1 | 79.8 | 88.1 | 93.3 | 23.6 | 10.3 | 99.8% | 0.1% | 1252 |
| 1000 | **52.2** | 68.2 | 81.0 | 89.4 | 93.3 | 23.6 | 9.0 | 99.6% | 0.1% | 1244 |

Same set, same metric (mean@16): Qwen3-4B-Base 21.0 (RL step 0), **4B distilled from 30B-RL 52.2**, 4B RL best 65.8 (@510), 30B RL teacher
75.75 (@320). Distillation recovers ~57% of the base->teacher gap but stays ~14 pts below direct 4B RL; most of the gain lands by step 250.

---

## 2026-10-08 — Qwen3-30B-A3B medium RL FINISHED (early stopping @420); 4B hard resume moved to GKE

`q3-30b-rl-med` early-stopped at step 420 (5 consecutive validations > 1 point below the best; 360-420 after the ckpt-fix relaunch):
best val mean@16 **0.7575 @ step 320**; run_outcome early_stopping / final 420 / best 320, best_hf + receipts verified on S3
(s3://scale-ml/genai/rl-distill/qwen3-30b-a3b-base-rl/qwen3-30b-a3b-medium-es5/). 4B hard 400->600 extension moved to GKE at the user's
request (e9b1136c: ST_CLUSTER=gke passthrough, NCCL AF_INET on GKE): job_db3e9nsf8c4g07jroe1g (a3.megagpu.8g:4), supervisor
`.scale_train_supervisors/q3-4b-rl-hard-20261008-gke`; first GKE launch of this code -- S3 access / log capture unverified at launch.

---

## 2026-10-08 — Qwen3-4B-Base hard-band RL FINISHED (max steps 400)

`q3-4b-rl-hard` reached its 400-step cap at 00:02Z (not early-stopped): best val mean@16 **0.3819 @ step 390** (0.371 @ 400, 1 miss);
completion receipts verified by the supervisor; best_hf under s3://scale-ml/genai/rl-distill/qwen3-4b-base-rl/qwen3-4b-hard-es5/.
**Extended to 600 steps (user, 00:30Z 10-08)** as for 4B medium: run_complete.json -> run_complete.step400.json, best_hf copied to
best_hf_at_step400/, best_hf/_REMOTE_COMPLETE.json removed; supervisor `.scale_train_supervisors/q3-4b-rl-hard-20261008` (TOTAL_TRAINING_STEPS=600,
same ES rule, same W&B id) resumes the step-400 permanent checkpoint (Adam + data cursor + ES state). First pod rejected by a bad node
("GPU is lost" at admission); supervisor auto-relaunches after its 900 s backoff.
**4B hard 400 -> 600 DONE 2026-10-09 06:29Z** (max steps, receipts verified): best val mean@16 **0.4117 @ step 560** (was 0.3819 @390 before
the extension; 0.4096 @440, 0.4098 @550); step 600 scored 0.362. best_hf (step 560) on S3; the step-390 best kept under best_hf_at_step400/.

---

## 2026-10-07 — Qwen3-4B-Base -> Qwen3-30B-A3B-Base distillation (step 2): PREPARED, NOT LAUNCHED

Recipe = the E4B-base -> 26B-A4B student runs (DISTILLATION_EXPERIMENTS.md §9): off-policy top-128 forward KL on the q4b-base-<band>
bundles (3000 q x 16 = 48k train rows, 128 teacher val generations), batch 128, lr 2e-6 (100 warmup, linear -> 2e-7), 1000 steps
(~2.7 epochs), val KL every 10, 1 seq/micro-batch (4096 padded-token cap), KL chunk 1024, fp32 master + Adam, bf16 param views, grad
ckpt; S3-only checkpoints (permanent every 250, rolling every 50 with the HF export under hf_exports/), no Hub pushes.
Layout: **8 GPUs** (FSDP2 DP8, no offload): 30.5B x 16 B = 61 GB/GPU of params/grads/Adam (~67 GB with activations); 4 GPUs would be
122 GB/GPU -> CPU offload, several x slower. Code: run_gemma4_distill_one.sh STUDENT=q30b (Qwen/Qwen3-30B-A3B-Base@1b75feb7,
Qwen3MoeDecoderLayer wrap, VERL_FSDP2_LOCAL_LOAD=0 for the untied head) + q4b-base-* trace family; transformer_impl skip_lm_head patch
extended to Qwen3/Qwen3-MoE (without it the chunked-LM-head distill forward cannot run on Qwen; CPU test
test_qwen3_distill_skip_lm_head_on_cpu.py); view preflight: q4b directions + Qwen stop strings/ids. No router aux loss
(output_router_logits=False in the 30B config; same as Gemma 26B-A4B). Launch: `BAND=medium|hard bash
rl-distill-scripts/scale_train/start_qwen3_4b_to_30b_distill_supervisor.sh` (jobs q3-dist-4b30b-{med,hard}, 8 GPUs, borrowing, high;
janitor tmux ckpt-janitor-q3-dist-4b30b) -> run_qwen3_4b_to_30b_distill_st.sh; checkpoints
s3://scale-ml/genai/rl-distill/qwen3-4b-base-distill-ckpts-v1/q4b-base-<band>-to-q30b-bs128-s1000-lr2e-6, W&B project
qwen3-4b-base-distill-v1. CPU dry run (VALIDATE_ONLY) passed for medium: view 48,000 / 128 rows, tokenizer + identities + direction OK.
Untested on GPU (first step will confirm memory; fallback FSDP_OFFLOAD=true).
**2026-10-08 launch issues (fixed):** (1) teacher top-k ids were bounded by tokenizer.vocab_size (Qwen 151,643 excludes
<|endoftext|>/<|im_end|>) -> first val batch failed; now len(tokenizer) (b3f89482). (2) Both runs then HUNG in the initial validation
(1 h+, all 8 GPUs 100% util at ~120 W = NCCL spin): the untied 30B lm_head was its own FSDP2 unit, and the loss calls lm_head once per
1024-token chunk outside the forward, so every call resharded/all-gathered and ranks with different token counts deadlocked (Gemma's
tied head lives in the root unit). Fix: VERL_FSDP2_WRAP_EMBED_LM_HEAD=0 keeps embed/lm_head in the root unit (set for STUDENT=q30b);
2-rank CPU repro tests/special_distributed/run_fsdp2_untied_lm_head_chunked.py hangs without it and passes with identical grads with it.
Relaunched as r4 (hourly retry on failure).
r4 TRAINING since 03:17Z 10-08 (8 GPUs, no OOM, ~20 s/step -> ~6 h for 1000 steps): initial val KL medium 0.0620 (teacher mass
0.989), hard 0.0983 (0.913); step-1 loss 0.079 / 0.086. W&B project qwen3-4b-base-distill-v1.
**Hard band DONE 2026-10-08 21:14Z** (exit 0, receipts verified): 1000 steps, val KL 0.0983 -> 0.0369 (400) -> 0.0355 (500) -> 0.0332 (750)
-> **0.0324 (1000)**; checkpoints 250/500/750/1000 (+HF exports) on S3. It was preempted 7x on borrowed capacity (only step 250 survived)
and finished on dedicated capacity (ALLOW_BORROWING=false, user 15:20Z); exact resume verified (val KL at 250 identical to 7 digits).
Medium moved to dedicated too (resumed from 500 at 19:17Z, exact).
**Medium band DONE 2026-10-08 22:47Z** (receipts verified): val KL 0.0620 -> 0.0419 (250) -> 0.0398 (500) -> 0.0389 (750) -> **0.0387 (1000)**;
checkpoints 250/500/750/1000 (+HF exports) on S3. Both 4B -> 30B-A3B students are done; evals of the students not yet requested.

---

## 2026-10-07 — Qwen3-30B-A3B RL: checkpoint-overhead fix + relaunch to 500 steps

Measured share of wall-clock spent checkpointing: 30B medium 34%, 30B hard 29%, 4B hard 6%, 4B medium 4% (30B: rolling save at
every step-5 ~8 min = 2.5 local + ~5 upload; permanent every 10 ~12 min incl. HF export + vLLM sleep/wake; 366-427 GB each).
Fixes (958bb6f9): (1) ROLLING_CHECKPOINT_ENABLED=False -> permanent saves every 10 only; (2) ASYNC_CHECKPOINT_UPLOAD=True ->
dapo_ray_trainer uploads each checkpoint on a background thread, joined before the next save and before the run outcome (S3
commit marker last, so preemption mid-upload falls back to the previous step). Validation can't overlap the local save (both
need vLLM's GPU memory in opposite states; ES state saved in the checkpoint must include that step's validation); the upload
overlaps the next steps instead. Expected 30B overhead ~10%. Cutover (`cutover_qwen3_30b_ckptfix.sh`, detached): medium at
permanent step 350 (best 0.7575 @320, 1 miss; job_db38bg4f8c4g08flbe70), hard cut over early at the user's request from rolling step 195
(best 0.5154 @190; job_db38lnjmjdjg088cefb0) -> new supervisors `.scale_train_supervisors/q3-30b-rl-{med,hard}-20261007-ckptfix/`,
TOTAL_TRAINING_STEPS=500, same S3 prefixes / W&B ids, exact resume (Adam + data cursor + ES state). 4B runs unchanged.

---

## 2026-10-07 — Qwen3-4B-Base -> 30B-A3B distillation, step 1: top-128 teacher traces (ScaleTrain, 1 GPU per band)

Same protocol as the E4B-base -> 12B/26B control (DISTILLATION_EXPERIMENTS.md §9): untrained Qwen3-4B-Base@906bfd4b teacher,
Qwen3-30B medium + hard bands (JWei05/DeepScaleR-Easy-Medium-Hard-Qwen3-30B-A3B-Base-10k@be4dcd8b), 3,000 train q x 16 + 300 val
q x 1, temp 1 / top-p 1 / top-k -1, 8192 response tokens, RL prompt (0-shot ChatML + boxed instruction), token ids + top-128
logprobs. Specs q4b-base-{medium,hard} in run_gemma4_bestckpt_trace_collection.sh; ScaleTrain jobs q4b-tr-{med,hard}
(job_db2stgbmjdjg07lcroq0 / job_db2stgcf8c4g08flbdug, 1 GPU, borrowing, supervisor `tr-q4b-tr-*`), bundles
`s3://scale-ml/genai/rl-distill/qwen3-4b-base-traces-topk128-v1/q4b-base-{medium,hard}/` (S3-mirrored per shard, resumable).
Found in smoke: the untrained 4B samples Qwen3's LM-head padding ids (151,669-151,935; ~1 per 800k tokens), which fail trace
validation -> --restrict-to-tokenizer-vocab (-100 logit bias on padding ids + processed_logprobs; vLLM caps allowed_token_ids at
1024). Throughput is CPU-bound on top-128 processing: ~940 output tok/s per GPU, 32 prompts/shard -> 94 shards x ~11 min ~= 17 h
and ~21 GB per band. Smoke shard (32 medium q x 16): teacher strict acc 21%, mean response 1,211 tokens, 2.3% hit 8192.
**DONE 2026-10-07** (relaunched on 4 GPUs per band, GPUS=4): both bundles VALID + COMPLETE.json on S3 -- medium 48,300 rows /
56.7M response tokens (20.6 GB train, COMPLETE 18:48 UTC), hard 48,300 rows / 60.6M tokens (22.1 GB, 19:17 UTC). The single-process
final validation alone takes ~80-85 min per bundle (4.3x the E4B bundles' 17 min, proportional to size).

---

## 2026-10-06 — Qwen3 medium-band DAPO RL LAUNCHED: 30B-A3B (8 GPUs) + 4B (4 GPUs), ScaleTrain borrowing

Supervisors `rl-q3-30b-rl-med` / `rl-q3-4b-rl-med` (+ `ckpt-janitor-*`), state `.scale_train_supervisors/q3-{30b,4b}-rl-med-20261006/`;
`start_qwen3_30b_a3b_medium_rl_supervisor.sh` (`QWEN_SIZE=4b` for 4B: FSDP2 DP4, vLLM DP4 TP1, no offload, no R3). W&B ids
`q3-30b-a3b-medium-es5-s42-v1`, `q3-4b-medium-es5-s42-v1`. Launch fixes on the way: VERL_FSDP2_LOCAL_LOAD per family (b9a57b31,
30B untied); torchcodec 0.15.0 pin (a9400d35/4467c8a9, PyPI 0.17 broke `import vllm` on cu129 pods); vLLM parked during checkpoint
saves (0c32176b) -- BOTH runs OOMed at their first permanent (HF-export) save at step 10 with vLLM awake after validation (63.75 /
66.15 GiB of 79 on GPU 0); after the fix both step-10 permanent saves completed. Step 0 -> 10 val mean@16: 30B 31.1 -> 32.8, 4B 21.0
-> 22.1 (30B val matches the teacher-band expectation 31.3). 30B: ~250 s/step (gen 80 / old-logprob 32 / update 113 / sync 16), R3
route agreement 98.6%, aux_loss 12.6 (balanced 8), 0.3% clipped; rolling ckpt 366 GB: local 2.5 min + upload 5.2 min (1.2 GB/s);
restore 17 min (0.36 GB/s download); permanent save ~13 min. 4B: ~150 s/step, 2.6% clipped, 48 GB checkpoints.

---

## 2026-10-06 — Qwen3-30B-A3B-Base medium-band DAPO RL: PREPARED, NOT LAUNCHED

Gemma 4 26B-A4B recipe (GRPO n=16, bsz 64 / mini 32, lr 1e-6 + 20 warmup, token-mean, clip 0.2/0.28, 8k + 2k overlong,
token TIS 2.0, val every 10 on the 300 x16 medium validation split, early stop patience 5 on mean@16, 400 steps max) on the
Qwen3-30B medium band (train 3,000 / val 300 of the new HF dataset; prompt = teacher's 0-shot ChatML + boxed instruction,
verified identical to the 30B sampler's prompt for all 3,300 questions). One p5 node, borrowing on, priority high.
Files: `scale_train/launch_qwen3_30b_a3b_medium_rl.sh`, `scale_train/start_qwen3_30b_a3b_medium_rl_supervisor.sh`
(supervisor + rolling-checkpoint janitor), `data/prepare_deepscaler_qwen3_30b_difficulty_rl_data.py`; the Gemma 4 run-file
gained a `Qwen/Qwen3-30B-A3B-Base` branch (Qwen3MoeDecoderLayer, qwen3 jinja, stop ids 151645/151643/151644 + "\nHuman:" /
"\nuser\n" via new `VERL_ROLLOUT_EXTRA_STOP_TOKEN_IDS` and `\n` escapes in `VERL_ROLLOUT_EXTRA_STOP`).
Deltas vs Gemma: GEN_TP=2, KV 12 GiB/GPU, MAX_ACTOR_CKPT_TO_KEEP=2. Update (same day): R3 router replay added for Qwen3-MoE
(`verl/models/transformers/qwen3_moe_router_replay.py`, CPU tests exact on native routes) and enabled; early stopping
`EARLY_STOPPING_MISS_TOLERANCE=0.01` (a miss = more than 1 point below the best; 5 consecutive misses stop; within 1 point
resets); rolling checkpoint every 5 steps, permanent every 10. MoE router metrics (monitoring only, not in the loss) on W&B as
`actor/moe/{aux_loss,z_loss,router_entropy,max_expert_load,max_expert_load_worst_layer,min_expert_load,dead_expert_frac,
r3_route_agreement}` from the actor-update forwards (padding masked, layer-averaged; aux_loss = top_k = 8 when balanced).
Checkpoints (original plan, superseded): rolling S3 slot after every step (~366 GB: fp32 shards + Adam) + permanent every 10; full_checkpoint_s3.py now
transfers files in parallel (16 files x 8 parts; devbox test 1.12 GB/s vs 0.69 single-file, devbox NIC-bound). Local
DRY_RUN composed the config end to end. Unvalidated on GPUs: memory layout, step time, checkpoint upload time on a pod.
S3: `s3://scale-ml/genai/rl-distill/qwen3-30b-a3b-base-rl{,-full-checkpoints}/…`; W&B id q3-30b-a3b-medium-es5-s42-v1.

---

## 2026-10-06 — HF dataset `JWei05/DeepScaleR-Easy-Medium-Hard-Qwen3-30B-A3B-Base-10k` (public)

Replica of `JWei05/DeepScaleR-Easy-Medium-Hard-Gemma-26B-PT-10k` with Qwen3-30B-A3B-Base (n=8 run above) as the teacher:
hard = 1/8, medium = 2-3/8, easy = 4-8/8 (pools 6,440 / 9,315 / 13,859); 3,300 per band by SHA-256 rank (seed 42), then
300 validation / 3,000 train (seed 43); same verl schema, manifest and README layout. Builder
`data/build_deepscaler_qwen3_30b_difficulty_dataset.py` (copy of the Gemma builder with teacher constants swapped);
grade source `/mnt/efs/jasonwei/verl/data/deepscaler_qwen3_30b_a3b_base_n8_dedup_v2_20261005/questions_with_current_rl_grades.parquet`
(sha 84965510…, also in the 30B S3 prefix); Hub commit be4dcd8b. Rows hold the plain problem (as in the Gemma version);
the teacher's 0-shot instruction is in `extra_info.teacher_prompt_instruction` and must be appended at tokenization.
Not the same questions as the 2026-10-05 300-q band eval sets (those used easy = 4-7/8 and a numpy RNG draw).

---

## 2026-10-05 — Qwen3-1.7B/4B-Base on 8B- and 30B-defined difficulty bands (n=16)

Bands from each teacher's own strict k/8 on full DeepScaleR: easy 4-7/8, medium 2-3/8, hard 1/8; 300 random questions
per (teacher, band), seed 42 (`data/build_qwen3_teacher_band_sets.py`; sets at
`s3://scale-ml/genai/rl-distill/qwen3-deepscaler-samples/band_sets/`). Students sampled with the identical protocol
(n=16) on ScaleTrain job `job_db1qvfpmen9g08cv9ih0` (4 GPUs, borrowing; `scale_train/run_qwen3_band_evals_st.sh`, ~37
min; a local attempt failed because other users' jobs held 72+ GiB on 7/8 GPUs). Strict grading
(`data/grade_band_eval_samples.py`, `data/summarize_qwen3_band_evals.py`); outputs + grades + `band_table.md` under
`.../qwen3-deepscaler-samples/band_evals/`. Teacher mean@8 is selection-biased (the sets were picked with those samples).

| set | teacher mean@8 | 1.7B mean@16 | 1.7B pass@16 | 4B mean@16 | 4B pass@16 |
|---|---|---|---|---|---|
| 8B easy | 64.6 | 29.2 | 88.3 | 39.4 | 98.3 |
| 8B medium | 30.5 | 11.7 | 63.7 | 21.2 | 90.7 |
| 8B hard | 12.5 | 4.0 | 37.7 | 9.4 | 68.0 |
| 30B easy | 65.9 | 26.2 | 84.3 | 34.7 | 97.3 |
| 30B medium | 30.6 | 8.3 | 54.3 | 16.6 | 82.0 |
| 30B hard | 12.5 | 3.7 | 35.7 | 7.6 | 58.3 |

pass^k (all k correct, C(c,k)/C(n,k)): pass^4 = 1.7B 2.7/0.4/0.0, 4B 4.6/0.9/0.1 on 8B easy/medium/hard and 1.7B 3.0/0.2/0.0, 4B 3.3/0.4/0.0 on 30B easy/medium/hard; pass^16 = 0 everywhere (best question 14/16; only 4-17 of 300
questions per set have all 16 samples boxed). Boxed ~74-78% for both students (teachers 88-89%), truncation 1-3%. 30B-defined bands are harder for both students
than 8B-defined bands at equal teacher accuracy (30B solves questions the 8B and smaller models don't).

---

## 2026-10-04 — Qwen3 base family chosen; Qwen3-8B-Base DeepScaleR sampling (n=8) on ScaleTrain

Switched the controlled student/teacher study to the Qwen3 Base family (1.7B / 4B / 8B; same 36T pretraining tokens
per the Qwen3 report). First job: Qwen3-8B-Base on the full deduplicated DeepScaleR set (38,796 questions,
`deepscaler_it_gen_dedup_v2.parquet`), 8 samples/question, temp 1.0, top_p 1.0, 8192 max response tokens, 0-shot native
ChatML prompt + "Please reason step by step, and put your final answer within \\boxed{}." (bare question boxed only 54%
in a smoke test, with Python-program/Chinese drift; the instruction gives ~87-89% boxed). Stops: `<|im_end|>` token id
plus plain-text "\\nHuman:" / "\\nuser\\n" turn starts. ScaleTrain 4 GPUs (borrowing, high), DP4 via
`scale_train/run_deepscaler_samples_st.sh` (S3-resumable chunk parquets, supervisor relaunch). Bugs fixed before the
real run: untracked generator not shipped; image .venv flash_attn.cute crash (pin TRITON_ATTN); `${VAR:-...}` default
mangled `\\boxed{}` to `\\boxed{.}`. First 1,024 questions: 89% boxed, strict mean@8 28.9%, pass@8 70.8%.
Output: `s3://scale-ml/genai/rl-distill/qwen3-deepscaler-samples/qwen3-8b-base-0shot-qwenmath-n8-8k/` (`all.parquet`,
`RUN_COMPLETE.json`). **Done 2026-10-05 05:05 UTC.** One prompt (source_index 32326, 3,320 tokens) exceeded the original 2048
prompt cap; cap raised to 4096 (resume accepts raised caps). Final strict (boxed-only) grading, 38,796 q x 8:
mean@8 28.52%, pass@8 71.10%, boxed 89.0%; per-question k/8 = 0: 28.9%, 1: 18.8%, 2: 13.4%, 3: 10.6%, 4: 8.9%, 5: 7.8%,
6: 6.2%, 7: 3.9%, 8: 1.4%. Lenient grading (whole-output extraction) agreed to within 0.3 pt; strict kept.
Qwen3-30B-A3B-Base (TP2 x DP2, same protocol) auto-launched 05:06 UTC, **done 10:22 UTC** ->
`.../qwen3-30b-a3b-base-0shot-qwenmath-n8-8k/`: strict mean@8 33.91%, pass@8 76.33%, boxed 87.7%, mean response 985
tokens; k/8 = 0: 23.7%, 1: 16.6%, 2: 12.6%, 3: 11.4%, 4: 10.0%, 5: 9.5%, 6: 8.0%, 7: 5.9%, 8: 2.3%. Per question vs 8B:
30B higher 42.5%, equal 33.5%, lower 24.0%; 4,570 solved only by 30B, 2,541 only by 8B; corr(k) 0.744.
30B response lengths (tokens; strict per-sample grades re-derived 2026-10-06 -> `grades_strict.parquet` in the same prefix):
all samples mean 985 / median 675 / std 1171 / p99 7574, 0.87% hit 8192; medium band (30B k=2-3, 9,315 q) mean 1026 /
median 725 / std 1128, 0.70% hit 8192 (correct samples 0.13%, incorrect 0.95%), 1.27% exceed 6144 (Gemma 4 RL's
overlong-penalty onset); ~60% of capped samples end in a low-entropy repetition loop (zlib tail ratio < 0.25).

---

## 2026-10-03 — Ouro looped-LM baseline evals (1 vs 4 loops, 12-shot vs 0-shot)

Start of a controlled study with ByteDance Ouro-1.4B / Ouro-2.6B (planned: small model with no loop vs larger model
with untied per-loop parameters). Baseline: both models × {1, 4 (shipped)} recurrent steps × {12-shot, 0-shot}, on the
three 300-question DeepScaleR band validation sets, 4 samples/question, temp 1.0, 8192 tokens, boxed-only math_verify.
Run locally on GPUs 3–7 via `run_ouro_loop_evals.py` (loop count through vLLM `hf_overrides`; HF-vs-vLLM parity
checked at 1 and 4 steps). **Done, 24/24.** Key results (mean@4, easy/medium/hard): 2.6B 4-step 0-shot
68.6/55.5/41.0, 1.4B 4-step 0-shot 62.2/51.4/38.2; 12-shot is 20–29 points worse on medium/hard; 1-step ≈ 0–11
because the 1-step distribution is near-uniform at temperature 1.0 (argmax still coherent). Full table + findings:
[`OURO_LOOP_STUDY.md`](OURO_LOOP_STUDY.md). Traces/metrics: `/tmp/ouro_eval/results_k4/` (local NVMe only).

---

## 2026-08-06 — Borrowing-enabled extra runs have durable full-checkpoint recovery

Three high-priority borrowing jobs were submitted: 12B Medium
`job_d9qhmp5cufhjo4csicv0`, E4B Medium `job_d9qhmp55herl6oiqnf6g`, and the
four-GPU E2B Easy continuation `job_d9qhpj55herl6oiqnf70`. All three were
accepted and initially `QUEUED` with no worker pod. Dedicated tmux supervisors
monitor pod phases and relaunch a failed/preempted attempt.

Every new step-20 checkpoint is uploaded to a run-specific S3 prefix with
model, Adam, scheduler/RNG extras, and `data.pt`; a completion marker is written
last and only manifest-complete checkpoints are restored. The old E2B Easy
step-200 full checkpoint was lost with its pod's `/tmp`, and the HF export is
weights-only, so its first extra-100-step attempt necessarily uses fresh Adam
plus a deterministic 200-batch dataloader fast-forward. Subsequent resumes are
exact after the first new full checkpoint. Focused checkpoint/recovery tests
pass (`4 passed`) and Ruff passes.

---

## 2026-08-05 — Repaired 12B production startup and step 1 passed

Replacement 12B production job `job_d9pibql5heri67n90crg` waited on its pinned
guaranteed-quota H100 node until 15:48 UTC, then started with zero pod restarts.
All eight vLLM servers initialized and the repaired text-only FSDP-to-vLLM sync
passed the exact point where the predecessor failed on
`embed_vision.pos_embedding`. W&B run `8t55mm42` started and the full initial
validation completed at 18.7375% in-distribution, 46.7887% GSM8K, and 16.7500%
MATH500.

The first real 12B optimizer update then completed in 824.7 seconds. Gradient
norm was finite at 12.94; token-TIS truncation was `9.9419e-06`; PPO ratio
metrics were present; and reward groups were 40/64 mixed, 24/64 all zero, and
0/64 all one. Generation took 90.8 seconds, old-log-probability computation
87.0 seconds, actor update 640.3 seconds, and weight synchronization 3.2
seconds. The repaired 12B verl path is therefore proven through step 1; its
required step-20 checkpoint/export remains under continuous monitoring.

Step 2 subsequently completed in 793.1 seconds and included an actual
8,192-token response. Gradient norm remained finite at 13.42, TIS truncation
was `8.3494e-06`, PPO ratio clipping was nonzero, and the pod still had zero
restarts. Reward groups were 35/64 mixed, 29/64 all zero, and 0/64 all one.

---

## 2026-08-05 — E4B production passed step-20 checkpoint/export

Replacement E4B production job `job_d9pht75cufhik662291g` completed step 20,
saved `global_step_20`, uploaded a new full HF export, completed the scheduled
in-distribution/GSM8K/MATH500 validation, and continued through step 21 with
zero pod restarts. The new HF commit is
`f5993c209242fca83cf42d271c9122763f18da78`; its `step_000020` tree contains
seven files totaling 17,366,959,359 bytes, including the 17,334,775,284-byte
`model.safetensors`. Step-20 validation reached 22.425% in-distribution,
50.634% GSM8K, and 20.425% MATH500. Token TIS, PPO ratio, and reward-group
metrics remained present; step 20 reported 37/64 mixed groups, 24/64 all zero,
and 3/64 all one.

---

## 2026-08-05 — Packed E2B production passed step-20 checkpoint/export

Both four-GPU children of `job_d9phvft5herivi4ihheg` completed the scheduled
step-10 in-distribution/GSM8K/MATH500 sweep and continued training. Easy changed
from 5.4125% to 5.7625% in-distribution, 7.7765% to 8.0797% on GSM8K, and
4.5625% to 5.0125% on MATH500. Medium changed from 3.2125% to 3.4000%
in-distribution and 8.1014% to 8.5888% on GSM8K; MATH500 was effectively flat
at 4.6875% versus 4.6500%.

Medium step 8 logged a non-finite pre-clip gradient norm while every loss,
rollout-correction, PPO-ratio, and reward metric remained finite. The active
FSDP engine's anomaly guard zeroed the gradients and skipped that optimizer
update, so weights were not corrupted. Steps 9 and 10 then completed with
finite gradient norms 5.19 and 5.29, respectively, including an 8,192-token
response at step 9. Easy continued through step 11 with finite metrics.

Both children later completed step 20, ran the second scheduled validation,
saved the local checkpoint, and uploaded a new full HF export. Easy's new commit
is `9ce950247f3e9e56df942f397b668a87a8dccb60`; Medium's is
`f329ac40b18154b4f46171145e31cb19eed008f4`. Each `step_000020` export has seven
files totaling 11,084,114,210 bytes, including the 11,051,930,326-byte
`model.safetensors` plus tokenizer, processor, config, generation config, and
chat template. Easy step-20 validation reached 6.3375% in-distribution, 10.0292%
GSM8K, and 5.8750% MATH500. Medium reached 3.7625%, 8.9028%, and 5.2125%,
respectively. Easy continued through step 21 after the save.

Both packed children subsequently reached step 40 and produced another full
HF export. Easy commit `a87df6d5c09abca1795fbbaf9089748b317b59f3`
landed at 15:34 UTC and Medium commit
`be22f986c8f7197521bb32177a599a77690da879` at 15:41 UTC. Each
`step_000040` tree contains seven files totaling 11,084,114,210 bytes, including
the 11,051,930,326-byte `model.safetensors`.

---

## 2026-08-05 — 12B text-only vLLM weight sync skips unused vision branch

The first replacement 12B job reached vLLM initialization but failed its
initial FSDP-to-vLLM synchronization because the training model emitted
`embed_vision.pos_embedding`, which vLLM's Gemma 4 Unified model intentionally
does not expose. The rollout is text-only, so the existing optional multimodal
reload filter now recognizes Gemma 4's `embed_vision` naming, and the 12B matrix
profile enables that filter explicitly. Language-model weights remain synced.
The failed job was `job_d9phq3t5herivi4ihhe0`; it was canceled successfully and
the tested replacement is `job_d9pibql5heri67n90crg` (borrowing disabled, high
priority). The focused vLLM-filter and matrix suite passes 22/22.

At 2026-08-05 13:46 UTC the replacement created worker pod
`g4-12b-easy-med-full-jasonwei-2026080-bsyyl-workers-0-0-vldn4`. The pod is
still `Pending` with zero restarts and no container startup: it is pinned to a
currently occupied H100 node, while the autoscaler reports max node-group size.
This is a capacity wait, not a recurrence of the weight-sync failure.

---

## 2026-08-05 — Replacement E4B step 1 confirms throughput improvement

E4B production job `job_d9pht75cufhik662291g` completed step 1 in 559.3
seconds, down from roughly 830–850 seconds/step on the preceding production
run. Generation took 26.2 seconds, old-log-probability computation 69.6 seconds,
and the actor update 454.8 seconds. The step processed 1,024 responses with
finite loss and gradient norm. TIS truncation was `1.51798e-05`, PPO ratio
metrics were present, and the reward-group split was 35/64 mixed, 29/64 all
zero, and 0/64 all one. After three completed steps, the new run's median is
562.9 seconds/step versus 837.9 seconds over 38 completed steps in its exact-name
predecessor W&B run `xlc5hevs`, a 32.8% step-time reduction.

---

## 2026-08-05 — Packed E2B microbatch-8 production step 1 passed

The Easy child of packed production job `job_d9phvft5herivi4ihheg` completed a
real four-GPU step 1 in 311.9 seconds with microbatch 8 and the 12,288-token
adaptive ceiling. It included an 8,192-token response, finite gradient norm
`4.90`, zero TIS truncation, and PPO clip fraction `2.96e-06`. Generation took
34.7 seconds, old-log-probability computation 37.1 seconds, and actor update
232.0 seconds. Reward groups were 18/64 mixed, 46/64 all zero, and 0/64 all
one. This is the production-scale confirmation of the local throughput canary.
The Medium child independently passed step 1 in 310.6 seconds with an 8,192-
token response, finite gradient norm `13.32`, zero TIS truncation, and a group
split of 10/64 mixed, 54/64 all zero, and 0/64 all one. The matched production
comparisons now establish the throughput gain beyond the one-step canary: Easy
has a 293.5-second median over its first six steps versus 893.3 seconds over 38
steps in exact-name predecessor `zcuagr1z` (67.1% lower), while Medium has a
299.6-second median over its first five steps versus 842.6 seconds over 38 steps
in exact-name predecessor `bsfow2oe` (64.4% lower).

---

## 2026-08-05 — E2B microbatch-8 / 12,288-token adaptive canary passed

A four-GPU Gemma 4 E2B canary completed a full optimizer update with compiled
vLLM rollout, an 8,192-token response limit, microbatch 8, and a 12,288 padded-
token adaptive ceiling. It wrote the complete `global_step_1` checkpoint and
ended with `RUN_DONE rc=0`. Step time was 140.0 seconds including 54.5 seconds
of checkpointing; generation took 46.7 seconds, actor update 31.0 seconds, and
old-log-probability computation 4.17 seconds. Token TIS, PPO ratio clipping, and
reward-group count/percentage metrics were present and finite. The E2B packed
production profile now uses microbatch 8 and the 12,288-token ceiling.

---

## 2026-08-05 — Math reward uses math_verify OR bounded Miles SymPy

The strict last-box reward now marks an answer correct when either Hugging Face
`math_verify` or the Miles/DeepScaleR SymPy equivalence grader accepts the same
extracted final box. `math_verify` runs first; SymPy runs only after a miss and
has a five-second worker-local timeout so pathological simplification cannot
stall the reward pool. A stratified 1,800-trace audit found 11 grader
disagreements, all manually valid equivalents; the OR recovered the one
SymPy-only case (`(15)x - 80.` versus `15x - 80`) while retaining the ten
`math_verify` acceptances. The verl focused suite passes 19/19 and the NeMo-RL
parity gate passes 8/8.

---

## 2026-08-05 — Math reward now grades the last well-formed box

The boxed-only `math_verify` path now grades the last balanced `\\boxed{}` or
`\\fbox{}` in a completion. A completion with no well-formed box still scores
zero, but an earlier boxed answer no longer invalidates a later correction. The
majority-vote prediction extractor and the NeMo-RL parity port use the same
selection rule. Historical results documented below retain the exactly-one-box
semantics used when those evaluations were run.

---

## 2026-08-05 — Production image repaired; E4B KV-cache failure fixed selectively

**Image repair.** The production image had excluded `verl/third_party` because
the Docker ignore rule `third_party` matched both the root dependency checkout
and the package directory. The rule is now scoped to `third_party/**`, and the
Dockerfile asserts that `verl/third_party/vllm/__init__.py` exists. A direct
container import verified the packaged module.

**E4B failure and recovery.** E4B job `job_d9p7utt5heri8vqpv6v0` failed at
vLLM startup because a 512 MiB explicit KV cache was too small for the
12,288-token window; vLLM required about 0.66 GiB. E4B and 12B now use a 1 GiB
explicit cache. The selectively relaunched E4B job is
`job_d9p8f4d5heri8vqpv720`; it passed the previous crash point, initialized all
eight vLLM servers and the agent-loop manager, and entered the 200-step trainer.
The focused matrix suite passes (`7 passed`).

**Current jobs.** Healthy jobs were not reset. The final monitored set is 12B
`job_d9p7ntd5heri67n90cd0` (`IN_PROGRESS`, worker capacity-pending), E2B packed
4+4 `job_d9p87355heri67n90ceg` (`IN_PROGRESS`, worker running), and E4B
`job_d9p8f4d5heri8vqpv720` (`IN_PROGRESS`, worker running). All retain EKS,
high priority, borrowing disabled, 200 steps, validation every 10 steps, and
checkpoint/Hugging Face export every 20 steps.

---

## 2026-08-03 — Local 8K gate passed; three production jobs launched

**Local gate.** Gemma 4 E2B Easy-10k completed ten real optimizer steps on four
H100s with the production 4,096/8,192 prompt/response window, microbatch 2, 64
prompts x16 responses, and explicit 512 MiB vLLM KV cache. All ten steps had
finite loss, gradient, and reward metrics; steps 1, 3, 4, 5, 8, 9, and 10 each
included at least one genuine 8,192-token response. Step 10 also completed the
full in-distribution/GSM8K/MATH500 validation sweep. The complete step-10 FSDP,
optimizer, extra-state, and Hugging Face checkpoint was written, the container
exited zero, and the log ended with `RUN_DONE rc=0`.

**Production topology.** Six logical 200-step runs resolve to three
borrowing-false, high-priority, full-node EKS jobs. One 12B job runs easy then
medium sequentially; one E4B job does the same; the E2B job runs easy and
medium concurrently on disjoint four-GPU halves. Validation remains every ten
steps and checkpoints/HF pushes every twenty. The three production jobs were
submitted after the local gate passed and were initially queued with no worker
pods admitted:

- 12B sequential: `job_d9o8g16rkvumpnejdjgg`
- E4B sequential: `job_d9o8hcurkvum8ubf6hn0`
- E2B packed 4+4: `job_d9o8i4ihu8ojb1b2cua0`

The previous remote smoke jobs were canceled successfully.

**Queue reset.** At the user's request on 2026-08-03, the three production jobs
above were canceled and immediately relaunched to move the matrix to the back
of the queue. Every cancellation returned `Result: True`; the briefly admitted
12B worker was still `Pending` and disappeared before relaunch. Replacement
jobs retain EKS, high priority, borrowing disabled, one full node per topology,
200 steps, validation every 10 steps, and save/export every 20 steps:

- 12B sequential: `job_d9oegh2hu8ojnqcvi9u0`
- E4B sequential: `job_d9oeimahu8ojnqcvi9ug`
- E2B packed 4+4: `job_d9oekfahu8ojnqcvi9v0`

All three replacements were initially `QUEUED` with no worker pods.

**Second queue reset.** At the user's request on 2026-08-04, those three
queued replacements were canceled and immediately resubmitted once more. All
three old job records now report `CANCELED`. The new jobs preserve the same
EKS/high-priority/borrowing-disabled production configuration and were
verified `QUEUED` after submission:

- 12B sequential: `job_d9ol3655hergioua77u0`
- E4B sequential: `job_d9ol47t5hergioua77ug`
- E2B packed 4+4: `job_d9ol4ut5hergioua77v0`

**Entry points.** The sequential 12B/E4B wrapper is
`scale_train/run_gemma4_easy_medium_sequential.sh`; E2B remains in
`scale_train/run_gemma4_e2b_easy_medium_packed.sh`. The matrix launcher resolves
all six runs into the three intended jobs, and focused tests pass.

---

## 2026-08-02 — Gemma 4 Easy-10k/Medium-20k six-run RL matrix prepared

**Scope.** Prepared but did not submit six seed-42 DAPO runs: Gemma 4 12B, E4B,
and E2B on both DeepScaleR Easy-10k and Medium-20k. The 12B/E4B runs reserve one
eight-GPU node each. The two E2B runs share one full node with four GPUs each,
isolated CUDA visibility, Ray directories, vLLM port ranges, caches, W&B dirs,
and checkpoint roots. This is five ScaleTrain jobs containing six logical runs.

**Validation contract.** Each run evaluates its matching 500-question
in-distribution validation set at x16 (8,000 rows), GSM8K's 1,319 questions at
x7 (9,233 rows), and MATH-500's 500 questions at x16 (8,000 rows), before
training and every ten steps. All repeated rows carry stable per-question UIDs.
The pinned preparation completed end to end and verified train/validation
non-overlap, exact row counts, uniform repeat counts, and a SHA-256 manifest.

**Training contract.** The matrix retains the current 8k Gemma 4 recipe:
64 prompts x16 responses, mini-batch 32, LR 1e-6 with 20 warmup steps, 4,096
prompt + 8,192 response tokens, 2,048-token soft-overlong buffer, strict boxed
reward, save every 25 steps, and a 500-step production horizon. 12B/E4B use
microbatch 1; E2B uses the validated configured microbatch 2 with adaptive
singleton splitting above a 4,096 padded-token ceiling. All use FSDP2 CPU
offload policy. The new 12B path selects `Gemma4UnifiedTextDecoderLayer` and is
gated behind a one-step canary because it has not previously trained in this
repository.

**Entry points.** See `GEMMA4_DIFFICULTY_RL_RUNS.md`,
`scale_train/launch_gemma4_difficulty_matrix.py`, and
`scale_train/run_gemma4_e2b_easy_medium_packed.sh`. The default launcher action
is dry-run; `--launch` is required for submission. Focused tests pass, and all
five smoke-job commands resolve against the full-node EKS build configuration.

---

## 2026-07-31 — Finite Gemma 4 gradient spikes accepted; deterministic-cuDNN experiment withdrawn

**Operator decision.** Large but finite Gemma 4 pre-clip gradient norms are not a stop condition for
this experiment. The scoped `torch.backends.cudnn.deterministic=True` change and its report-v5
contract were withdrawn. Its clean eight-GPU gate produced a finite norm of `127.41`; this is now
retained as an observed Gemma 4 training value rather than classified as numerical corruption.

**Production contract.** Training returns to the normal verified stack from `9a3e103f`: cuDNN SDPA
for training target parity, non-cuDNN SDPA for interleaved validation, singleton microbatches, and
the 4,096 padded-token ceiling. The finite gradient-magnitude ceiling is disabled by default while
gradient norms remain logged. Non-finite loss, non-finite gradients, invalid probability masses,
crashes, stalls, checkpoint failures, and upload failures still fail closed and are monitored by the
supervisor. A fresh clean receipt is required because production receipts bind the repository
commit and audited source hashes.

---

## 2026-07-31 — Validation-interleaved cuDNN backward anomaly isolated and fixed

**Failure beyond singleton batching.** A clean-HEAD three-step W&B smoke (`5b475fb2`) used the
audited singleton-microbatch contract and still reached gradient norm `74.66` on step 3 after
validation ran at steps 0, 1, and 2. The fail-closed gate stopped before the optimizer update; no
checkpoint from that run is resumable.

**Isolation.** The same exact three train batches with only step-0 and final validation passed in
W&B run `0ea93c05` at gradient norms `14.78`, `21.02`, and `12.61`, proving that the third batch and
optimizer sequence are safe without an intervening validation forward. Disabling cuDNN SDPA for
all forwards also kept gradients finite (`14.73`, `18.63`, `23.20`) but failed target parity:
weighted log-probability drift `0.01826` and sampled-token p95 `0.09447`.

**Fix and gate.** Training retains cuDNN SDPA so it matches the immutable BF16 target overlay;
in-process validation uses the non-cuDNN SDPA backend so it cannot contaminate the next backward.
The audit now executes the real optimizer/scheduler sequence and exact validation cadence. The
mixed-backend diagnostic passed with exact ordered top-128 support, weighted drift `0.000168`,
sampled-token p95 `0.000864`, and sequential gradient norms `14.78`, `21.20`, `14.95`. A clean-HEAD
receipt and fresh three-step W&B checkpoint smoke remain required before production relaunch.

---

## 2026-07-31 — E2B-to-E4B paired-microbatch anomaly isolated; singleton production contract

**Failure.** Fresh W&B smoke `b50c8b0e` used the BF16-forward/FP32-master stack and the previously
authorized microbatch-2/5,120-padded-token policy. Gradient norms were `13.62`, `20.16`, then
`356.42`; layers 0-1 `per_layer_input_gate` dominated the third step. The supervisor stopped before
the anomalous optimizer update, so this run is not resumable.

**Isolation.** An exact replay of the same first three deterministic train batches with per-
microbatch diagnostics reproduced the third-batch failure at gradient norm `62.96`. The different
magnitude establishes nondeterminism in the bad backward path. Microbatch 1 on the same batches
produced `14.84`, `21.26`, and `16.10`, completed validation at loss `0.20946`, and saved a complete
step-3 checkpoint. Disabling activation checkpointing or vocabulary-projection chunk checkpointing
did not repair paired batches; the evidence points to BF16 cuDNN SDPA backward when two padded
sequences share a microbatch, not the earlier NeMoRL shared-KV/use-cache bug.

**Fix and gate.** Production now requires microbatch 1, keeps cuDNN SDPA for target parity, and uses
a defensive 4,096 padded-token ceiling. The training-engine receipt is upgraded to bind the exact
runtime contract and exercise all first three seed-42 batches (384 distinct rows) rather than only
the first batch. Focused audit/verifier/launcher tests pass (`34 passed`), and the broader Gemma 4
data, launcher, supervisor, checkpoint, batching, gradient-diagnostic, and engine suite passes
(`252 passed`). A clean eight-GPU receipt and fresh three-step W&B smoke are still required before
launching the 750-step run.

---

## 2026-07-31 — Gemma 4 source traces complete; cross-engine target policy and preparation tooling

**Scope.** Preparation only for the two Gemma 4 distillation-vs.-RL lines. No 750-step
distillation, benchmark production, post-distillation RL, or model training run was started. The
canonical status and ambiguity ledger are in
[`GEMMA4_DISTILL_VS_RL_EXPERIMENTS.md`](GEMMA4_DISTILL_VS_RL_EXPERIMENTS.md).

**Complete local vLLM source bundles.** Both directions contain 48,615 train plus 1,000 validation
rows (1,216 train and 25 validation shards), five responses per registered question unit, exact
stored token IDs/response masks, and top-128 full-vocabulary-normalized vLLM targets:

| Direction | Dataset index | Experiment | Response tokens | Local root |
|---|---|---|---:|---|
| E4B-RL step 100 → E2B | `8b5712e0f5dea3388340a9bc91a6ceee40ff2ff990e66b7238090e240daeda6c` | `b515965fcdc31caebae1e71cf696731f4271182bdaca8af8326888d0b196af92` | 14,049,865 | `/lambda/nfs/Jason-scale/rl-distill-traces/gemma4-e4b-rl100-topk128` |
| E2B base → E4B | `d170166abad89588880f5c0a9eac43006f9a32bbe27cec28d7eb97c65288dbcc` | `f77e9dd611371030cc7752d4f4d0c92d4890448d9fe48594efdf3e223a5a409e` | 8,257,057 | `/lambda/nfs/Jason-scale/rl-distill-traces/gemma4-e2b-base-topk128` |

Both indexes preserve the historical 9,723-row train roster, 200 validation UIDs, 180 duplicate-text
train occurrences, and seven train/validation text overlaps. Clean evaluation must also report the
193-question non-overlap subset. Sampling is temperature/top-p 1.0, top-k disabled, with
4,096/8,192/12,288 prompt/response/context caps. The tokenizer and template hashes are
`f3ab24e73e9022f7b8d77113f543debd3779ef1e96c6452c68aaa9f3e6b81d17` and
`27b8801d8b61a413a9bb3b54b6f55e16217eff3e55f7c560377c8a162dd63c1c`.

**HF publication status.** Both source bundles are public and independently verified at immutable
commits:

| Direction | Public repository | Commit | Dataset index |
|---|---|---|---|
| E4B-RL step 100 → E2B | `JWei05/gemma4-e4b-rl100-topk128-traces` | `2b6e49a0a456ee9d67b16a1dc61785562bee90c9` | `8b5712e0f5dea3388340a9bc91a6ceee40ff2ff990e66b7238090e240daeda6c` |
| E2B base → E4B | `JWei05/gemma4-e2b-base-topk128-traces` | `e32aaa02681ae83b3d7256b1b155c9084da2f289` | `d170166abad89588880f5c0a9eac43006f9a32bbe27cec28d7eb97c65288dbcc` |

Each commit contains exactly 2,485 registered files. Independent remote verification compared every
path and size, LFS SHA-256 for every Parquet, and Git blob SHA-1 for every non-LFS file. The initial
private E2B upload had previously failed closed with `Private repository storage limit reached`, and
the corresponding early E4B upload-result log was empty; those facts remain historical rather than
current publication status.

**Cross-engine audit.** Expanded vLLM-versus-unsharded-HF diagnostics used exact stored sequences and
covered 32 traces per teacher: 1,870 E2B positions and 1,975 E4B positions. The two unsharded-HF
paths (native Gemma 4 forward and manual projection) had max absolute error 0. Tie-safe top-1 was 0.99572/0.99747,
top-128 overlap 0.98504/0.97654, weighted absolute log-probability delta mean 0.017996/0.010180,
probability-L1 mean 0.017343/0.010221, and sampled-token delta p95 0.09419/0.06082 for E2B/E4B.
Both pass the calibrated diagnostic thresholds, but the engines are not bit-identical.

**Target policy.** Keep each vLLM bundle immutable as the generation record. The intended primary
distillation path preserves the exact stored IDs and derives a separate unsharded-HF top-128 target overlay
with full-sequence teacher forcing, BF16+SDPA, causal predecessor scoring, Gemma 4 softcapping, and
FP32 full-vocabulary normalization. This remains offline precomputation: no online teacher and no
re-tokenization. A real FSDP2-engine target audit is still required before claiming numerical
equivalence to the distributed training forward.

**Prepared code.** Added the immutable-index-bound cross-engine audit, a disjoint resumable overlay
rescorer with exact chunked/native parity mode, focused CPU tests, and guarded operator documentation.
Added `preflight_gemma4_training_topk_overlay.py` and schema-based launcher routing: an overlay is
accepted only with its exact source index, parity receipt, target/student identities, and verified
one-to-one shard/row/token binding. The combined source/overlay/launcher suite passed 48 tests. The
trace generator now returns a distinct status for deterministic schema failures, and the supervisor
refuses identical-seed retries. The source-bundle uploader now commits only content-verified staged
snapshots to an otherwise empty destination branch, pins the observed parent commit, defaults to
private visibility, and suppresses raw provider exceptions so tokens cannot reappear in formatted
tracebacks. Bulk GPU rescoring, overlay generation/finalization, and overlay upload were not run.

**Status.** Both immutable source bundles are publicly published and remotely verified. Repository
review/test/commit/push remains preparation only. Integrated GPU audit, bulk rescore, student smoke,
both experimental lines, production evaluation, and post-distillation RL remain held.

---

## 2026-07-30 (fixed run, 100 steps) — verl val-curve parity EXACT; truncation bimodality does NOT reproduce

**Run.** `e1du1oyu` (`nemorl-dapo-gemma4-e2b-pt-DeepScaleR-4of4strict-seed42-8k-local4g-v2`), the
relaunch on the fixed stack (see ROOT CAUSE entry below), local 4×H100, ~4.5–6 min/step, ckpts at
25/50/75/100.

**Validation parity vs verl (`recbw9dcxso`) — every PARITY_CHECKLIST milestone matched:**

| step | 0 | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nemo (fixed) | 5.91 | 6.47 | 7.28 | 10.78 | 13.50 | 14.22 | 13.56 | 14.97 | 15.28 | 16.69 | **16.78** |
| verl ref | 6.16 | 6.00 | 6.78 | 10.0 | 12.59 | 14.19 | — | — | — | — | **16.59** |

Step-1 `probs_ratio` 1.000020; grad_norm sane throughout. The corrupted run (`ihdn67bj`) had
flatlined at 5.78 by step 30 — the fix is what unlocked learning, and the two frameworks now
agree to within mean@16 noise across the entire curve.

**Bimodality verdict (the repro's purpose), 99 training steps:** NeMo-RL clean steps (n=55)
grad_norm 0.25–3.81; truncated steps (n=44) 0.44–15.0. Spikes >3 concentrate on truncated steps
(7 vs 2), so a mild truncation-keyed heavy tail exists — but the regimes **overlap**, vs verl's
disjoint clean 1.3–6.8 / truncated 83–3108. `compare_grad_norms.py`: "regimes overlap — differs
from verl". With 44 truncated steps and zero events near verl's regime, on a run with exact val
parity, the evidence points to verl's extreme bimodality being **framework-specific behavior on
truncated sequences, not a property of the DAPO algorithm or the data**. Caveats: NeMo logs the
last of 2 mini-batch grad norms (hides ~half the candidate events, not all 44); vLLM sampler
numerics differ; distributional comparison only. (Supersedes the retracted `ihdn67bj` verdict —
same conclusion, but that one was measured on a corrupted objective and inadmissible.)

**Status.** Run continues open-ended past step 100 for curve extension; next verl-side step is
deciding whether to hunt the verl-only mechanism (e.g. rmpad/padded-path logit handling on at-cap
sequences) or accept the framework-specific classification.

---

## 2026-07-30 — NeMo-RL repro running LOCALLY on 4×H100 (gate PASSED 0.0591; 2 config/doc fixes)

**Goal.** Stand up the gemma-4 E2B NeMo-RL repro on a local 4×H100 box (192-222-53-241, driver 580 /
CUDA 13 — no cuda-compat needed) and get the full training run going, per the nemo_rl_repro README.

**Setup (fresh box, ~50 min end-to-end).** sudo apt cuda-toolkit-13-0 + cudnn9/libcudnn9-dev-cuda-13 +
librdmacm-dev; Kitware cmake 4.0.3; uv 0.12. Driver venv `/tmp/nemo-rl-venv`
(`UV_PROJECT_ENVIRONMENT=... uv sync --locked --extra automodel --no-install-package deep-ep`, torch
2.11.0+cu130, TE 2.15.0), worker venvs `/tmp/nemo-rl-worker-venvs`, HF cache `/tmp/hf-home` (E2B
prefetched), data via `prepare_deepscaler_4of4strict_rl_data.sh` → `/tmp/verl/data`. Launch wrappers
on the box: `/tmp/nemo-rl-local/{env.sh,build_venv.sh,run_gate.sh,run_train.sh}`. CPU gates re-passed
here: strict reward 6/6, tokenization 5/5 (via the real nemo_rl processor path in the py3.13 venv).

**Fixes (each root-caused).**
1. README prereq gap: `git submodule update --init third_party/nemo-rl` is not enough — the uv sync
   dies on `nemo-gym ... not a workspace member`. Nested submodules required: **`--recursive`**
   (Gym/Automodel/Megatron-Bridge). README updated.
2. `checkpointing.keep_top_k: null` in the committed repro yaml **fails MasterConfig pydantic
   validation** on nemo-rl 5f89b3ae (`CheckpointingConfig.keep_top_k` is `NotRequired[int]` — absent
   OK, explicit null not; the pod gate must have run a pre-commit working copy). Fixed in-config:
   `keep_top_k: 1000000` (keep-all intent, valid int).
3. First 4-GPU gate: val PASSED then the train step **OOM'd** (12.0 GiB fp32 alloc, 11.45 free).
   Root cause chain: at 4 ranks the FSDP shards are 2× the 8-GPU pod run's, and the 12 GiB transient
   is a **full fp32 [12288, 262144] logits tensor in train()** — upstream nemo-rl bug:
   `LossPostProcessor.__call__` (`nemo_rl/models/automodel/train.py`) never forwards `chunk_size` to
   `prepare_loss_input`, so `policy.logprob_chunk_size: 4096` is silently ignored in the training
   loss path (honored only in the logprob pass). Zero-edit fix: **`policy.dtensor_cfg.cpu_offload=true`**
   (CLI override in the local wrappers; gradient-identical, exercised in their own grpo-deepscaler
   recipes) — frees the static shard memory so the transient fits.

**Gate (4 GPUs, full 3200-sample val): PASSED.** step-0 `validation/accuracy = 0.0591` ∈ [0.045,
0.075] (pod 0.0550, old local 2-GPU 0.0508); reference policy auto-skipped; train step 1 completed:
loss 0.2581, avg reward 0.0576, mean gen length 194.7 tok, 346 s/step (81% policy_training — the
cpu_offload tax; generation 41 s).

**Run `ihdn67bj` (36 steps) — KILLED: training was silently corrupted from step 1** (see the
follow-up entry below). Its val curve (5.91→6.22→6.25→5.78 at 0/10/20/30 vs verl 6.16→6.00→6.78→10.0)
is a *no-learning* baseline, not a comparison point; step-25 ckpt is garbage-trained (discardable).

---

## 2026-07-30 (later) — ROOT CAUSE: act-ckpt + gemma-4 KV-shared layers = garbage training forward

**Symptom (user-flagged).** `train/probs_ratio` bulk 0.05–0.45 with min 0 / max up to 1.7e6, and
`train/probs_ratio_clamped` pinned at ~0.80 (the lower PPO clip bound) — deterministic from step 1
(gate + full run logged identical values), while `gen_kl_error`/`policy_kl_error` ≈ 8e-4 (tiny).
Since those two metrics compare prev↔generation (NOT curr), and the step-1 train-data dump shows
prev↔gen agreeing to −0.0008 ± 0.04 nats over 199k tokens, the **training forward (`curr`) was the
corrupted side** — and the loss consumes the same `ratios` tensor, so training optimized garbage.
(Step-1 LR is 1e-14 under LinearLR warmup: both mini-batches are at numerically identical weights,
so real policy movement is excluded.)

**Root cause (proven by offline CPU repro, `/tmp/nemo-rl-local/repro_curr_vs_prev.py`).** gemma-4
E2B has KV-shared layers (`num_kv_shared_layers > 0`). nemo-rl PR #2224 makes every trainer forward
request `use_cache=True` because on transformers < 5.5.2 the shared layers otherwise fall back to
untrained K/V projections (upstream **Automodel#1705** — the exact same bug, root-caused by NVIDIA in
April). But HF transformers 5.5.0 **force-disables the cache in grad-mode when gradient checkpointing
is on** (our worker logs: "`use_cache=True` is incompatible with gradient checkpointing. Setting
`use_cache=False`"). Repro on identical base weights, dumped step-1 sequence: use_cache=False forward
diverges from the dumped logprob-pass values by **−10.4 nats mean** (98.9% of loss ratios < 0.8, min
0 — the exact wandb signature); flipping only use_cache=True matches to **−0.0001 ± 0.04** (ratios
1.0007). So `activation_checkpointing: true` (the 07-29 pod OOM "fix") silently poisons ALL gemma-4
E2B training on the locked transformers 5.5.0; no pod run ever logged a train step, so it went unseen.
**The 8-GPU ScaleTrain config/image has the same poison — do not launch pods until fixed there.**

**Fix stack (local, all verified):**
1. transformers **5.5.0 → 5.5.4** in the driver + DTensor-policy-worker venvs (within nemo-rl's
   `>=5.5.0,<5.6.0` pin; HF #45312 in ≥5.5.2 dissociates KV sharing from the cache — nemo-rl's own
   TODO endorses this). Repro on 5.5.4: use_cache=False now matches (−0.0001 ± 0.04); tokenization
   parity re-passed 5/5. NOTE: a bare worker-venv rebuild reverts to the locked 5.5.0 — reapply.
2. Act-ckpt must stay ON for memory (without recompute, activations at 12288 seq = 75.1 GiB → OOM
   even with cpu_offload; measured). Now safe on 5.5.4.
3. **`nemo_rl_repro/sitecustomize.py` (new adapter file, zero vendored edits):** on ≥ 5.5.2 it
   disables nemo-rl's `_needs_kv_cache_for_shared_layers` workaround via a lazy meta-path hook
   (loads in every Ray actor through the PYTHONPATH the launcher already exports; also copied into
   the worker venv site-packages). Needed because the workaround's use_cache=True path CRASHES on
   5.5.4 (`modeling_gemma4.py` `shared_kv_states` KeyError 13 in `get_logprobs`); with the patch all
   passes run use_cache=False, which is now correct.
4. Also fixed en route: `keep_top_k: null` → int (pydantic), cpu_offload=true (4-rank static memory),
   and documented the (memory-only, numerically-inert) `LossPostProcessor` chunk_size gap — upstream
   fixed the Megatron twin in #2871/#2872 but never the automodel side.

**Cross-check vs upstream/PR #2224 (user question).** Our pin (5f89b3ae, 2026-07-27) already contains
PR #2224 (merged 2026-06-14) + follow-ups (#3297, #3124); config matches every gemma-4 correctness
knob in their E2B recipe; nothing in `5f89b3ae..origin/main` touches the affected files. NVIDIA's
recipe never hit this because it runs 4096-token sequences with act-ckpt OFF; their known-good recipe
also runs TIS on (we deliberately match verl: off).

**Two more landmines found while re-gating on 5.5.2+ (each root-caused, both defused by
`nemo_rl_repro/sitecustomize.py`, zero vendored edits):**
- 5.5.4's new KV-sharing passes anchor K/V between layers by MUTATING a plain `shared_kv_states`
  dict kwarg. **FSDP2's pre-forward input cast (`MixedPrecisionPolicy.cast_forward_inputs=True`
  default) rebuilds kwargs containers** (`_apply_to_tensors`), so each fully-sharded decoder layer
  receives a fresh COPY — anchors write into copies, shared layers read empty dicts →
  `modeling_gemma4.py` KeyError 13 in get_logprobs, with either use_cache mode, custom or stock
  class. Fix: force `cast_forward_inputs=False` (value-wise a no-op: model/activations already
  bf16; fp32 output cast gated separately on output_dtype).
- The Automodel custom gemma4 class predates the 5.5.2 mechanism entirely →
  `+policy.dtensor_cfg.automodel_kwargs.force_hf=true` (stock HF class; refit/generation validated
  through the gate). Dead ends tried and documented: TP=2 (E2B forward breaks under the dtensor TP
  plan: mixed Tensor/DTensor aten.where), act-ckpt off (75.1 GiB activations OOM).

**GATE PASSED end-to-end (run `4ndtkcry`): step-1 `probs_ratio = 1.000017`,
`probs_ratio_clamped = 1.000017` (was pinned 0.80), ratio min/max 0.795/1.298 (was 0/1.7e6),
grad_norm 0.74, val 0.0600 ∈ [0.045, 0.075], rc=0.** Full run relaunched as wandb
`DAPO/nemorl-dapo-gemma4-e2b-pt-DeepScaleR-4of4strict-seed42-8k-local4g-v2` (ckpts/logs
`/tmp/verl/{ckpts,logs}/...-local4g-v2`). Final local recipe = locked stack + transformers 5.5.4
(driver + policy-worker venvs; reverts on worker-venv rebuild — reapply) + sitecustomize (both
patches) + act-ckpt ON + cpu_offload ON + force_hf. verl-comparison caveats unchanged (vLLM venv
untouched; training math identical). Upstream-worthy reports: Automodel#1705-class bug reachable
via nemo-rl on the LOCKED stack whenever act-ckpt is on for E2B/E4B; the FSDP2
cast_forward_inputs dict-copy breaking 5.5.2+ gemma-4 KV sharing; `LossPostProcessor` chunk_size
gap (automodel twin of #2871).

---

## 2026-07-29 — NeMo-RL repro on ScaleTrain: parity gate PASSED at scale; baked image; OOM→act-ckpt

**Goal.** Run the 10-step NeMo-RL comparison (`MAX_STEPS=10`) on ScaleTrain to collect cross-framework
grad norms vs the verl reference (recovery `recbw9dcxso`, bimodal law: clean steps 1.3–6.8, ≥1
truncated response → 83–3108). Directive: keep relaunching, borrowing-true + borrowing-false pair.

**Headline result.** On 8×H100 with the full 3200-sample val, the step-0 GO/NO-GO gate returned
**validation/accuracy = 0.0550** — inside the verl baseline band [0.045, 0.075] and matching the local
2-GPU gate (0.0508). The whole parity stack (12-shot chat template byte-identical tokenization, strict
boxed-only reward port, PT model, sampling params, stop strings) is now validated at production scale.

**Pod-attempt ledger** (each fixed one blocker; all fixes in `scale_train/run_nemorl_gemma4_e2b_repro.sh`):
1. `job_d9kofjhq` — TE build: `cudnn.h` not found (pip cudnn wheel lands mid-`uv sync`) → apt
   `libcudnn9-dev-cuda-13` before the sync.
2. (launch killed) — docker context swept a stale multi-GB `third_party/nemo-rl/venvs/` from an early
   local run → dockerignored + deleted.
3. (launcher crash) — my `rm -rf` of that dir raced the launcher's repo staging copy → sequenced.
4. `job_d9kpfd1q` — data prep died on `python3: command not found` (pod login shell resets PATH; image
   venv dropped) and was non-fatal → gate crashed 1 h later on missing parquet. Fix: absolute venv
   python + hard fail-fast (`DATA_PREP_OK` marker).
5. `job_d9kqj9hq` — **GATE_PASS 0.0550**, then the first training step **OOM'd** (71.8 GiB PyTorch
   alloc / 80 GiB H100). Root cause: verl-parity `max_total_sequence_length=12288` is 3× NVIDIA's own
   E2B recipe (4096) with activation recompute off. Fix (parity-neutral, gradient-identical):
   `policy.dtensor_cfg.activation_checkpointing: true` (their 26B/31B recipes use it) +
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
6. Current pair on the fix: `job_d9l2cgpq` (borrow-on) + `job_d9l2huer` (borrow-off), queued 16:24Z.

**Baked image** (user: no more hour-long pod setup). New `st_config/Dockerfile.nemorl` + build config
`train-rl-distill-nemorl` (78.9 GB): CUDA-13 toolkit/compat/cuDNN-dev, Kitware cmake, driver venv with
TE compiled, and both worker venvs (`VllmGenerationWorker`, `DTensorPolicyWorkerV2`) prefetched to
`/opt/ray_venvs` via their `prefetch_venvs.py` (positional filters; deep-ep needs
`/usr/local/cuda-13.0/lib64/stubs` on `LIBRARY_PATH` in the GPU-less build). Heavy layers depend only
on `third_party/nemo-rl` → cached across launches; run script auto-detects baked artifacts and
fast-skips (startup ~95 → ~35–40 min, dominated by HF download + gate).

**Analysis prep.** `nemo_rl_repro/compare_grad_norms.py` — pulls the newest nemorl wandb run +
`recbw9dcxso`, tabulates `train/grad_norm` vs `train/truncation_rate`, prints a bimodality verdict.
Verified both frameworks log the same quantity (pre-clip global L2 per optimizer step; NeMo-RL:
`nemo_automodel scale_grads_and_clip_grad_norm`, no PP/EP scaling in our dense no-PP case).

**Status.** Attempt-6 pair queued; monitors on job status + pod logs. Next: gate re-pass → 10 steps →
run `compare_grad_norms.py` for the cross-framework verdict.

**2026-07-30 UPDATE — first 10 NeMo-RL steps (local 4×H100, run `ihdn67bj`).** Attempt-6 nb pod
crashed at vLLM init (`expandable_segments` incompatible with vLLM's memory pool — removed; act-ckpt
is the real OOM fix). Attempt-7 ScaleTrain pair queued for hours (cluster packed); meanwhile local
4-GPU gates (`m2jrli33` failed step 1, `b9b03y2f` finished: val 0.0591, grad_norm 2.43) proved 4-way
training fits with activation checkpointing (~346 s/step), and the full run `ihdn67bj` delivered
steps 1–10. **Result: verl's truncation bimodality does NOT reproduce** — grad norms 1.4–37.7 with
6/10 steps containing 1–2 truncated responses and zero spikes near verl's 83–3108 regime
(`compare_grad_norms.py` verdict: "regimes overlap"). Caveats: NeMo logs last-of-2 optimizer-step
grad norm (verl averaged both); truncated-in-all-wrong-group coincidence may not have occurred yet
at these truncation rates. ScaleTrain pair cancelled as redundant (jobs d9l8k5hq/d9l8tdpq); the
local run continues past step 10 to sharpen the test as responses lengthen. Env images published:
ECR `tmp:rl-distill-nemorl-cu130-20260729` (full) and hf.co/JWei05/nemorl-gemma4-cu130-env
(env-only, public).

> **⚠ RETRACTED (2026-07-30, see the ROOT CAUSE entry above): the bimodality verdict from
> `ihdn67bj` is INVALID.** That run's training forward was silently corrupted from step 1
> (act-ckpt × KV-shared layers on transformers 5.5.0 — `train/probs_ratio_clamped` pinned at
> 0.80), so its grad norms are gradients of a garbage objective, not the DAPO loss. The
> gate `b9b03y2f` cited here had the same poison in its train step (its VAL number remains
> valid). The bimodality test restarts from scratch on the fixed run `e1du1oyu` (`-local4g-v2`).

---

## 2026-07-28 — NeMo-RL exact-replication adapters for the gemma-4 E2B DAPO run (CPU gates green)

**Goal.** Reproduce `DAPO-gemma4-e2b-pt-DeepScaleR-4of4strict-seed42` (**8k variant**: 8192 max
response + 2048 overlong buffer) in NVIDIA NeMo-RL (`third_party/nemo-rl`, commit `5f89b3ae`) with
zero edits to the vendored checkout, so framework-level differences can be isolated from recipe
effects. Follows the verified interface study (workflow `wf_e4fa6102-3e1`).

**What was built** (all under `rl-distill-scripts/nemo_rl_repro/`):
- `rl_distill_nemo/deepscaler_dataset.py` — `DeepScalerStrictParquet`, verl-parquet → NeMo response
  dataset (dotted-path reference; no registry edit).
- `rl_distill_nemo/strict_math_env.py` — `math_strict` env: verbatim port of our strict boxed-only
  scorer from `verl/utils/reward_score/math_verify.py` (exactly-one-`\boxed{}`, LatexExtractionConfig
  only, spawn-pool verify with 30s timeout + pool pre-warm) wrapped in NeMo's
  `BaseMathEnvironment`/worker pattern.
- `run_grpo_repro.py` — wrapper that registers the env + `PY_EXECUTABLES.SYSTEM` actor entry and
  execs nemo-rl's `examples/run_grpo.py:main()`; exports PYTHONPATH so Ray actors import the adapters.
- `config/dapo_gemma4_e2b_pt_repro.yaml` — inherits their `grpo_math_1B.yaml` (NOT the gemma4-it
  recipe: its TE FusedAdam kwargs would merge into and crash `torch.optim.AdamW` — `_override_` is
  top-level-only in their loader); encodes every verl knob (64×16, gbs 512, token-mean, 0.2/0.28/c10,
  AdamW 1e-6/wd0.1, warmup 20, shaping 2048/1.0/**8192 explicit**, temp/top_p 1.0 top_k null, stops
  `<end_of_turn>/<start_of_turn>`, val 3200@10 + at-start, seed 42, save 25) + the recipe's gemma-4
  deltas (freeze towers, mbs/logprob 1, chunk 4096, packing off/dyn-batching on, gpu_mem_util 0.5
  flagged). Wandb: `DAPO/nemorl-dapo-gemma4-e2b-pt-DeepScaleR-4of4strict-seed42-8k`, entity rl-distill.
- `PARITY_CHECKLIST.md` — matched-knob table, unmatchable caveats (approx-entropy, grad_norm
  last-of-2, vLLM sampler numerics, stop-string in response length), go/no-go gate.

**Results (CPU gates).** `tests/test_strict_reward.py`: **PASS 6/6**, port and verl reference agree
(correct/wrong/two/no box, equivalent latex `1/2`≡`\frac{1}{2}`, malformed latex no-crash).
`tests/test_tokenization_parity.py`: **PASS 5/5** byte- and token-identical vs the verl reference
(12-shot template ~1.6k tok prompts, single BOS) — via the marked 3-line replicated processor path
(`import nemo_rl.models.policy` asserts transformers <5.12; our `.venv-gemma4*` has 5.14.1 — their
own uv venv is unaffected). Ran from `/tmp/.venv-gemma4-cu129` (EFS was RPC-saturated, load ~800).
Config verified end-to-end through their inheritance loader (all overrides + interpolations resolve).

**Status / next.** Adapters + gates done; **launch pending** the separately-built nemo-rl uv venv.
Launch: see `run_grpo_repro.py` header; then the PARITY_CHECKLIST go/no-go — step-0
`validation/accuracy ∈ [0.045, 0.075]` (verl baseline 6.16%@n=3200) with `grpo.max_num_steps=1`
before the full run. Deviation from the study: `grpo.max_num_epochs=100` (verl ran
`trainer.total_epochs=100`; study said 1, which would stop at ~151 steps).

**Goal.** Get verl DAPO RL working on `google/gemma-4-E2B`/`E4B` (PT), then land one production E2B
+ one E4B run: DeepScaleR strict-4/4 split, 12-shot prompt, 20k max response, seed 42, 1 node.

**Stack.** New `.venv-gemma4` (`setup_env_gemma4.sh`): torch 2.11 + vllm 0.25.1 + transformers 5.14.1,
**cu129 variant** (runs on the p5 fleet's CUDA-12.8 driver — gemma-4 does *not* need CUDA 13).
ScaleTrain runner: `scale_train/run_gemma4_pt_deepscaler_4of4strict_rl.sh` (reuses the
`train-rl-distill` image; builds the venv in /tmp at job start).

**Fix ladder (each root-caused; all in-repo, gemma-3 paths untouched).**
1. cu129 venv variant (driver compat).
2. `VERL_FSDP2_LOCAL_LOAD=1` — FSDP2 loader NCCL broadcast desync (rank0 +8 collectives) → deadlock.
3. `VERL_ROLLOUT_EXTRA_STOP` — gemma-4 has no `<end_of_turn>` token; without stop strings responses
   never terminate (20480-token rambles). With stops: mean response ~193 tokens, 87% single-boxed.
4. math-verify dep + `MATH_VERIFY_OK` fail-fast guard (missing dep silently scored 0.0).
5. flash-attn-free padding fallback in `verl/utils/attention_utils.py` (transformers `_unpad_input`).
6. Chunked entropy: flags on `actor_rollout_ref.actor.fsdp_config.*` + non-rmpad branch patch in
   `verl/workers/engine/fsdp/transformer_impl.py` (fp32 entropy blob was 21–40GB).
7. `use_dynamic_bsz=False` + all `*_micro_batch_size_per_gpu=1` (padded-forward logits OOM).
8. `free_cache_engine=False` + `enable_sleep_mode=False` (vLLM cumem sleep/wake broken on cu129);
   resident engine at util 0.25 (E2B/4×H100) / 0.3+OFFLOAD (E4B/8×H100).
9. v16: fused `F.cross_entropy` logprobs — killed `logprobs_from_logits_v2`'s fp32 autograd retention…
10. …but `F.cross_entropy` is on **autocast's fp32 cast-list**: one 21.5GiB fp32 upcast of the
    [22k, 262k] logits on any near-max-length sample (killed both v16 twins, steps 1–3). **v17:
    `_ChunkedLogprobsFromLogits`** in `verl/utils/torch_functional.py` — chunked fp32 gather−logsumexp
    forward, softmax-recompute backward (2GiB fwd transient, zero extra retention).
11. v17 died on its own backward's 10.8GB `empty_like` grad buffer → **v18: in-place backward**
    (grad written into the logits buffer; flash-attn `inplace_backward` contract).
12. v18 still ~51-53GB retained at step-1 backward → the non-rmpad NO_PADDING branch of
    `prepare_model_outputs` itself: `cat([logits.unbind()])` logits duplicate (+10.6GB) + its padded
    base-grad in backward (+10.6GB), unconditional `div_(temperature)` DivBackward full-vocab grad at
    temp 1.0, chunked entropy bypassing `entropy_checkpointing`, and graph-attached meta tensors across
    the accumulation loop. **v19 fixes all four** (padded-direct logprobs — bit-identical to packed;
    div skip at temp 1.0; checkpointed chunked entropy; detached metas). Composed update-flow peak at
    [22057, 262144]: base+3GiB. E2B arms also gain OFFLOAD=True for Adam headroom at steps ≥2.
13. v19 still died at step 1 — **gemma-4's own forward applies `final_logit_softcapping=30.0`**
    (`logits/cap → tanh → *cap`): 2-3 extra full-vocab tensors per forward, tanh retains its 10.8GB
    output for backward. Found in minutes by the new **local harness**
    `gemma4_local_update_memtest.py`, which runs the real TrainingWorker + prod config on
    a worst-case batch (4× 22,057-token samples) through old-logprob → update → post-Adam update, with
    memory-history snapshots and a 55.7GB budget check (79.19 − 23.5 resident vLLM). **v20: build_model
    nulls `text_config.final_logit_softcapping` and prepare_model_outputs re-applies it fused into the
    chunked logprobs (backward ×(1−tanh²), exact to 8e-6) or chunked in-place for no-grad passes; vLLM
    reads the untouched checkpoint config.** Harness green: old-logprob 26.4GB, update 44.1GB with real
    pg_loss (first-ever completed worst-case update), post-Adam 61.2GB at world-1 (optimizer term
    quarters at prod world-4 → ~33GB/rank projected). **Standing rule: no ScaleTrain submission without
    a green harness run.**

**Baselines (strict-4/4 val, temp 1.0, 12-shot, step-0; from authoritative single-line `step:0` logs
— earlier numbers parsed from wandb's wrapped pprint dumps were wrong).** E2B acc mean@16 5.2–6.1%,
best@16 26–30%, resp len ~170–185 tok (many reproductions); E4B acc mean@16 **12.4–12.6%**, best@16
41–45%, resp len ~250–270 tok (6×). Training confirmed healthy when it runs: E2B 4 steps (entropy
0.94–1.06, sane pg_loss), E4B 2 steps — pre-v19 durability was luck of batch lengths. ~8 min/step at
micro=1 (perf debt: flash-attn CE build / padding-aware budgets later).

**Update (2026-07-27 ~14:00Z): 8k runs learning fast.** E2B-8k (`bw9dcxso`): val 6.16→6.00→6.78→
10.0→**12.59%** (step 40) — >2× baseline; step-25 ckpt pushed (NOTE: same HF repo/path as the 20k
run's step-25 — superseded it; old snapshot only in HF git history. Add a config suffix to
HF_PUSH_REPO if configs must be kept separate). E4B-8k (`1tyrqscv`): val 12.06→13.97→**22.97%**
(step 20), step-25 ckpt pushed. 8k curves dominate the 20k config's early trajectory
(E2B-20k was 7.25% at step 20; E4B-20k 13.41% at step 10). ~3-4 min/step at 8k vs ~8-12 at 20k.

**Update (2026-07-27 ~06:55Z): restarted at 8k response length (user request).** The 20k v20 pair
trained cleanly (E2B: 26 steps, val 5.03→5.63→7.25%, step-25 ckpt pushed to HF
`JWei05/DAPO-gemma4-e2b-PT-DeepScaleR-4of4strict-seed42/step_000025`; E4B: 16 steps, val
12.97→13.41%) and was then replaced by a `MAX_RESPONSE_LENGTH=8192` + `OVERLONG_BUFFER_LEN=2048`
pair (env-only change, same image): E2B-8k job_d9jg45k89ikduq9ss6fg (4×H100), E4B-8k
job_d9jg4q9q1fpdvq22m7u0 (8×H100), borrowing false, priority high.

**Status (2026-07-27 ~01:15Z). v20 WORKS — production runs live.** The harness-validated v20 quad all
passed step 0 val at baseline and trained: E2B twins cleared steps 1–5+ (past v14's all-time record of
4 luck-steps; step-2 post-Adam regime included), E4B twins cleared steps 1–3. Consolidated per the
goal (one run per model, borrowing false): **E2B = job_d9j9k0c89ikd2e6dlipg / wandb `bo3fu4ow`
(4×H100, util 0.25, offload), E4B = job_d9j9kh9q1fpdkf56p82g / wandb `fsrp64mz` (8×H100, util 0.3,
offload)** — DeepScaleR strict-4/4, 12-shot, 20k max response, seed 42, priority high. -brw twins
cancelled healthy (flushed steps 5/3 on exit). Watchpoints: step-10 val, step-25 checkpoint + HF push.
Deferred: math-verify trace spot-check + response-length distribution; perf debt ~8 min/step (micro=1).
Full forensic detail in memory `gemma4-verl-rl-enablement.md`.

---

## 2026-07-22 — 1B-PT RL relaunch on the **strict 4/4** split + heavy wandb trace logging

**Goal.** Relaunch the two 1B-PT DeepScaleR-4/4 RL runs on a fresh split cut from the **strict-graded**
4/4 pool, under the strict boxed-only grader (now the math_verify default), and upload far more traces
to wandb: 100 random **train** rollouts every train step + 100 random **val** traces every eval.

**Data.** New split from `deepscaler_acc4of4_strict.parquet` (9,923 strict-4/4 questions), shuffle seed 42:
**train 9,723 / val 200** → `deepscaler_4of4strict_rl_{train,val200_x16}.parquet` (val 200 uids ×16 = 3,200,
`val_kwargs.n=1` → pass/maj/mean@16). **Local only** (1B reads local files; not pushed to HF).

**Code change (fork).** `dapo/dapo_ray_trainer.py` (symlink → `rl-distill-scripts/dapo_ray_trainer.py`):
added `RayDAPOTrainer._log_train_generations_to_wandb(batch)` — a **fresh per-step** `train/generations`
wandb Table (one row per trace: step/input/output/score; no unbounded accumulation), gated on
`trainer.log_train_generations`, called in `fit()` right after the disk-dump block. Core launcher
`gemma3_pt_fewshot_math_rl.sh` exposes it via `LOG_TRAIN_GENERATIONS` → `+trainer.log_train_generations`
(new key, `+`). Val traces use the existing `LOG_VAL_GENERATIONS` (bumped to 100). Strict grading is the
math_verify default (`VERL_MATH_VERIFY_STRICT_BOXED`), so no extra flag.

**Runs** — same recipe (GRPO n=16, max_response 8192 + 2048 overlong buffer, temp 1.0 train=val, 12-shot
prompt, SAVE_FREQ=25, HF push /25, wandb, val_before_train), via
`launch_1b_deepscaler_4of4strict_seed_sweep.sh`: seed 42 → GPUs 2,3 / port 52000; seed 43 → GPUs 4,5 /
port 53000; isolated Ray `/tmp/ray_1b_ds4of4s_seed{42,43}`; both `LOG_{VAL,TRAIN}_GENERATIONS=100`.
wandb `DAPO-gemma3-1b-pt-DeepScaleR-4of4strict-seed{42,43}`; HF
`JWei05/DAPO-Gemma3-1B-PT-DeepScaleR-4of4strict-seed{42,43}`.

**Ray startup timeout — root cause + fix (two parts).** Both seeds repeatedly died with Ray "The current
node timed out during startup" (even seed 42 alone, first — so NOT the 150s-stagger collision). Real root
cause: **the shared box is oversubscribed** — `load average ~180` with 83 users vs 192 CPUs (memory was
fine, 1.4 TB free). `ray/_private/node.py` gives the raylet only a **hardcoded 30s** to register with the
GCS (`raylet_start_wait_time_s = 30`); under that CPU starvation the raylet can't register in time and the
driver SIGKILLs it (the raylet.err "crash" trace is just that teardown, not a self-crash → a longer wait
succeeds). Two-part fix, both in `gemma3_1b_pt_fewshot_math_rl.sh`:
  1. **Widen the window.** Patched `.venv/.../ray/_private/node.py:411` to read an env var:
     `raylet_start_wait_time_s = int(os.environ.get("RAY_RAYLET_START_WAIT_TIME_S", "30"))`, and the wrapper
     exports `RAY_RAYLET_START_WAIT_TIME_S=300`. ⚠️ **This node.py edit is venv-local and is lost on a fresh
     `setup_env.sh`** — reapply it then (the wrapper's env export is a harmless no-op without it).
  2. **Cut our own startup churn.** `+ray_kwargs.ray_init.num_cpus=32` — the raylet prestarts one Python
     worker per CPU, so capping CPUs (a 2-GPU FSDP run needs only ~8: `max_colocate_count=3` →
     `{CPU:3,GPU:1}` bundles) avoids a 192-way EFS-venv import storm on top of the existing load.
  3. **Two more startup timeouts, same root cause.** With the driver-wait widened, the raylet then aborted
     `node_manager.cc:3362 Check failed ... Timed out waiting for file .../metrics_agent_port` — the
     metrics/dashboard agent (another EFS-venv Python import) was too slow to write its port file. These are
     real C++ RayConfig fields (env-overridable, confirmed via `strings _raylet.so`), so the wrapper now also
     exports `RAY_agent_register_timeout_ms=300000` and `RAY_worker_register_timeout_seconds=600`. (Stuck
     drivers go D-state on EFS I/O, so `kill -9` is pending until the syscall returns — expect slow cleanup.)

  4. **The un-fixable one (why it still won't start right now).** Even solo, the raylet keeps aborting on
     `Timed out waiting for file .../metrics_agent_port`: the metrics/dashboard agent (yet another EFS-venv
     Python import) takes ~90–117s to start under load but the raylet's wait for its port file is a **fixed
     ~86s hardcoded C++ timeout with NO env override** (confirmed `agent_register_timeout_ms` reaches the
     raylet via `os.environ.copy()` but doesn't govern this wait; can't patch a compiled `.so`). So the only
     real cure is the **external CPU load dropping** — box was at load ~210 when first flagged and climbed to
     ~280 (81 users) over the session; GPUs 1–5 are free the whole time, so it's purely CPU/EFS startup
     starvation, not our config.

All of this is captured in memory `ray-local-startup-timeouts-loaded-box`.

**Status — ARMED, waiting on the box.** Everything is staged and correct; the sole blocker is the overloaded
shared box. Deployed a **load-gated retry orchestrator** `launch_1b_ds4of4strict_retry.sh` (nohup, deadline
180min): it polls `/proc/loadavg` and only attempts a launch when 1-min load < `LOAD_GATE=160` (so it never
piles onto a peak-load box), brings seeds up **one at a time** (seed 42 on GPUs 2,3 then seed 43 on 4,5), and
stops the instant each is past Ray bringup. A persistent monitor watches its output. It will auto-launch both
runs the moment the box calms below 160 — nothing further to do on our side until then.

---

## 2026-07-22 — Strict boxed-only grading + full strict re-eval of GEMMA3_PT_EVAL_REPLICATION.md

**Change.** `verl/utils/reward_score/math_verify.py` now scores **strict** by default
(`VERL_MATH_VERIFY_STRICT_BOXED`): exactly one `\boxed{}` (0 or ≥2 → 0), verify only that box — vs the
old lenient whole-output `parse` + max-over-candidates that credited bare-LaTeX echoes / hedges. Tests:
`tests/utils/reward_score/test_math_verify_strict_boxed_on_cpu.py` (14 pass).

**Re-eval.** Rescored saved traces (sampled base/RL matrix `~/verl/eval_traces`; 4B-IT DeepScaleR ×4)
and regenerated the untraced ones on GPUs 2–5 with `--enforce_eager --trace_dir ~/verl/eval_traces_strict`
(greedy base/RL tables; per-difficulty 4B-PT/1B-PT on the **strict** difficulty buckets; few-shot sweep;
longest-2-shot; DeepScaleR-250). Scripts in scratch: `strict_rescore_traces.py`, `rescore_itgen_strict.py`,
`regen_strict.py`, `regen_strict2.py`, `sweep_strict.py`.

**Findings.** Strict ≈ lenient for **greedy** (single clean box); **meaningfully lower for sampled/temp-1.0**
(degeneration → no-box/echo false positives removed, e.g. DeepScaleR-250 4B temp-1.0 4.0→1.6). 4B-IT barely
changes (mean@4 38.8→37.9). Base→RL conclusions hold (RL still ~doubles DAPO-val pass@16). The 4B-IT
difficulty ranking transfers to PT models (4B-PT greedy 0/4→0% … 4/4→27%). Verified: all 2,530
scored-correct of 25,401 traces have exactly one box (0 violations); no false negatives spotted. All in the
new "Strict grading" section of `GEMMA3_PT_EVAL_REPLICATION.md`.

---

## 2026-07-22 — DeepScaleR **4/4-subset** RL: 1B ×2 seeds (local) + 4B ×1 (ScaleTrain)

**Goal.** Replace the prior full-DeepScaleR RL experiments with RL on the **4/4 subset** — the 10,404
questions Gemma-3-4B-IT solved 4/4 (from the difficulty filtering below). Clean, verified-solvable
problems; the 4B-PT only gets ~19% greedy on them, so there's RL headroom.

**Data.** 4/4 split (shuffle seed 42): **train 10,204 / val 200**. Files `deepscaler_4of4_rl_train.parquet`,
`deepscaler_4of4_rl_val200_x16.parquet` (200×16=3,200, `val_kwargs.n=1` → **pass/mean/maj@16**). Uploaded
to HF **`JWei05/DeepScaleR-4of4-RL`** (both local + pod pull the identical split).

**Runs** — same recipe as the DeepScaleR runs (GRPO, n=16, **max_response 8192 + 2048 overlong buffer**
[penalty ramps 6144→8192; changed from 20k/4k on the 2026-07-22 restart], temp 1.0 train=val,
**12-shot prompt** `gemma3_it_fewshot_math.jinja`, SAVE_FREQ=25, HF push /25, wandb, val_before_train):

- **1B ×2 seeds, local** (`launch_1b_deepscaler_4of4_seed_sweep.sh`): seed 42 → GPUs 2,3 / port 52000;
  seed 43 → GPUs 4,5 / port 53000; isolated Ray `/tmp/ray_1b_ds4of4_seed{42,43}`. wandb
  `DAPO-gemma3-1b-pt-DeepScaleR-4of4-seed{42,43}`; HF `JWei05/DAPO-Gemma3-1B-PT-DeepScaleR-4of4-seed{42,43}`.
- **4B ×1 seed, ScaleTrain** (`scale_train/run_gemma3_4b_pt_deepscaler_4of4_rl.sh` +
  `data/prepare_deepscaler_4of4_rl_data.sh`): 1×8-GPU H100, **priority high + borrowing**. wandb
  `DAPO-gemma3-4b-pt-DeepScaleR-4of4`; HF `JWei05/DAPO-Gemma3-4B-PT-DeepScaleR-4of4`.

**Status.** Launched. 1B seed 42 → GPUs 2,3 (fine); seed 43 hit a transient Ray-startup timeout when both
local Ray clusters started at once under EFS load → relaunched alone on GPUs 4,5 (fix next time: start the
two local seeds fully sequentially, not 45s-staggered). 4B ScaleTrain job (QUEUED, priority high + borrowing). Data split was cut from `deepscaler_acc4of4.parquet`
(see entry below).

**Restart (2026-07-22):** all 3 restarted to change **max_response 20480→8192** and **overlong buffer
4096→2048** (penalty ramps 6144→8192). 1B stagger bumped 45s→150s to avoid the Ray-startup collision.
Active jobs: 1B seeds 42 (GPUs 2,3) / 43 (GPUs 4,5) local; **4B ScaleTrain `job_d9g2rupq1fp5np2n185g`**
(QUEUED). (Superseded IDs: canceled `job_d9g1o4hq1fp5np2n1850`.)

---

## 2026-07-21/22 — DeepScaleR difficulty filtering (via Gemma-3-4B-IT pass rate) + 4B-PT check

**Goal.** Partition `agentica-org/DeepScaleR-Preview-Dataset` (40,315 competition-math problems) by
difficulty so we can select learnable-difficulty subsets for RL (drop always-solved / never-solved,
keep the middle), rather than training on the full, mostly-too-hard set.

**Approach — what we settled on.**
- First tried bucketing by the dataset's own `solution` length → **abandoned: 81.7% of `solution`
  fields are empty** (only 7,391 of 40,315 non-empty), so it's not a usable difficulty signal.
- Switched to a **model-based difficulty measure**: generate with **Gemma-3-4B-IT**, 4 samples per
  question, and use the per-question pass rate (`n_correct` out of 4) as difficulty.

### Step 1 — Inference: Gemma-3-4B-IT over all of DeepScaleR, 4×
- Model `google/gemma-3-4b-it`; **every** question (40,315) × **4 samples**.
- Sampling: **temp 1.0, top_p 1.0, top_k −1**; **8k context (max_prompt 2048 + max_response 6144)**.
  (Started at 12k/8192, tightened to 8k on request — smaller KV → higher concurrency.)
- Prompt: **short 2-shot** (`data/gemma3_it_fewshot_math_2shot.jinja`, domain + 15-trees).
- **DP=4** across GPUs 2,3,4,5 (dataset sharded 4 ways, TP=1 each). ~250–300 concurrent seqs/GPU,
  ~68–84% util, ~40k output tok/s aggregate, ~1.5 h wall.
- Script: `data/deepscaler_it_gen.py` (feeds token-id prompts, single BOS; saves per question the 4
  `response_texts` + `response_lens` + math_verify `scores`/`accs` + `n_correct`/`any_correct`).
- Output: `~/verl/data/deepscaler_it_gen/shard_{0..3}.parquet` (full traces kept).
- Note: run each shard's launch **staggered** or with a per-shard compile cache next time — 4 vLLM
  procs sharing the HF cache on EFS made the weight-load slow (~10 min D-state), and sharing the
  torch.compile inductor cache later corrupted it (see Step 3 caveat).

**IT-gen results (40,315 q):** overall **mean@4 = 38.8%**, **pass@4 = 53.7%**. Response length is a
strong difficulty signal — mean@4 by response-length quintile: **79.5 → 45.8 → 28.4 → 22.5 → 17.8%**
(shortest→longest). Lengths: median 1,264 tok, p95 2,021, max 4,252 (nothing hit the 6,144 cap).

### Step 2 — The 5 accuracy subsets (by `n_correct` of 4)
Merged (`merge_and_analyze_deepscaler_it_gen.py`) → `deepscaler_it_gen_merged.parquet` (all traces),
plus 5 **RL-ready** parquets `~/verl/data/deepscaler_it_gen/deepscaler_acc{0..4}of4.parquet`
(data_source=`math`, `\boxed{}` prompt, ground-truth):

| subset | questions | share | median resp len (IT) |
|---|---|---|---|
| 0/4 (never solved) | 18,670 | 46.3% | 1,405 |
| 1/4 | 4,755 | 11.8% | 1,472 |
| 2/4 | 3,251 | 8.1% | 1,419 |
| 3/4 | 3,235 | 8.0% | 1,260 |
| 4/4 (always solved) | 10,404 | 25.8% | 617 |

Bimodal: ~46% never solved, ~26% always solved, ~28% in the learnable middle (1–3/4).

### Step 3 — Sanity eval: Gemma-3-4B-PT on the subsets (greedy + temp 1.0)
- `google/gemma-3-4b-pt`, **12-shot** prompt (`data/gemma3_it_fewshot_math.jinja`), **100 random
  questions/bin** (seed 42), both **greedy (temp 0)** and **temp 1.0**, max_tokens 4096, max_model_len
  8192, via `eval_math_passk.py --enforce_eager`. Samples: `~/verl/data/deepscaler_acc{0..4}of4_s100.parquet`.

**4B-PT accuracy by IT-difficulty subset (mean@1, %):**

| IT subset | greedy | temp 1.0 |
|---|---|---|
| 0/4 | 3.0 | 2.0 |
| 1/4 | 5.0 | 0.0 |
| 2/4 | 4.0 | 1.0 |
| 3/4 | 7.0 | 2.0 |
| 4/4 | **19.0** | **8.0** |

- **Difficulty transfers across models**: 4B-PT accuracy rises with the 4B-IT pass rate; clearly
  highest on the 4/4 bin, near-floor on 0/4. Confirms the IT-pass-rate binning is a valid difficulty axis.
- But **compressed/low**: even the easiest (4/4) bin is only 19% greedy for the PT — DeepScaleR is hard
  for the 4B-PT, and IT-4/4 (stronger model + its own 2-shot prompt) ≠ easy-for-PT. Middle bins (1–3/4)
  are ~3–7% and statistically flat at n=100 (±~3–5%); the real signal is the 4/4 jump.
- **Greedy > temp 1.0** everywhere (base-PT pattern seen throughout, e.g. GEMMA3_PT_EVAL_REPLICATION.md).
- **Caveat:** needed `--enforce_eager` (flag added to `eval_math_passk.py`) — the DP=4 IT-gen corrupted
  the shared torch.compile inductor cache (`UnpicklingError: pickle data was truncated`); eager skips
  compilation and sidesteps it.

**Artifacts.**
- Scripts: `data/deepscaler_it_gen.py`, `data/merge_and_analyze_deepscaler_it_gen.py`,
  `eval_math_passk.py` (now supports `--enforce_eager`).
- Data (local): `~/verl/data/deepscaler_it_gen/` (shards, merged, 5 `acc*of4` subsets),
  `~/verl/data/deepscaler_acc*of4_s100.parquet`.
- Related earlier: `data/build_deepscaler_rl_data.py` + `data/prepare_deepscaler_rl_data.sh` built the
  DeepScaleR RL train/val split, uploaded to HF `JWei05/DeepScaleR-RL`.

**Status.** IT-gen + subset build + 4B-PT sanity eval all complete; nothing running. The earlier
DeepScaleR RL runs (1B local, 4B ScaleTrain) were stopped/cancelled — superseded by this filtering
work. Next: decide which subset(s) to train RL on (likely the learnable middle 1–3/4).

## 2026-09-13 — Hugging Face Hub cleanup (user request)

**Rule:** delete model repos last modified before 2026-07-30 (datasets untouched; repos created before but modified after the cutoff kept); the two Gemma 4 `4of4strict-seed42` RL repos (2026-07-27/28) were explicitly kept. 37 repos deleted (irreversible), 77 model repos remain.

Earlier the same day: `JWei05/Distill-gemma4-e4b-base-medium-to-{12b,26b}-base` pruned to `step_001000` with the other steps' LFS blobs purged (0.47 TB + 1.01 TB); their `step_001000` exports copied to `s3://scale-ml/genai/rl-distill/gemma4-e4b-base-distill-final-exports/`.

Deleted repos (several are still referenced by legacy Gemma 3 launch/eval scripts, which now point at nothing):

- `JWei05/dapo-gemma3-1b-pt`
- `JWei05/gemma3-4b-pt-sft-nemotron-cascade2-16k`
- `JWei05/gemma3-4b-pt-sft-distill-from-27b`
- `JWei05/gemma3-4b-pt-sft-distill-from-12b`
- `JWei05/dapo-gemma3-4b-pt-sft-nc2-16k`
- `JWei05/gemma3-1b-pt-sft-distill-from-4b`
- `JWei05/gemma3-1b-pt-sft-distill-from-12b`
- `JWei05/gemma3-1b-pt-sft-distill-from-27b`
- `JWei05/gemma3-1b-pt-onpolicy-distill-from-dapo4b-step60`
- `JWei05/gemma3-1b-pt-onpolicy-distill-from-dapo12b-step80`
- `JWei05/gemma3-1b-pt-onpolicy-distill-from-dapo27b-step80`
- `JWei05/gemma3-4b-pt-onpolicy-distill-from-dapo12b-step80`
- `JWei05/gemma3-4b-pt-onpolicy-distill-from-dapo27b-step80`
- `JWei05/gemma3-4b-pt-moe-2e-top1-sft-16k`
- `JWei05/gemma3-4b-pt-moe-4e-top1-sft-16k`
- `JWei05/dapo-gemma3-4b-pt`
- `JWei05/gemma3-4b-pt-sft-distill-from-27b-rl-step40-seed43`
- `JWei05/gemma3-4b-pt-sft-distill-from-12b-rl-step20-seed43`
- `JWei05/gemma3-4b-pt-sft-distill-from-27b-rl-step40-seed43-all33296-n4`
- `JWei05/gemma3-4b-pt-sft-distill-from-12b-rl-step20-seed43-all33296-n4`
- `JWei05/qwen35-9b-swe-save-smoke`
- `JWei05/Qwen3-4B-Instruct-2507-Scalar-Critic`
- `JWei05/DAPO-Gemma3-4B-PT-RL-DAPO17k`
- `JWei05/Qwen3-4B-Instruct-2507-Hybrid-Scalar-Reasoning-Critic`
- `JWei05/Qwen3-4B-Instruct-2507-Reasoning-Critic`
- `JWei05/Qwen2.5-3B-Instruct-DeepScaleR-Canonical-Critic`
- `JWei05/dapo-gemma3-4b-pt-moe-2e-megatron-rl`
- `JWei05/DAPO-Gemma3-1B-PT-FewShotMath-seed43`
- `JWei05/DAPO-Gemma3-1B-PT-FewShotMath-seed44`
- `JWei05/DAPO-Gemma3-1B-PT-FewShotMath-seed42`
- `JWei05/DAPO-Gemma3-4B-PT-FewShotMath`
- `JWei05/DAPO-Gemma3-4B-PT-FewShotMath-seed43`
- `JWei05/DAPO-Gemma3-1B-PT-DeepScaleR`
- `JWei05/DAPO-Gemma3-4B-PT-DeepScaleR`
- `JWei05/DAPO-Gemma3-1B-PT-DeepScaleR-4of4-seed42`
- `JWei05/DAPO-Gemma3-1B-PT-DeepScaleR-4of4-seed43`
- `JWei05/DAPO-Gemma3-1B-PT-DeepScaleR-4of4strict-seed42-local`

## 2026-09-14 — 12B distilled RL (medium) finished
- `g4-12b-distill-rl-med` (RL on top of the E4B-base→12B off-policy distilled student, DeepScaleR medium, seed 42, ES patience 5): early stopping at step 240, best val mean@16 **0.497 @ step 190** (from ≈0.08 at step 0). Best HF snapshot + receipt on S3 (`gemma4-12b-from-e4bbase-distill-rl/gemma4-12b-medium-from-e4bbase-distill-es5/best_hf/`). Details: DISTILLATION_EXPERIMENTS.md §9.0.
- On-policy distillation take 3 (student-top-128 reverse KL + tail bucket) running: job_dajtd33ns19g0896plog, step 40+, no flattening so far (§9.0f). Plain student-top-k run (take 2) collapsed by step 150 and was cancelled; its step-50 results stand (§9.0e).

## 2026-09-16 — devbox-008 restart backup

Infra is stop/starting devbox-008 (erases `/opt/dlami/nvme` = `/tmp` and `/dev/shm`). Everything of ours on `/tmp` that had no S3/Hub copy
was synced to `s3://scale-ml/genai/rl-distill/devbox-008-backup-20260916/` (≈62 GB): `gemma4_12bd_evals/passk`, `gemma4_distill_study_eval/results`,
`gemma4_e4b_val32`, `gemma4_stk50_kl`, `local_eval_raw`, `gemma4_distill_views`, `jasonwei_hf_stage/onpolicy_stk_step50`, the paper draft
(`tmax-research.*/paper`, `paper.tar`), two eval jsonl dumps and the local 12B-resume logs. Not uploaded (recoverable elsewhere): 26B checkpoint
copies (S3), Hub staging dirs (uploads finished), trace bundles (S3), HF/vLLM caches, agent scratch. `local_eval_models` (569 GB of Qwen2.5-3B
GRPO eval exports) was left for the user to decide. `/dev/shm` held only 238 MB of ours. The local venv `/tmp/.venv-gemma4` is rebuilt by
`setup_env_gemma4.sh` + `rl-distill-scripts/patches/`. GitHub `main` pushed through `dd6647dc`.
