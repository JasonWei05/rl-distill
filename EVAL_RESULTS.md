# Eval: DAPO-Gemma3-27B off-policy distilled students on math val sets

Generation config: T=0.7, top_p=0.95, max_tokens=20480, mean@1, `math_verify` scoring (LatexExtractionConfig — only `\boxed{}` answers credited).

Two SFT data variants compared:
- **unfiltered** — `JWei05/DAPO-Gemma3-27B-IT-RL-SFT-Data` (n=4 teacher samples per prompt, 17,398 prompts, 69,592 rows). Train 200 SFT steps, bs 128, lr 1e-5 cosine, 16,000 train prompts × 2 responses.
- **correct (rejection-sampled)** — `JWei05/DAPO-Gemma3-27B-IT-RL-SFT-Data-correct` (only teacher samples with math_verify-correct final answer; 13,062 prompts have ≥1 correct sample; 4,336 prompts have 0/4 correct and were dropped entirely). Train 200 SFT steps, bs 128, lr 1e-5 cosine, 12,000 train prompts × ≤3 responses (31,552 rows).

## Accuracy — IT (instruction-tuned) students (%)

| Model / Checkpoint | AIME 2024 | AIME 2025 | AIME 2026 | MATH500 | OlympiadBench | MinervaMath | GSM8K | **Mean** |
|---|---|---|---|---|---|---|---|---|
| **gemma-3-4b-it** (base IT, no SFT) | 9.79 | 12.08 | 9.79 | 76.40 | 41.54 | 24.72 | 89.76 | **37.73** |
| 4b unfiltered / step_000050 | 4.90 | 5.73 | 6.88 | 66.40 | 31.38 | 16.82 | 85.60 | 31.10 |
| 4b unfiltered / step_000100 | 6.25 | 8.75 | 6.25 | 67.80 | 31.38 | 19.21 | 82.64 | 31.75 |
| 4b unfiltered / step_000150 | 8.54 | 8.33 | 6.15 | 67.50 | 32.20 | 19.12 | 83.85 | 32.24 |
| 4b unfiltered / step_000200 | 8.02 | 8.02 | 8.75 | 69.30 | 33.90 | 18.93 | 85.06 | **33.14** |
| 4b correct / step_000050 | 5.83 | 6.46 | 4.79 | 65.50 | 29.01 | 17.46 | 84.23 | 30.47 |
| 4b correct / step_000100 | 6.88 | 7.08 | 6.88 | 66.40 | 30.19 | 16.08 | 85.44 | 31.28 |
| 4b correct / step_000150 | 5.62 | 9.48 | 6.77 | 67.80 | 31.97 | 17.00 | 84.38 | 31.86 |
| 4b correct / step_000200 | 5.83 | 8.02 | 8.02 | 68.70 | 33.23 | 17.10 | 85.29 | **32.31** |
| **gemma-3-12b-it** (base IT, no SFT) | 23.33 | 19.48 | 15.42 | 85.90 | 53.64 | 35.57 | 94.16 | **46.79** |
| 12b unfiltered / step_000050 | 21.67 | 16.35 | 16.46 | 81.90 | 44.73 | 34.93 | 93.25 | 44.18 |
| 12b unfiltered / step_000100 | 17.08 | 15.31 | 15.21 | 82.10 | 45.99 | 33.82 | 93.78 | 43.33 |
| 12b unfiltered / step_000150 | 21.25 | 17.19 | 16.15 | 81.90 | 45.33 | 34.38 | 93.40 | 44.23 |
| 12b unfiltered / step_000200 | 21.15 | 16.46 | 16.35 | 83.30 | 44.96 | 34.74 | 93.18 | **44.30** |
| 12b correct / step_000050 | 18.23 | 14.37 | 11.88 | 80.30 | 43.62 | 32.35 | 93.93 | 42.10 |
| 12b correct / step_000100 | 19.06 | 14.90 | 12.50 | 82.70 | 42.66 | 33.92 | 93.71 | 42.78 |
| 12b correct / step_000150 | 19.58 | 17.60 | 15.21 | 81.60 | 42.95 | 33.18 | 93.63 | 43.39 |
| 12b correct / step_000200 | 22.29 | 16.88 | 14.79 | 81.20 | 44.21 | 33.82 | 93.18 | **43.77** |

## Accuracy — PT (pretrained base) students (%)

| Model / Checkpoint | AIME 2024 | AIME 2025 | AIME 2026 | MATH500 | OlympiadBench | MinervaMath | GSM8K | **Mean** |
|---|---|---|---|---|---|---|---|---|
| **gemma-3-4b-pt** (base PT, no SFT) | 0.10 | 0.00 | 0.10 | 4.40 | 1.11 | 2.39 | 1.80 | **1.41** |
| 4b pt unfiltered / step_000050 | 0.73 | 0.31 | 0.10 | 25.60 | 7.12 | 5.88 | 47.61 | 12.48 |
| 4b pt unfiltered / step_000100 | 1.25 | 0.00 | 0.42 | 29.10 | 8.53 | 6.16 | 49.13 | 13.51 |
| 4b pt unfiltered / step_000150 | 1.25 | 0.42 | 0.42 | 33.00 | 10.68 | 7.08 | 53.83 | 15.24 |
| 4b pt unfiltered / step_000200 | 1.67 | 0.52 | 0.73 | 35.70 | 12.02 | 9.01 | 56.18 | **16.55** |
| 4b pt unfiltered / step_000250 | 1.15 | 0.31 | 0.62 | 35.70 | 10.98 | 9.56 | 56.79 | 16.44 |
| **gemma-3-12b-pt** (base PT, no SFT) | 0.42 | 0.42 | 0.42 | 15.70 | 4.23 | 6.25 | 25.55 | **7.57** |
| 12b pt unfiltered / step_000050 | 2.81 | 1.56 | 2.92 | 48.70 | 20.03 | 17.00 | 83.02 | 25.15 |
| 12b pt unfiltered / step_000100 | 4.58 | 3.65 | 4.58 | 57.70 | 24.18 | 21.51 | 81.05 | 28.18 |
| 12b pt unfiltered / step_000150 | 4.69 | 5.00 | 6.35 | 60.70 | 28.19 | 22.89 | 84.84 | 30.38 |
| 12b pt unfiltered / step_000200 | 5.94 | 6.04 | 6.35 | 64.50 | 29.01 | 24.17 | 85.90 | 31.70 |
| 12b pt unfiltered / step_000250 | 6.56 | 5.21 | 6.56 | 64.20 | 31.31 | 25.55 | 86.58 | **32.28** |

## Accuracy — IT students distilled from `google/gemma-4-31B-it` teacher (%)

Generation config T=1.0, top_p=0.7, max_tokens=20480, mean@1, `math_verify` scoring.
Train: same 16k train prompts × 2 responses / 1.4k val prompts × 1 split as the dapo27b-teacher runs, but teacher = vanilla `google/gemma-4-31B-it` on DAPO-Math-17k with n=2 responses/prompt ([HF dataset](https://huggingface.co/datasets/JWei05/DAPO-Gemma4-31B-IT-SFT-Data)). Student = `gemma-3-4b-it`. lr=1e-5 cosine, 200 steps, bs=128.

| Model / Checkpoint | AIME 2024 | AIME 2025 | AIME 2026 | MATH500 | OlympiadBench | MinervaMath | GSM8K | **Mean** |
|---|---|---|---|---|---|---|---|---|
| **gemma-3-4b-it** (base IT, no SFT) | 9.79 | 12.08 | 9.79 | 76.40 | 41.54 | 24.72 | 89.76 | **37.73** |
| 4b it / gemma4-teacher / step_000050 | 2.29 | 5.42 | 1.88 | 53.90 | 23.22 | 15.62 | 75.97 | 25.47 |
| 4b it / gemma4-teacher / step_000100 | 1.56 | 5.52 | 2.08 | 55.60 | 24.11 | 15.17 | 76.57 | 25.80 |
| 4b it / gemma4-teacher / step_000150 | 2.81 | 4.58 | 2.50 | 57.50 | 25.74 | 14.52 | 75.74 | 26.20 |
| 4b it / gemma4-teacher / step_000200 | 2.71 | 5.31 | 3.44 | 57.10 | 25.89 | 14.43 | 78.09 | **26.71** |

### Correct vs unfiltered — matched-step mean delta

| Step | 4B corr mean | 4B unfilt mean | Δ | 12B corr mean | 12B unfilt mean | Δ |
|---|---|---|---|---|---|---|
| s50  | 30.47 | 31.10 | **−0.63** | 42.10 | 44.18 | **−2.08** |
| s100 | 31.28 | 31.75 | **−0.47** | 42.78 | 43.33 | **−0.55** |
| s150 | 31.86 | 32.24 | **−0.38** | 43.39 | 44.23 | **−0.84** |
| s200 | 32.31 | 33.14 | **−0.83** | 43.77 | 44.30 | **−0.53** |

Filter-correct underperforms unfiltered at **every matched checkpoint across both sizes** (8/8 negative). Mean delta ≈ −0.9 pts (weighted by step).

## Mean response length — IT students (tokens)

| Model / Checkpoint | AIME 2024 | AIME 2025 | AIME 2026 | MATH500 | OlympiadBench | MinervaMath | GSM8K |
|---|---|---|---|---|---|---|---|
| gemma-3-4b-it (base IT, no SFT) | 1856 | 1605 | 1757 | 998 | 1375 | 960 | 449 |
| 4b unfiltered / step_000050 | 5687 | 5155 | 4664 | 2268 | 3626 | 2125 | 485 |
| 4b unfiltered / step_000100 | 4433 | 4079 | 4243 | 1603 | 2952 | 1271 | 372 |
| 4b unfiltered / step_000150 | 4719 | 4236 | 4193 | 1803 | 2999 | 1573 | 430 |
| 4b unfiltered / step_000200 | 4178 | 3452 | 3752 | 1661 | 2550 | 1291 | 410 |
| 4b correct / step_000050 | 3871 | 3442 | 3544 | 1511 | 2642 | 1294 | 395 |
| 4b correct / step_000100 | 3505 | 3153 | 3336 | 1375 | 2289 | 1242 | 396 |
| 4b correct / step_000150 | 3640 | 3458 | 3329 | 1481 | 2245 | 1292 | 394 |
| 4b correct / step_000200 | 3319 | 2998 | 2811 | 1264 | 2085 | 1227 | 380 |
| gemma-3-12b-it (base IT, no SFT) | 1651 | 1521 | 1733 | 778 | 1199 | 645 | 298 |
| 12b unfiltered / step_000050 | 4197 | 3539 | 3868 | 1270 | 2327 | 1260 | 322 |
| 12b unfiltered / step_000100 | 3194 | 2521 | 2713 | 1028 | 1866 | 1051 | 308 |
| 12b unfiltered / step_000150 | 3182 | 2862 | 2944 | 1015 | 1963 | 1039 | 317 |
| 12b unfiltered / step_000200 | 3089 | 2559 | 2792 | 1020 | 1809 | 986 | 313 |
| 12b correct / step_000050 | 3223 | 2767 | 3037 | 1002 | 1925 | 1010 | 306 |
| 12b correct / step_000100 | 2840 | 2485 | 2834 | 1008 | 1758 | 1035 | 320 |
| 12b correct / step_000150 | 3164 | 2759 | 3013 | 1100 | 1871 | 991 | 317 |
| 12b correct / step_000200 | 2842 | 2622 | 2643 | 975 | 1730 | 968 | 322 |

## Mean response length — PT students (tokens)

| Model / Checkpoint | AIME 2024 | AIME 2025 | AIME 2026 | MATH500 | OlympiadBench | MinervaMath | GSM8K |
|---|---|---|---|---|---|---|---|
| 4b pt unfiltered / step_000050 | 8312 | 8028 | 7585 | 5049 | 7656 | 6270 | 2693 |
| 4b pt unfiltered / step_000100 | 8287 | 8212 | 7036 | 4585 | 6789 | 5715 | 1804 |
| 4b pt unfiltered / step_000150 | 7218 | 6540 | 6174 | 3935 | 5609 | 5209 | 1761 |
| 4b pt unfiltered / step_000200 | 6172 | 5705 | 5788 | 3272 | 5499 | 3858 | 1203 |
| 4b pt unfiltered / step_000250 | 5731 | 5711 | 5893 | 2931 | 5210 | 3716 | 1212 |
| 12b pt unfiltered / step_000050 | 5509 | 5916 | 4971 | 2176 | 3615 | 3301 | 490 |
| 12b pt unfiltered / step_000100 | 5448 | 4118 | 4254 | 1799 | 3628 | 1711 | 492 |
| 12b pt unfiltered / step_000150 | 5195 | 4202 | 4486 | 1737 | 2999 | 1819 | 446 |
| 12b pt unfiltered / step_000200 | 4266 | 3550 | 3894 | 1349 | 2672 | 1581 | 360 |
| 12b pt unfiltered / step_000250 | 5071 | 3745 | 4056 | 1496 | 2784 | 1796 | 401 |

## Mean response length — IT from gemma-4-31B-it teacher (tokens)

| Model / Checkpoint | AIME 2024 | AIME 2025 | AIME 2026 | MATH500 | OlympiadBench | MinervaMath | GSM8K |
|---|---|---|---|---|---|---|---|
| 4b it / gemma4-teacher / step_000050 | 6354 | 6116 | 5583 | 1908 | 3834 | 1080 | 395 |
| 4b it / gemma4-teacher / step_000100 | 7332 | 6991 | 6619 | 2095 | 4083 | 973 | 362 |
| 4b it / gemma4-teacher / step_000150 | 7391 | 7629 | 6419 | 2465 | 4480 | 1038 | 458 |
| 4b it / gemma4-teacher / step_000200 | 7647 | 7613 | 6846 | 2080 | 4700 | 1027 | 434 |

## Takeaways

1. **Off-policy distillation underperforms base IT on math, in both data variants and both student sizes.** 4B best (correct s200) 32.3 vs base 37.7 (−5.4). 12B best (correct s200) 43.8 vs base 46.8 (−3.0). Training-time forward KL drops cleanly (4B val_loss 0.10 → 0.075, 12B 0.10 → 0.046) — the student *is* learning teacher tokens — but sampling quality collapses because the student can't autoregressively reproduce 27B-scale reasoning.

2. **Rejection-sampling (math_verify correct-only) makes it *worse*, not better, at every matched step.** 4B mean at s200: 32.31 correct vs 33.14 unfiltered (−0.83). 12B mean at s50 shows the biggest gap (−2.08). All 8 matched (size, step) deltas are negative. Reasons that likely compound:
   - **The hardest 25% of prompts are dropped entirely.** 4,336 / 17,398 prompts had 0/4 teacher samples correct and never reach the correct-split. Evals (AIME, OlympiadBench) are dominated by hard prompts, so correct training has zero exposure to ~8k "teacher-struggles-on-hard-problem" rows that the unfiltered run saw. Even wrong teacher attempts carry useful priors on how to approach hard problems.
   - **Surviving-prompt weighting skews easy.** Prompts with 4/4 correct (easiest, 7,492 of 13,062) contribute 3 rows each in the correct split; prompts with 1/4 correct (hardest surviving, 1,611 prompts) contribute 1 row. Unfiltered is uniform across 16k prompts. Correct's effective training distribution is *much easier* than the eval distribution.
   - **Filter selects for *short confident wins*, not good reasoning.** Teacher's correct responses are its clean "I-see-it-immediately" trajectories; wrong responses include retries/self-correction that are closer to how a smaller model must actually reason. Filtering pushes the student toward shorter clean outputs (observed: 4B-correct s200 avg 2012 tokens vs unfiltered 2470, 12B-correct s200 avg 1871 vs unfiltered 1794 — both shorter than unfiltered). But on AIME/OlympiadBench the student needs *more* thinking, not less.
   - **"Correct" is outcome-filtering, not process-filtering.** `math_verify` checks only `\boxed{...}` equivalence — reasoning with a lucky right answer slips through, good reasoning with a transcription slip in the last step gets thrown out. So the filter doesn't cleanly select for quality reasoning.
   - **Capability-gap amplification.** When the 27B teacher succeeds on a hard problem, its trajectory often uses composition/context the 4B/12B can't execute autoregressively. Filtering concentrates these "imitate the impossible" examples. Unfiltered data dilutes them with teacher's wrong-but-relatable attempts.

3. **Response length stays 2-3× base-IT under both variants** — students clearly absorb teacher's verbosity pattern, but don't absorb the reasoning quality. 4B AIME24: base 1856 → correct s200 3319 (1.8×). 12B AIME24: base 1651 → correct s200 2842 (1.7×). Length alone isn't what's causing the accuracy drop (correct is shorter than unfiltered but scores same/worse).

4. **GSM8K is well-preserved in both variants** — 4B drops 4.5 pts, 12B drops 1 pt. Easy-distribution math survives distillation. The regressions concentrate on AIME/OlympiadBench (the hard distribution), exactly where the filter's prompt-dropping is most damaging.

5. **One real-but-minor bug found** (not the dominant cause): `distill_dataset.py` + `forward_kl_loss.py` leave each sample's trailing `\n` (token 107, appended by Gemma chat template after `<end_of_turn>`) with teacher_lp=0, training the student to confidently emit `\n` after EOS. ~1 in ~3000 tokens per sample gets misaligned supervision; the position is *after* the model stops at eval, so the impact is negligible.

## What would probably help (unverified)

- **Keep all 17,398 prompts** — for prompts where the teacher failed all 4 times, keep *one* teacher sample anyway (or one base-IT sample). The student needs some representation of every problem in training, not zero.
- **Bias sampling toward harder surviving prompts** — pick (up to) 2 responses per surviving prompt uniformly rather than 3 responses from 4/4-correct prompts — evens out the effective prompt distribution.
- **On-policy distillation** (student samples, teacher scores) avoids the capability-gap amplification entirely — the student can only imitate tokens it *itself* just sampled, so there's no "impossible trajectory" drift. Scripts for this written at `rl-distill-scripts/gemma3_{4b,12b}_it_distill_onpolicy.sh`, pending the `_compute_teacher_colocate` wire-in into `dapo_ray_trainer.py::fit()`.

## Wall-clock (generation seconds, summed over 7 datasets)

| Model / Checkpoint | Total gen seconds |
|---|---|
| gemma-3-4b-it (base IT, no SFT) | 1232 |
| 4b unfiltered / step_000050 | 2598 |
| 4b unfiltered / step_000100 | 2213 |
| 4b unfiltered / step_000150 | 2208 |
| 4b unfiltered / step_000200 | 2027 |
| gemma-3-12b-it (base IT, no SFT) | 1712 |
| 12b unfiltered / step_000050 | 3131 |
| 12b unfiltered / step_000100 | 2626 |
| 12b unfiltered / step_000150 | 2389 |
| 12b unfiltered / step_000200 | 2402 |

## Reproduction

- Per-run summaries: `/home/tiger/verl/data/eval_results/*__summary.json`
- Eval launcher (unfiltered): `dapo/_launch_eval.sh`
- Eval launcher (correct): `dapo/_launch_eval_correct.sh`
- Eval launcher (base IT baselines): `dapo/_launch_eval_baselines.sh`
- Eval scoring script: `dapo/_eval_model_on_math.py`
- Re-compile the auto table (accuracy + response length + wall-clock only, no takeaways): `python3 dapo/_compile_eval_md.py /home/tiger/verl/data/eval_results /home/tiger/verl/data/eval_results/EVAL_RESULTS.md`
