# Oct 22, 2026

## Qwen3: On-policy Distillation on top of Off-policy Distillation

Each off-policy-distilled student (step 1000, forward KL top-128 on teacher traces) is continued with on-policy distillation (OPD) from the same RL teacher on the same band. The student samples, the teacher scores every sampled token, and the per-token reverse-KL estimate is used as a policy-gradient advantage. This is the recipe most frontier reports use (Thinking Machines, MiMo / MOPD, GLM-5, Nemotron-Cascade 2, Nemotron 3 Ultra, Kimi K3). MOPD (<https://arxiv.org/pdf/2606.30406>) finds it matches top-k logit distillation and is more robust when teacher and student distributions differ.

### Configs

- Teachers: best RL checkpoints (30B A3B: medium step 320, hard 410, very hard 460; 4B: medium 510, hard 560, very hard 620)
- Students: the off-policy distilled checkpoints at step 1000 (4B ⇒ 1.7B, 30B ⇒ 1.7B, 30B ⇒ 4B; medium, hard, very hard)
- Advantage A_t = sg[log π_teacher(y_t) − log π_student(y_t)] (sampled-token k1), clipped to [−5, 5] (MOPD); no task reward, no advantage normalization
- IcePop token importance-sampling mask between the trainer and vLLM: weight r_t if 0.5 ≤ r_t ≤ 2, else 0 (Nemotron-Cascade 2, GLM-5)
- 128 prompts × 1 sample per step, one gradient update per step (so the PPO ratio clip is inactive)
- LR 2e-6, warmup steps 20, constant; token-mean loss; temperature 1.0, top-p 1.0, max response length 8192
- 120 steps; validation every 10 steps (mean@16 on the 300 in-distribution validation questions); best checkpoint = best validation step ≤ 120
- vLLM TP 1 for the student and the 4B teacher, TP 2 for the 30B A3B teacher; GPUs: 1.7B student 2, 4B student 4

### W&B runs (all Qwen3 runs)

Direct RL (best = the validation checkpoint used in all tables below; very hard includes the +100-step continuation)

- 30B A3B RL Medium https://wandb.ai/rl-distill/DAPO/runs/q3-30b-a3b-medium-es5-s42-v1 (final step 420, best step 320)
- 30B A3B RL Hard https://wandb.ai/rl-distill/DAPO/runs/q3-30b-a3b-hard-es5-s42-v1 (final step 500, best step 410)
- 30B A3B RL Very Hard https://wandb.ai/rl-distill/DAPO/runs/q3-30b-a3b-very_hard-es5-s42-v1 (final step 550, best step 460)
- 4B RL Medium https://wandb.ai/rl-distill/DAPO/runs/q3-4b-medium-es5-s42-v1 (final step 590, best step 510)
- 4B RL Hard https://wandb.ai/rl-distill/DAPO/runs/q3-4b-hard-es5-s42-v1 (final step 600, best step 560)
- 4B RL Very Hard https://wandb.ai/rl-distill/DAPO/runs/q3-4b-very_hard-es5-s42-v1 (final step 700, best step 620)
- 1.7B RL Medium https://wandb.ai/rl-distill/DAPO/runs/q3-1.7b-medium-es5-s42-v1 (final step 600, best step 520)
- 1.7B RL Hard https://wandb.ai/rl-distill/DAPO/runs/q3-1.7b-hard-es5-s42-v1 (final step 600, best step 550)
- 1.7B RL Very Hard https://wandb.ai/rl-distill/DAPO/runs/q3-1.7b-very_hard-es5-s42-v1 (final step 1000, best step 810)

Off-policy distillation (forward KL top-128, 1,000 steps; evaluated at step 1000)

- 30B RL Medium ⇒ 4B https://wandb.ai/rl-distill/qwen3-30b-rl-distill-v1/runs/7d4h349i
- 30B RL Hard ⇒ 4B https://wandb.ai/rl-distill/qwen3-30b-rl-distill-v1/runs/su6s8dee
- 30B RL Very Hard ⇒ 4B https://wandb.ai/rl-distill/qwen3-30b-rl-distill-v1/runs/djbjgba4
- 30B RL Medium ⇒ 1.7B https://wandb.ai/rl-distill/qwen3-30b-rl-distill-v1/runs/gt36keuc
- 30B RL Hard ⇒ 1.7B https://wandb.ai/rl-distill/qwen3-30b-rl-distill-v1/runs/ppt88udx
- 30B RL Very Hard ⇒ 1.7B https://wandb.ai/rl-distill/qwen3-30b-rl-distill-v1/runs/dr48mkts
- 4B RL Medium ⇒ 1.7B https://wandb.ai/rl-distill/qwen3-4b-rl-distill-v1/runs/kowg1frk
- 4B RL Hard ⇒ 1.7B https://wandb.ai/rl-distill/qwen3-4b-rl-distill-v1/runs/r9gsjy95
- 4B RL Very Hard ⇒ 1.7B https://wandb.ai/rl-distill/qwen3-4b-rl-distill-v1/runs/cn9a1nh2

Off-policy distillation from the untrained 4B Base into 30B A3B (small ⇒ big control; earlier runs are preempted segments, the last one finished at step 1000)

- 4B Base Medium ⇒ 30B A3B https://wandb.ai/rl-distill/qwen3-4b-base-distill-v1/runs/dojowpu7 (earlier segments: https://wandb.ai/rl-distill/qwen3-4b-base-distill-v1/runs/8ya3dzj8, https://wandb.ai/rl-distill/qwen3-4b-base-distill-v1/runs/5iwqpskq, https://wandb.ai/rl-distill/qwen3-4b-base-distill-v1/runs/8kau7cff, https://wandb.ai/rl-distill/qwen3-4b-base-distill-v1/runs/jdvzd4m4)
- 4B Base Hard ⇒ 30B A3B https://wandb.ai/rl-distill/qwen3-4b-base-distill-v1/runs/px7u3gbe (earlier segments: https://wandb.ai/rl-distill/qwen3-4b-base-distill-v1/runs/ksrks5ae, https://wandb.ai/rl-distill/qwen3-4b-base-distill-v1/runs/d26ja3r3)

On-policy distillation (120 steps; the 4B RL Hard and Very Hard ⇒ 1.7B runs show "crashed" only because they were stopped at step 120 from a 150-step launch)

- 4B RL Medium ⇒ 1.7B https://wandb.ai/rl-distill/DAPO/runs/q3-opd-q4bm-to-q1p7b-pgk1-v1-s42 (best step 120)
- 4B RL Hard ⇒ 1.7B https://wandb.ai/rl-distill/DAPO/runs/q3-opd-q4bh-to-q1p7b-pgk1-v1-s42 (best step 110)
- 4B RL Very Hard ⇒ 1.7B https://wandb.ai/rl-distill/DAPO/runs/q3-opd-q4bvh-to-q1p7b-pgk1-v1-s42 (best step 110)
- 30B RL Medium ⇒ 1.7B https://wandb.ai/rl-distill/DAPO/runs/q3-opd-q30bm-to-q1p7b-pgk1-v1-s42 (best step 110)
- 30B RL Hard ⇒ 1.7B https://wandb.ai/rl-distill/DAPO/runs/q3-opd-q30bh-to-q1p7b-pgk1-v1-s42 (best step 100)
- 30B RL Very Hard ⇒ 1.7B https://wandb.ai/rl-distill/DAPO/runs/q3-opd-q30bvh-to-q1p7b-pgk1-v1-s42 (best step 100)
- 30B RL Medium ⇒ 4B https://wandb.ai/rl-distill/DAPO/runs/q3-opd-q30bm-to-q4b-pgk1-v1-s42 (best step 110)
- 30B RL Hard ⇒ 4B https://wandb.ai/rl-distill/DAPO/runs/q3-opd-q30bh-to-q4b-pgk1-v1-s42 (best step 120)
- 30B RL Very Hard ⇒ 4B https://wandb.ai/rl-distill/DAPO/runs/q3-opd-q30bvh-to-q4b-pgk1-v1-s42 (best step 100)

### Matrix update (diagonal is RL)

In-distribution pass@1 / pass@16. "Off" = off-policy distillation (existing entries, best step); "On" = after on-policy distillation (best step ≤ 120, started from off-policy step 1000).

Medium

| | 1.7B Teacher | 4B Teacher | 30B Teacher |
|---|---|---|---|
| 1.7B Student | **33.1 / 80.0 (RL step 520)** | Off: 28.6 / 78.3 (step 750); **On: 39.1 / 84.0 (step 120)** | Off: 25.9 / 75.7 (step 1000); On: 33.9 / 80.7 (step 110) |
| 4B Student | | **65.5 / 93.7 (RL step 510)** | Off: 52.2 / 93.3 (step 1000); On: 59.7 / 94.3 (step 110) |
| 30B Student | | | 74.3 / 96.3 (RL step 320) |

Hard

| | 1.7B Teacher | 4B Teacher | 30B Teacher |
|---|---|---|---|
| 1.7B Student | 12.1 / 51.0 (RL step 550) | Off: 11.9 / 54.0 (step 1000); **On: 19.4 / 62.3 (step 110)** | Off: 11.3 / 53.0 (step 1000); On: 18.1 / 61.7 (step 100) |
| 4B Student | | 40.6 / 85.7 (RL step 560) | Off: 34.4 / 86.7 (step 1000); **On: 41.8 / 87.0 (step 120)** |
| 30B Student | | | 61.4 / 89.0 (RL step 410) |

Very hard

| | 1.7B Teacher | 4B Teacher | 30B Teacher |
|---|---|---|---|
| 1.7B Student | 12.1 / 42.7 (RL step 810) | Off: 10.0 / 46.0 (step 1000); **On: 12.7 / 50.3 (step 110)** | Off: 7.5 / 42.3 (step 750); On: 10.2 / 49.0 (step 100) |
| 4B Student | | 23.7 / 63.3 (RL step 620) | Off: 19.6 / 61.7 (step 1000); **On: 24.0 / 66.0 (step 100)** |
| 30B Student | | | 38.4 / 76.3 (RL step 460) |

### Extensive Math Eval by student and training difficulty

mean@k / pass@k, where k is 32 for AIME and 16 everywhere else; Average is over the 8 sets. Each table is one student size and one training band. RL = best validation checkpoint; Off = off-policy distillation step 1000; On = on-policy distillation best step ≤ 120. Bold = best in the column within the table.

1.7B Medium

| Model | Easy | Medium | Hard | Very Hard | GSM8K | MATH500 | AIME 2025 | AIME 2026 | Average |
|---|---|---|---|---|---|---|---|---|---|
| 1.7B Base | 31.4 / 85.7 | 9.6 / 55.7 | 3.8 / 32.3 | 3.3 / 30.3 | 40.3 / 94.6 | 30.1 / 81.0 | 1.0 / 13.3 | 0.5 / 16.7 | 15.0 / 51.2 |
| 1.7B RL Medium | 70.7 / 93.7 | 33.1 / 79.7 | 14.6 / 54.0 | 12.1 / 50.0 | 83.1 / **97.0** | 66.2 / 89.4 | 5.4 / **33.3** | 2.5 / 13.3 | 36.0 / 63.8 |
| 4B RL Medium ⇒ 1.7B Off | 67.2 / 95.3 | 27.8 / 74.7 | 12.1 / 49.7 | 9.4 / 44.0 | 82.6 / 96.7 | 64.9 / 89.2 | 3.5 / 30.0 | 2.5 / **23.3** | 33.7 / 62.9 |
| 4B RL Medium ⇒ 1.7B On | **75.6** / **97.0** | **39.1** / **84.0** | **19.2** / **59.7** | **13.2** / 49.7 | **85.9** / **97.0** | **72.2** / **92.2** | **6.7** / 30.0 | **5.4** / 20.0 | **39.7** / **66.2** |
| 30B RL Medium ⇒ 1.7B Off | 64.2 / 95.0 | 25.9 / 75.7 | 11.5 / 49.3 | 8.8 / 46.7 | 80.2 / 96.9 | 61.4 / 89.4 | 2.9 / 20.0 | 2.1 / **23.3** | 32.1 / 62.0 |
| 30B RL Medium ⇒ 1.7B On | 72.5 / 96.0 | 33.9 / 80.7 | 15.5 / 55.3 | 11.7 / **51.0** | 84.1 / 96.7 | 67.9 / 89.8 | 4.9 / 26.7 | 4.4 / **23.3** | 36.9 / 64.9 |

1.7B Hard

| Model | Easy | Medium | Hard | Very Hard | GSM8K | MATH500 | AIME 2025 | AIME 2026 | Average |
|---|---|---|---|---|---|---|---|---|---|
| 1.7B Base | 31.4 / 85.7 | 9.6 / 55.7 | 3.8 / 32.3 | 3.3 / 30.3 | 40.3 / 94.6 | 30.1 / 81.0 | 1.0 / 13.3 | 0.5 / 16.7 | 15.0 / 51.2 |
| 1.7B RL Hard | 66.4 / 94.7 | 28.4 / 75.0 | 12.1 / 51.0 | 11.2 / 48.7 | 80.9 / 97.0 | 62.6 / 88.4 | 2.7 / 23.3 | 3.3 / 26.7 | 33.5 / 63.1 |
| 4B RL Hard ⇒ 1.7B Off | 64.2 / 93.7 | 26.4 / 74.7 | 11.9 / 54.0 | 9.7 / 47.7 | 81.0 / 97.0 | 63.4 / 89.2 | 4.2 / 20.0 | 2.0 / 13.3 | 32.8 / 61.2 |
| 4B RL Hard ⇒ 1.7B On | **74.7** / 95.3 | **37.9** / **84.0** | **19.4** / **62.3** | **14.1** / **56.0** | **85.5** / **97.5** | **73.1** / 91.8 | 7.8 / **33.3** | 5.6 / 23.3 | **39.8** / 68.0 |
| 30B RL Hard ⇒ 1.7B Off | 61.8 / 95.3 | 24.9 / 77.7 | 11.3 / 52.7 | 9.0 / 49.3 | 79.1 / 97.1 | 62.3 / 89.4 | 3.8 / 26.7 | 2.4 / 20.0 | 31.8 / 63.5 |
| 30B RL Hard ⇒ 1.7B On | 73.3 / **96.3** | 36.0 / **84.0** | 18.1 / 61.7 | 13.2 / 53.0 | 84.7 / 97.4 | 72.1 / **92.0** | **9.2** / 26.7 | **7.3** / **33.3** | 39.2 / **68.1** |

1.7B Very Hard

| Model | Easy | Medium | Hard | Very Hard | GSM8K | MATH500 | AIME 2025 | AIME 2026 | Average |
|---|---|---|---|---|---|---|---|---|---|
| 1.7B Base | 31.4 / 85.7 | 9.6 / 55.7 | 3.8 / 32.3 | 3.3 / 30.3 | 40.3 / 94.6 | 30.1 / 81.0 | 1.0 / 13.3 | 0.5 / 16.7 | 15.0 / 51.2 |
| 1.7B RL Very Hard | 62.5 / 89.7 | 27.8 / 67.3 | 10.4 / 40.0 | 12.1 / 42.7 | 80.4 / 96.6 | 60.1 / 84.2 | 2.0 / 20.0 | 2.0 / 16.7 | 32.2 / 57.1 |
| 4B RL Very Hard ⇒ 1.7B Off | 61.1 / 92.0 | 25.3 / 72.3 | 10.9 / 49.0 | 10.0 / 45.7 | 78.6 / 96.9 | 59.7 / 88.4 | 2.5 / 23.3 | 2.2 / **30.0** | 31.3 / 62.2 |
| 4B RL Very Hard ⇒ 1.7B On | **69.5** / 93.3 | **32.4** / **78.3** | **14.8** / **53.0** | **12.7** / **50.3** | 83.1 / 96.4 | **66.3** / 88.6 | **4.3** / **26.7** | 3.8 / 23.3 | **35.9** / **63.7** |
| 30B RL Very Hard ⇒ 1.7B Off | 56.8 / 92.0 | 20.9 / 69.0 | 9.2 / 48.3 | 7.4 / 40.0 | 76.9 / **97.0** | 54.7 / 86.6 | 2.2 / 23.3 | 1.5 / 23.3 | 28.7 / 59.9 |
| 30B RL Very Hard ⇒ 1.7B On | 67.8 / **94.0** | 28.9 / 74.7 | 12.9 / 50.0 | 10.2 / 49.0 | **83.2** / 96.6 | 64.0 / **89.0** | 3.1 / 16.7 | **4.3** / **30.0** | 34.3 / 62.5 |

4B Medium

| Model | Easy | Medium | Hard | Very Hard | GSM8K | MATH500 | AIME 2025 | AIME 2026 | Average |
|---|---|---|---|---|---|---|---|---|---|
| 4B Base | 44.8 / 97.3 | 21.0 / 83.3 | 8.3 / 52.3 | 6.9 / 50.7 | 42.2 / 97.3 | 39.9 / 90.4 | 2.8 / 30.0 | 2.7 / 16.7 | 21.1 / 64.8 |
| 4B RL Medium | **89.1** / 98.7 | **64.5** / **95.3** | **37.4** / **78.0** | **25.5** / **68.7** | **93.3** / **98.3** | **83.7** / 93.4 | **17.3** / 36.7 | **14.0** / 30.0 | **53.1** / **74.9** |
| 30B RL Medium ⇒ 4B Off | 85.2 / **99.7** | 52.1 / 93.3 | 27.1 / 73.7 | 20.6 / 67.7 | 91.8 / 98.2 | 77.4 / **94.2** | 10.7 / **40.0** | 9.1 / 23.3 | 46.7 / 73.8 |
| 30B RL Medium ⇒ 4B On | 88.9 / **99.7** | 59.7 / 94.3 | 32.6 / 74.7 | 23.4 / **68.7** | 92.6 / 97.6 | 80.0 / 93.8 | 11.8 / 36.7 | 10.9 / **33.3** | 50.0 / 74.8 |

4B Hard

| Model | Easy | Medium | Hard | Very Hard | GSM8K | MATH500 | AIME 2025 | AIME 2026 | Average |
|---|---|---|---|---|---|---|---|---|---|
| 4B Base | 44.8 / 97.3 | 21.0 / 83.3 | 8.3 / 52.3 | 6.9 / 50.7 | 42.2 / 97.3 | 39.9 / 90.4 | 2.8 / 30.0 | 2.7 / 16.7 | 21.1 / 64.8 |
| 4B RL Hard | 87.6 / 98.3 | 65.2 / 94.7 | 39.4 / 83.7 | 29.2 / **74.7** | 92.8 / 98.4 | 84.0 / 95.4 | **19.1** / **53.3** | 15.8 / **36.7** | 54.1 / **79.4** |
| 30B RL Hard ⇒ 4B Off | 85.2 / **99.0** | 57.3 / **96.7** | 34.4 / 86.7 | 23.2 / 73.0 | 91.6 / **98.6** | 81.4 / 95.8 | 16.1 / 43.3 | 11.2 / 30.0 | 50.1 / 77.9 |
| 30B RL Hard ⇒ 4B On | **88.8** / **99.0** | **65.6** / 94.7 | **41.8** / **87.0** | **30.6** / 73.7 | **93.0** / 98.1 | **85.1** / **96.0** | 18.6 / 40.0 | **16.0** / 33.3 | **54.9** / 77.7 |

4B Very Hard

| Model | Easy | Medium | Hard | Very Hard | GSM8K | MATH500 | AIME 2025 | AIME 2026 | Average |
|---|---|---|---|---|---|---|---|---|---|
| 4B Base | 44.8 / 97.3 | 21.0 / 83.3 | 8.3 / 52.3 | 6.9 / 50.7 | 42.2 / 97.3 | 39.9 / 90.4 | 2.8 / 30.0 | 2.7 / 16.7 | 21.1 / 64.8 |
| 4B RL Very Hard | 84.3 / 98.0 | **55.5** / 89.7 | **30.6** / 71.7 | 23.8 / 63.7 | **92.5** / **98.0** | 77.7 / **92.6** | 8.9 / 30.0 | **10.0** / **33.3** | **47.9** / 72.1 |
| 30B RL Very Hard ⇒ 4B Off | 79.9 / **99.7** | 46.8 / 90.0 | 22.3 / **72.3** | 19.6 / 62.0 | 90.5 / 97.9 | 73.5 / 92.2 | 7.7 / 23.3 | 6.6 / **33.3** | 43.4 / 71.3 |
| 30B RL Very Hard ⇒ 4B On | **85.0** / 98.7 | 54.5 / **91.0** | 28.5 / 72.0 | **24.0** / **66.0** | 92.2 / 97.8 | **77.8** / 92.4 | **11.1** / **36.7** | 9.0 / 30.0 | 47.8 / **73.1** |

### Mode covering (forward KL, off-policy) vs mode seeking (reverse KL, on-policy)

Per-token entropy, measured as the mean NLL of each model's own temperature-1 samples on its band's training prompts (nats). For students it comes from OPD training logs: the start of OPD is the off-policy student, the end is after OPD.

| Pair | Teacher | Off-policy student (forward KL) | After OPD (reverse KL) |
|---|---|---|---|
| 4B RL ⇒ 1.7B (medium / hard / very hard) | 0.52 / 0.58 / 0.60 | 0.76 / 0.96 / 1.05 | 0.47 / 0.64 / 0.85 |
| 30B RL ⇒ 1.7B | 0.29 / 0.42 / 0.59 | 0.54 / 0.86 / 0.99 | 0.39 / 0.63 / 0.85 |
| 30B RL ⇒ 4B | 0.29 / 0.42 / 0.59 | 0.36 / 0.58 / 0.76 | 0.28 / 0.41 / 0.64 |

Answer diversity: distinct final answers per question out of 16 samples, on the hard validation set.

| | Off-policy | After OPD | Direct RL |
|---|---|---|---|
| 1.7B students | 9.6 – 11.3 | 7.7 – 9.4 | 7.4 – 9.1 |
| 4B students | 6.9 – 8.0 | 5.6 – 6.3 | 5.2 – 5.7 |

- Forward-KL (off-policy) students are 1.2–2.3× more entropic than their teacher and give the most distinct answers: classic mode covering. This is why off-policy distillation holds pass@k well despite a lower mean@k.
- Reverse-KL OPD pulls entropy back to about the teacher's level (slightly below it in 2 of 9 runs), cuts distinct answers by 20–30% to roughly the RL level, and narrows pass@16 − pass@1 (e.g. 30B RL Medium ⇒ 4B on medium: 41.2 → 34.6): mode seeking.
- pass@16 still rises at 1.7B because pass@1 rises far more than the spread shrinks. At 4B, pass@16 is flat, which is where the classic trade-off shows.

### Takeaways

- At 1.7B, on-policy distillation > RL > off-policy distillation on mean@k in every band. The best OPD students (4B teacher, medium or hard) average 39.7–39.8 vs 36.0 for the best 1.7B RL model, and OPD also has the best pass@k (68.0–68.1 for the hard-band students vs 63.8).
- At 4B (30B teacher only), OPD ≈ RL: hard 54.9 vs 54.1, very hard 47.8 vs 47.9, medium 50.0 vs 53.1. On pass@k, RL Hard is still highest (79.4 vs 77.7).
- OPD adds 3–7 points of mean@k on top of off-policy distillation within ≤ 120 steps. KL drops 15–25% and then plateaus around step 40–60, and validation plateaus with it.
- For the 1.7B student the 4B teacher is as good as or better than the 30B teacher (39.7 vs 36.9 on medium). Hard is the best training band for both teacher sizes.
- Caveats: OPD starts from the off-policy checkpoint, so OPD > Off is close to guaranteed; the real comparison is OPD vs RL. Compute is not matched (OPD = teacher RL + 1000 off-policy steps + ≤ 120 OPD steps). Best checkpoints are chosen on the in-distribution validation set, a ~0.5–1 point bias on that column. AIME is 30 questions, so 1 question = 3.3 points of pass@32.
