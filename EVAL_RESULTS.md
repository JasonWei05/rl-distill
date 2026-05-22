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

## Accuracy — PT students distilled from `JWei05/dapo-gemma3-27b-pt-warmup20` @ step_000080 teacher (%)

Generation config T=1.0, top_p=0.7, top_k=-1, max_tokens=20480, mean@1, `math_verify` scoring (LatexExtractionConfig only).
Teacher: `JWei05/dapo-gemma3-27b-pt-warmup20` at revision `step_000080` (the 27B PT base after 80 DAPO RL steps). Data: n=2 responses/prompt on DAPO-Math-17k (34,796 rows at [`JWei05/DAPO-Gemma3-27B-PT-warmup20-step80-SFT-Data`](https://huggingface.co/datasets/JWei05/DAPO-Gemma3-27B-PT-warmup20-step80-SFT-Data)). Split: 16k train prompts × 2 / 1.4k val prompts × 1.
Student: `gemma-3-4b-pt` with `gemma-3-4b-it` chat template patched in. lr=2e-5 cosine, 250 steps, bs=128.

| Model / Checkpoint | AIME 2024 | AIME 2025 | AIME 2026 | MATH500 | OlympiadBench | MinervaMath | GSM8K | **Mean** |
|---|---|---|---|---|---|---|---|---|
| **gemma-3-4b-pt** (base PT, no SFT) | 0.10 | 0.00 | 0.10 | 4.40 | 1.11 | 2.39 | 1.80 | **1.41** |
| 4b pt / 27bptw20-step80 / step_000050 | 0.21 | 0.10 | 0.31 | 19.00 | 4.01 | 7.35 | 40.86 | 10.26 |
| 4b pt / 27bptw20-step80 / step_000100 | 0.00 | 0.00 | 0.10 | 15.60 | 5.12 | 7.63 | 38.74 | 9.60 |
| 4b pt / 27bptw20-step80 / step_000150 | 0.10 | 0.00 | 0.00 | 20.20 | 6.23 | 9.19 | 41.32 | 11.01 |
| 4b pt / 27bptw20-step80 / step_000200 | 0.52 | 0.21 | 0.21 | 21.20 | 5.93 | 9.93 | 43.67 | 11.67 |
| 4b pt / 27bptw20-step80 / step_000250 | 0.00 | 0.00 | 0.10 | 23.90 | 5.56 | 9.28 | 46.47 | **12.19** |
| **gemma-3-12b-pt** (base PT, no SFT) | 0.42 | 0.42 | 0.42 | 15.70 | 4.23 | 6.25 | 25.55 | **7.57** |
| 12b pt / 27bptw20-step80 / step_000150 | 0.62 | 0.42 | 0.21 | 31.40 | 9.35 | 14.71 | 68.61 | 17.90 |
| 12b pt / 27bptw20-step80 / step_000200 | 1.15 | 0.21 | 0.21 | 31.10 | 11.42 | 13.97 | 70.58 | **18.38** |
| 12b pt / 27bptw20-step80 / step_000250 | 1.56 | 0.42 | 0.21 | 33.70 | 9.94 | 15.26 | 73.46 | **19.22** |

12B PT variant (same setup, `gemma-3-12b-pt` student) is evaluating steps 150/200/250 — step_050/100 ckpts were evicted locally before HF upload completed during the 12B run.

## Accuracy — PT students distilled from raw `google/gemma-3-27b-pt` base teacher (%)

Generation config T=1.0, top_p=0.7, top_k=-1, max_tokens=20480, mean@1, `math_verify` scoring (LatexExtractionConfig only).
Teacher: raw `google/gemma-3-27b-pt` (no RL, no fine-tuning). Data: n=2 responses/prompt on DAPO-Math-17k (34,796 rows). Split: 16k train prompts × 2 / 1.4k val prompts × 1.
Student: `gemma-3-4b-pt` with `gemma-3-4b-it` chat template patched in. lr=2e-5 cosine, 250 steps, bs=128.

| Model / Checkpoint | AIME 2024 | AIME 2025 | AIME 2026 | MATH500 | OlympiadBench | MinervaMath | GSM8K | **Mean** |
|---|---|---|---|---|---|---|---|---|
| **gemma-3-4b-pt** (base PT, no SFT) | 0.10 | 0.00 | 0.10 | 4.40 | 1.11 | 2.39 | 1.80 | **1.41** |
| 4b pt / 27b-pt-base / step_000050 | 0.10 | 0.00 | 0.00 | 5.60 | 1.26 | 2.57 | 5.08 | **2.09** |
| 4b pt / 27b-pt-base / step_000100 | 0.10 | 0.00 | 0.00 | 7.10 | 1.11 | 2.48 | 3.56 | **2.05** |
| 4b pt / 27b-pt-base / step_000150 | 0.21 | 0.10 | 0.10 | 6.30 | 1.71 | 2.39 | 4.40 | **2.17** |
| 4b pt / 27b-pt-base / step_000200 | 0.00 | 0.21 | 0.10 | 7.10 | 0.89 | 2.94 | 5.38 | **2.38** |
| 4b pt / 27b-pt-base / step_000250 | 0.21 | 0.00 | 0.00 | 7.00 | 1.71 | 2.30 | 5.23 | **2.35** |

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

## Mean response length — PT from 27bptw20-step80 teacher (tokens)

| Model / Checkpoint | AIME 2024 | AIME 2025 | AIME 2026 | MATH500 | OlympiadBench | MinervaMath | GSM8K |
|---|---|---|---|---|---|---|---|
| 4b pt / 27bptw20-step80 / step_000050 | 13535 | 14373 | 14068 | 13423 | 14345 | 15521 | 10236 |
| 4b pt / 27bptw20-step80 / step_000100 | 4662 | 5918 | 4830 | 4792 | 6046 | 6111 | 1180 |
| 4b pt / 27bptw20-step80 / step_000150 | 7040 | 9258 | 6992 | 5849 | 7456 | 7299 | 1674 |
| 4b pt / 27bptw20-step80 / step_000200 | 5077 | 7039 | 4716 | 4182 | 5549 | 5626 | 1420 |
| 4b pt / 27bptw20-step80 / step_000250 | 6477 | 8711 | 6985 | 5096 | 6745 | 6120 | 1515 |
| 12b pt / 27bptw20-step80 / step_000150 | 4636 | 7890 | 6111 | 3921 | 5863 | 4602 | 728 |
| 12b pt / 27bptw20-step80 / step_000200 | 4408 | 7310 | 6422 | 3761 | 5107 | 4481 | 701 |
| 12b pt / 27bptw20-step80 / step_000250 | 4657 | 7384 | 6113 | 3655 | 5047 | 4312 | 829 |

## Mean response length — PT from raw 27b-pt-base teacher (tokens)

| Model / Checkpoint | AIME 2024 | AIME 2025 | AIME 2026 | MATH500 | OlympiadBench | MinervaMath | GSM8K |
|---|---|---|---|---|---|---|---|
| 4b pt / 27b-pt-base / step_000050 | 13627 | 13837 | 14261 | 12638 | 13330 | 13735 | 13031 |
| 4b pt / 27b-pt-base / step_000100 | 19342 | 18800 | 18756 | 18728 | 19045 | 19165 | 18620 |
| 4b pt / 27b-pt-base / step_000150 | 16575 | 16428 | 16162 | 16683 | 16546 | 16729 | 14827 |
| 4b pt / 27b-pt-base / step_000200 | 16482 | 15833 | 16072 | 15757 | 16054 | 16586 | 15186 |
| 4b pt / 27b-pt-base / step_000250 | 16336 | 16703 | 15382 | 16008 | 16317 | 15979 | 15617 |

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

---

# Eval: DAPO-Gemma3 *RL-trained* teacher distilled into Gemma 3 *PT* students (round 2)

This sweep uses the **RL-trained** teachers (DAPO checkpoints, not the underlying SFT/IT models) to off-policy distill into Gemma 3 PT students of multiple sizes. Six pairs total:

| Teacher (DAPO ckpt) | Teacher dataset (HF)                         | Student bases trained |
|---|---|---|
| 27B @ step_000080 (`JWei05/dapo-gemma3-27b-pt`) | `JWei05/DAPO-Gemma3-27B-PT-RL-SFT-Data` (n=2/prompt, 34,796 prompts, 69,592 rows) | 1B, 4B, 12B PT |
| 12B @ step_000080 (`JWei05/dapo-gemma3-12b-pt`) | `JWei05/DAPO-Gemma3-12B-PT-RL-SFT-Data` (n=2/prompt, same 34,796 prompts) | 1B, 4B PT |
| 4B  @ step_000060 (`JWei05/dapo-gemma3-4b-pt`)  | `JWei05/DAPO-Gemma3-4B-PT-RL-SFT-Data`  (n=2/prompt, same 34,796 prompts) | 1B PT |

## Teacher-data generation

- Teacher repos above were sampled with vLLM 0.11 (in-process engine, `VLLM_ENABLE_V1_MULTIPROCESSING=0`), TP=1, 8 shards/node, on `dapo_openmath2_mix.parquet` (34,796 prompts).
- Sampling: `temperature=1.0, top_p=1.0, top_k=-1, max_tokens=20480, max_model_len=22528`, `n=2` per prompt → 69,592 rows.
- For each row we recorded the assistant text plus per-token `teacher_log_probs` and `teacher_token_ids` (for forward-KL distillation loss).
- Generation script: `rl-distill-scripts/data/generate_teacher_data.py`; per-node launcher: `rl-distill-scripts/data/launch_teacher_gen.sh`; merger: `rl-distill-scripts/data/merge_teacher_shards.py`.

## SFT (off-policy distillation)

- **Train/val split** (`rl-distill-scripts/data/split_sft_dataset.py`, `seed=43`):
  - Group rows by `prompt_idx`, shuffle prompt-ids deterministically.
  - Train: **32,000 prompts × 2 outputs = 64,000 rows**.
  - Val: **1,000 of the remaining prompts × 1 output = 1,000 rows** (held out for in-loop forward-KL `val/loss`, *not* generative pass@1).
- **Loss**: per-token forward-KL `teacher_log_prob − student_log_prob` over assistant tokens (`rl-distill-scripts/forward_kl_loss.py`, `DistillSFTDataset` from `rl-distill-scripts/distill_dataset.py`).
- **Trainer**: `verl.trainer.sft_trainer.SFTTrainer` subclass `DistillSFTTrainer` in `rl-distill-scripts/main_distill_offpolicy.py`, FSDP2 across 8×B200, `transformer_layer_cls_to_wrap=["Gemma3DecoderLayer"]`, gradient checkpointing on.
- **Optim**: AdamW, `lr_max=5e-6`, `lr_warmup_steps=100` linear, `lr_scheduler_type=cosine`, `min_lr_ratio=0.1` (so end LR = 5e-7), `weight_decay=0.1`, `betas=[0.9,0.98]`, `clip_grad=1.0`.
- **Batch**: `train_batch_size=128`, `micro_batch_size_per_gpu=16` (8 for 12B student), `max_length=22528`, **1 epoch, 500 steps total**.
- **Tokenizer**: Gemma 3 PT repos lack a chat template, so the launcher pulls the IT counterpart's `chat_template.json` (or extracts from `tokenizer_config.json` for 1B), strips the alternating-role `raise_exception` (verl probes it with two consecutive user messages), and inlines into the student PT's `tokenizer_config.json`.
- **Checkpoints**: `save_freq=250` → saves at step 250 and 500. Each save async-pushed to `JWei05/gemma3-{1,4,12}b-pt-sft-distill-from-{4,12,27}b/step_NNNNNN`, with `delete_local_after=true` so on-device disk is reclaimed (`max_to_keep=2` on HF).
- **In-loop val**: `test_freq=5` → forward-KL loss on the 1k held-out rows every 5 SFT steps.
- **Pipeline driver**: `rl-distill-scripts/distill_sft_eval.sh` (env-var-parameterised by `TEACHER_REPO`, `TEACHER_PARQUET_NAME`, `STUDENT_HF_REPO`, `STUDENT_TAG`, `TEACHER_TAG`, plus optional `SEED`, `WARMUP_STEPS`, `LR_MAX`, `LR_SCHEDULER`, `MIN_LR_RATIO`, `TEST_FREQ`, `MICRO_BSZ`).

## Generative eval (post-training)

After each run's training finishes, the same script runs Phase C: two parallel `dapo/_eval_model_on_math.py` instances on the same node, **TP=4 each** (GPUs 0–3 / 4–7), one per saved checkpoint. Each pulls its `step_NNNNNN/` from the HF push repo, patches in the Gemma 3 multimodal preprocessor configs (`preprocessor_config.json`, `processor_config.json`) from `google/gemma-3-{1,4,12}b-pt`, loads vLLM, sweeps all 7 val parquets, scores with `math_verify` (LatexExtractionConfig, only `\boxed{}` answers credited), writes `<repo>__step_NNNNNN__summary.json`.

- **Sampling**: `temperature=1.0, top_p=0.7, top_k=-1, max_tokens=20480, max_model_len=22528, gpu_memory_utilization=0.80`, `n=1` (mean@1).
- **Val files** (per worker, all on `/home/tiger/verl/data/`):
  - `math__aime2024_repeated_32x_960.parquet` (n=960)
  - `math__aime2025_repeated_32x_960.parquet` (n=960)
  - `math__aime2026_repeated_32x_960.parquet` (n=960)
  - `math__math_500_repeated_2x_1000.parquet` (n=1000)
  - `math__olympiadbench_repeated_2x.parquet`  (n=1348)
  - `math__minervamath_repeated_4x.parquet`    (n=1088)
  - `math__gsm8k_test.parquet`                 (n=1319)
- Per-run JSONs: `/home/tiger/verl/data/eval_results/{EXP_NAME}/<repo>__step_NNNNNN__summary.json`.

## Accuracy — round-2 PT students (mean@1, %)

| Pair (teacher → student) | Step | AIME 2024 | AIME 2025 | AIME 2026 | MATH500 | OlympiadBench | MinervaMath | GSM8K | **Mean** |
|---|---|---|---|---|---|---|---|---|---|
| 27b → 4b PT  | 250 | 0.10 | 0.21 | 0.00 | 27.70 |  6.75 |  8.09 | 59.59 | **14.63** |
| 27b → 4b PT  | 500 | 0.42 | 0.21 | 0.21 | 30.70 |  7.72 |  8.73 | 61.18 | **15.59** |
| 12b → 4b PT  | 250 | 0.62 | 0.21 | 0.00 | 30.70 |  8.09 | 12.04 | 59.59 | **15.89** |
| 12b → 4b PT  | 500 | 0.73 | 0.21 | 0.10 | 32.60 |  8.23 | 12.59 | 58.91 | **16.20** |
| 27b → 12b PT | 250 | 1.46 | 0.10 | 0.10 | 42.30 | 12.76 | 15.72 | 78.54 | **21.57** |
| 27b → 12b PT | 500 | 0.62 | 0.83 | 1.77 | 42.70 | 14.69 | 17.00 | 79.30 | **22.42** |

Replicate 12b→4b run on a 2nd worker (same hyperparams, same seed=43, separate FSDP RNG) at step 500: 0.42 / 0.21 / 0.10 / 31.40 / 9.42 / 12.04 / 59.44 / **16.15** — within ~0.05 mean of run A, confirming run-to-run determinism is tight.

**1B-student evals** (completed via FLASH_ATTN backend workaround — see note below):

| Pair (teacher → student) | Step | AIME 2024 | AIME 2025 | AIME 2026 | MATH500 | OlympiadBench | MinervaMath | GSM8K | **Mean** |
|---|---|---|---|---|---|---|---|---|---|
| 4b → 1b PT  | 250 | 0.00 | 0.00 | 0.00 |  4.90 | 2.37 | 1.93 | 5.23 | **2.06** |
| 4b → 1b PT  | 500 | 0.00 | 0.00 | 0.00 |  4.30 | 3.04 | 1.19 | 4.70 | **1.89** |
| 12b → 1b PT | 250 | 0.00 | 0.10 | 0.10 |  3.60 | 1.78 | 1.10 | 3.94 | **1.52** |
| 12b → 1b PT | 500 | 0.00 | 0.21 | 0.00 |  3.60 | 1.71 | 1.56 | 4.70 | **1.68** |
| 27b → 1b PT | 250 | 0.00 | 0.21 | 0.10 |  2.60 | 0.96 | 1.29 | 3.41 | **1.23** |
| 27b → 1b PT | 500 | 0.00 | 0.00 | 0.00 |  2.80 | 1.41 | 1.19 | 3.64 | **1.29** |

Capability-gap observation: stronger teacher → weaker 1B student. The 27B teacher produces long reasoning trajectories (~10 k tokens) that the 1B model can't faithfully imitate (its own responses stay at ~1.5–3 k tokens), degrading the distilled student. 4b → 1b wins (mean ≈ 1.89–2.06), and gsm8k in particular tracks teacher-size *inversely* (5.23 → 3.94 → 3.41 at step 250 for 4b/12b/27b teachers). Step-500 is not strictly better than step-250 for the 1B runs (vs monotonic improvement for 4B/12B students).

**1B eval workaround.** The default vLLM V1 backend on B200 is FlashInfer, which has a known bug for `block_size=16 + head_size=256` (see `vllm/v1/attention/backends/flashinfer.py:623` referencing [flashinfer-ai#1993](https://github.com/flashinfer-ai/flashinfer/issues/1993)). Gemma 3 1B has `head_dim=256` which trips the assertion at default `block_size=16`. Passing `--block-size 32|64` triggers a FlashInfer JIT compile of `fmha_gen` which fails because `nvcc` on these pods doesn't recognize `compute_100a`. Neither `VLLM_ATTENTION_BACKEND=FLASH_ATTN` nor `VLLM_USE_V1=0` nor `enforce_eager=True` switches the backend. **Fix**: pass `attention_backend="FLASH_ATTN"` as a kwarg to `vllm.LLM(...)` (threads into `EngineArgs.attention_backend` before backend selection). `dapo/_eval_model_on_math.py` now exposes `--attention_backend`; use it with TP=4 for Gemma 3 1B evals.

## Wall-clock (eval gen seconds, summed over 7 datasets, TP=4)

| Pair | Step | Total gen seconds |
|---|---|---|
| 27b → 4b PT  | 250 | 4250 |
| 27b → 4b PT  | 500 | 3854 |
| 12b → 4b PT  | 250 | 2010 |
| 12b → 4b PT  | 500 | 2005 |
| 27b → 12b PT | 250 | 7523 |
| 27b → 12b PT | 500 | 6129 |
| 4b → 1b PT   | 250 | 1325 |
| 4b → 1b PT   | 500 | 1367 |
| 12b → 1b PT  | 250 | 2660 |
| 12b → 1b PT  | 500 | 2365 |
| 27b → 1b PT  | 250 | 3485 |
| 27b → 1b PT  | 500 | 3553 |

## Reproduction (round 2)

- Per-pair pipeline: `rl-distill-scripts/distill_sft_eval.sh`
- Queueing was done with local one-off wrappers around `rl-distill-scripts/distill_sft_eval.sh`.
- Eval scoring: `dapo/_eval_model_on_math.py` (now supports `--block_size`, `--enforce_eager`, `--attention_backend`)
- 1B-student eval workaround: call `dapo/_eval_model_on_math.py` with TP=4 and `--attention_backend FLASH_ATTN`
- HF model repos: `JWei05/gemma3-{1,4,12}b-pt-sft-distill-from-{4,12,27}b`

<!-- DISTILL_SFT_RESULTS_START -->
## Accuracy - Current PT SFT Distillation (8 eval sets)

Generation config: T=1.0, top_p=0.7, top_k=-1, max_tokens=20480, mean@1, `math_verify` scoring.
Checkpoint coverage depends on the run; the 32k x4 reruns evaluate `step_000250`, `step_000500`, `step_000750`, and `step_001000`.
Updated: 2026-05-10 17:23:54 UTC.

Summary JSON archive: `/mlx_devbox/users/jason.wei/playground/rl-distill/eval_results/distill_sft`.

### Accuracy (%)

| Run / Checkpoint | DAPO val | AIME 2024 | AIME 2025 | AIME 2026 | MATH500 | OlympiadBench | MinervaMath | GSM8K | Mean (8) |
|---|---|---|---|---|---|---|---|---|---|
| 12B PT <- 27B RL step40, all33296 x4 / step_000250 | 28.27 | 3.12 | 0.94 | 2.19 | 54.20 | 22.03 | 27.76 | 86.66 | 28.15 |
| 12B PT <- 27B RL step40, all33296 x4 / step_000500 | 29.60 | 2.60 | 2.29 | 1.88 | 56.90 | 24.70 | 27.21 | 87.11 | 29.04 |
| 12B PT <- 27B RL step40, all33296 x4 / step_000750 | 30.40 | 2.81 | 1.56 | 2.50 | 57.50 | 23.74 | 28.86 | 88.40 | 29.47 |
| 12B PT <- 27B RL step40, all33296 x4 / step_001000 | 30.73 | 3.02 | 2.29 | 2.29 | 57.50 | 24.48 | 28.77 | 87.72 | 29.60 |
| 12B PT <- 27B RL step40, lr 2.5e-6 / step_000250 | 27.80 | 1.67 | 1.56 | 0.52 | 53.90 | 22.55 | 26.56 | 83.78 | 27.29 |
| 12B PT <- 27B RL step40, lr 2.5e-6 / step_000500 | 29.27 | 2.50 | 1.15 | 1.04 | 55.40 | 21.88 | 27.76 | 85.97 | 28.12 |
| 4B PT <- 12B RL step20 / step_000250 | 17.00 | 0.52 | 0.52 | 0.10 | 33.10 | 9.57 | 12.68 | 62.17 | 16.96 |
| 4B PT <- 12B RL step20 / step_000500 | 18.73 | 0.83 | 0.21 | 0.83 | 35.20 | 9.94 | 11.95 | 66.87 | 18.07 |
| 4B PT <- 12B RL step20, all33296 x4 / step_000250 | 14.20 | 0.31 | 0.00 | 0.31 | 31.80 | 7.05 | 11.03 | 55.65 | 15.04 |
| 4B PT <- 12B RL step20, all33296 x4 / step_000500 | 17.73 | 0.31 | 0.42 | 0.31 | 32.70 | 9.72 | 12.04 | 60.80 | 16.75 |
| 4B PT <- 12B RL step20, all33296 x4 / step_000750 | 17.33 | 0.73 | 0.31 | 0.31 | 34.30 | 9.72 | 12.04 | 60.58 | 16.92 |
| 4B PT <- 12B RL step20, all33296 x4 / step_001000 | 18.07 | 0.52 | 0.21 | 0.31 | 35.30 | 9.87 | 12.78 | 61.64 | 17.34 |
| 4B PT <- 27B RL step40 / step_000250 | 16.53 | 0.73 | 0.42 | 0.00 | 34.20 | 8.68 | 11.76 | 59.74 | 16.51 |
| 4B PT <- 27B RL step40 / step_000500 | 18.13 | 0.83 | 0.10 | 0.52 | 34.90 | 10.09 | 11.40 | 60.88 | 17.11 |
| 4B PT <- 27B RL step40, all33296 x4 / step_000250 | 15.67 | 0.83 | 0.52 | 0.31 | 30.80 | 7.79 | 11.12 | 58.53 | 15.70 |
| 4B PT <- 27B RL step40, all33296 x4 / step_000500 | 18.07 | 0.94 | 0.10 | 0.31 | 34.30 | 8.53 | 12.32 | 62.09 | 17.08 |
| 4B PT <- 27B RL step40, all33296 x4 / step_000750 | 19.07 | 1.04 | 0.21 | 0.62 | 33.10 | 11.42 | 11.58 | 64.37 | 17.68 |
| 4B PT <- 27B RL step40, all33296 x4 / step_001000 | 18.40 | 0.94 | 0.21 | 0.42 | 37.30 | 10.24 | 11.03 | 63.84 | 17.80 |

### Mean Response Length (tokens)

| Run / Checkpoint | DAPO val | AIME 2024 | AIME 2025 | AIME 2026 | MATH500 | OlympiadBench | MinervaMath | GSM8K |
|---|---|---|---|---|---|---|---|---|
| 12B PT <- 27B RL step40, all33296 x4 / step_000250 | 2533 | 3636 | 3631 | 2872 | 1640 | 2370 | 2680 | 471 |
| 12B PT <- 27B RL step40, all33296 x4 / step_000500 | 2608 | 4019 | 3862 | 3334 | 1420 | 2105 | 2217 | 332 |
| 12B PT <- 27B RL step40, all33296 x4 / step_000750 | 2299 | 3501 | 3022 | 2941 | 1528 | 2272 | 2535 | 413 |
| 12B PT <- 27B RL step40, all33296 x4 / step_001000 | 2467 | 4013 | 4095 | 3170 | 1509 | 2553 | 2578 | 393 |
| 12B PT <- 27B RL step40, lr 2.5e-6 / step_000250 | 2849 | 4659 | 3692 | 3640 | 1591 | 2826 | 1554 | 277 |
| 12B PT <- 27B RL step40, lr 2.5e-6 / step_000500 | 2364 | 4197 | 3156 | 3296 | 1564 | 2393 | 1784 | 250 |
| 4B PT <- 12B RL step20 / step_000250 | 5032 | 7360 | 6646 | 6939 | 3307 | 5455 | 4782 | 1102 |
| 4B PT <- 12B RL step20 / step_000500 | 4612 | 7207 | 6348 | 6838 | 3495 | 4760 | 5957 | 911 |
| 4B PT <- 12B RL step20, all33296 x4 / step_000250 | 7047 | 9755 | 9480 | 8970 | 6416 | 6889 | 9837 | 4005 |
| 4B PT <- 12B RL step20, all33296 x4 / step_000500 | 6274 | 8377 | 8012 | 6772 | 4979 | 6210 | 7423 | 2624 |
| 4B PT <- 12B RL step20, all33296 x4 / step_000750 | 4881 | 7171 | 6773 | 6174 | 3281 | 4933 | 6125 | 1005 |
| 4B PT <- 12B RL step20, all33296 x4 / step_001000 | 4975 | 6937 | 7403 | 6561 | 3449 | 5359 | 5942 | 1239 |
| 4B PT <- 27B RL step40 / step_000250 | 3269 | 4996 | 4308 | 4679 | 2174 | 3571 | 2550 | 417 |
| 4B PT <- 27B RL step40 / step_000500 | 3203 | 4541 | 4245 | 4259 | 2400 | 3634 | 2458 | 492 |
| 4B PT <- 27B RL step40, all33296 x4 / step_000250 | 3557 | 4885 | 4863 | 4009 | 2289 | 3845 | 3885 | 839 |
| 4B PT <- 27B RL step40, all33296 x4 / step_000500 | 3906 | 5682 | 5292 | 4583 | 2596 | 4060 | 3943 | 689 |
| 4B PT <- 27B RL step40, all33296 x4 / step_000750 | 3768 | 5683 | 4786 | 4531 | 2377 | 3916 | 4125 | 684 |
| 4B PT <- 27B RL step40, all33296 x4 / step_001000 | 3819 | 5729 | 4933 | 4251 | 2377 | 3847 | 3819 | 534 |

### HF Repos

- 12B PT <- 27B RL step40, all33296 x4: `JWei05/gemma3-12b-pt-sft-distill-from-27b-rl-step40-seed43-all33296-n4`
- 12B PT <- 27B RL step40, lr 2.5e-6: `JWei05/gemma3-12b-pt-sft-distill-from-27b-rl-step40-seed43-lr2p5e-6`
- 4B PT <- 12B RL step20: `JWei05/gemma3-4b-pt-sft-distill-from-12b-rl-step20-seed43`
- 4B PT <- 12B RL step20, all33296 x4: `JWei05/gemma3-4b-pt-sft-distill-from-12b-rl-step20-seed43-all33296-n4`
- 4B PT <- 27B RL step40: `JWei05/gemma3-4b-pt-sft-distill-from-27b-rl-step40-seed43`
- 4B PT <- 27B RL step40, all33296 x4: `JWei05/gemma3-4b-pt-sft-distill-from-27b-rl-step40-seed43-all33296-n4`

<!-- DISTILL_SFT_RESULTS_END -->
