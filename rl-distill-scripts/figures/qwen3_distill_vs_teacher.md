# Distilled student vs teacher

pass@k (unbiased, n=32 samples per question on the band's 300-q val set, strict boxed grading), mean@32, entropy = own-sample
NLL per token, forward KL = top-128 KL(teacher || student) per token on the teacher's held-out validation generations (W&B
val/loss; at step 0 it is the KL between the two untouched bases). Ideal: student rows match the teacher row.

## A. 4B base -> 30B base (per band)

### medium

| model | mean@32 | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 | entropy (NLL/tok) | fwd KL |
|---|---|---|---|---|---|---|---|---|---|
| **teacher: Qwen3-4B-Base** | 20.8 | 20.8 | 34.9 | 52.9 | 70.9 | 84.8 | 92.7 | 0.957 | 0.0000 |
| start: Qwen3-30B-A3B-Base | 27.9 | 27.9 | 45.9 | 67.3 | 85.6 | 95.8 | 99.0 | 0.406 | 0.0620 |
| 30B student @250 | 18.0 | 18.0 | 30.8 | 47.9 | 66.2 | 81.5 | 91.0 | 0.459 | 0.0419 |
| 30B student @500 | 17.8 | 17.8 | 30.8 | 48.2 | 66.9 | 82.1 | 91.7 | 0.424 | 0.0398 |
| 30B student @750 | 16.8 | 16.8 | 29.0 | 45.5 | 63.5 | 78.9 | 89.3 | 0.437 | 0.0389 |
| 30B student @1000 | 18.2 | 18.2 | 31.1 | 48.1 | 66.0 | 80.6 | 90.3 | 0.426 | 0.0387 |

### hard

| model | mean@32 | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | pass@32 | entropy (NLL/tok) | fwd KL |
|---|---|---|---|---|---|---|---|---|---|
| **teacher: Qwen3-4B-Base** | 8.6 | 8.6 | 15.7 | 26.7 | 41.5 | 57.7 | 72.0 | 0.973 | 0.0000 |
| start: Qwen3-30B-A3B-Base | 13.0 | 13.0 | 23.2 | 38.1 | 56.2 | 73.1 | 86.0 | 0.458 | 0.0983 |
| 30B student @250 | 7.5 | 7.5 | 13.7 | 23.5 | 37.2 | 53.3 | 69.7 | 0.451 | 0.0410 |
| 30B student @500 | 7.2 | 7.2 | 13.3 | 23.1 | 37.1 | 53.8 | 70.3 | 0.461 | 0.0355 |
| 30B student @750 | 7.7 | 7.7 | 14.2 | 24.5 | 39.1 | 56.6 | 73.0 | 0.447 | 0.0332 |
| 30B student @1000 | 7.9 | 7.9 | 14.5 | 25.0 | 39.6 | 56.8 | 73.7 | 0.445 | 0.0324 |

