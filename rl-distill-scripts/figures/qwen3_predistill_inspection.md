# Pre-distillation inspection (base -> base on-policy distillation)

Per band: the student by step vs the SOURCE base it imitates (target) and the base it started from. mean@16 / best@16 on
the band's val300 x16 (best@16 = verl bootstrap, ~pass@10); len = mean response tokens; NLL = training NLL/token of its own
samples (entropy estimate, +-5-step window); k1 = mean per-token log-ratio to the teacher on its samples.
`closest` = step >= 50 minimizing |d mean@16| + 20 |d NLL| to the source base (a suggestion, not a choice).

## 4B' (4B base <- 1.7B base), easy (finished, last step 250)

| step | mean@16 | best@16 | len | NLL/tok | k1 |
|---|---|---|---|---|---|
| **target: source base** | **32.0** | **82.3** | **793** | **0.970** | 0 |
| start: student base | 43.8 | 96.6 | 991 | 0.952 | |
| 50 | 49.0 | 94.5 | 734 | 0.393 | 0.047 |
| 100 | 43.7 | 91.2 | 701 | 0.456 | 0.048 |
| 150 | 44.4 | 91.3 | 732 | 0.446 | 0.046 |
| 200 | 44.0 | 89.6 | 746 | 0.444 | 0.046 |
| 240 (closest) | 42.5 | 89.4 | 715 | 0.514 | 0.046 |
| 250 | 44.2 | 88.6 | 747 | 0.447 | 0.042 |

## 4B' (4B base <- 1.7B base), medium (finished, last step 250)

| step | mean@16 | best@16 | len | NLL/tok | k1 |
|---|---|---|---|---|---|
| **target: source base** | **9.2** | **45.1** | **980** | **0.892** | 0 |
| start: student base | 20.9 | 77.5 | 1193 | 0.870 | |
| 50 | 15.7 | 60.9 | 943 | 0.476 | 0.053 |
| 100 | 15.8 | 60.0 | 944 | 0.449 | 0.048 |
| 150 | 14.4 | 57.7 | 995 | 0.496 | 0.050 |
| 170 (closest) | 13.7 | 55.6 | 1000 | 0.497 | 0.047 |
| 200 | 14.3 | 56.6 | 980 | 0.479 | 0.046 |
| 250 | 14.6 | 56.5 | 1014 | 0.521 | 0.047 |

## 4B' (4B base <- 1.7B base), hard (crashed, last step 240)

| step | mean@16 | best@16 | len | NLL/tok | k1 |
|---|---|---|---|---|---|
| **target: source base** | **3.3** | **21.8** | **1017** | **1.059** | 0 |
| start: student base | 8.8 | 47.1 | 1192 | 0.837 | |
| 50 | 6.4 | 35.3 | 988 | 0.506 | 0.054 |
| 100 | 5.8 | 33.0 | 951 | 0.537 | 0.053 |
| 150 | 5.7 | 30.6 | 1047 | 0.519 | 0.051 |
| 200 | 6.0 | 31.0 | 1068 | 0.548 | 0.050 |
| 220 (closest) | 5.4 | 32.1 | 1026 | 0.565 | 0.052 |

## 1.7B' (1.7B base <- 4B base), easy (finished, last step 250)

| step | mean@16 | best@16 | len | NLL/tok | k1 |
|---|---|---|---|---|---|
| **target: source base** | **43.8** | **96.6** | **991** | **0.952** | 0 |
| start: student base | 32.0 | 82.3 | 793 | 0.970 | |
| 50 | 38.9 | 85.1 | 678 | 0.444 | 0.071 |
| 100 | 42.5 | 87.4 | 742 | 0.433 | 0.065 |
| 150 | 42.1 | 86.4 | 686 | 0.479 | 0.067 |
| 200 | 43.0 | 86.2 | 716 | 0.422 | 0.063 |
| 220 (closest) | 43.7 | 88.0 | 712 | 0.457 | 0.064 |
| 250 | 44.5 | 85.2 | 722 | 0.488 | 0.066 |

## 1.7B' (1.7B base <- 4B base), medium (finished, last step 250)

| step | mean@16 | best@16 | len | NLL/tok | k1 |
|---|---|---|---|---|---|
| **target: source base** | **20.9** | **77.5** | **1193** | **0.870** | 0 |
| start: student base | 9.2 | 45.1 | 980 | 0.892 | |
| 50 | 14.1 | 56.0 | 932 | 0.488 | 0.072 |
| 100 | 13.5 | 54.5 | 977 | 0.509 | 0.072 |
| 130 (closest) | 14.3 | 53.8 | 994 | 0.507 | 0.068 |
| 150 | 13.5 | 51.4 | 945 | 0.499 | 0.069 |
| 200 | 14.2 | 53.7 | 972 | 0.509 | 0.068 |
| 250 | 13.7 | 54.3 | 1003 | 0.521 | 0.068 |

## 1.7B' (1.7B base <- 4B base), hard (finished, last step 250)

| step | mean@16 | best@16 | len | NLL/tok | k1 |
|---|---|---|---|---|---|
| **target: source base** | **8.8** | **47.1** | **1192** | **0.837** | 0 |
| start: student base | 3.3 | 21.8 | 1017 | 1.059 | |
| 50 | 5.8 | 32.6 | 971 | 0.548 | 0.076 |
| 100 | 5.2 | 28.7 | 1020 | 0.543 | 0.071 |
| 150 | 5.2 | 29.7 | 993 | 0.556 | 0.071 |
| 200 | 5.9 | 31.1 | 1038 | 0.530 | 0.071 |
| 250 (closest) | 5.6 | 32.0 | 1028 | 0.595 | 0.072 |

