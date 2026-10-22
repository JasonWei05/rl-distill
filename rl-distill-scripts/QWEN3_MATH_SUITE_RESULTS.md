# Qwen3 math suite results

_Generated 2026-10-22 14:42Z by `rl-distill-scripts/summarize_qwen3_math_suite.py` from `s3://scale-ml/genai/rl-distill/qwen3-math-suite-v1/<model>/<set>/summary.json`._

Protocol: 0-shot native ChatML + "Please reason step by step, and put your final answer within \boxed{}.", temp 1, top-p 1,
8192 response tokens, strict last-box grading (= the RL reward). n = 16 samples per question (AIME 32); mean@n = average
accuracy, pass@n = unbiased any-of-n. Sets: the Qwen3-30B-defined DeepScaleR easy / medium / hard / very_hard validation
splits (300 q each), GSM8K test (1,319), MATH500 (500), AIME 2025 / 2026 (30 each). avg = plain mean over the 8 sets.
Teachers are the RL best checkpoints (30B-A3B: medium @320, hard @410, very_hard @460; 4B: @510 / @560 / @620).
Noise: ~0.4-0.6 pts SE on a 300-q set's mean@16; AIME pass@32 moves 3.3 pts per question.

## 1.7B students — mean@n

| group | model | easy | medium | hard | very_hard | gsm8k | math500 | aime2025 | aime2026 | avg |
|---|---|---|---|---|---|---|---|---|---|---|
| base | Qwen3-1.7B-Base | 31.4 | 9.6 | 3.8 | 3.3 | 40.3 | 30.1 | 1.0 | 0.5 | 15.0 |
| RL (best val ckpt) | RL medium | 70.7 | 33.1 | 14.6 | 12.1 | 83.1 | 66.2 | 5.4 | 2.5 | 36.0 |
| RL (best val ckpt) | RL hard | 66.4 | 28.4 | 12.1 | 11.2 | 80.9 | 62.6 | 2.7 | 3.3 | 33.5 |
| RL (best val ckpt) | RL very_hard | 62.5 | 27.8 | 10.4 | 12.1 | 80.4 | 60.1 | 2.0 | 2.0 | 32.2 |
| off-policy distill (step 1000) | 4B medium -> 1.7B | 67.2 | 27.8 | 12.1 | 9.4 | 82.6 | 64.9 | 3.5 | 2.5 | 33.7 |
| off-policy distill (step 1000) | 4B hard -> 1.7B | 64.2 | 26.4 | 11.9 | 9.7 | 81.0 | 63.4 | 4.2 | 2.0 | 32.8 |
| off-policy distill (step 1000) | 4B very_hard -> 1.7B | 61.1 | 25.3 | 10.9 | 10.0 | 78.6 | 59.7 | 2.5 | 2.2 | 31.3 |
| off-policy distill (step 1000) | 30B medium -> 1.7B | 64.2 | 25.9 | 11.5 | 8.8 | 80.2 | 61.4 | 2.9 | 2.1 | 32.1 |
| off-policy distill (step 1000) | 30B hard -> 1.7B | 61.8 | 24.9 | 11.3 | 9.0 | 79.1 | 62.3 | 3.8 | 2.4 | 31.8 |
| off-policy distill (step 1000) | 30B very_hard -> 1.7B | 56.8 | 20.9 | 9.2 | 7.4 | 76.9 | 54.7 | 2.2 | 1.5 | 28.7 |
| on-policy distill (best val <= 120) | 4B medium -> 1.7B | 75.6 | 39.1 | 19.2 | 13.2 | 85.9 | 72.2 | 6.7 | 5.4 | 39.7 |
| on-policy distill (best val <= 120) | 4B hard -> 1.7B | 74.7 | 37.9 | 19.4 | 14.1 | 85.5 | 73.1 | 7.8 | 5.6 | 39.8 |
| on-policy distill (best val <= 120) | 4B very_hard -> 1.7B | 69.5 | 32.4 | 14.8 | 12.7 | 83.1 | 66.3 | 4.3 | 3.8 | 35.9 |
| on-policy distill (best val <= 120) | 30B medium -> 1.7B | 72.5 | 33.9 | 15.5 | 11.7 | 84.1 | 67.9 | 4.9 | 4.4 | 36.9 |
| on-policy distill (best val <= 120) | 30B hard -> 1.7B | 73.3 | 36.0 | 18.1 | 13.2 | 84.7 | 72.1 | 9.2 | 7.3 | 39.2 |
| on-policy distill (best val <= 120) | 30B very_hard -> 1.7B | 67.8 | 28.9 | 12.9 | 10.2 | 83.2 | 64.0 | 3.1 | 4.3 | 34.3 |

## 1.7B students — pass@n

| group | model | easy | medium | hard | very_hard | gsm8k | math500 | aime2025 | aime2026 | avg |
|---|---|---|---|---|---|---|---|---|---|---|
| base | Qwen3-1.7B-Base | 85.7 | 55.7 | 32.3 | 30.3 | 94.6 | 81.0 | 13.3 | 16.7 | 51.2 |
| RL (best val ckpt) | RL medium | 93.7 | 79.7 | 54.0 | 50.0 | 97.0 | 89.4 | 33.3 | 13.3 | 63.8 |
| RL (best val ckpt) | RL hard | 94.7 | 75.0 | 51.0 | 48.7 | 97.0 | 88.4 | 23.3 | 26.7 | 63.1 |
| RL (best val ckpt) | RL very_hard | 89.7 | 67.3 | 40.0 | 42.7 | 96.6 | 84.2 | 20.0 | 16.7 | 57.1 |
| off-policy distill (step 1000) | 4B medium -> 1.7B | 95.3 | 74.7 | 49.7 | 44.0 | 96.7 | 89.2 | 30.0 | 23.3 | 62.9 |
| off-policy distill (step 1000) | 4B hard -> 1.7B | 93.7 | 74.7 | 54.0 | 47.7 | 97.0 | 89.2 | 20.0 | 13.3 | 61.2 |
| off-policy distill (step 1000) | 4B very_hard -> 1.7B | 92.0 | 72.3 | 49.0 | 45.7 | 96.9 | 88.4 | 23.3 | 30.0 | 62.2 |
| off-policy distill (step 1000) | 30B medium -> 1.7B | 95.0 | 75.7 | 49.3 | 46.7 | 96.9 | 89.4 | 20.0 | 23.3 | 62.0 |
| off-policy distill (step 1000) | 30B hard -> 1.7B | 95.3 | 77.7 | 52.7 | 49.3 | 97.1 | 89.4 | 26.7 | 20.0 | 63.5 |
| off-policy distill (step 1000) | 30B very_hard -> 1.7B | 92.0 | 69.0 | 48.3 | 40.0 | 97.0 | 86.6 | 23.3 | 23.3 | 59.9 |
| on-policy distill (best val <= 120) | 4B medium -> 1.7B | 97.0 | 84.0 | 59.7 | 49.7 | 97.0 | 92.2 | 30.0 | 20.0 | 66.2 |
| on-policy distill (best val <= 120) | 4B hard -> 1.7B | 95.3 | 84.0 | 62.3 | 56.0 | 97.5 | 91.8 | 33.3 | 23.3 | 68.0 |
| on-policy distill (best val <= 120) | 4B very_hard -> 1.7B | 93.3 | 78.3 | 53.0 | 50.3 | 96.4 | 88.6 | 26.7 | 23.3 | 63.7 |
| on-policy distill (best val <= 120) | 30B medium -> 1.7B | 96.0 | 80.7 | 55.3 | 51.0 | 96.7 | 89.8 | 26.7 | 23.3 | 64.9 |
| on-policy distill (best val <= 120) | 30B hard -> 1.7B | 96.3 | 84.0 | 61.7 | 53.0 | 97.4 | 92.0 | 26.7 | 33.3 | 68.1 |
| on-policy distill (best val <= 120) | 30B very_hard -> 1.7B | 94.0 | 74.7 | 50.0 | 49.0 | 96.6 | 89.0 | 16.7 | 30.0 | 62.5 |

## 4B students — mean@n

| group | model | easy | medium | hard | very_hard | gsm8k | math500 | aime2025 | aime2026 | avg |
|---|---|---|---|---|---|---|---|---|---|---|
| base | Qwen3-4B-Base | 44.8 | 21.0 | 8.3 | 6.9 | 42.2 | 39.9 | 2.8 | 2.7 | 21.1 |
| RL (best val ckpt) | RL medium | 89.1 | 64.5 | 37.4 | 25.5 | 93.3 | 83.7 | 17.3 | 14.0 | 53.1 |
| RL (best val ckpt) | RL hard | 87.6 | 65.2 | 39.4 | 29.2 | 92.8 | 84.0 | 19.1 | 15.8 | 54.1 |
| RL (best val ckpt) | RL very_hard | 84.3 | 55.5 | 30.6 | 23.8 | 92.5 | 77.7 | 8.9 | 10.0 | 47.9 |
| off-policy distill (step 1000) | 30B medium -> 4B | 85.2 | 52.1 | 27.1 | 20.6 | 91.8 | 77.4 | 10.7 | 9.1 | 46.7 |
| off-policy distill (step 1000) | 30B hard -> 4B | 85.2 | 57.3 | 34.4 | 23.2 | 91.6 | 81.4 | 16.1 | 11.2 | 50.1 |
| off-policy distill (step 1000) | 30B very_hard -> 4B | 79.9 | 46.8 | 22.3 | 19.6 | 90.5 | 73.5 | 7.7 | 6.6 | 43.4 |
| on-policy distill (best val <= 120) | 30B medium -> 4B | 88.9 | 59.7 | 32.6 | 23.4 | 92.6 | 80.0 | 11.8 | 10.9 | 50.0 |
| on-policy distill (best val <= 120) | 30B hard -> 4B | 88.8 | 65.6 | 41.8 | 30.6 | 93.0 | 85.1 | 18.6 | 16.0 | 54.9 |
| on-policy distill (best val <= 120) | 30B very_hard -> 4B | 85.0 | 54.5 | 28.5 | 24.0 | 92.2 | 77.8 | 11.1 | 9.0 | 47.8 |

## 4B students — pass@n

| group | model | easy | medium | hard | very_hard | gsm8k | math500 | aime2025 | aime2026 | avg |
|---|---|---|---|---|---|---|---|---|---|---|
| base | Qwen3-4B-Base | 97.3 | 83.3 | 52.3 | 50.7 | 97.3 | 90.4 | 30.0 | 16.7 | 64.8 |
| RL (best val ckpt) | RL medium | 98.7 | 95.3 | 78.0 | 68.7 | 98.3 | 93.4 | 36.7 | 30.0 | 74.9 |
| RL (best val ckpt) | RL hard | 98.3 | 94.7 | 83.7 | 74.7 | 98.4 | 95.4 | 53.3 | 36.7 | 79.4 |
| RL (best val ckpt) | RL very_hard | 98.0 | 89.7 | 71.7 | 63.7 | 98.0 | 92.6 | 30.0 | 33.3 | 72.1 |
| off-policy distill (step 1000) | 30B medium -> 4B | 99.7 | 93.3 | 73.7 | 67.7 | 98.2 | 94.2 | 40.0 | 23.3 | 73.8 |
| off-policy distill (step 1000) | 30B hard -> 4B | 99.0 | 96.7 | 86.7 | 73.0 | 98.6 | 95.8 | 43.3 | 30.0 | 77.9 |
| off-policy distill (step 1000) | 30B very_hard -> 4B | 99.7 | 90.0 | 72.3 | 62.0 | 97.9 | 92.2 | 23.3 | 33.3 | 71.3 |
| on-policy distill (best val <= 120) | 30B medium -> 4B | 99.7 | 94.3 | 74.7 | 68.7 | 97.6 | 93.8 | 36.7 | 33.3 | 74.8 |
| on-policy distill (best val <= 120) | 30B hard -> 4B | 99.0 | 94.7 | 87.0 | 73.7 | 98.1 | 96.0 | 40.0 | 33.3 | 77.7 |
| on-policy distill (best val <= 120) | 30B very_hard -> 4B | 98.7 | 91.0 | 72.0 | 66.0 | 97.8 | 92.4 | 36.7 | 30.0 | 73.1 |

## Checkpoints evaluated

| model | source |
|---|---|
| `q1p7b-base` | `hf:Qwen/Qwen3-1.7B-Base@ea980cb0a6c2ae4b936e82123acc929f1cec04c1` |
| `q1p7b-rl-medium-best` | `s3:s3://scale-ml/genai/rl-distill/qwen3-1.7b-base-rl/qwen3-1.7b-medium-es5/best_hf` |
| `q1p7b-rl-hard-best` | `s3:s3://scale-ml/genai/rl-distill/qwen3-1.7b-base-rl/qwen3-1.7b-hard-es5/best_hf` |
| `q1p7b-rl-very_hard-best` | `s3:s3://scale-ml/genai/rl-distill/qwen3-1.7b-base-rl/qwen3-1.7b-very_hard-es5/best_hf` |
| `q4bm-to-q1p7b-s1000` | `s3:s3://scale-ml/genai/rl-distill/qwen3-4b-rl-distill-ckpts-v1/q4bm-medium-to-q1p7b-bs128-s1000-lr2e-6/global_step_1000/huggingface` |
| `q4bh-to-q1p7b-s1000` | `s3:s3://scale-ml/genai/rl-distill/qwen3-4b-rl-distill-ckpts-v1/q4bh-hard-to-q1p7b-bs128-s1000-lr2e-6/global_step_1000/huggingface` |
| `q4bvh-to-q1p7b-s1000` | `s3:s3://scale-ml/genai/rl-distill/qwen3-4b-rl-distill-ckpts-v1/q4bvh-very_hard-to-q1p7b-bs128-s1000-lr2e-6/global_step_1000/huggingface` |
| `q30bm-to-q1p7b-s1000` | `s3:s3://scale-ml/genai/rl-distill/qwen3-30b-rl-distill-ckpts-v1/q30bm-medium-to-q1p7b-bs128-s1000-lr2e-6/global_step_1000/huggingface` |
| `q30bh-to-q1p7b-s1000` | `s3:s3://scale-ml/genai/rl-distill/qwen3-30b-rl-distill-ckpts-v1/q30bh-hard-to-q1p7b-bs128-s1000-lr2e-6/global_step_1000/huggingface` |
| `q30bvh-to-q1p7b-s1000` | `s3:s3://scale-ml/genai/rl-distill/qwen3-30b-rl-distill-ckpts-v1/q30bvh-very_hard-to-q1p7b-bs128-s1000-lr2e-6/global_step_1000/huggingface` |
| `q4bm-to-q1p7b-opd120` | `s3:s3://scale-ml/genai/rl-distill/qwen3-opd/q4bm-to-q1p7b-pgk1-v1/best_hf` |
| `q4bh-to-q1p7b-opd120` | `s3:s3://scale-ml/genai/rl-distill/qwen3-opd/q4bh-to-q1p7b-pgk1-v1/best_hf` |
| `q4bvh-to-q1p7b-opd120` | `s3:s3://scale-ml/genai/rl-distill/qwen3-opd/q4bvh-to-q1p7b-pgk1-v1/best_hf` |
| `q30bm-to-q1p7b-opd120` | `s3:s3://scale-ml/genai/rl-distill/qwen3-opd/q30bm-to-q1p7b-pgk1-v1/best_hf` |
| `q30bh-to-q1p7b-opd120` | `s3:s3://scale-ml/genai/rl-distill/qwen3-opd/q30bh-to-q1p7b-pgk1-v1/best_hf` |
| `q30bvh-to-q1p7b-opd120` | `s3:s3://scale-ml/genai/rl-distill/qwen3-opd/q30bvh-to-q1p7b-pgk1-v1/best_hf` |
| `q4b-base` | `hf:Qwen/Qwen3-4B-Base@906bfd4b4dc7f14ee4320094d8b41684abff8539` |
| `q4b-rl-medium-best` | `s3:s3://scale-ml/genai/rl-distill/qwen3-4b-base-rl/qwen3-4b-medium-es5/best_hf` |
| `q4b-rl-hard-best` | `s3:s3://scale-ml/genai/rl-distill/qwen3-4b-base-rl/qwen3-4b-hard-es5/best_hf` |
| `q4b-rl-very_hard-best` | `s3:s3://scale-ml/genai/rl-distill/qwen3-4b-base-rl/qwen3-4b-very_hard-es5/best_hf` |
| `q30bm-to-q4b-s1000` | `s3:s3://scale-ml/genai/rl-distill/qwen3-30b-rl-distill-ckpts-v1/q30bm-medium-to-q4b-bs128-s1000-lr2e-6/global_step_1000/huggingface` |
| `q30bh-to-q4b-s1000` | `s3:s3://scale-ml/genai/rl-distill/qwen3-30b-rl-distill-ckpts-v1/q30bh-hard-to-q4b-bs128-s1000-lr2e-6/global_step_1000/huggingface` |
| `q30bvh-to-q4b-s1000` | `s3:s3://scale-ml/genai/rl-distill/qwen3-30b-rl-distill-ckpts-v1/q30bvh-very_hard-to-q4b-bs128-s1000-lr2e-6/global_step_1000/huggingface` |
| `q30bm-to-q4b-opd120` | `s3:s3://scale-ml/genai/rl-distill/qwen3-opd/q30bm-to-q4b-pgk1-v1/best_hf` |
| `q30bh-to-q4b-opd120` | `s3:s3://scale-ml/genai/rl-distill/qwen3-opd/q30bh-to-q4b-pgk1-v1/best_hf` |
| `q30bvh-to-q4b-opd120` | `s3:s3://scale-ml/genai/rl-distill/qwen3-opd/q30bvh-to-q4b-pgk1-v1/best_hf` |
