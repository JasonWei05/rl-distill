# Qwen3 math suite results

_Generated 2026-10-22 19:14Z by `rl-distill-scripts/summarize_qwen3_math_suite.py` from `s3://scale-ml/genai/rl-distill/qwen3-math-suite-v1/<model>/<set>/summary.json`._

Protocol: 0-shot native ChatML + "Please reason step by step, and put your final answer within \boxed{}.", temp 1, top-p 1,
8192 response tokens, strict last-box grading (= the RL reward). n = 16 samples per question (AIME 32); mean@n = average
accuracy, pass@n = unbiased any-of-n. Sets: the Qwen3-30B-defined DeepScaleR easy / medium / hard / very_hard validation
splits (300 q each), GSM8K test (1,319), MATH500 (500), AIME 2025 / 2026 (30 each). avg = plain mean over the 8 sets.
Teachers are the RL best checkpoints (30B-A3B: medium @320, hard @410, very_hard @460; 4B: @510 / @560 / @620).
Noise: ~0.4-0.6 pts SE on a 300-q set's mean@16; AIME pass@32 moves 3.3 pts per question.

## 1.7B students — mean@n / pass@n

| group | model | easy | medium | hard | very_hard | gsm8k | math500 | aime2025 | aime2026 | avg |
|---|---|---|---|---|---|---|---|---|---|---|
| base | Qwen3-1.7B-Base | 31.4 / 85.7 | 9.6 / 55.7 | 3.8 / 32.3 | 3.3 / 30.3 | 40.3 / 94.6 | 30.1 / 81.0 | 1.0 / 13.3 | 0.5 / 16.7 | 15.0 / 51.2 |
| RL (best val ckpt) | RL medium | 70.7 / 93.7 | 33.1 / 79.7 | 14.6 / 54.0 | 12.1 / 50.0 | 83.1 / 97.0 | 66.2 / 89.4 | 5.4 / 33.3 | 2.5 / 13.3 | 36.0 / 63.8 |
| RL (best val ckpt) | RL hard | 66.4 / 94.7 | 28.4 / 75.0 | 12.1 / 51.0 | 11.2 / 48.7 | 80.9 / 97.0 | 62.6 / 88.4 | 2.7 / 23.3 | 3.3 / 26.7 | 33.5 / 63.1 |
| RL (best val ckpt) | RL very_hard | 62.5 / 89.7 | 27.8 / 67.3 | 10.4 / 40.0 | 12.1 / 42.7 | 80.4 / 96.6 | 60.1 / 84.2 | 2.0 / 20.0 | 2.0 / 16.7 | 32.2 / 57.1 |
| off-policy distill (step 1000) | 4B medium -> 1.7B | 67.2 / 95.3 | 27.8 / 74.7 | 12.1 / 49.7 | 9.4 / 44.0 | 82.6 / 96.7 | 64.9 / 89.2 | 3.5 / 30.0 | 2.5 / 23.3 | 33.7 / 62.9 |
| off-policy distill (step 1000) | 4B hard -> 1.7B | 64.2 / 93.7 | 26.4 / 74.7 | 11.9 / 54.0 | 9.7 / 47.7 | 81.0 / 97.0 | 63.4 / 89.2 | 4.2 / 20.0 | 2.0 / 13.3 | 32.8 / 61.2 |
| off-policy distill (step 1000) | 4B very_hard -> 1.7B | 61.1 / 92.0 | 25.3 / 72.3 | 10.9 / 49.0 | 10.0 / 45.7 | 78.6 / 96.9 | 59.7 / 88.4 | 2.5 / 23.3 | 2.2 / 30.0 | 31.3 / 62.2 |
| off-policy distill (step 1000) | 30B medium -> 1.7B | 64.2 / 95.0 | 25.9 / 75.7 | 11.5 / 49.3 | 8.8 / 46.7 | 80.2 / 96.9 | 61.4 / 89.4 | 2.9 / 20.0 | 2.1 / 23.3 | 32.1 / 62.0 |
| off-policy distill (step 1000) | 30B hard -> 1.7B | 61.8 / 95.3 | 24.9 / 77.7 | 11.3 / 52.7 | 9.0 / 49.3 | 79.1 / 97.1 | 62.3 / 89.4 | 3.8 / 26.7 | 2.4 / 20.0 | 31.8 / 63.5 |
| off-policy distill (step 1000) | 30B very_hard -> 1.7B | 56.8 / 92.0 | 20.9 / 69.0 | 9.2 / 48.3 | 7.4 / 40.0 | 76.9 / 97.0 | 54.7 / 86.6 | 2.2 / 23.3 | 1.5 / 23.3 | 28.7 / 59.9 |
| on-policy distill (best val <= 120) | 4B medium -> 1.7B | 75.6 / 97.0 | 39.1 / 84.0 | 19.2 / 59.7 | 13.2 / 49.7 | 85.9 / 97.0 | 72.2 / 92.2 | 6.7 / 30.0 | 5.4 / 20.0 | 39.7 / 66.2 |
| on-policy distill (best val <= 120) | 4B hard -> 1.7B | 74.7 / 95.3 | 37.9 / 84.0 | 19.4 / 62.3 | 14.1 / 56.0 | 85.5 / 97.5 | 73.1 / 91.8 | 7.8 / 33.3 | 5.6 / 23.3 | 39.8 / 68.0 |
| on-policy distill (best val <= 120) | 4B very_hard -> 1.7B | 69.5 / 93.3 | 32.4 / 78.3 | 14.8 / 53.0 | 12.7 / 50.3 | 83.1 / 96.4 | 66.3 / 88.6 | 4.3 / 26.7 | 3.8 / 23.3 | 35.9 / 63.7 |
| on-policy distill (best val <= 120) | 30B medium -> 1.7B | 72.5 / 96.0 | 33.9 / 80.7 | 15.5 / 55.3 | 11.7 / 51.0 | 84.1 / 96.7 | 67.9 / 89.8 | 4.9 / 26.7 | 4.4 / 23.3 | 36.9 / 64.9 |
| on-policy distill (best val <= 120) | 30B hard -> 1.7B | 73.3 / 96.3 | 36.0 / 84.0 | 18.1 / 61.7 | 13.2 / 53.0 | 84.7 / 97.4 | 72.1 / 92.0 | 9.2 / 26.7 | 7.3 / 33.3 | 39.2 / 68.1 |
| on-policy distill (best val <= 120) | 30B very_hard -> 1.7B | 67.8 / 94.0 | 28.9 / 74.7 | 12.9 / 50.0 | 10.2 / 49.0 | 83.2 / 96.6 | 64.0 / 89.0 | 3.1 / 16.7 | 4.3 / 30.0 | 34.3 / 62.5 |

## 4B students — mean@n / pass@n

| group | model | easy | medium | hard | very_hard | gsm8k | math500 | aime2025 | aime2026 | avg |
|---|---|---|---|---|---|---|---|---|---|---|
| base | Qwen3-4B-Base | 44.8 / 97.3 | 21.0 / 83.3 | 8.3 / 52.3 | 6.9 / 50.7 | 42.2 / 97.3 | 39.9 / 90.4 | 2.8 / 30.0 | 2.7 / 16.7 | 21.1 / 64.8 |
| RL (best val ckpt) | RL medium | 89.1 / 98.7 | 64.5 / 95.3 | 37.4 / 78.0 | 25.5 / 68.7 | 93.3 / 98.3 | 83.7 / 93.4 | 17.3 / 36.7 | 14.0 / 30.0 | 53.1 / 74.9 |
| RL (best val ckpt) | RL hard | 87.6 / 98.3 | 65.2 / 94.7 | 39.4 / 83.7 | 29.2 / 74.7 | 92.8 / 98.4 | 84.0 / 95.4 | 19.1 / 53.3 | 15.8 / 36.7 | 54.1 / 79.4 |
| RL (best val ckpt) | RL very_hard | 84.3 / 98.0 | 55.5 / 89.7 | 30.6 / 71.7 | 23.8 / 63.7 | 92.5 / 98.0 | 77.7 / 92.6 | 8.9 / 30.0 | 10.0 / 33.3 | 47.9 / 72.1 |
| off-policy distill (step 1000) | 30B medium -> 4B | 85.2 / 99.7 | 52.1 / 93.3 | 27.1 / 73.7 | 20.6 / 67.7 | 91.8 / 98.2 | 77.4 / 94.2 | 10.7 / 40.0 | 9.1 / 23.3 | 46.7 / 73.8 |
| off-policy distill (step 1000) | 30B hard -> 4B | 85.2 / 99.0 | 57.3 / 96.7 | 34.4 / 86.7 | 23.2 / 73.0 | 91.6 / 98.6 | 81.4 / 95.8 | 16.1 / 43.3 | 11.2 / 30.0 | 50.1 / 77.9 |
| off-policy distill (step 1000) | 30B very_hard -> 4B | 79.9 / 99.7 | 46.8 / 90.0 | 22.3 / 72.3 | 19.6 / 62.0 | 90.5 / 97.9 | 73.5 / 92.2 | 7.7 / 23.3 | 6.6 / 33.3 | 43.4 / 71.3 |
| on-policy distill (best val <= 120) | 30B medium -> 4B | 88.9 / 99.7 | 59.7 / 94.3 | 32.6 / 74.7 | 23.4 / 68.7 | 92.6 / 97.6 | 80.0 / 93.8 | 11.8 / 36.7 | 10.9 / 33.3 | 50.0 / 74.8 |
| on-policy distill (best val <= 120) | 30B hard -> 4B | 88.8 / 99.0 | 65.6 / 94.7 | 41.8 / 87.0 | 30.6 / 73.7 | 93.0 / 98.1 | 85.1 / 96.0 | 18.6 / 40.0 | 16.0 / 33.3 | 54.9 / 77.7 |
| on-policy distill (best val <= 120) | 30B very_hard -> 4B | 85.0 / 98.7 | 54.5 / 91.0 | 28.5 / 72.0 | 24.0 / 66.0 | 92.2 / 97.8 | 77.8 / 92.4 | 11.1 / 36.7 | 9.0 / 30.0 | 47.8 / 73.1 |

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
