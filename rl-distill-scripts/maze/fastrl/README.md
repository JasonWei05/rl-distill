# fastrl: single-GPU GRPO + TIS for the text-maze models

> **2026-11-17:** the verl comparison suite described below (`test_fastrl.py`) was removed (last at `e642401a`); synth/ no
> longer depends on verl. synth/fastrl.py now runs the math recipe's PPO clip x token TIS loss with mini-batches
> (`--minibatches`); its tests are in synth/tests. This page records the maze grids' trainer as it was.

fastrl is the maze RL recipe of `../rl/rl_tis_run.sh` (the t3 grids: GRPO with 16 samples per prompt, reward 1 for a
shortest path, token-level TIS `min(pi / pi_rollout, 2)`, lr 1e-5 with a 20-step warmup, validation every 20 steps on 500
mazes x 16 samples) in one process, without Ray, FSDP, HF generate or reward workers. It now lives in `../../synth/`
(`fastrl.py`, the model and sampler in `model.py`, the maze in `tasks.py`), where it runs any synth task;
`test_fastrl.py` here checks it piece by piece against the verl stack it replaces (TailRL `experiments/text_maze` +
`../tailrl_rl_tis.patch`) on the 25x25 study's data and bases. The single-file version, its benchmark and its
ScaleTrain launcher (used for the results below) are at commit `a1aca9f5` (`maze/fastrl/`).

## What it does the same way as verl

- Prompt order: verl's `RandomSampler` (generator seeded with the run seed), `drop_last` batches.
- Sampling at temperature 1 from the fp16 policy; the log-prob of each sampled token under the sampler is the TIS
  behaviour policy (verl `skip_old_log_prob`).
- Reward: TailRL's `judge_maze_binary_shortest` on `decode(response, skip_special_tokens=True)`, reimplemented on token ids.
- GRPO advantage, the TIS loss (token-mean per micro-batch, divided by the number of micro-batches), fp32 master weights
  with fp16 forward / backward (FSDP mixed precision, log-softmax in fp32 as under verl's autocast), the dynamic loss
  scale (2^16, growth every 400 steps), grad-norm clip 1.0, AdamW (0.9, 0.999, wd 0.01), the warmup (update k uses
  `lr * min(1, (k - 1) / 20)`), the FLOPs ledger formula.

## Where the speed comes from

- No Ray / DataProto transfers, FSDP hooks, HF generate, per-row Python log-softmax or multiprocess reward.
- Decode: each prompt's KV cache is shared by its 16 samples; the prompt part of attention is one flash-attention call
  per group, merged with the per-row part through the log-sum-exp; finished rows are compacted away every 32 tokens;
  each 32-token segment's step is one CUDA graph; small elementwise ops are compiled into single kernels.
- Reward and BFS shortest paths are batched on the GPU.
- Update: one prompt forward per group (as verl's `PREFIX_FORWARD`), a lower-right causal mask instead of a dense one.

## Tests

```bash
cd rl-distill-scripts/maze/fastrl
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../../synth:$TAILRL/experiments/text_maze:../rl $TAILRL/.venv/bin/python -m pytest -q test_fastrl.py
```

16 tests, ~90 s on an H100 (the train / resume test is in `synth/tests`). Exact: the synth maze generator reproduces the
study's tokenized prompts and pretraining sequences, the response mask, maze parsing + BFS (TailRL `MazeEnv`, data
`optimal_path_length`), the judge on 31k real and adversarial responses (every outcome field), GRPO advantage, TIS loss
and its gradient, prompt order over several epochs, FLOPs. Up to float rounding, measured against float64: the forward
(fp32: as close as HF's fp32; fp16: as close as verl's prefix forward), decode log-probs (as close as verl's
`PrefixSharedDecoder`), one full update against verl's `DataParallelPPOActor` on FSDP over 6 batches (fp32 and fp16).
Statistical: first-token frequencies (chi-square), validation outcomes against verl's decoder (paired z < 4).

## Results (2026-11-13)

Speed, one dedicated H100 pod (`MODE=bench`, `fastrl-bench-v1d`), 2XS base at 3%, t3 recipe; seconds per training step
(median), validation = 500 mazes x 16 samples:

| prompts / step | verl (TailRL, same pod) | fastrl | speedup |
|---|---|---|---|
| 16 | 0.535 | 0.252 | 2.1x |
| 64 | 1.18 | 0.340 | 3.5x |
| 256 | 3.72 | 0.638 | 5.8x |
| 1024 | 8.19 on 4 GPUs (grid pods) | 1.80 on 1 GPU | 4.5x (18x per GPU) |
| validation | 5.9 | 0.43 | 14x |

fastrl by model size (same pod): 3XS 0.26 / 0.36 / 0.60 / 1.61 s, small 0.27 / 0.39 / 0.78 / 2.39 s at 16 / 64 / 256 /
1024 prompts. Peak GPU memory (allocated), 2XS: 4.0 / 5.1 / 10.2 / 10.3 GB against verl's 3.9 / 7.6 / 15.1 GB, and
15.1 GB on each of 4 GPUs at 1024. At 16-64 prompts the GPU is mostly idle (the step is decode latency over up
to 300 tokens), so packing several runs per GPU raises throughput.

Learning curves: fastrl reruns of finished verl runs (2XS, 3% base, seeds 2 / 3 / 4, same bases, seeds and prompt order),
validation is_shortest averaged over each 1000-step window (seed mean; the ranges overlap throughout):

| steps | fastrl, 16 prompts | verl, 16 prompts | fastrl, 64 prompts | verl, 64 prompts |
|---|---|---|---|---|
| 0-999 | 0.0747 | 0.0748 | 0.0917 | 0.0928 |
| 1000-1999 | 0.1038 | 0.1016 | 0.1298 | 0.1301 |
| 2000-2999 | 0.1198 | 0.1181 | 0.1494 | 0.1494 |
| 3000-3999 | 0.1294 | 0.1297 | 0.1610 | 0.1627 |
| 4000-4999 | 0.1402 | 0.1384 | 0.1690 | 0.1711 |
| 5000-5999 | 0.1476 | 0.1450 | 0.1763 | 0.1803 |
| 6000-6999 | 0.1524 | 0.1507 | 0.1821 | 0.1853 |
| 7000-7999 | 0.1583 | 0.1538 | 0.1891 | 0.1938 |
| 8000-8999 | 0.1623 | 0.1621 | 0.1978 | 0.1991 |
| 9000-9999 | 0.1655 | 0.1670 | 0.2036 | 0.2049 |
| best (seeds 2 / 3 / 4) | 0.145 / 0.206 / 0.180 | 0.147 / 0.209 / 0.184 | 0.180 / 0.235 / 0.230 | 0.177 / 0.243 / 0.226 |

256 prompts per step (ScaleTrain, one dedicated H100 each, 10k steps) and 1024 prompts (seed 3):

| steps | fastrl, 256 | verl, 256 | fastrl, 1024 (s3) | verl, 1024 (s3) |
|---|---|---|---|---|
| 0-999 | 0.1099 | 0.1103 | 0.1470 | 0.1476 |
| 1000-1999 | 0.1553 | 0.1558 | 0.2067 | 0.2081 |
| 2000-2999 | 0.1758 | 0.1757 | 0.2302 | 0.2313 |
| 3000-3999 | 0.1908 | 0.1916 | 0.2427 | 0.2443 |
| 4000-4999 | 0.2037 | 0.2042 | 0.2565 | 0.2561 |
| 5000-5999 | 0.2115 | 0.2118 | 0.2699 | 0.2687 |
| 6000-6999 | 0.2205 | 0.2210 | 0.2790 | 0.2770 |
| 7000-7999 | 0.2267 | 0.2263 | 0.2855 | 0.2876 |
| 8000-8999 | 0.2327 | 0.2334 | 0.2914 | 0.2910 |
| 9000-9999 | 0.2393 | 0.2411 | 0.2974 | 0.2970 |
| best (seeds 2 / 3 / 4) | 0.207 / 0.274 / 0.270 | 0.208 / 0.274 / 0.274 | 0.307 (s3) | 0.307 (s3) |
| wall clock, 10k steps | 1.47 h, 1 GPU | 11.7-12.2 h, 1 GPU | 4.30 h, 1 GPU | 24.0 h, 4 GPUs |

The FLOPs ledgers agree too (256, seed 2: 417.6k vs 416.7k TFLOPs; 1024: 1.681M vs 1.677M).

Runs: `/tmp/fastrl/runs/e2e` on the devbox (W&B `fastrl-maze`, `fr-x2t3-p3-b{16,64}-s{2,3,4}`); 256 / 1024 prompts per step
on ScaleTrain (`launch_fastrl_st.sh`, `s3://scale-ml/genai/rl-distill/maze/fastrl_v1/runs/`).
