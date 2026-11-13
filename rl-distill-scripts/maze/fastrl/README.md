# fastrl: single-GPU GRPO + TIS for the text-maze models

`fastrl.py` is the maze RL recipe of `../rl/rl_tis_run.sh` (the t3 grids: GRPO with 16 samples per prompt, reward 1 for a
shortest path, token-level TIS `min(pi / pi_rollout, 2)`, lr 1e-5 with a 20-step warmup, validation every 20 steps on 500
mazes x 16 samples) in one process and one file, without Ray, FSDP, HF generate or reward workers. `test_fastrl.py` checks
it piece by piece against the verl stack it replaces (TailRL `experiments/text_maze` + `../tailrl_rl_tis.patch`).

## Run

Needs the TailRL venv (torch 2.8, safetensors, pyarrow, wandb) and one GPU; prompts are tokenized once into
`<parquet>.tokens.npy` next to the parquet (1.3M train prompts: ~5 min, 850 MB).

```bash
python fastrl.py --base <HF dir of the start policy> --train <dir>/main_parquet/train.parquet \
  --val <dir>/eval500/test.parquet --out /tmp/runs/<name> --batch 64 --seed 3 --micro 512 --wandb-id <name>
```

`--micro` caps the samples per update micro-batch (verl `PPO_MICRO_SAMPLES`; the t3 grids used 512 for 16 / 64 prompts
per step and 1024 for 256 / 1024). The run dir gets `ckpt.pt` (resumed in place), `best.json` / `best_hf/` /
`COMPLETE.json` in verl's format, and the W&B files; W&B metric names are verl's (`val-aux/maze_17/is_shortest/mean@16`,
`tflops/cumulative_total`, ...). Several runs can share a GPU.

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
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.:$TAILRL/experiments/text_maze:../rl $TAILRL/.venv/bin/python -m pytest -q test_fastrl.py
```

16 tests, ~2 min on an H100. Exact: prompt tokens (HF tokenizer), maze parsing + BFS (TailRL `MazeEnv`, data
`optimal_path_length`), the judge on 29.5k real and adversarial responses (every outcome field), GRPO advantage, TIS loss
and its gradient, prompt order over several epochs, FLOPs. Up to float rounding, measured against float64: the forward
(fp32: as close as HF's fp32; fp16: as close as verl's prefix forward), decode log-probs (as close as verl's
`PrefixSharedDecoder`), one full update against verl's `DataParallelPPOActor` on FSDP over 6 batches (fp32 and fp16).
Statistical: first-token frequencies (chi-square), validation outcomes against verl's decoder (paired z < 4). Plus a
train / resume run.

## Results

(filled in below)
