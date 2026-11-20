# synth: synthetic-task pipeline (data → SFT → base tuning → GRPO + TIS RL)

One config file per environment (`configs/<name>.yaml`) drives every stage; the task (maze, sudoku or chess) is a
plug-in in `tasks.py`. Everything runs on one GPU per job, with many jobs packed per GPU on ScaleTrain pods.

| File | What |
|---|---|
| `tasks.py` | `Maze` (depth-first + wall knockout, ε-noisy shortest-path demonstrations, shortest-path reward), `Sudoku` (solver-trace demonstrations, solved-grid reward) and `Chess` (pre2post-chess, arXiv 2607.16097: Lichess puzzles with their search-trace CoT, tokenizer and benchmark; scripted multi-turn answers, their reward): vocabulary, examples (CPU), `parse` + `judge` (GPU) |
| `chess_hf/` | their released tokenizer files (MIT), copied into exported chess models |
| `data.py` | config → `<split>.npy` uint8 token arrays (dev 1000 / test 1000 / val 256 / train) + `meta.json` |
| `model.py` | Qwen2 / Qwen3 run functionally on a weight dict; init, HF model directories (+ `synth.json`), the prefix-shared CUDA-graph sampler (response caches grow as the responses do; chess: the environment's replies fed after each call) |
| `sft.py` | SFT from scratch or from an HF model (`--init`; chess: their recipe), checkpoints and pass@1 screens on a schedule, the validation loss every `sft.val_every` steps |
| `eval.py`, `eval_curve.py`, `export.py` | pass@k of a model or checkpoint on a split (per group: chess B1–B5); every checkpoint of a running SFT; a checkpoint as an HF model directory |
| `theirs.py` | their SFT trainer (pre2post-chess, unchanged) as a pack job, for parity |
| `tune.py` | an SFT run → one RL base per pass@1 level (checkpoint averaging + weight interpolation, sampled pass@1) |
| `fastrl.py` | GRPO with the math recipe's PPO clip x token TIS loss; `--minibatches k` updates per step on k mini-batches; `--gpus K`: data parallel, same update |
| `plan.py`, `pack.py` | job plans (SFT+tune, RL grids, evals, split over pods) and the pod-side packer (GPU shares packed best-fit, multi-GPU jobs, S3 mirror from hard-link snapshots, resume) |
| `chess_rl_plan.py`, `chess_report.py` (+ `.html`) | the chess RL grid's plans (lr x model x batch, split over a borrowing-false and a borrowing-true pod) and its results page (bases, RL curves, the bigger-set SFT) |
| `report.py`, `bench.py`, `bench_decode.py`, `files.py` | maze / sudoku result tables from S3; RL step timing; decoder A/B; atomic run-dir writes |
| `run_synth_st.sh`, `launch_synth_st.sh` | ScaleTrain run-file and supervised launcher |

## Configs

`configs/maze33.yaml` (ε 0.3), `maze49.yaml` (ε 0.2), `sudoku9.yaml` (40–75 clues), `sudoku9hard.yaml` (30–55),
`maze25.yaml` (reproduces the 25×25 study's data, seed base 60M), `chess.yaml` (their 85 SFT trace shards and recipe),
`chess_big.yaml` (all 180 shards, 2.3x the traces). Sections: `task` (the task's constructor arguments), `max_response` (data / SFT
width), `data`, `sft`, `tune`, `decode` (sampling cap and KV budget for screens, tuning and RL), `rl`. A dataset's
`meta.json`, every model directory's `synth.json` and every SFT run's `run.json` carry the config; stages refuse inputs
built from a different one in a frozen section (`task`, `max_response`, `data`, `sft`): the `tune`, `decode` and `rl`
sections may change after the data is built (`data.py --config ... --out ... --refresh` updates `meta.json`; sync it to
S3 only while no pod uses the dataset).

## Run

```bash
cd rl-distill-scripts/synth; PY=/mnt/efs/jasonwei/rl-distill/.venv/bin/python
$PY data.py --config configs/sudoku9.yaml --out /tmp/synth/data/sudoku9 --workers 48       # ~30 s (maze49: ~25 min)
$PY sft.py --data /tmp/synth/data/sudoku9 --size 2XS --seed 0 --out /tmp/synth/runs/sudoku9/sft/2XS_s0 \
  --wandb-project synth-sudoku9-sft --wandb-id sudoku9-2XS-s0
$PY tune.py --run /tmp/synth/runs/sudoku9/sft/2XS_s0 --data /tmp/synth/data/sudoku9
$PY fastrl.py --base /tmp/synth/runs/sudoku9/sft/2XS_s0/tune/p3 --data /tmp/synth/data/sudoku9 \
  --out /tmp/synth/runs/sudoku9/rl/2XS_p3_s0_b64 --batch 64 --seed 0 --wandb-project synth-sudoku9-rl --wandb-id x
```

On ScaleTrain (data under `s3://scale-ml/genai/rl-distill/synth/data/<config>/`, runs under `.../synth/runs/`):

```bash
$PY plan.py sft --configs maze33 --sizes 2XS XS S --seeds 0 1 2 --out /tmp/synth/plans/sft-maze33.json
CODE_REF=<commit> TAG=synth-sft-maze33 GPUS=4 PLAN=/tmp/synth/plans/sft-maze33.json bash launch_synth_st.sh
```

## Speed (H100)

SFT, batch 128, ms per step alone: sudoku 7.6 (any size: launch-bound); maze33 2XS 20, S 34; maze49 2XS 50, S 83 (40k
steps: 5 min to 55 min; TailRL: ~80 for 2XS on 25×25). Maze SFT is compute-bound: two runs on one GPU take ~2.7x
longer each, so packing them buys little.

RL, s per step alone: sudoku 2XS 0.15 / 0.20 / 0.31 / 0.84 (16 / 64 / 256 / 1024 prompts); maze49 S (a 0% model)
0.79 at 16, 3.5 at 256, 9.4 at 1024 (decode 3.8 with a 16 GB KV budget, 7.5 with 4 GB; update 5.4, compute-bound,
most of it padding each micro-batch to its longest response). On a pod with ~4 sudoku runs per GPU each step takes
~1 s (packing ~10% better than running them one after another); the b1024 maze runs use data parallelism.

## Tests

```bash
cd rl-distill-scripts/synth && CUDA_VISIBLE_DEVICES=0,1 AWS_PROFILE=ml-worker $PY -m pytest -q tests   # ~20 min
# on the shared devbox the 2-GPU tests hang in NCCL peer-to-peer: add NCCL_P2P_DISABLE=1
```

On a pod, `tests/run.py [pytest args]` runs them as a pack job. The chess tests need the devbox copies of their files
(skipped elsewhere).

Tasks (sudoku solver against sudoku-transfomer's traces, judge against a plain-Python reference, dataset files,
repeats), model (HF layout / logits / tokenizer, decoder log-probs with and without chunking), SFT (batches, loss against
HF, resume, stop rule), tuning (the search on a run with a known pass@1 curve), fastrl (train / resume, mini-batches), pack (S3
integration), fastrl's loss against a plain-Python reference and its update's invariance to micro-batch size and GPU count.
