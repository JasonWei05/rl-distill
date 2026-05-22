# Megatron RL setup notes

Standalone venv for running verl + DAPO + Megatron/mbridge on B200.
The default venv is separate from the normal FSDP2 venv, so `.venv` is left
untouched.

## Quickstart

```bash
bash setup_megatron.sh
MEGATRON_VENV="$PWD/.venv-megatron" bash rl-distill-scripts/gemma3_4b_pt_megatron_20k.sh
```

Useful overrides:

```bash
VENV_DIR=/shared/path/.venv-megatron CUDA_HOME=/tmp/cuda-12.9 MAX_JOBS=64 bash setup_megatron.sh
MEGATRON_VENV=/shared/path/.venv-megatron CUDA_HOME_OVERRIDE=/tmp/cuda-12.9 bash rl-distill-scripts/gemma3_4b_pt_megatron_20k.sh
```

## Layout

| Path | Purpose |
|---|---|
| `.venv-megatron` | Default Megatron venv (Python 3.12). Override with `VENV_DIR` / `MEGATRON_VENV` when shared storage is needed. |
| `/tmp/cuda-12.9` or `/usr/local/cuda` | CUDA toolkit. Override with `CUDA_HOME` at setup time or `CUDA_HOME_OVERRIDE` at launch time. |
| `setup_megatron.sh` | Reproducible install of the Megatron venv. The local frozen dependency dump is not committed because it can contain private editable URLs. |
| `rl-distill-scripts/gemma3_4b_pt_megatron_20k.sh` | Single-node Gemma3-4B-PT DAPO entrypoint |

## Pinned versions (in the megatron venv)

- `torch==2.10.0+cu129`
- `vllm==0.18.0`
- `transformers==5.3.0`
- `Megatron-LM @ core_v0.16.0`
- `mbridge @ 641a5a01de71080b2200d10e369090e40c9a351c`
- `flash-attn==2.8.3`
- `TransformerEngine @ release_v2.12`
- `flash-linear-attention==0.4.1`
- `peft==0.18.1`
- `trl==0.27.0`

Set `RAY_VERSION=2.54.0` when you need to match an existing Ray cluster. The
default launcher uses an isolated local Ray instance, so it does not need to
match the normal FSDP2 venv's Ray version.

## Isolated ray cluster

There is already a long-lived FSDP2-venv ray cluster on the worker (started by
the rl-distill .venv). Connecting to it would pin ray workers to the wrong
Python interpreter (no megatron installed) and produce
`AssertionError: Unknown backend: megatron`.

The training script forces a fresh single-node ray instance with its own
temp dir, so the FSDP2 cluster is left running:

```bash
export RAY_ADDRESS=local
python3 -m dapo.main_dapo \
    --config-name=dapo_megatron_trainer \
    +ray_kwargs.ray_init.address=local \
    +ray_kwargs.ray_init._temp_dir=/tmp/ray_megatron \
    +ray_kwargs.ray_init.include_dashboard=False \
    ...
```

## Required env vars

Set in `gemma3_4b_pt_megatron_20k.sh` and forwarded into ray's
`runtime_env.env_vars` so worker actors inherit them:

| Var | Value | Why |
|---|---|---|
| `CUDA_HOME` | setup/launch CUDA path | Override host CUDA when it cannot target sm_100/B200 |
| `LD_LIBRARY_PATH` | venv's `nccl/lib`, `nvjitlink/lib`, `$CUDA_HOME/lib64` | Resolve nccl + nvjitlink without poisoning system paths |
| `TORCH_CUDA_ARCH_LIST` | `"10.0"` (quoted!) | B200 = sm_100. megatron-core's `unified_memory.py` JITs a CUDA ext at import time and otherwise crashes with `IndexError: list index out of range`. Quote it so Hydra parses it as str, not float. |
| `NCCL_SOCKET_IFNAME=lo`, `NCCL_SOCKET_FAMILY=AF_INET`, `GLOO_SOCKET_IFNAME=lo` | — | Single-node loopback transport |
| `TORCH_NCCL_AVOID_RECORD_STREAMS=1`, `CUDA_DEVICE_MAX_CONNECTIONS=1` | — | Standard Megatron NCCL hygiene |

## Single-node 4B parallelism

Gemma3-4B fits comfortably on one B200 (≈56 GB total at fp16 + AdamW state vs
180 GB device memory), so the script defaults to pure DP=8 — no TP/PP/CP/EP:

```
ACTOR_TP=1  ACTOR_PP=1  ACTOR_CP=1  ACTOR_EP=1
REF_TP=1    REF_PP=1    REF_CP=1    REF_EP=1
```

VPP is set to `null` whenever PP=1 (Megatron's interleaved schedule is invalid
otherwise).

## Launch

```bash
MEGATRON_VENV="$PWD/.venv-megatron" \
CUDA_HOME_OVERRIDE=/tmp/cuda-12.9 \
bash rl-distill-scripts/gemma3_4b_pt_megatron_20k.sh
```

Use the repo-local `MEGATRON_VENV` default when running directly from the
checkout. Use an absolute shared path for `MEGATRON_VENV` when running through
a remote worker command.

## Common errors → fixes (history)

| Error | Cause | Fix |
|---|---|---|
| `RuntimeError: Version mismatch: cluster Ray 2.54 vs process 2.55` | Connected to existing FSDP2-venv ray cluster | Pin venv ray to 2.54, force `address=local` |
| `AssertionError: Unknown backend: megatron` | Ray workers spawned in FSDP2 venv (no megatron) | Isolated local ray (above) |
| `TypeError: Gemma3TransformerLayer.__init__() got an unexpected keyword argument 'pg_collection'` | mbridge / megatron-core API drift | Use the pinned mbridge commit and Megatron-LM ref from `setup_megatron.sh` |
| `IndexError: list index out of range` in `_get_cuda_arch_flags` | torch JIT can't detect arch on cold ray worker | Set `TORCH_CUDA_ARCH_LIST="10.0"` (quoted) |
| `runtime_env['env_vars'] must be Dict[str,str], but value 10.0 is float` | Hydra coerced unquoted `10.0` to float | Quote the value in the override: `…=\"10.0\"` |
