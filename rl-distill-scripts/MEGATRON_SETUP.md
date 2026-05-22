# Megatron RL setup notes

Standalone venv + patches for running verl + DAPO + megatron-core on B200.
The existing FSDP2 venv (`/mlx_devbox/.../rl-distill/.venv`) is left untouched.

## Layout

| Path | Purpose |
|---|---|
| `/mlx_devbox/users/jason.wei/playground/rl_distill_megatron_env/.venv` | Megatron venv (Python 3.12, on shared storage so all workers see it) |
| `/tmp/cuda-12.9` | Userspace CUDA 12.9 toolkit (host nvcc is 12.6 and can't target sm_100/B200) |
| `setup_megatron.sh` / `setup_megatron_frozen_dep.txt` | Reproducible install of the venv |
| `rl-distill-scripts/gemma3_4b_pt_megatron_20k.sh` | Single-node Gemma3-4B-PT DAPO entrypoint |
| `rl-distill-scripts/_ops/_kill_and_relaunch_4b.sh` | Helper: kill any prior run, then relaunch |

## Pinned versions (in the megatron venv)

- `ray==2.54.0` — must match the ray version of any cluster you connect to
  (the FSDP2 venv ships 2.55; we don't connect to it, but pinning avoids surprises)
- `megatron-core==0.15.3` — 0.16+ has API churn that breaks more of mbridge
- `mbridge==0.15.1` — latest published; needs the patch below

Reinstall after rebuilding the venv:

```bash
VENV=/mlx_devbox/users/jason.wei/playground/rl_distill_megatron_env/.venv
"$VENV/bin/pip" install --no-deps "ray==2.54.0" "megatron-core==0.15.3"
```

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
| `CUDA_HOME` | `/tmp/cuda-12.9` | Override host CUDA 12.6 (no sm_100 support) |
| `LD_LIBRARY_PATH` | venv's `nccl/lib`, `nvjitlink/lib`, `$CUDA_HOME/lib64` | Resolve nccl + nvjitlink without poisoning system paths |
| `TORCH_CUDA_ARCH_LIST` | `"10.0"` (quoted!) | B200 = sm_100. megatron-core's `unified_memory.py` JITs a CUDA ext at import time and otherwise crashes with `IndexError: list index out of range`. Quote it so Hydra parses it as str, not float. |
| `NCCL_SOCKET_IFNAME=lo`, `NCCL_SOCKET_FAMILY=AF_INET`, `GLOO_SOCKET_IFNAME=lo` | — | Single-node loopback transport |
| `TORCH_NCCL_AVOID_RECORD_STREAMS=1`, `CUDA_DEVICE_MAX_CONNECTIONS=1` | — | Standard Megatron NCCL hygiene |

## mbridge patch (vendored)

mbridge 0.15.1's `Gemma3TransformerLayer.__init__` and
`MemoryEfficientAttention.__init__` only accept the legacy
`model_comm_pgs=` kwarg. megatron-core ≥ 0.15 renamed it to
`pg_collection=`, so megatron's `build_module` now blows up with
`TypeError: ... unexpected keyword argument 'pg_collection'`.

Fix: a venv-wide auto-loaded monkey-patch that rebinds both `__init__`s to
accept either name.

| Path | Purpose |
|---|---|
| `.venv/.../site-packages/mbridge_pg_collection_patch.py` | Patch module. Idempotent (`_pg_patched` sentinel), per-class try/except so non-megatron interpreters no-op silently. |
| `.venv/.../site-packages/mbridge_pg_collection_patch.pth` | Single-line `import mbridge_pg_collection_patch`. Python's `site.py` runs this at every interpreter start — including ray worker actors — so the patch applies before mbridge is consumed. |

Verify with `bash rl-distill-scripts/_ops/_verify_pg_patch.sh` — should print
`patched: True` for both classes.

If mbridge ever ships an upstream fix, delete those two files and the patch
disappears.

## Single-node 4B parallelism

Gemma3-4B fits comfortably on one B200 (≈56 GB total at fp16 + AdamW state vs
180 GB device memory), so the script defaults to pure DP=8 — no TP/PP/CP/EP:

```
ACTOR_TP=1  ACTOR_PP=1  ACTOR_CP=1  ACTOR_EP=1
REF_TP=1    REF_PP=1    REF_CP=1    REF_EP=1
```

VPP is set to `null` whenever PP=1 (Megatron's interleaved schedule is invalid
otherwise).

## Launch / stop

```bash
# Launch (detaches via setsid+nohup so mlx worker login can return):
mlx worker login 883483 -- "bash /mlx_devbox/users/jason.wei/playground/rl-distill/rl-distill-scripts/_ops/_kill_and_relaunch_4b.sh"

# Tail:
mlx worker login 883483 -- "tail -f /mlx_devbox/users/jason.wei/playground/rl-distill/rl-distill-scripts/_logs/gemma3_4b_pt_mega.log"

# Stop:
mlx worker login 883483 -- "bash /mlx_devbox/users/jason.wei/playground/rl-distill/rl-distill-scripts/_ops/_stop_4b_pt_megatron.sh"
```

PID lives at `rl-distill-scripts/_logs/gemma3_4b_pt_mega.pid`.

## Common errors → fixes (history)

| Error | Cause | Fix |
|---|---|---|
| `RuntimeError: Version mismatch: cluster Ray 2.54 vs process 2.55` | Connected to existing FSDP2-venv ray cluster | Pin venv ray to 2.54, force `address=local` |
| `AssertionError: Unknown backend: megatron` | Ray workers spawned in FSDP2 venv (no megatron) | Isolated local ray (above) |
| `TypeError: Gemma3TransformerLayer.__init__() got an unexpected keyword argument 'pg_collection'` | mbridge ↔ megatron-core API drift | Vendored monkey-patch (above) |
| `IndexError: list index out of range` in `_get_cuda_arch_flags` | torch JIT can't detect arch on cold ray worker | Set `TORCH_CUDA_ARCH_LIST="10.0"` (quoted) |
| `runtime_env['env_vars'] must be Dict[str,str], but value 10.0 is float` | Hydra coerced unquoted `10.0` to float | Quote the value in the override: `…=\"10.0\"` |
