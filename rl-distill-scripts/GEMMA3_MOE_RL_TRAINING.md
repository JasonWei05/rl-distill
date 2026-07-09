# Gemma 3 4B PT MoE — Setup & RL Training Guide

End-to-end, self-contained instructions to **replicate on a fresh machine**:
DAPO RL on a **Gemma 3 4B PT model upcycled from a dense MLP into a top-1
Mixture-of-Experts model** (2 or 4 experts), with a **Megatron-Core
actor/reference** and a **vLLM rollout**, using this verl fork.

Contents: [1. What this is](#1-what-this-is) ·
[2. Prerequisites](#2-prerequisites) ·
[3. Get the code](#3-get-the-code) ·
[4. The Megatron-Bridge fork](#4-the-megatron-bridge-fork) ·
[5. Build the environment](#5-build-the-environment) ·
[6. Verify the build](#6-verify-the-build) ·
[7. Data](#7-data) ·
[8. Checkpoints & remote code](#8-checkpoints--remote-code) ·
[9. Router replay (R2 / R3)](#9-router-replay-r2--r3) ·
[10. Run locally](#10-run-locally) ·
[11. Run on ScaleTrain](#11-run-on-scaletrain) ·
[12. Validation](#12-validation) ·
[13. Pinned versions](#13-pinned-versions) ·
[14. Troubleshooting](#14-troubleshooting)

---

## 1. What this is

The dense Gemma 3 4B PT feed-forward block is replaced by a top-1 MoE layer:
a bias-free linear router picks **one** expert per token, and each expert is a
full-size copy of the original dense MLP (including Gemma's per-expert post-MLP
RMSNorm). Routing uses post-top-k softmax, so the combine weight is exactly
`1.0` and a freshly upcycled model reproduces the dense model's logits; the
router then specializes during training through the auxiliary load-balancing
loss. Design details: [`../gemma_3_4b_moe_upcycling.md`](../gemma_3_4b_moe_upcycling.md).

| Variant | Experts | Default single-node parallelism |
|---|---|---|
| 2E | 2 | `TP=1, EP=2` |
| 4E | 4 | `TP=2, EP=4` |

Stack: **actor/reference** = Megatron-Core through the **Megatron-Bridge** fork
([§4](#4-the-megatron-bridge-fork)); **rollout** = vLLM 0.18 via its Transformers
backend loading the custom remote-code model in [`gemma3_moe_hf/`](gemma3_moe_hf);
**algorithm** = DAPO/GRPO via `dapo.main_dapo`.

---

## 2. Prerequisites

- **GPU host.** Single node, ≥2 GPUs for 2E, ≥4 for 4E (80 GB cards; H100/B200
  tested). Actor+ref+rollout are colocated per GPU.
- **CUDA toolkit** with `nvcc`, CUDA **12.9**. Set `TORCH_CUDA_ARCH_LIST=9.0`
  for H100, `10.0` for B200.
- **`uv`**: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **`git`**, **`cmake`**, `build-essential` (gcc/g++). `cmake` is also pip-installed
  by the setup script for the TransformerEngine build.
- **Disk:** ~40 GB for the venv, tens of GB of transient build cache, plus
  13 GB (2E) / 23 GB (4E) per checkpoint snapshot.
- **Hugging Face access** to gated `google/gemma-3-4b-pt` and the MoE repos
  (see [§8](#8-checkpoints--remote-code)); put `HF_TOKEN` in `.env` ([§3](#3-get-the-code)).
- **Time:** the first environment build source-compiles Apex, flash-attn, and
  TransformerEngine — budget **1–3 h**.

---

## 3. Get the code

```bash
git clone git@github.com:JasonWei05/rl-distill.git
cd rl-distill
```

The Megatron-Bridge fork is **vendored** at `third_party/Megatron-Bridge`, so the
clone is self-contained — no submodule init needed.

Create `.env` at the repo root (it is gitignored — never commit it):

```bash
cat > .env <<'EOF'
HF_TOKEN=hf_xxx
WANDB_API_KEY=xxx            # optional; runs default to console logging
WANDB_BASE_URL=https://...   # optional
EOF
```

---

## 4. The Megatron-Bridge fork

verl's Megatron path builds the model through NVIDIA's **Megatron-Bridge**
(`megatron.bridge`), which upstream has no Gemma3-MoE support. The fork at
`third_party/Megatron-Bridge` (base commit in its `VENDORED_COMMIT`) adds:

- `gemma3_moe_layer_spec` + `Gemma3MoEModelProvider{,4B,4B4E}` — Gemma3 attention
  with a Megatron-Core `MoELayer` (`SequentialMLP` experts, each with a torch
  `Gemma3RMSNorm` post-MLP norm; `moe_grouped_gemm=False`).
- `Gemma3MoeBridge` — the `Gemma3MoeForCausalLM` ↔ Megatron parameter mapping
  (router; per-expert gate/up→`linear_fc1`, down→`linear_fc2`; per-expert
  post-norm), EP-aware.
- A `TransformerConfig` relaxation allowing `moe_router_topk=1` with post-top-k
  softmax when aux-loss balancing is on (combine weight is exactly 1.0).
- Compatibility shims + a small vendored `megatron.training` subset so it
  imports against a pip-installed **`megatron-core` 0.16** (which lacks
  `megatron.training`, `_rank_utils`, `_slurm_utils`, the fault injector, etc.).

`setup_megatron.sh` installs it editable from `third_party/Megatron-Bridge`
(override with `MEGATRON_BRIDGE_PATH`).

---

## 5. Build the environment

```bash
cd rl-distill
UV_NO_CONFIG=1 PIP_INDEX_URL=https://pypi.org/simple \
  TORCH_CUDA_ARCH_LIST=9.0 \
  CUDA_HOME=/usr/local/cuda-12.9 \
  MAX_JOBS=64 \
  bash setup_megatron.sh
```

This creates `.venv-megatron/` and installs the full stack (see
[§13](#13-pinned-versions)), source-building Apex, flash-attn, and
TransformerEngine, then installs the Megatron-Bridge fork and `verl` editable.

- `UV_NO_CONFIG=1 PIP_INDEX_URL=https://pypi.org/simple` — **use this on any host
  whose pip/uv is pointed at a private mirror** (e.g. Scale CodeArtifact, whose
  token expires and will 401 mid-build). Harmless on a clean machine.
- `TORCH_CUDA_ARCH_LIST` — `9.0` (H100) or `10.0` (B200). Quote it if you pass it
  through Hydra later.
- `CUDA_HOME` — the script auto-detects `/tmp/cuda-12.9` then `/usr/local/cuda`;
  pass `CUDA_HOME` explicitly if your toolkit is elsewhere.

Key overrides:

| Var | Default | Notes |
|---|---|---|
| `VENV_DIR` | `./.venv-megatron` | Target venv. |
| `CUDA_HOME` | `/tmp/cuda-12.9` → `/usr/local/cuda` | CUDA toolkit. |
| `TORCH_CUDA_ARCH_LIST` | `10.0` | `9.0` for H100. |
| `MAX_JOBS` | `32` | Parallel compile jobs (flash-attn/TE). |
| `MEGATRON_BRIDGE_PATH` | `third_party/Megatron-Bridge` | Bridge fork checkout. |
| `SKIP_GPU_CHECK` | `0` | `1` for GPU-less (Docker) builds. |
| `RUN_HEAVY_BUILDS` | `1` | `0` skips the Apex/flash-attn/TE source builds. |
| `RUN_SMOKE_TEST` | `1` | `0` skips the post-build import check. |

The TE build needs `cmake` and the pip `nvidia-cudnn` include path — both are
wired into the script; you don't set them manually.

---

## 6. Verify the build

```bash
cd rl-distill
SP=.venv-megatron/lib/python3.12/site-packages
LD_LIBRARY_PATH="$SP/nvidia/cudnn/lib:$SP/nvidia/nccl/lib:${CUDA_HOME:-/usr/local/cuda-12.9}/lib64" \
  .venv-megatron/bin/python - <<'PY'
import torch, vllm, transformers, megatron.core, mbridge, flash_attn
import transformer_engine.pytorch, apex, math_verify
import megatron.bridge
from megatron.bridge import AutoBridge
from megatron.bridge.models.gemma.gemma3_provider import gemma3_moe_layer_spec, Gemma3MoEModelProvider4B
from megatron.bridge.models.gemma import Gemma3MoeBridge
print("torch", torch.__version__, "| vllm", vllm.__version__, "| transformers", transformers.__version__)
print("ENV_OK")
PY
```

Expect `ENV_OK`. The `LD_LIBRARY_PATH` prefix is required for the
TransformerEngine import to find cuDNN (the launchers set this for you).

---

## 7. Data

The launchers do **not** auto-create datasets locally (only the ScaleTrain
entrypoint does). Prepare them once:

```bash
# Smoke dataset (DAPO-17k train/test):
DATA_DIR="${HOME}/verl/data" bash rl-distill-scripts/data/prepare_dapo_17k_split.sh
#   -> ${HOME}/verl/data/dapo_17k_{train,test}.parquet

# Full-run dataset (the launcher's default TRAIN/VAL: openmath2 mix):
DATA_DIR="${HOME}/verl/data" bash rl-distill-scripts/data/prepare_dapo_openmath2_mix_split.sh
```

Or point `TRAIN_FILE` / `VAL_FILE` at any existing parquet with a `prompt`
chat-list column and a `reward_model.ground_truth` field.

---

## 8. Checkpoints & remote code

Upcycled SFT checkpoints on the Hub (text-only `Gemma3MoeForCausalLM`):

| Variant | Repo | Pinned revision |
|---|---|---|
| 2E | `JWei05/gemma3-4b-pt-moe-2e-top1-sft-16k` | `952a11b802b63ef091f20ec2dfe08eb66376794c` |
| 4E | `JWei05/gemma3-4b-pt-moe-4e-top1-sft-16k` | `cd87f6e541b1bc0fba8caef218c55601fbb0c533` |

The launcher downloads the snapshot automatically (needs `HF_TOKEN` with access),
or set `HF_MOE_LOCAL_DIR` to a local copy. Each layer stores `mlp.router.weight`
and per expert `gate_proj` / `up_proj` / `down_proj` / `post_layernorm`.

**Remote-code model** (`gemma3_moe_hf/`, installed into the checkpoint dir at
launch by the main script):

- Expert count is stored as **`gemma3_moe_num_experts`**, not the generic
  `num_experts` — deliberately, so vLLM's `is_moe` autodetection stays off and
  the model runs vLLM's **plain** Transformers backend. The fused
  (`TransformersMoEForCausalLM`) path can't represent the per-expert post-MLP
  RMSNorm and hard-codes SiLU, so it must be avoided.
- `_tied_weights_keys` is a dict and `tie_weights()` is overridden so
  `lm_head.weight` ties to the embeddings under transformers 5.x.
- The R3 capture hook uses a **lazy** vLLM import (see [§9](#9-router-replay-r2--r3)).

Convert a fresh Megatron dist-ckpt to this HF layout:

```bash
.venv-megatron/bin/python rl-distill-scripts/gemma3_moe_hf/convert_gemma3_moe_distckpt_to_hf.py \
  --hf-repo-id JWei05/gemma3-4b-pt-moe-4e-top1-sft-16k \
  --output-dir /tmp/gemma3-4b-pt-moe-4e-hf --num-experts 4
```

---

## 9. Router replay (R2 / R3)

Top-1 routing is a discrete argmax, so log-probs of the same tokens can differ
between the rollout engine and the trainer if routing flips. `ROUTER_REPLAY_MODE`
controls consistency:

- **`R2`** (default): the trainer records routes during its own forward-only
  (old-log-prob) pass and replays them in the update passes. No rollout-engine
  dependency. Simplest; recommended default.
- **`R3`**: routes are captured **in vLLM during generation** and replayed in the
  trainer — the strongest consistency. See the enablement recipe below.
- **`disabled`**: every forward routes independently.

A guardrail in `verl/workers/engine/megatron/transformer_impl.py` raises if the
replayed routing map is entirely zero (capture silently failed) instead of
training on garbage.

### Enabling R3

R3 works on this model even though vLLM only auto-binds its routed-experts
capturer to `FusedMoE` modules (which this architecture can't use):
`gemma3_moe_hf/modeling_gemma3_moe.py` feeds vLLM's process-global
`RoutedExpertsCapturer` directly via a lazy hook.

**To run with R3**, set the mode — everything else is already the default:

```bash
ROUTER_REPLAY_MODE=R3 NUM_EXPERTS=2 SMOKE_GPUS=4,5 \
  bash rl-distill-scripts/gemma3_4b_pt_moe_megatron_rl_local_smoke.sh
# or on the full launcher:
ROUTER_REPLAY_MODE=R3 bash rl-distill-scripts/gemma3_4b_pt_moe_2e_megatron_rl_20k.sh
```

**Requirements (all satisfied by the defaults — do not disable):**

1. **`enforce_eager` rollout** (`ROLLOUT_ENFORCE_EAGER=True`, the default). CUDA
   graphs would bypass the per-layer capture call.
2. **The patched `modeling_gemma3_moe.py`** must be the file vLLM serves. The
   launcher installs it into `MODEL_PATH` automatically; if you serve a
   checkpoint by hand, copy `rl-distill-scripts/gemma3_moe_hf/modeling_gemma3_moe.py`
   into it.
3. **The vLLM import in the hook must stay lazy** (resolved at forward time). A
   module-level `import vllm …` stalls vLLM's trust-remote-code inspection
   subprocess at startup. It is already lazy in the committed file — don't
   "clean it up" to a top-level import.
4. **DP=1 per rollout replica** (`GEN_TP=1`, the default): the capturer's
   token-slice bookkeeping assumes single-DP per replica.

**Verify R3 is actually capturing** during a run — in the trainer logs you should
see, per actor/ref worker:

```
routing replay layers: 34
```

and the run must **not** raise the all-zero guardrail
(`routed_experts from the rollout are entirely zero …`). If it does, capture
failed — check the four requirements above, or fall back to `R2`.

> Status: R3 capture is validated against the HF router's own argmax (~99.5%
> agreement; the residual is the last sampled token, which has no route, plus
> rare bf16 argmax ties). `R2` remains the default as the simpler, dependency-free
> choice. If the colocated actor+rollout OOMs with R3, lower
> `ROLLOUT_GPU_MEMORY_UTILIZATION` (e.g. `0.6`).

---

## 10. Run locally

### 10a. Smoke (recommended first)

Two-step DAPO round trip on a subset of GPUs (refuses busy GPUs). Prepare the
smoke data first ([§7](#7-data)), then:

```bash
# 2 experts on 2 GPUs (TP=1, EP=2):
NUM_EXPERTS=2 SMOKE_GPUS=4,5 \
  bash rl-distill-scripts/gemma3_4b_pt_moe_megatron_rl_local_smoke.sh

# 4 experts on 4 GPUs (TP=2, EP=4):
NUM_EXPERTS=4 SMOKE_GPUS=4,5,6,7 ACTOR_TP=2 REF_TP=2 \
  bash rl-distill-scripts/gemma3_4b_pt_moe_megatron_rl_local_smoke.sh
```

The wrapper forces a single-node local Ray instance, small batches, short
generations, `test_freq=save_freq=-1`, `ROUTER_REPLAY_MODE=disabled` (override to
`R2`/`R3`), and pins the checkpoint revisions. **Success** = both steps complete
with finite `actor/grad_norm` and a nonzero `actor/train/router_loss`.

### 10b. Full RL run

Prepare the full dataset first ([§7](#7-data)), then:

```bash
MEGATRON_VENV="$PWD/.venv-megatron" \
  bash rl-distill-scripts/gemma3_4b_pt_moe_2e_megatron_rl_20k.sh   # or _4e_
```

Common env vars (full list in the script): `NUM_EXPERTS`, `ACTOR_TP`, `ACTOR_EP`,
`REF_TP`, `REF_EP`, `ROUTER_REPLAY_MODE`, `TRAIN_FILE`, `VAL_FILE`, `MODEL_PATH`,
`HF_MOE_LOCAL_DIR`, `ROLLOUT_GPU_MEMORY_UTILIZATION`, `RAY_ADDRESS`, `NNODES`,
`GPUS_PER_NODE`, `SAVE_FREQ`, `TEST_FREQ`.

---

## 11. Run on ScaleTrain (dedicated k8s GPUs)

Use ScaleTrain when your local box is shared — co-tenant processes on the same
physical GPU will OOM a colocated actor+rollout run. Auth first:

```bash
export PATH="$HOME/.local/bin:$PATH" AWS_PROFILE=ml-admin \
  AWS_DEFAULT_REGION=us-west-2 AWS_REGION=us-west-2
aws sso login
```

**First build** (bakes the whole venv into the image via `setup_megatron.sh` —
slow, one-time):

```bash
python rl-distill-scripts/scale_train/launch_st_job.py \
  --cluster eks --n-instances 1 --gpus-per-instance 4 \
  --priority high --allow-borrowing \
  --build-config-key train-rl-distill-megatron \
  --job-name gemma3-moe-4e-smoke \
  --run-file run_gemma3_moe_rl_smoke.sh \
  --env-vars "NUM_EXPERTS=4"
```

- `--gpus-per-instance {1,2,4,8}` uses the fractional `p5.48xlarge:N` preset
  (`<8` requires `--n-instances 1`).
- `st_config/Dockerfile.megatron` runs `setup_megatron.sh` in-image with
  `SKIP_GPU_CHECK=1`.
- `run_gemma3_moe_rl_smoke.sh` prepares DAPO-17k data in-container and runs the
  smoke across all allocated GPUs. It puts the venv on `PATH` up front because
  the `sudo` entrypoint's `secure_path` otherwise hides it (and the CUDA base
  image has no system `python3`).
- Pass `ROUTER_REPLAY_MODE=R3` (or `R2`) in `--env-vars` to exercise replay.

**Iterate on scripts without the ~3 h rebuild** — the
`train-rl-distill-megatron-overlay` config layers only the refreshed
`rl-distill-scripts/` onto the already-pushed heavy image. Set the base tag in
`st_config/Dockerfile.megatron-overlay` to the last pushed image tag, then:

```bash
python rl-distill-scripts/scale_train/launch_st_job.py \
  --cluster eks --n-instances 1 --gpus-per-instance 4 \
  --priority high --allow-borrowing \
  --build-config-key train-rl-distill-megatron-overlay \
  --job-name gemma3-moe-4e-smoke \
  --run-file run_gemma3_moe_rl_smoke.sh --env-vars "NUM_EXPERTS=4"
```

**Monitor:**

```bash
scale-train get job <job_id>
kubectl -n train get pods | rg gemma3-moe
kubectl -n train logs -f <pod_name> --tail=50
```

---

## 12. Validation

**Logit parity** (HF vs Megatron-Bridge load, real SFT weights):

```bash
SP=.venv-megatron/lib/python3.12/site-packages
CUDA_VISIBLE_DEVICES=4 LD_LIBRARY_PATH="$SP/nvidia/cudnn/lib" \
  .venv-megatron/bin/python rl-distill-scripts/logit_parity_gemma3_moe_hf_vs_megatron.py \
  <snapshot_dir>                       # single rank

CUDA_VISIBLE_DEVICES=4,5 LD_LIBRARY_PATH="$SP/nvidia/cudnn/lib" \
  .venv-megatron/bin/torchrun --nproc-per-node 2 \
  rl-distill-scripts/logit_parity_gemma3_moe_hf_vs_megatron.py <snapshot_dir> --ep 2
```

Passing = top-1 token agreement `1.0`, mean logit diff `< 0.05`. Validated: 2E at
EP=1/2, 4E at EP=1/2/4.

**Smoke:** [§10a](#10a-smoke-recommended-first). Validated locally (2E
`disabled`/`R2`, 4E `TP2/EP4`) and a 4-GPU 4E run to completion on ScaleTrain.
R3 capture validated to ~99.5% HF-argmax agreement.

---

## 13. Pinned versions

Installed by `setup_megatron.sh` into `.venv-megatron` (Python 3.12):

| Package | Version / ref |
|---|---|
| torch / torchvision / torchaudio | `2.10.0+cu129` / `0.25.0` / `2.10.0` |
| vllm | `0.18.0` |
| transformers | `5.3.0` |
| Megatron-LM (megatron-core) | `core_v0.16.0` |
| mbridge | `641a5a01de71080b2200d10e369090e40c9a351c` |
| Megatron-Bridge | vendored fork, `third_party/Megatron-Bridge` |
| flash-attn | `2.8.3` (source build) |
| TransformerEngine | `release_v2.12` (source build) |
| Apex | latest (source build) |
| flash-linear-attention / peft / trl | `0.4.1` / `0.18.1` / `0.27.0` |
| nvidia-modelopt, math-verify, tensorboard | (latest at build time) |

Override any of these via the matching env var at the top of `setup_megatron.sh`.

---

## 14. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `uv`/pip `401` mid-build | Private mirror token expired — build with `UV_NO_CONFIG=1 PIP_INDEX_URL=https://pypi.org/simple`. |
| `ModuleNotFoundError: megatron.bridge` | Bridge fork not installed — ensure `third_party/Megatron-Bridge` exists and re-run `setup_megatron.sh`. |
| `Could not find CMake executable` (TE build) | `cmake` missing — it's in the pip deps; install system `cmake` and re-run. |
| `cudnn.h: No such file` (TE build) | cuDNN include path — handled by setup; ensure `nvidia-cudnn-cu12` installed. |
| `ImportError … transformer_engine … cudnn` at runtime | Add `.venv-megatron/.../nvidia/cudnn/lib` to `LD_LIBRARY_PATH` (launchers do this). |
| `ValueError: Please use --moe-router-pre-softmax when topk is 1` | Missing the fork's `TransformerConfig` relaxation — you're on stock Megatron-Bridge. |
| vLLM loads `TransformersMoEForCausalLM` / crashes on `post_layernorm` | Config exposed a generic `num_experts` key — must be `gemma3_moe_num_experts`. |
| vLLM rollout server hangs at startup (R3) | The capture import in `modeling_gemma3_moe.py` must be **lazy**, not module-level. |
| `routed_experts … are entirely zero` (guardrail raises) | R3 capture failed — check the [§9](#9-router-replay-r2--r3) requirements, or use `R2`. |
| `python3: command not found` (ScaleTrain pod, exit 127) | venv not on `PATH` under `sudo`; `run_gemma3_moe_rl_smoke.sh` fixes it — rebuild via the overlay config. |
| OOM on a "free" shared GPU | A co-tenant landed on the same physical GPU. Use ScaleTrain, or lower `ROLLOUT_GPU_MEMORY_UTILIZATION`. |
