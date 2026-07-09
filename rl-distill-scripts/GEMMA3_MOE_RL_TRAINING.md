# Gemma 3 4B PT MoE — Setup & RL Training Guide

This guide covers running DAPO RL on a **Gemma 3 4B PT model upcycled from a
dense MLP into a top-1 Mixture-of-Experts model** (2 or 4 experts), with a
**Megatron-Core actor/reference** and a **vLLM rollout**, via this verl fork.

If you only want the short version, jump to [Quickstart](#quickstart).

---

## 1. What this is

The dense Gemma 3 4B PT feed-forward block is replaced by a top-1 MoE layer:
a bias-free linear router selects **one** expert per token, and each expert is
a full-size copy of the original dense MLP (including Gemma's per-expert
post-MLP RMSNorm). Routing uses post-top-k softmax, so the combine weight is
exactly `1.0` and a freshly upcycled model reproduces the dense model's
logits; the router then specializes during training through the auxiliary
load-balancing loss. Design details: [`../gemma_3_4b_moe_upcycling.md`](../gemma_3_4b_moe_upcycling.md).

Two variants are supported and validated:

| Variant | Experts | Default parallelism (single node) |
|---|---|---|
| 2E | 2 | `TP=1, EP=2` |
| 4E | 4 | `TP=2, EP=4` |

The training stack:

- **Actor / reference:** Megatron-Core (`use_mbridge=True, vanilla_mbridge=False`)
  built through **Megatron-Bridge** — specifically the forked provider/bridge
  vendored at [`../third_party/Megatron-Bridge`](../third_party/Megatron-Bridge)
  (see [§4](#4-the-megatron-bridge-fork)).
- **Rollout:** vLLM 0.18 via its **Transformers backend** (`model_impl=transformers`),
  loading the custom remote-code model in [`gemma3_moe_hf/`](gemma3_moe_hf).
- **Algorithm:** DAPO / GRPO, driven by `dapo.main_dapo`.

---

## 2. Repository layout

| Path | Purpose |
|---|---|
| `setup_megatron.sh` | Builds the `.venv-megatron` environment (torch/vLLM/TE/flash-attn/Apex/Megatron-Core + the Bridge fork). |
| `third_party/Megatron-Bridge/` | Vendored Megatron-Bridge fork adding the Gemma3 MoE provider/bridge and Megatron-Core 0.16 compat shims. |
| `rl-distill-scripts/gemma3_moe_hf/` | Custom HF remote-code MoE model (`Gemma3MoeForCausalLM`) + dist-ckpt→HF converter. |
| `rl-distill-scripts/gemma3_4b_pt_moe_megatron_rl_20k.sh` | Main DAPO RL launcher (reads `NUM_EXPERTS`, parallelism, replay mode, paths). |
| `rl-distill-scripts/gemma3_4b_pt_moe_2e_megatron_rl_20k.sh` / `_4e_` | Thin wrappers pinning the 2E / 4E checkpoint + parallelism. |
| `rl-distill-scripts/gemma3_4b_pt_moe_megatron_rl_local_smoke.sh` | Local 2-step smoke on a subset of GPUs (busy-GPU guarded). |
| `rl-distill-scripts/logit_parity_gemma3_moe_hf_vs_megatron.py` | HF-vs-Megatron logit parity test (weight-mapping regression). |
| `rl-distill-scripts/_ops/ensure_gemma3_processor_files.sh` | Idempotently backfills processor/tokenizer metadata into a checkpoint dir. |
| `rl-distill-scripts/scale_train/` | ScaleTrain (k8s) launcher, image Dockerfiles, run entrypoint. |

---

## 3. Prerequisites

- An NVIDIA GPU host. Single-node H100/B200 is the tested target
  (`TORCH_CUDA_ARCH_LIST=9.0` for H100, `10.0` for B200).
- A CUDA 12.9 toolkit (`nvcc`). `setup_megatron.sh` auto-detects `/tmp/cuda-12.9`
  (else `/usr/local/cuda`); if your toolkit is elsewhere, pass `CUDA_HOME`
  explicitly (e.g. `CUDA_HOME=/usr/local/cuda-12.9`, as in §5).
- `uv` (`curl -LsSf https://astral.sh/uv/install.sh | sh`).
- `cmake` (used by the TransformerEngine build — installed via pip by `setup_megatron.sh`, but a system `cmake` is also fine).
- A `.env` at the repo root with `HF_TOKEN` (gated Gemma access) and optionally `WANDB_API_KEY`. **`.env` is gitignored — never commit it.**

---

## 4. The Megatron-Bridge fork

verl's Megatron path builds the model through NVIDIA's **Megatron-Bridge**
(`megatron.bridge`), which upstream has no Gemma3-MoE support. The fork vendored
at `third_party/Megatron-Bridge` (base commit in `VENDORED_COMMIT`) adds:

- `gemma3_moe_layer_spec` + `Gemma3MoEModelProvider{,4B,4B4E}` — Gemma3 attention
  with a Megatron-Core `MoELayer` (`SequentialMLP` experts, each with a torch
  `Gemma3RMSNorm` post-MLP norm; `moe_grouped_gemm=False`).
- `Gemma3MoeBridge` — the `Gemma3MoeForCausalLM` ↔ Megatron parameter mapping
  (router, per-expert gate/up→`linear_fc1`, down→`linear_fc2`, per-expert
  post-norm), EP-aware.
- A `TransformerConfig` relaxation allowing `moe_router_topk=1` with post-top-k
  softmax when aux-loss balancing is active (the combine weight is exactly 1.0).
- Compatibility shims + a small vendored `megatron.training` subset so the
  bridge imports against a **pip-installed `megatron-core` 0.16** (which lacks
  `megatron.training`, `_rank_utils`, `_slurm_utils`, the fault injector, etc.).

`setup_megatron.sh` installs it editable from `third_party/Megatron-Bridge`
by default (override with `MEGATRON_BRIDGE_PATH`).

---

## 5. Environment setup

```bash
cd /path/to/rl-distill
# H100:
TORCH_CUDA_ARCH_LIST=9.0 CUDA_HOME=/usr/local/cuda-12.9 MAX_JOBS=64 \
  bash setup_megatron.sh
```

This creates `.venv-megatron/` and source-builds Apex, flash-attn, and
TransformerEngine (expect a long first run — 1–3 h depending on the host). It
installs, among others: `torch 2.10.0+cu129`, `vllm 0.18.0`,
`transformers 5.3.0`, `megatron-core @ core_v0.16.0`, `mbridge`, the
Megatron-Bridge fork, `nvidia-modelopt`, `math-verify`, `tensorboard`.

Useful overrides:

| Var | Default | Notes |
|---|---|---|
| `VENV_DIR` | `./.venv-megatron` | Target venv. |
| `CUDA_HOME` | `/tmp/cuda-12.9` or `/usr/local/cuda` | CUDA toolkit. |
| `TORCH_CUDA_ARCH_LIST` | `10.0` | Set `9.0` for H100. |
| `MAX_JOBS` | `32` | Parallel compile jobs. |
| `MEGATRON_BRIDGE_PATH` | `third_party/Megatron-Bridge` | Bridge fork checkout. |
| `SKIP_GPU_CHECK` | `0` | Set `1` for GPU-less (Docker) builds. |
| `RUN_HEAVY_BUILDS` | `1` | `0` skips the Apex/flash-attn/TE source builds. |

**Build gotchas already handled in the script** (documented so you know why):
the TransformerEngine build needs `cmake` and the pip `nvidia-cudnn` include
path (both wired in); if your pip index is a private mirror with an expiring
token, run with `UV_NO_CONFIG=1 PIP_INDEX_URL=https://pypi.org/simple`.

---

## 6. Model checkpoints & remote code

The upcycled SFT checkpoints live on the Hub (text-only `Gemma3MoeForCausalLM`):

| Variant | Repo | Pinned revision |
|---|---|---|
| 2E | `JWei05/gemma3-4b-pt-moe-2e-top1-sft-16k` | `952a11b802b63ef091f20ec2dfe08eb66376794c` |
| 4E | `JWei05/gemma3-4b-pt-moe-4e-top1-sft-16k` | `cd87f6e541b1bc0fba8caef218c55601fbb0c533` |

The launcher pulls the snapshot automatically (or point `HF_MOE_LOCAL_DIR` at a
local copy). Each layer stores `mlp.router.weight` and, per expert,
`gate_proj` / `up_proj` / `down_proj` / `post_layernorm`.

**Remote-code notes** (in `gemma3_moe_hf/`, installed into the checkpoint dir at
launch by the main script):

- The expert count is stored as **`gemma3_moe_num_experts`**, not the generic
  `num_experts`. This is deliberate: it keeps vLLM's `is_moe` autodetection off
  so the model runs vLLM's **plain** Transformers backend. The fused
  (`TransformersMoEForCausalLM`) path cannot represent the per-expert post-MLP
  RMSNorm and hard-codes SiLU, so it must be avoided for this architecture.
- `_tied_weights_keys` is a dict and `tie_weights()` is overridden so
  `lm_head.weight` ties to the embeddings under transformers 5.x (the checkpoint
  never serializes `lm_head`).

To convert a fresh Megatron dist-ckpt to this HF layout:

```bash
python rl-distill-scripts/gemma3_moe_hf/convert_gemma3_moe_distckpt_to_hf.py \
  --hf-repo-id JWei05/gemma3-4b-pt-moe-4e-top1-sft-16k \
  --output-dir /tmp/gemma3-4b-pt-moe-4e-hf --num-experts 4
```

---

## 7. Router replay (R2 / R3)

Top-1 routing is a discrete argmax, so log-probs of the same tokens can differ
between the rollout engine and the trainer if routing flips. `ROUTER_REPLAY_MODE`
controls how routes are kept consistent:

- **`R2`** (default): the trainer records routes during its own forward-only
  (old-log-prob) pass and replays them in the update passes. No rollout-engine
  dependency. Recommended default.
- **`R3`**: routes are captured **in vLLM during generation** and replayed in the
  trainer — the strongest consistency. Works here via a lazy hook in
  `gemma3_moe_hf/modeling_gemma3_moe.py` that feeds vLLM's process-global
  `RoutedExpertsCapturer` directly (vLLM only auto-binds capture to `FusedMoE`
  modules, which this model doesn't use). **The vLLM import in that hook must
  stay lazy** — a top-level import stalls vLLM's trust-remote-code inspection
  subprocess.
- **`disabled`**: every forward routes independently.

A guardrail in `verl/workers/engine/megatron/transformer_impl.py` raises if the
replayed routing map is entirely zero (i.e. capture silently failed) instead of
training on garbage.

---

## 8. Quickstart

### 8a. Local smoke (recommended first)

First prepare the smoke dataset (the local scripts do **not** auto-create it;
only the ScaleTrain entrypoint does):

```bash
DATA_DIR="${HOME}/verl/data" bash rl-distill-scripts/data/prepare_dapo_17k_split.sh
# -> ${HOME}/verl/data/dapo_17k_{train,test}.parquet (the smoke's default TRAIN/VAL)
```

Then run the two-step DAPO round trip on a subset of GPUs (refuses busy GPUs):

```bash
# 2 experts on 2 GPUs (TP=1, EP=2):
NUM_EXPERTS=2 SMOKE_GPUS=4,5 \
  bash rl-distill-scripts/gemma3_4b_pt_moe_megatron_rl_local_smoke.sh

# 4 experts on 4 GPUs (TP=2, EP=4):
NUM_EXPERTS=4 SMOKE_GPUS=4,5,6,7 ACTOR_TP=2 REF_TP=2 \
  bash rl-distill-scripts/gemma3_4b_pt_moe_megatron_rl_local_smoke.sh
```

The wrapper forces a single-node local Ray instance, small batches, short
generations, `test_freq=save_freq=-1`, `ROUTER_REPLAY_MODE=disabled`, and pins
the checkpoint revisions. Success = both steps complete with finite
`actor/grad_norm` and a nonzero `actor/train/router_loss`.

### 8b. Full RL run

The full launcher defaults `TRAIN_FILE`/`VAL_FILES` to the `dapo_openmath2_mix_*`
parquets under `${HOME}/verl/data`; prepare those (or point `TRAIN_FILE`/`VAL_FILE`
at existing parquets) before launching.

```bash
MEGATRON_VENV="$PWD/.venv-megatron" \
  bash rl-distill-scripts/gemma3_4b_pt_moe_2e_megatron_rl_20k.sh   # or _4e_
```

Key env vars (see the script for the full list): `NUM_EXPERTS`, `ACTOR_TP`,
`ACTOR_EP`, `ROUTER_REPLAY_MODE`, `TRAIN_FILE`, `VAL_FILE`, `MODEL_PATH`,
`HF_MOE_LOCAL_DIR`, `ROLLOUT_GPU_MEMORY_UTILIZATION`, `RAY_ADDRESS`, `NNODES`,
`GPUS_PER_NODE`.

---

## 9. Running on ScaleTrain (dedicated k8s GPUs)

Use ScaleTrain when the local box is oversubscribed (co-tenant processes on
shared GPUs will OOM a colocated actor+rollout run).

**First build** (source-builds the whole stack into the image — slow, one-time):

```bash
export PATH="$HOME/.local/bin:$PATH" AWS_PROFILE=ml-admin \
  AWS_DEFAULT_REGION=us-west-2 AWS_REGION=us-west-2
aws sso login   # if needed

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
- `run_gemma3_moe_rl_smoke.sh` prepares the DAPO-17k data in-container and runs
  the local smoke wrapper across all allocated GPUs. It puts the venv on `PATH`
  up front because the `sudo` entrypoint's `secure_path` otherwise hides it (and
  the CUDA base image has no system `python3`).

**Iterating on scripts** (fast — no source rebuild): the
`train-rl-distill-megatron-overlay` build config layers only the refreshed
`rl-distill-scripts/` on top of the already-pushed heavy image. Update the base
tag in `st_config/Dockerfile.megatron-overlay` to the last pushed image, then:

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

## 10. Validation

**Logit parity** (HF vs Megatron-Bridge load, real SFT weights):

```bash
SP=.venv-megatron/lib/python3.12/site-packages
CUDA_VISIBLE_DEVICES=4 LD_LIBRARY_PATH="$SP/nvidia/cudnn/lib" \
  .venv-megatron/bin/python rl-distill-scripts/logit_parity_gemma3_moe_hf_vs_megatron.py \
  <snapshot_dir>                       # single rank

# expert-parallel (EP=2 across 2 GPUs):
CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc-per-node 2 \
  rl-distill-scripts/logit_parity_gemma3_moe_hf_vs_megatron.py <snapshot_dir> --ep 2
```

Passing = top-1 token agreement `1.0` and mean logit diff `< 0.05`. Validated:
2E at EP=1/2, 4E at EP=1/2/4.

**Smoke:** §8a. Validated 2E (`disabled`, `R2`) and 4E (`TP2/EP4`) locally, and
a 4-GPU 4E run to completion on ScaleTrain.

---

## 11. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ModuleNotFoundError: megatron.bridge` | Bridge fork not installed — `MEGATRON_BRIDGE_PATH=third_party/Megatron-Bridge` in `setup_megatron.sh`. |
| `Could not find CMake executable` (TE build) | Install `cmake` (apt) / it's in the pip deps; re-run setup. |
| `cudnn.h: No such file` (TE build) | cuDNN include path — handled by setup; ensure `nvidia-cudnn-cu12` is installed. |
| `ValueError: Please use --moe-router-pre-softmax when topk is 1` | Missing the fork's `TransformerConfig` relaxation — you're on stock Megatron-Bridge. |
| vLLM loads `TransformersMoEForCausalLM` / crashes on `post_layernorm` | Config exposed a generic `num_experts` key — must be `gemma3_moe_num_experts`. |
| vLLM rollout server hangs at startup (R3) | The capture import in `modeling_gemma3_moe.py` must be **lazy**, not module-level. |
| Megatron replays expert 0 for every token | Rollout capture returned all zeros; the guardrail now raises. Use `R2`, or ensure the R3 hook + `enforce_eager` are active. |
| `python3: command not found` (ScaleTrain pod, exit 127) | venv not on `PATH` under `sudo`; `run_gemma3_moe_rl_smoke.sh` fixes it — rebuild via the overlay config. |
| OOM on a "free" shared GPU | A co-tenant landed on the same physical GPU. Use ScaleTrain (dedicated) or lower `ROLLOUT_GPU_MEMORY_UTILIZATION`. |
