# Gemma 3 4B dense-to-MoE upcycling and RL

This is the reproducible workflow for converting `google/gemma-3-4b-pt` into a
2- or 4-expert top-1 MoE, proving that initialization is correct, running it
through native vLLM, and starting Megatron DAPO training.

The architecture rationale is in
[`../gemma_3_4b_moe_upcycling.md`](../gemma_3_4b_moe_upcycling.md). Do not start
a long RL run until the gates in [Validation](#validation) pass.

## What is supported

| Component | Supported path |
|---|---|
| Dense source | `google/gemma-3-4b-pt` or a local snapshot of it |
| MoE shape | 2E or 4E, top-1, full-size duplicated MLPs |
| Actor/reference | Megatron-Core through the vendored Megatron-Bridge fork |
| Rollout | vLLM 0.18 native Gemma3-MoE plugin, Triton attention |
| Router consistency | R2 replay (supported default) |
| Exact initialization test | Canonical dense-init checkpoint view |
| Normal training | Ordinary sparse checkpoint only |

The generic vLLM Transformers backend is not a production path. It previously
produced very long, non-terminating MoE responses and entropy near `9`, while
the native plugin restored the expected response-length and entropy range.

## 1. Prerequisites

- Linux host with CUDA 12.9 and `nvcc`.
- H100 (`TORCH_CUDA_ARCH_LIST=9.0`) or B200 (`10.0`).
- At least 2 GPUs for a small 2E smoke test; the validated 20k-response recipe
  uses one node of 8x80 GB H100s with TP=4/EP=2.
- `uv`, `git`, `cmake`, gcc/g++, and access to gated
  `google/gemma-3-4b-pt`.
- Approximately 40 GB for the environment and about 13 GB/23 GB for a 2E/4E
  checkpoint. A canonical view normally uses hard links and adds negligible
  weight storage.

Install `uv` if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone the repository and create an untracked `.env`:

```bash
git clone https://github.com/JasonWei05/rl-distill.git
cd rl-distill

cat > .env <<'EOF'
HF_TOKEN=hf_xxx
WANDB_API_KEY=xxx
EOF
```

The launcher sources `.env`. For standalone conversion/validation commands,
export it into the current shell once:

```bash
set -a
source .env
set +a
```

## 2. Build the environment

The setup script creates `.venv-megatron`, installs the pinned CUDA stack, the
vendored bridge, and this repository editable. Editable installation is
required for vLLM to discover the `verl_gemma3_moe` plugin entry point.

H100:

```bash
UV_NO_CONFIG=1 PIP_INDEX_URL=https://pypi.org/simple \
  CUDA_HOME=/usr/local/cuda-12.9 \
  TORCH_CUDA_ARCH_LIST=9.0 \
  MAX_JOBS=64 \
  bash setup_megatron.sh
```

B200 uses `TORCH_CUDA_ARCH_LIST=10.0`. Useful setup overrides are:

| Variable | Default | Purpose |
|---|---|---|
| `VENV_DIR` | `./.venv-megatron` | Environment location |
| `CUDA_HOME` | auto-detected | CUDA toolkit containing `nvcc` |
| `MEGATRON_BRIDGE_PATH` | `third_party/Megatron-Bridge` | Bridge checkout |
| `MAX_JOBS` | `32` | Source-build parallelism |
| `RUN_HEAVY_BUILDS` | `1` | Build Apex, flash-attn, and TransformerEngine |
| `RUN_SMOKE_TEST` | `1` | Run setup import checks |

Verify the plugin after setup:

```bash
VIRTUAL_ENV="$PWD/.venv-megatron" uv run --active --no-sync python - <<'PY'
from vllm import ModelRegistry
from vllm.plugins import load_general_plugins

load_general_plugins()
assert "Gemma3MoeForCausalLM" in ModelRegistry.get_supported_archs()
print("NATIVE_GEMMA3_MOE_PLUGIN_OK")
PY
```

If that fails after pulling new code, refresh the editable install:

```bash
uv pip install --python .venv-megatron/bin/python -e .
```

## 3. Upcycle the dense checkpoint

Choose shared checkpoint paths. The command accepts either the Hub ID below or
an already-downloaded local snapshot.

### 2 experts

```bash
MOE_DIR=/shared/checkpoints/gemma3-4b-pt-moe-2e
CANONICAL_DIR=/shared/checkpoints/gemma3-4b-pt-moe-2e-canonical

VIRTUAL_ENV="$PWD/.venv-megatron" uv run --active --no-sync python \
  rl-distill-scripts/gemma3_moe_hf/create_gemma3_moe_from_dense_hf.py \
  --dense-model google/gemma-3-4b-pt \
  --output-dir "$MOE_DIR" \
  --canonical-output-dir "$CANONICAL_DIR" \
  --num-experts 2
```

### 4 experts

```bash
MOE_DIR=/shared/checkpoints/gemma3-4b-pt-moe-4e
CANONICAL_DIR=/shared/checkpoints/gemma3-4b-pt-moe-4e-canonical

VIRTUAL_ENV="$PWD/.venv-megatron" uv run --active --no-sync python \
  rl-distill-scripts/gemma3_moe_hf/create_gemma3_moe_from_dense_hf.py \
  --dense-model google/gemma-3-4b-pt \
  --output-dir "$MOE_DIR" \
  --canonical-output-dir "$CANONICAL_DIR" \
  --num-experts 4
```

The converter resolves the effective dense text configuration, copies every
non-MLP tensor, duplicates the gate/up/down/post-RMSNorm tensors into every
expert, initializes one deterministic random router per layer, and verifies
all tensor values and keys. Do not proceed unless it prints:

```text
GEMMA3_MOE_CHECKPOINT_VERIFIED
```

The two outputs have distinct roles:

| Path | Config flag | Use |
|---|---:|---|
| `$MOE_DIR` | `false` | Normal sparse smoke/RL training |
| `$CANONICAL_DIR` | `true` | Exact initialization tests only |

All experts are exact copies, so sparse routing is mathematically
dense-equivalent. Ordinary bf16 dispatch is not necessarily bit-exact because
token partitioning changes GEMM shapes and kernel fusion. The canonical view
runs expert 0 over the full batch while still evaluating the router; this
isolates conversion correctness from sparse bf16 rounding. Never train the
canonical view.

## 4. Prepare data

For the bounded smoke and H100 wrapper:

```bash
DATA_DIR="${HOME}/verl/data" \
  bash rl-distill-scripts/data/prepare_dapo_17k_split.sh
```

This creates `dapo_17k_train.parquet` and `dapo_17k_test.parquet`. The generic
launcher also supports the OpenMath2 mix:

```bash
DATA_DIR="${HOME}/verl/data" \
  bash rl-distill-scripts/data/prepare_dapo_openmath2_mix_split.sh
```

Custom parquet files need a `prompt` chat-list column and
`reward_model.ground_truth`. Set `TRAIN_FILE` and `VAL_FILE` explicitly.

## 5. Validation

### Gate A: Hugging Face weights and every layer boundary

Run the canonical checkpoint on one idle GPU:

```bash
CUDA_VISIBLE_DEVICES=0 \
VIRTUAL_ENV="$PWD/.venv-megatron" uv run --active --no-sync python \
  rl-distill-scripts/check_gemma3_dense_moe_activations.py \
  --dense-model google/gemma-3-4b-pt \
  --moe-model "$CANONICAL_DIR" \
  --seq-len 2048 --all-components
```

Required result:

```text
logits exact=True ... top1_agreement=1.00000000
first_nonexact=none
HF_DENSE_MOE_ACTIVATION_PARITY_OK
```

### Gate B: real Megatron actor graph at production topology

For the validated 2E 8xH100 layout:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
VIRTUAL_ENV="$PWD/.venv-megatron" uv run --active --no-sync \
  torchrun --standalone --nproc-per-node 8 \
  rl-distill-scripts/check_gemma3_mcore_dense_moe_activations.py \
  --dense-model google/gemma-3-4b-pt \
  --moe-model "$CANONICAL_DIR" \
  --tp 4 --ep 2 --seq-len 2048 --check-parameters
```

This independently bridge-loads dense and MoE weights and checks every layer's
FC1, FC2, norms, attention output, MLP output, residual output, logits, and
top-1 decisions. Required result:

```text
first_nonexact_parameter=none
first_nonexact=none
top1_agreement=1.00000000
MCORE_DENSE_MOE_ACTIVATION_PARITY_OK
```

`99.02%` is not a passing result. The canonical gate is deliberately exact so
a mapping or graph error cannot be dismissed as harmless numerical drift.

### Gate C: native vLLM

Run one backend per process on idle GPUs. Use the same prompt, tokenizer, stop
IDs, attention backend, and max-token limit:

```bash
CUDA_VISIBLE_DEVICES=0 \
VIRTUAL_ENV="$PWD/.venv-megatron" uv run --active --no-sync python \
  rl-distill-scripts/diagnose_gemma3_vllm_parity.py \
  --model dense --backend native \
  --dense-model google/gemma-3-4b-pt \
  --attention-backend TRITON_ATTN --max-tokens 2048 \
  --output-json /tmp/gemma3-dense-native.json

CUDA_VISIBLE_DEVICES=1 \
VIRTUAL_ENV="$PWD/.venv-megatron" uv run --active --no-sync python \
  rl-distill-scripts/diagnose_gemma3_vllm_parity.py \
  --model moe --backend native \
  --dense-model google/gemma-3-4b-pt \
  --moe-model "$CANONICAL_DIR" \
  --attention-backend TRITON_ATTN --max-tokens 2048 \
  --output-json /tmp/gemma3-moe-native.json

VIRTUAL_ENV="$PWD/.venv-megatron" uv run --active --no-sync python - <<'PY'
import json

with open("/tmp/gemma3-dense-native.json") as handle:
    dense = json.load(handle)
with open("/tmp/gemma3-moe-native.json") as handle:
    moe = json.load(handle)
for key in ("prompt_token_sha256", "output_token_sha256", "finish_reason", "output_token_count"):
    assert dense[key] == moe[key], (key, dense[key], moe[key])
print("NATIVE_VLLM_DENSE_MOE_PARITY_OK")
PY
```

The normal sparse checkpoint can differ slightly from dense in bf16. Use the
canonical view for the proof, then the normal view for training.

### Gate D: bounded end-to-end RL round trip

This exercises the production sparse Megatron actor/reference, native vLLM
rollout, R2 replay, old log-probs, an update, and post-update weight
resynchronization. Gates A--C already prove the canonical initialization; this
gate deliberately tests the execution path used for training:

```bash
UPCYCLED_MOE_DIR="$MOE_DIR" \
TOTAL_TRAINING_STEPS=2 \
  bash rl-distill-scripts/gemma3_4b_pt_moe_megatron_correctness_1node_h100.sh
```

Only after this finishes should the normal sparse checkpoint enter a long run.

## 6. Start normal sparse training

The one-node H100 wrapper is the known-good 2E recipe:

```bash
UPCYCLED_MOE_DIR="$MOE_DIR" \
MEGATRON_VENV="$PWD/.venv-megatron" \
  bash rl-distill-scripts/gemma3_4b_pt_moe_2e_megatron_rl_1node_h100.sh
```

Important defaults:

| Variable | Default | Meaning |
|---|---:|---|
| `ACTOR_TP` / `ACTOR_EP` | `4` / `2` | Validated 8xH100 topology |
| `ROUTER_REPLAY_MODE` | `R2` | Trainer old-route record/replay |
| `MOE_AUX_LOSS_COEFF` | `1e-3` | Router load balancing |
| `ROLLOUT_MODEL_IMPL` | `native` | Registered vLLM implementation |
| `ROLLOUT_ATTENTION_BACKEND` | `TRITON_ATTN` | Same native attention family as dense probe |
| `ROLLOUT_ENFORCE_EAGER` | `True` | Validated execution mode for routed experts |
| `SAVE_FREQ` | `25` | Local model/Adam/extra/HF export checkpoint |
| `TEST_FREQ` | `5` | Validation every five steps |
| `VAL_BEFORE_TRAIN` | `True` | Validation at global step 0 |
| `ACTOR_LR_WARMUP_STEPS` | `20` | Fresh-run optimizer warmup |

The wrapper refuses to fall back to the historical SFT checkpoint and rejects
the canonical test view. Both Gate D and normal training require the ordinary
sparse output.

For 4E or another topology, call the generic launcher explicitly:

```bash
NUM_EXPERTS=4 \
HF_MOE_LOCAL_DIR="$MOE_DIR" \
ACTOR_TP=2 ACTOR_EP=4 REF_TP=2 REF_EP=4 \
  bash rl-distill-scripts/gemma3_4b_pt_moe_megatron_rl_20k.sh
```

The older `gemma3_4b_pt_moe_{2e,4e}_megatron_rl_20k.sh` wrappers explicitly
select published SFT checkpoints. They are useful for those checkpoints but
are not fresh dense-upcycle parity runs.

### Metrics to watch

- `actor/entropy`: compare against a dense run with identical prompts,
  sampling, and attention backend. At initialization/early in the validated
  2E run it was roughly `1.9-2.1`; entropy near `9` is a correctness alarm.
- `response_length/mean`: should track the dense baseline. Repeatedly reaching
  `MAX_RESPONSE_LENGTH` indicates stop/cache/model-path trouble.
- `actor/train/router_loss`: a balanced two-expert auxiliary loss is near
  `1.0`; values such as `1.09` are plausible. This does not prove logit parity.
- `actor/train/router_loss_scaled`: should be the raw router loss times the
  configured auxiliary coefficient.
- `actor/grad_norm`, KL, reward, and validation: must remain finite.

Do not tune entropy coefficients or response limits to mask an initialization
or rollout mismatch.

## 7. Local checkpoints and Hugging Face uploads

Every positive `SAVE_FREQ` event writes:

```text
global_step_N/
  actor/                 # Megatron model, Adam optimizer, RNG/extra state
  actor/huggingface/     # weight-only HF snapshot for inference
  data.pt                # StatefulDataLoader cursor
```

The launcher saves `model`, `optimizer`, `extra`, and `hf_model`, but only
loads `model`, `optimizer`, and `extra` on resume. The HF directory is an
inference artifact, not an Adam checkpoint.

Enable weight-only Hub uploads at the local save cadence:

```bash
HF_PUSH_ENABLE=True \
HF_PUSH_REPO=your-org/gemma3-4b-pt-moe-2e-rl \
SAVE_FREQ=25 HF_PUSH_FREQ=25 \
UPCYCLED_MOE_DIR="$MOE_DIR" \
  bash rl-distill-scripts/gemma3_4b_pt_moe_2e_megatron_rl_1node_h100.sh
```

`HF_PUSH_FREQ` is evaluated only after a local save. Make it a multiple of
`SAVE_FREQ`; `25/25` uploads steps 25, 50, 75, and so on. `HF_TOKEN` is read
from `.env` by the launcher.

## 8. Resume without changing the data position

An exact resume restores model, Adam, scheduler/RNG/extra state, global step,
and `data.pt`:

```bash
RUN_DIR=/shared/ckpts/DAPO/my-run
STEP_DIR="$RUN_DIR/global_step_25"

CKPTS_DIR="$RUN_DIR" \
RESUME_MODE=resume_path \
UPCYCLED_MOE_DIR="$MOE_DIR" \
  bash rl-distill-scripts/gemma3_4b_pt_moe_2e_megatron_rl_1node_h100.sh \
  trainer.resume_from_path="$STEP_DIR"
```

Keep dataset paths, shuffle seed, batch size, response count, and data-parallel
layout unchanged. The logs should say that global step 25 and the dataloader
state were restored. If `data.pt` is absent, the loader warns and starts from
the beginning.

To deliberately keep Adam moments but replace the saved LR schedule with zero
warmup, use:

```bash
CKPTS_DIR="$RUN_DIR" \
RESUME_MODE=resume_path \
ACTOR_LR_WARMUP_STEPS=0 \
UPCYCLED_MOE_DIR="$MOE_DIR" \
  bash rl-distill-scripts/gemma3_4b_pt_moe_2e_megatron_rl_1node_h100.sh \
  trainer.resume_from_path="$STEP_DIR" \
  actor_rollout_ref.actor.optim.use_checkpoint_opt_param_scheduler=False
```

That is a deliberate scheduler change, not an exact resume. Adam tensors and
the dataloader cursor still come from step 25.

## 9. Router replay

- `R2` records routes during the actor's old-log-prob forward and replays them
  for the update. It has no rollout-engine dependency and is supported.
- `disabled` reroutes each pass independently and is useful only for diagnosis.
- `R3` asks vLLM to capture generation routes and transfer them to the actor.
  The native model contains the capture call, but the colocated handoff has not
  passed the full correctness gate. The launcher requires
  `ALLOW_EXPERIMENTAL_R3=True`, and the trainer raises if the replay map is all
  zeros.

Use R2 for production until R3 completes the same end-to-end gate.

## 10. Pinned stack

`setup_megatron.sh` currently installs Python 3.12 with:

| Package | Version/ref |
|---|---|
| PyTorch | `2.10.0+cu129` |
| vLLM | `0.18.0` |
| Transformers | `5.3.0` |
| Megatron-Core | `core_v0.16.0` |
| Megatron-Bridge | vendored `third_party/Megatron-Bridge` fork |
| TransformerEngine | `release_v2.12` |
| flash-attn | `2.8.3` |

Changing these versions requires rerunning the activation and native-vLLM
gates; the plugin imports vLLM model internals whose API is version-sensitive.

## 11. Troubleshooting

| Symptom | Action |
|---|---|
| Converter cannot access Gemma | Accept the gated model and export `HF_TOKEN`. |
| `GEMMA3_MOE_CHECKPOINT_VERIFIED` is absent | Treat conversion as failed; do not launch. |
| Launcher selects no model | Set `MODEL_PATH`, `HF_MOE_LOCAL_DIR`, or explicit `HF_MOE_REPO`. |
| Canonical mode mismatch | Use canonical only for Gates A--C; Gate D and normal RL use the sparse output. |
| Native plugin not registered | Run `uv pip install --python .venv-megatron/bin/python -e .`. |
| vLLM selects `TransformersMoEForCausalLM` | Remove generic expert keys and reconvert with the current script. |
| Entropy is near 9 or generations hit the cap | Confirm `model_impl=native`, `TRITON_ATTN`, stop IDs, and the intended checkpoint. |
| Canonical top-1 is below 1.0 | Stop: inspect the first mismatching parameter/layer; this is not acceptable rounding. |
| Router loss is near 1 | Usually healthy balance for 2E; check parity separately. |
| R3 replay is entirely zero | Use R2; the guardrail prevented invalid training. |
| Resume starts at the wrong samples | Confirm `global_step_N/data.pt` exists and data/batch/topology settings are unchanged. |
| HF step was not uploaded | Uploads only run on local-save steps; align `HF_PUSH_FREQ` with `SAVE_FREQ`. |
| TransformerEngine cannot find cuDNN | Use the launcher or add the venv's `nvidia/cudnn/lib` to `LD_LIBRARY_PATH`. |
