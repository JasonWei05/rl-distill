# NeMo-RL exact replication of the gemma-4 E2B DAPO run

Reproduces the verl run **`DAPO-gemma4-e2b-pt-DeepScaleR-4of4strict-seed42` (8k variant:
8192 max response + 2048 overlong buffer)** on NVIDIA NeMo-RL, with **zero edits** to the
vendored `third_party/nemo-rl` checkout (pinned submodule @ `5f89b3ae`). Purpose: test
whether the truncation-driven grad-norm bimodality seen in verl (clean steps 1.3–6.8;
steps with ≥1 at-cap response 83–3108) reproduces in an independent framework —
i.e. behavior of the *algorithm*, not a verl bug.

verl reference data: wandb `rl-distill/DAPO` run `recbw9dcxso` (recovery of `bw9dcxso`,
285 steps).

## Layout

| File | Role |
|---|---|
| `config/dapo_gemma4_e2b_pt_repro.yaml` | Full parity config; every knob comments its verl counterpart |
| `run_grpo_repro.py` | Wrapper around their `examples/run_grpo.py`: registers the `math_strict` env before setup; `NEMORL_FORCE_LOCAL_RAY=1` shim for shared devboxes |
| `sitecustomize.py` | Auto-loads in every Ray actor via the launcher's PYTHONPATH: on transformers >= 5.5.2 applies the two gemma-4 KV-sharing fixes (use_cache workaround removal + FSDP2 `cast_forward_inputs=False`); on the locked 5.5.0 it **hard-fails** any act-ckpt launch instead of letting training silently corrupt (see Gotchas) |
| `rl_distill_nemo/strict_math_env.py` | Verbatim port of the strict boxed-only scorer (`verl/utils/reward_score/math_verify.py`) as a NeMo-RL environment |
| `rl_distill_nemo/deepscaler_dataset.py` | verl-parquet → NeMo response-dataset adapter |
| `tests/` | Reward parity (6/6) and tokenization parity (5/5 byte-identical) vs verl |
| `PARITY_CHECKLIST.md` | The matched-knob table + caveats |
| `compare_grad_norms.py` | Pulls the newest nemorl wandb run, tabulates grad_norm vs truncation_rate, prints a bimodality verdict vs the verl reference |
| `../scale_train/run_nemorl_gemma4_e2b_repro.sh` | Pod run file (works on both images; see below) |
| `../scale_train/st_config/Dockerfile.nemorl` | Baked image (`train-rl-distill-nemorl` build config) |

## Prebuilt image (pull instead of building)

The baked image is in Scale's ECR under a stable tag (launch-generated tags in the same
repo are ephemeral/overwritten — use this one):

```bash
aws ecr get-login-password --region us-west-2 | \
  docker login --username AWS --password-stdin 692474966980.dkr.ecr.us-west-2.amazonaws.com
docker pull 692474966980.dkr.ecr.us-west-2.amazonaws.com/scale_train/shared/training/tmp:rl-distill-nemorl-cu130-20260729
# digest sha256:370c4cd9e8185f82d08439aabba7349d56c223c1eb1edb32ad451f40130ddcc2 (78.9GB)
```

Needs AWS creds for account 692474966980 (us-west-2). External registries (GHCR/Docker
Hub) are not an option without restructuring the Dockerfile: the venv layers are 38.4GB
and 19.9GB, past their ~10GB layer caps. Worst case, `Dockerfile.nemorl` rebuilds it
from scratch in ~1h (H100-arch only, no GPU needed at build time).

**No-AWS option (public)**: the environment-only variant (`--target env` — everything
except the private repo COPY layer; only public/redistributable content) is on the HF Hub
as a `docker save` tarball, downloadable without any auth:

```bash
huggingface-cli download JWei05/nemorl-gemma4-cu130-env rl-distill-nemorl-env-cu130-20260729.docker.tar.zst --local-dir .
zstd -d -c rl-distill-nemorl-env-cu130-20260729.docker.tar.zst | docker load   # 29.5GB download
# materialize the repo around the baked third_party/nemo-rl (do NOT bind-mount over it):
# cd /workspace/rl-distill && git init && git remote add origin git@github.com:JasonWei05/rl-distill.git \
#   && git fetch --depth 1 origin main && git checkout -f FETCH_HEAD
```

## Prereqs

```bash
git submodule update --init --recursive third_party/nemo-rl   # pinned @ 5f89b3ae
#   --recursive is REQUIRED: nemo-rl's own 3rdparty submodules (Gym/Automodel/...) are uv
#   workspace members; without them `uv sync` fails with "nemo-gym ... not a workspace member"
# repo-root .env with HF_TOKEN (gated google/gemma-4-E2B) + WANDB_API_KEY
# data: bash rl-distill-scripts/data/prepare_deepscaler_4of4strict_rl_data.sh
#   (downloads the exact train 9,723 / val 200x16 split from JWei05/DeepScaleR-4of4-strict-RL)

# only needed by the sparse-delta checkpoint materializer; 0.25.0 is the verified version
uv pip install --python /path/to/venv/bin/python zstandard==0.25.0
```

The pinned E4B step-100 chain can then be validated or materialized with
`local/materialize_hf_checkpoint_chain.py`; its default lock is
`config/e4b_step100_delta_chain.lock.json`. Materialization refuses to overwrite an existing output
directory and verifies every link and the final packaged SHA256.

## ScaleTrain (the intended path)

```bash
# baked image: pod startup ~35-40 min (HF download + vLLM init dominate).
# MAX_STEPS caps the run (e.g. 10 for a grad-norm comparison run); omit for open-ended.
python rl-distill-scripts/scale_train/launch_st_job.py --cluster eks --n-instances 1 \
  --gpus-per-instance 8 --priority high --job-name gemma4-e2b-nemorl-repro \
  --build-config-key train-rl-distill-nemorl --team egp --product train.enterprise_rlvr \
  --run-file run_nemorl_gemma4_e2b_repro.sh --env-vars "MAX_STEPS=10" --allow-borrowing
```

The run file is idempotent across both images: on the plain `train-rl-distill` image it
does the full toolchain setup in-pod (~60 min: CUDA-13 toolkit, cuDNN dev headers, cmake,
uv sync with TransformerEngine source build); on `train-rl-distill-nemorl` every step
fast-skips (markers still print). Pod-log marker chain to watch:

```
DATA_PREP_OK -> CUDA_COMPAT_OK -> CUDNN_DEV_OK -> CUDA13_TOOLKIT_OK -> CMAKE_OK
  -> UV_ENV_OK -> CUDNN_HOME_OK/NCCL_HEADERS_OK -> WORKER_VENVS_OK -> GEMMA4_KV_PATCH_OK
  -> CUDA_OK -> MATH_VERIFY_OK -> GATE_PASS validation/accuracy=X
  -> (training steps) -> RUN_DONE rc=0
```

`GEMMA4_KV_PATCH_OK` (via `nemo_rl_repro/local/patch_gemma4_kv_venvs.sh`) is the gemma-4
KV-sharing correctness stack — REQUIRED for training; without it the repro's
`sitecustomize.py` hard-fails any act-ckpt launch on the locked transformers 5.5.0
rather than let training silently corrupt (see Gotchas).

**Go/no-go gate**: a `max_num_steps=1` run whose step-0 validation accuracy (full
3200-sample val, mean@16) must land in **[0.045, 0.075]** — the verl baseline band
(5.03–6.16%). Anything outside means prompt/reward/sampling parity is broken; do not
proceed to training. Passed at **0.0550** on 8xH100 (2026-07-29), matching the local
2-GPU gate (0.0508).

## Local (devbox) gate

Driver venv on NVMe (`/tmp/nemo-rl-venv`, built with `UV_PROJECT_ENVIRONMENT=... uv sync
--locked --extra automodel --no-install-package deep-ep ...`); see PROGRESS_LOG 2026-07-28
for the full env recipe (CUDA-13 user-prefix toolkit at /tmp/cuda-13.0, CCCL/NCCL include
wiring, Ray timeout patch). Key invocation shape:

```bash
NEMORL_FORCE_LOCAL_RAY=1 HF_HOME=/tmp/hf-home VLLM_USE_DEEP_GEMM=0 \
CUDA_VISIBLE_DEVICES=5,6 ... /tmp/nemo-rl-venv/bin/python run_grpo_repro.py \
  --config config/dapo_gemma4_e2b_pt_repro.yaml \
  cluster.gpus_per_node=2 grpo.max_num_steps=1 \
  grpo.num_prompts_per_step=8 policy.train_global_batch_size=128 \
  grpo.max_val_samples=512 grpo.val_batch_size=512 \
  checkpointing.enabled=false logger.wandb_enabled=false \
  policy.tokenizer.chat_template=$PWD/../data/gemma3_it_fewshot_math.jinja \
  policy.generation.vllm_cfg.gpu_memory_utilization=0.22
```

Validation/generation works at 2 GPUs; **the training step does not fit** below ~4 clean
H100s (fp32 AdamW states; see PROGRESS_LOG). Use ScaleTrain for training.

### 4xH100 training recipe (verified 2026-07-30; see PROGRESS_LOG for the forensics)

Training at 12288-token parity **requires activation checkpointing** (activations alone are
~75 GiB without recompute; TP=2 is not an escape — E2B's forward breaks under the dtensor TP
plan) — but act-ckpt on the **locked transformers 5.5.0 silently corrupts gemma-4 E2B training**
(Gotchas below). The full working stack:

The whole stack is scripted under **`local/`**:

```bash
bash rl-distill-scripts/nemo_rl_repro/local/build_venv.sh   # driver+worker venvs, transformers
                                                            # 5.5.4 patch, sitecustomize copy
bash rl-distill-scripts/nemo_rl_repro/local/run_gate.sh     # 1-step go/no-go (criteria below)
bash rl-distill-scripts/nemo_rl_repro/local/run_train.sh    # full run (MAX_STEPS caps; RUN_SUFFIX,
                                                            # GEMMA4_VARIANT=e2b|e4b, GPUS_PER_NODE)
```

The correctness knobs (act-ckpt + cpu_offload + `automodel_kwargs.force_hf`) live in the
config yamls; `local/patch_gemma4_kv_venvs.sh` applies the venv-level pieces (transformers
5.5.4 into driver + policy-worker venvs — a worker-venv rebuild reverts to the locked
5.5.0, re-run it — plus the `sitecustomize.py` copy).

Verified end-to-end on 4xH100 (gate `4ndtkcry`): step-0 val 0.0600 ∈ [0.045, 0.075], step-1
`train/probs_ratio = 1.000017`, grad_norm 0.74, ~5.5 min/step. **Gate criterion added by the
2026-07-30 incident: step-1 `train/probs_ratio` and `train/probs_ratio_clamped` must be ≈ 1.0
(±0.01). Val accuracy alone is NOT sufficient — the corrupted-training failure mode passes the
val band while `probs_ratio_clamped` pins at 0.80.**

## Comparing grad norms

```bash
python compare_grad_norms.py            # auto-finds the newest nemorl run in rl-distill/DAPO
```

Prediction if verl is correct: `train/grad_norm` bimodal, keyed on `train/truncation_rate`
(> 0 → high regime). Both frameworks log the same quantity: pre-clip global L2 over all
trainable params per optimizer step.

## Hard-won gotchas (all already encoded in the run file / Dockerfile / config)

- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` **must not** be set globally — it
  propagates into the vLLM workers whose memory pool asserts at engine init.
- `policy.dtensor_cfg.activation_checkpointing: true` is required: our 12288-token
  parity sequence length is 3x NVIDIA's own E2B recipe (4096); without recompute the
  training step OOMs on 80GB H100s.
- `policy.refit_buffer_size_gb: 12` (yaml, not CLI — CLI override was silently dropped):
  the ping-pong halves must fit the 4.7GB per-layer embedding tensor.
- `grpo.reward_shaping.max_response_length: 8192` must be explicit — the default
  interpolates to `max_total_sequence_length` (12288) and silently moves the penalty onset.
- gemma-4 has no `<end_of_turn>` token: termination comes from `stop_strings` in the
  config (the model emits the literal text otherwise and never stops).
- ScaleTrain pods reset PATH in the run shell: never rely on bare `python3` from image ENV.
- **act-ckpt + transformers 5.5.0 = silently corrupted gemma-4 E2B training** (root-caused
  2026-07-30). E2B's KV-shared layers need `use_cache=True` in every forward (their PR #2224 /
  Automodel#1705); HF disables the cache in grad mode under gradient checkpointing, so the
  TRAINING forward computes garbage logits while the no-grad logprob pass stays correct:
  `train/probs_ratio_clamped` pins at 0.80, val stays at baseline, nothing crashes.
  `sitecustomize.py` now hard-fails this combination. **The pod image bakes 5.5.0 — do not
  launch ScaleTrain runs until the image gets the same transformers bump + sitecustomize.**
- transformers >= 5.5.2 alone is NOT enough: its new KV-sharing passes anchor K/V by mutating a
  plain dict kwarg, which FSDP2's default input-cast (`cast_forward_inputs=True`) container-copies
  at every fully-sharded layer → `shared_kv_states` KeyError. And the Automodel custom gemma4
  class predates the mechanism entirely. Hence the other two pieces of the recipe:
  `sitecustomize.py`'s `cast_forward_inputs=False` patch (value-wise a no-op) and
  `+policy.dtensor_cfg.automodel_kwargs.force_hf=true`.
- `checkpointing.keep_top_k` must be an int on this pin (`NotRequired[int]`): an explicit
  `null` fails MasterConfig pydantic validation at startup. The config uses 1000000 = keep all.
- `policy.logprob_chunk_size` is silently ignored in the training loss path (upstream
  `LossPostProcessor` never forwards it; the Megatron twin was their #2871/#2872, automodel side
  unfixed) — memory-only, numerically inert; `cpu_offload=true` covers the 4-GPU shortfall.
