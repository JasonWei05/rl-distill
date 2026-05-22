# Gemma 3 4B Depth Upscaling — Plan & Implementation Guide

## Goal

Take the pretrained **Gemma 3 4B** model (34 transformer layers) and convert it into a deeper model by **inserting additional transformer layers** that act as *identity* at initialization. The upscaled model's forward pass at step 0 must be numerically equivalent to the original model's forward pass. Training afterward proceeds identically to training the original — same data, same optimizer, same hyperparameters — just on a taller stack.

This is the LLaMA-Pro / SOLAR-style "depth upcycling" technique, specialized to Gemma 3's 5:1 sliding/global attention rhythm and applied in the ModelChef repo using the existing **Megatron-Bridge** tooling (which already has first-class Gemma 3 4B support, including the VL variant).

---

## Insertion schedule — "after each block, insert two new blocks"

Gemma 3's layers alternate **5 local-sliding + 1 global** = a cycle of 6. The natural "block" is this 6-layer cycle. The model's 34 layers factor as:

```
Block 1:  layers  1– 6   (5 local + 1 global)
Block 2:  layers  7–12   (5 local + 1 global)
Block 3:  layers 13–18   (5 local + 1 global)
Block 4:  layers 19–24   (5 local + 1 global)
Block 5:  layers 25–30   (5 local + 1 global)
Tail:     layers 31–34   (4 local — no global cap at the end)
```

Inserting **two new 6-layer blocks after each of the five complete blocks**:

```
B1 | N1a N1b | B2 | N2a N2b | B3 | N3a N3b | B4 | N4a N4b | B5 | N5a N5b | tail
 6     12    | 6     12    | 6     12    | 6     12    | 6     12    |  4
```

**Total: 15 blocks × 6 + 4 tail = 94 layers** (34 original + 60 inserted).

### Why this schedule is the clean case

Because every insertion is a **multiple of 6**, two critical properties hold:

1. **Existing layers keep their type.** Any original layer at 1-indexed position `p` with `p % 6 == 0` (global) stays global; any with `p % 6 ∈ {1..5}` (local) stays local, because `(p + 6k) % 6 == p % 6` for any integer `k`.
2. **New layers inherit the correct type from their position.** We set each new block as a full copy of the preceding original block (intra-block position `j` → same `j` in the new block), which makes the new layer's attention type (local/global) match the source layer it was copied from.

So the 5:1 rhythm is preserved with **zero type mismatches**.

### Source of weights for each new block

```
N_i_a[j] := deepcopy(B_i[j])    for j in 0..5
N_i_b[j] := deepcopy(B_i[j])    for j in 0..5
```

Then zero the two output projections in every one of those 60 inserted layers (next section).

### PP divisibility

`94 = 2 × 47`. PP ∈ {1, 2, 47, 94} divide evenly; **PP=4 and PP=8 do not**. Options if you need PP≥4:

- **Uneven PP** (Megatron-LM supports `decoder_first_pipeline_num_layers` / `decoder_last_pipeline_num_layers`).
- **Adjust insertion count** to land on a friendlier total. E.g., insert 1 new block after each of the 5 blocks → 5 × 6 = 30 new layers → **64 layers total** (divisible by 4, 8, 16, 32, 64). This is probably the cleanest fallback if PP=8 is desired.
- **Mix**: insert 2 blocks after most, 1 after some, to hit 88 (= 8 × 11) or another multiple-of-8.

Decide PP before you scope the insertion count.

---

## Tooling in this repo

| Tool | Path | Relevance |
|---|---|---|
| **Megatron-Bridge** (NVIDIA, official) | `Megatron-Bridge/` | **First-class Gemma 3 support**. Providers for text-only 1B/4B/12B/27B and VL 4B/12B/27B. Bridges for both text-only and VL. Recipes for pretrain / SFT / PEFT. **This is the recommended path.** |
| **mbridge** (ByteDance / ISEEKYAN) | via `pip install verl[mcore]` | Supports Qwen, Llama, Mixtral, DeepSeek-V3 but **no Gemma**. Not the right tool here. |
| **Megatron-LM** | `Megatron-LM/` | Has MoE upcycling but **no depth-upcycling utility**. Not needed — we do surgery at the HF level. |
| **veomni-recipes** | `veomni-recipes/` | FSDP training on HF models directly. Backup path if Megatron-scale parallelism isn't needed. |

**Chosen path:** do the depth-upscaling surgery at the **HuggingFace checkpoint level**, then hand the modified checkpoint to Megatron-Bridge, which transparently loads a 94-layer Gemma 3 into Megatron-Core with full TP/PP/CP support. No new bridge code, no new model code.

### Key Megatron-Bridge files for this work

- `Megatron-Bridge/src/megatron/bridge/models/gemma/gemma3_provider.py` — text-only Gemma 3 provider. Attention pattern logic at `_is_local_attn_layer()` (~line 400–405) uses `layer_number % 6 != 0`, so it works for arbitrary layer counts automatically.
- `Megatron-Bridge/src/megatron/bridge/models/gemma/gemma3_bridge.py` — text-only HF↔mcore bridge. Weight mapping is regex-based (`decoder.layers.*.self_attention...`), so layer count is not hard-coded. Registered for `Gemma3ForCausalLM`.
- `Megatron-Bridge/src/megatron/bridge/models/gemma_vl/gemma3_vl_provider.py` and `gemma3_vl_bridge.py` — multimodal versions.
- `Megatron-Bridge/src/megatron/bridge/recipes/gemma/gemma3.py` — pretrain / SFT / PEFT recipes. Parameterized — instantiate `Gemma3ModelProvider4B()` or let the bridge read config from the HF checkpoint.

---

## Gemma 3 4B architecture (relevant pieces)

- **34 decoder layers**, hidden size 2560, 8 attention heads, 4 KV groups (GQA)
- **FFN intermediate** proportional to the Gemma 3 4B spec (verify via `config.intermediate_size`)
- **Sliding window** = 1024 tokens for 4B; applies only to local layers
- **Attention pattern**: `(5 local, 1 global)` cycle. Local layers use short-window attention + RoPE with local base 10K. Global layers use full attention + RoPE with global base 1M (linear scaling applied to global only).
- **Sandwich norms**: RMSNorm *both before and after* each sublayer:
  ```
  a  = self_attn(input_layernorm(x))
  a' = post_attention_layernorm(a)          # RMSNorm on attn OUTPUT
  h  = x + a'
  m  = mlp(pre_feedforward_layernorm(h))
  m' = post_feedforward_layernorm(m)
  y  = h + m'
  ```
- **RMSNorm with zero-centered gamma** (`x * (1 + w)`, not `x * w`), `eps = 1e-6`
- **QK-norm**: `q_norm` and `k_norm` applied to Q and K separately inside attention
- **Tied embeddings** (`share_embeddings_and_output_weights = True`)
- **4B is multimodal on HF** — top-level HF class is `Gemma3ForConditionalGeneration` (wraps `Gemma3ForCausalLM` language model + SigLIP vision tower + multimodal projector)

### Per-layer HF module names

- `self_attn.{q_proj, k_proj, v_proj, o_proj, q_norm, k_norm}`
- `mlp.{gate_proj, up_proj, down_proj}`
- `input_layernorm`, `post_attention_layernorm`
- `pre_feedforward_layernorm`, `post_feedforward_layernorm`

Language-model layer list is at:
- **VLM wrapper**: `model.language_model.model.layers`
- **Text-only**: `model.model.layers`

---

## The identity-init trick — why it works cleanly for Gemma 3

For the Gemma 3 block with sandwich norms, zeroing the two output projections forces both sublayer contributions to zero even with the extra norms:

| Weight | Action | Effect chain |
|---|---|---|
| `self_attn.o_proj.weight` | **zero** | `a = W_O · (softmax(QKᵀ/√d)·V) = 0` → `a' = RMSNorm(0) = 0` → `h = x + 0 = x` |
| `mlp.down_proj.weight` | **zero** | `m = W_D · act(gate ⊙ up) = 0` → `m' = RMSNorm(0) = 0` → `y = h + 0 = h = x` |

`RMSNorm(0) = 0` holds because `RMSNorm(z) = z / √(mean(z²) + eps) · γ`, and `z = 0` gives `0 / √eps · γ = 0`. This is the key reason the sandwich norm architecture doesn't break the identity trick.

Everything else in the layer stays at its trained values:
- Q/K/V, gate/up projections: copied from source
- Four layernorms (input_ln, post_attn_ln, pre_ff_ln, post_ff_ln): copied from source
- `q_norm`, `k_norm` (QK-norms): copied from source

Result: each inserted block is **exactly** identity at step 0.

### Gradient-flow caveat (one-step delay)

With `o_proj = 0` and `down_proj = 0` at step 0:

- **Output projections themselves get nonzero gradients at step 0** — they start learning immediately.
- **Q/K/V/gate/up get zero gradient at step 0** — their gradient must pass through `o_proj` or `down_proj` which is zero.
- After one optimizer step, output projections are nonzero → Q/K/V/gate/up begin receiving gradient from step 1.
- **Upstream layers (the 34 original Gemma 3 layers) are unaffected** — the residual path carries their gradients normally throughout.

This is the known LLaMA-Pro / SOLAR behavior; not an issue in practice.

**Alternative if the delay bothers you**: initialize output projections with a tiny nonzero std (e.g., 1e-5 or 1e-6). Block is ε-close to identity; all parameters get real gradient at step 0. Default recommendation is still **exact zero** — cleaner, proven, negligible downside.

---

## Implementation steps

### Step 0 — Decisions to make upfront

1. **Text-only or multimodal?**
   - **Text-only**: extract `Gemma3ForCausalLM` from the checkpoint (or use a text-only variant if one is published), use `gemma3_bridge.py`. Simpler; faster to iterate.
   - **Multimodal**: keep `Gemma3ForConditionalGeneration` with vision tower + projector intact, use `gemma3_vl_bridge.py`. Operate on `model.language_model.model.layers` during surgery; leave vision tower untouched.
2. **Base vs instruction-tuned**: `google/gemma-3-4b-pt` for continued pretraining; `google/gemma-3-4b-it` for preserving the instruction prior.
3. **PP target** (see PP divisibility above): confirms 94 is OK or forces a schedule adjustment.
4. **Exact-zero vs near-zero output projections**: default exact zero.

### Step 1 — Write the surgery script (HF level)

```python
import copy, torch
from transformers import AutoModelForImageTextToText, AutoProcessor
# Or: AutoModelForCausalLM if you extract text-only first.

SRC = "google/gemma-3-4b-pt"
DST = "/local/path/gemma3-4b-upscaled-94L"

model     = AutoModelForImageTextToText.from_pretrained(SRC, torch_dtype=torch.bfloat16)
processor = AutoProcessor.from_pretrained(SRC)

# For the VLM wrapper, LM layers are under language_model.model.layers.
# For text-only, it's model.model.layers — adjust accordingly.
layers = model.language_model.model.layers
assert len(layers) == 34

# Identify original-block boundaries (1-indexed):
# Block 1 = layers[0:6], Block 2 = [6:12], ..., Block 5 = [24:30], tail = [30:34]
ORIG_BLOCK_BOUNDS = [(0, 6), (6, 12), (12, 18), (18, 24), (24, 30)]  # 5 blocks

# Insert in REVERSE order so earlier indices don't shift during insertion.
for start, end in reversed(ORIG_BLOCK_BOUNDS):
    # Snapshot the 6 source layers of this block (deep-copy before inserting
    # so both new blocks come from the *pre-insertion* block).
    source_layers = [copy.deepcopy(layers[i]) for i in range(start, end)]

    # Build two fresh identity blocks from this source.
    new_block_a = [copy.deepcopy(l) for l in source_layers]
    new_block_b = [copy.deepcopy(l) for l in source_layers]

    for new_layer in (*new_block_a, *new_block_b):
        new_layer.self_attn.o_proj.weight.data.zero_()
        new_layer.mlp.down_proj.weight.data.zero_()
        # Gemma 3 uses bias=False throughout — nothing else to zero.
        # q_norm, k_norm, four layernorms, Q/K/V/gate/up are left intact.

    # Insert 12 new layers right after `end` (the end of the original block).
    # Because we iterate in reverse, `end` indexes are still valid for each step.
    for offset, layer in enumerate([*new_block_a, *new_block_b]):
        layers.insert(end + offset, layer)

assert len(layers) == 94

# Re-index layer_idx for KV-cache routing.
for new_idx, layer in enumerate(layers):
    layer.self_attn.layer_idx = new_idx

# Update config.
cfg = model.config
# The text config may be nested under cfg.text_config for the VLM.
text_cfg = getattr(cfg, "text_config", cfg)
text_cfg.num_hidden_layers = 94

# Regenerate layer_types if your transformers version stores an explicit list.
# Rule: type of layer i (0-indexed) follows HF's convention — verify empirically
# for your transformers version before committing to a regenerated list. In
# Megatron-Bridge the types are computed at runtime via `layer_number % 6`,
# so no server-side list is needed on the mcore side.
if hasattr(text_cfg, "layer_types"):
    # HF Gemma 3 layer_types: typically repeats
    # ["sliding_attention"] * 5 + ["full_attention"]
    # for each 6-layer cycle. Verify against the original 34-entry list before
    # regenerating for 94 entries.
    base_cycle = text_cfg.layer_types[:6]  # e.g. ['sliding_attention']*5 + ['full_attention']
    text_cfg.layer_types = (base_cycle * ((94 // 6) + 1))[:94]

model.save_pretrained(DST, safe_serialization=True)
processor.save_pretrained(DST)
```

**Notes:**
- Snapshotting `source_layers` via `deepcopy` **before** building `new_block_a`/`new_block_b` guarantees both new blocks are copies of the *pre-insertion* original block, not of each other or of an already-mutated list.
- Reverse-iteration order over the block boundaries keeps indexing valid during insertion.
- `copy.deepcopy` duplicates all submodule weights including the four layernorms, `q_norm`, `k_norm`, and all projections.
- Gemma 3 uses `bias=False` on all linear projections — no bias-zeroing needed. Verify by inspecting `list(model.named_parameters())`.

### Step 2 — Verify identity behavior

Before any training, confirm the surgery is exact:

```python
orig = AutoModelForImageTextToText.from_pretrained(SRC, torch_dtype=torch.bfloat16).eval().cuda()
new  = AutoModelForImageTextToText.from_pretrained(DST, torch_dtype=torch.bfloat16).eval().cuda()

prompt = "The quick brown fox"
ids = processor(text=prompt, return_tensors="pt").input_ids.cuda()

with torch.no_grad():
    lo = orig(input_ids=ids).logits
    ln = new(input_ids=ids).logits

print("max abs logit diff:", (lo - ln).abs().max().item())
# Expected: ~1e-3 or smaller (bf16 roundoff). If larger, a projection wasn't zeroed
# OR the sandwich norm chain broke somewhere (rarely — double-check post_attention_layernorm
# and post_feedforward_layernorm on the source layers).
```

Also run with a long-context prompt (> 1024 tokens) to exercise both sliding and global layers. Logit match must hold for both.

### Step 3 — Do NOT upload to HF Hub (yet)

**Local save is sufficient.** `AutoBridge.from_hf_pretrained(path)` accepts local directories just like HF's `from_pretrained`. Upload to the Hub only if you need to share the upscaled init across machines or publish reproducibility artifacts later.

### Step 4 — Configure training with Megatron-Bridge

```python
from megatron.bridge import AutoBridge

bridge   = AutoBridge.from_hf_pretrained("/local/path/gemma3-4b-upscaled-94L")
provider = bridge.to_megatron_provider()
# provider will have num_layers=94 (read automatically from HF config)
# _is_local_attn_layer uses layer_number % 6, so the sliding/global pattern
# is assigned correctly across all 94 layers without any extra config.
```

The bridge:
- Reads `num_hidden_layers=94` from the HF config → constructs a 94-layer Megatron-Core model.
- Maps weights element-wise via regex (`decoder.layers.*.self_attention...`) — layer count is not hard-coded.
- Preserves zeroed output projections exactly.
- Routes each layer's attention to local-window vs. global based on `layer_number % 6`.

### Step 5 — Launch training "like the original"

Adapt a Gemma 3 recipe under `Megatron-Bridge/src/megatron/bridge/recipes/gemma/gemma3.py`:
- Swap in `Gemma3ModelProvider4B()` or, preferably, let the bridge infer the provider from the upscaled HF checkpoint (so `num_layers=94` propagates automatically).
- Same optimizer, same LR schedule, same data, same batch size, same sequence length as the base-model recipe.
- **PP**: choose from {1, 2, 47, 94} or configure uneven PP if you need 4/8.
- **VPP**: also needs to divide per-rank layer count.
- **Optimizer state**: start fresh. Don't try to reuse optimizer state from the base 34-layer run.

### Step 6 — Post-training export

After training, Megatron-Bridge converts the trained Megatron-Core checkpoint back to HF format, producing a standard 94-layer `Gemma3ForCausalLM` (or `Gemma3ForConditionalGeneration` if you kept vision) loadable by vanilla transformers / vLLM / downstream inference stacks.

---

## Sanity checks before a long training job

1. **Forward-pass logit parity** vs. original Gemma 3 4B on the same prompts (Step 2). Must hold for both short (< 1024 tokens) and long (> 1024 tokens) contexts so both local and global layers are exercised.
2. **First training steps**: initial loss near the original model's eval loss on the same data — no spike. Spike = identity didn't hold somewhere.
3. **Gradient norms on inserted layers**:
   - `o_proj.weight`, `down_proj.weight`: nonzero gradient at step 0 ✓
   - `q/k/v_proj.weight`, `gate_proj.weight`, `up_proj.weight`: ~zero at step 0, nonzero from step 1 ✓
   - Four layernorms + `q_norm`/`k_norm`: nonzero gradient at step 0 (they receive gradient through the residual stream) ✓
4. **Round-trip through the bridge**: save untrained upscaled checkpoint → HF → Megatron → HF → forward pass → logits unchanged. Confirms the bridge doesn't silently alter the zeroed projections or mis-route any layer's attention type.

---

## Files that will live in this repo

Recommended locations:

- Surgery script: `experimental/depth_upscaling/upscale_gemma3_4b.py`
- Identity verification script: colocated with the surgery script
- Training config: under `Megatron-Bridge/src/megatron/bridge/recipes/gemma/` (a small wrapper that points at the upscaled checkpoint) or a ModelChef-side launcher under `veomni-recipes/configs/gemma/`

No changes to Megatron-Bridge, mbridge, or any submodule are required.

---

## Open questions / things to confirm before starting

1. **VLM vs text-only** — determines which HF class, which bridge, and whether `model.language_model.model.layers` vs `model.model.layers`.
2. **Base vs instruction-tuned** source (`gemma-3-4b-pt` vs `gemma-3-4b-it`).
3. **PP target** — 94 is fine for PP ∈ {1, 2, 47}. For PP=4/8, adjust the schedule (e.g., 34 → 64 via one block insert each) or use uneven PP.
4. **Training regime** — pretraining, SFT, PEFT, RLHF? Determines which Megatron-Bridge recipe to start from.
5. **HF `layer_types` regeneration** — verify your installed transformers version stores this as an explicit list; if it does, make sure the regenerated 94-entry list matches the `layer_number % 6` convention end-to-end (surgery script + HF forward pass + Megatron-Bridge).
6. **Exact-zero vs near-zero** output projection init — default exact zero.

---

## TL;DR

1. **Megatron-Bridge already supports Gemma 3 4B** (both text-only and VL). Attention-type assignment uses `layer_number % 6`, so arbitrary layer counts work without code changes.
2. Write a short HF surgery script: for each of the 5 complete 6-layer blocks, deep-copy it twice, zero `o_proj` and `down_proj` in every copied layer, and insert the 12 new layers after the block. **34 → 94 layers.**
3. `save_pretrained` locally. **No HF Hub upload needed.**
4. Verify logit parity vs. base Gemma 3 4B on short AND long prompts.
5. Load with `AutoBridge.from_hf_pretrained(...)` and train with any Gemma 3 recipe, mindful of PP divisibility (94 = 2 × 47; use uneven PP or adjust schedule if PP=4/8).
6. One-step gradient delay on Q/K/V/gate/up is expected and benign; residual path carries gradients to the 34 original layers from step 0.
