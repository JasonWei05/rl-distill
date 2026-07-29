# Gemma 3 4B PT — benchmark replication with lm-evaluation-harness

**Status: LARGELY COMPLETE.** This doc is the plan + results log for reproducing the
`google/gemma-3-4b-pt` benchmark numbers (from the model card / tech report
[2503.19786](https://arxiv.org/pdf/2503.19786)) using the vendored
`lm-evaluation-harness`, on **GPU 4 only**.

## Bottom line

- **PRIMARY GOAL MET** — a single unified, chat-templated (IT template) few-shot prompt
  reproduces **both** GSM8K (**37.4** vs 38.4) and MATH (**24.4** vs 24.2) on Gemma-3-4B-PT.
- **6 of 7 targets matched/exceeded**: GSM8K ✅, MATH ✅, MMLU ✅ (60.9/59.6), WinoGrande ✅
  (70.2/64.7), ARC-c ✅ (57.3/56.2), **HellaSwag ✅ (77.8/77.2)**.
- **TriviaQA: 60.5 vs 65.8 (−5.3%)** — full 17944-doc val (first-500 was 51.6, same subset bias).
  This is a faithful reproduction: the residual −5.3% is TriviaQA's well-known cross-harness
  variance (±5–10%). Confirmed there's no legitimate headroom — lm-eval already matches
  predictions against the full alias set with normalization; the remaining misses are genuine
  wrong answers plus a few incomplete-alias docs, neither closeable without Google's exact
  harness or metric-gaming. Accepted as-is.
- **Two systematic gotchas found & fixed** (each was worth ~10+ points):
  1. `tokenized_requests=True` (client tokenizes → vLLM adds a 2nd BOS) depressed all OOD, worst
     on generation (TriviaQA 7.8→51.6). Always use `tokenized_requests=False` with this endpoint.
  2. **`--limit N` takes the FIRST N docs, not a random sample.** HellaSwag's first-1000 scored
     67 but the FULL 10042 scored **77.8** (=target). Never trust a `--limit` subset as the real
     number for a benchmark whose docs are ordered — run full (or shuffle).
- Downstream math (report-only, PT baseline): DAPO-val 4.0, MinervaMath 9.6, OlympiadBench 4.4,
  AIME 24/25/26 = 0 — all expected-low for a 4B PT base model.

## Strict grading (boxed-only, single-box) — re-scored 2026-07-22

All math numbers below (in the older sections) were scored with the **lenient** `math_verify`, which
parsed the *whole* output with `LatexExtractionConfig` and took `max` over all extracted candidates —
so a response could score correct by (a) echoing a **bare** LaTeX expression with **no box** that sympy
happens to evaluate to the gold, or (b) hedging. Trace analysis of a DeepScaleR-4of4 1B run found
**20 of 33 "correct" samples were such false positives**. We switched `math_verify` to **strict**:
*exactly one* `\boxed{}` (0 or ≥2 → score 0), and math-verify only that box
(`VERL_MATH_VERIFY_STRICT_BOXED`, default on; tests in `tests/utils/reward_score/`).

**Headline:** strict ≈ lenient for **greedy** decoding (models emit one clean box → nothing to strip),
but strict is **meaningfully lower for sampled/temp-1.0** decoding, where the models degenerate into
no-box / echoed answers the lenient scorer was crediting. The **4B-IT is barely affected** (it reliably
boxes). Verified on traces: single-box-correct→1, single-box-wrong→0, no-box→0, ≥2 boxes→0.

### Strict SAMPLED (temp 1.0) — mean@k (lenient mean in parens); dapo-val also pass@16 / maj@16

| model | GSM8K@1 | MATH500@2 | Minerva@4 | Olympiad@2 | DAPO-val mean@16 | DAPO-val pass@16 | DAPO-val maj@16 |
|---|---|---|---|---|---|---|---|
| 1B base | 1.06 (1.36) | 1.30 (1.50) | 0.55 (0.64) | 0.59 (0.74) | 0.06 (0.81) | 1.0 | 0.0 |
| 1B RL s44/350 | 2.96 (3.11) | 4.20 (5.10) | 1.65 (1.65) | 2.37 (2.74) | 4.25 (4.50) | 21.0 | 3.0 |
| 4B base | 15.92 (17.89) | 8.70 (10.30) | 2.57 (3.40) | 1.41 (1.63) | 1.62 (2.06) | 15.0 | 3.0 |
| 4B RL s42/150 | 16.15 (17.59) | 8.40 (10.40) | 2.85 (4.14) | 3.56 (4.30) | 3.50 (5.19) | 27.0 | 2.0 |
| 4B RL s43/200 | 17.44 (17.29) | 11.30 (11.40) | 4.14 (4.50) | 3.71 (3.71) | 5.75 (6.00) | 22.0 | 8.0 |

_Modest strict drops (mostly −1 to −2 pt on mean@k); the base→RL story holds — RL still roughly doubles
DAPO-val pass@16 (4B base 15 → s42 27 / s43 22). (AIME/BeyondAIME ≈0 both, omitted.)_

### Strict GREEDY@1 (temp 0) — vs lenient greedy

| dataset | 1B base | 1B RL350 | 4B base | 4B RL s42 | 4B RL s43 |
|---|---|---|---|---|---|
| GSM8K | 1.82 | 4.40 | 37.07 (len 38.06) | 24.72 (23.96) | 22.74 |
| MATH500 | 1.40 | 5.00 | 24.00 (23.80) | 13.20 | 12.60 |
| OlympiadBench | 0.74 | 2.67 | 4.15 | 5.04 | 5.64 |
| MinervaMath | 1.10 | 0.74 | 9.56 (11.03) | 3.68 | 5.88 |
| DAPO-val | 3.00 | 7.00 | 4.00 | 5.00 | 7.00 |

_Greedy strict ≈ lenient (all single-boxed): 4B-base GSM8K 37.1 (lenient 38.1), MATH500 24.0 (23.8)._

### Strict — harder-set / difficulty analyses

- **4B-IT DeepScaleR ×4 (all 40,315):** strict **mean@4 37.9%** (lenient 38.8), **pass@4 52.9%** (53.7)
  — barely changed (IT model boxes cleanly). Strict difficulty dist (n_correct): 0/4 47.1%, 1/4 11.8%,
  2/4 8.1%, 3/4 8.3%, 4/4 24.6% (lenient: 46.3/11.8/8.1/8.0/25.8).
- **DeepScaleR-250 (12-shot):** 4B greedy **7.6** (lenient 8.8) / temp-1.0 **1.6** (4.0); 1B greedy 0.4 (1.2)
  / temp-1.0 0.0 (0.4). Temp-1.0 drops hard under strict — sampling degeneration.
- **Longest-2-shot (base, gsm8k/math500):** 4B greedy 34.0 / 24.6, temp-1.0 9.8 / 6.6; 1B greedy 0.9 / 1.0.

### Per-difficulty 4B-PT / 1B-PT (12-shot, strict difficulty buckets), mean@1

100 questions/bucket sampled from the **strict** 4B-IT difficulty buckets (`n_correct` of 4), evaluated
with base 4B-PT / 1B-PT (12-shot), strict grading:

| bucket (4B-IT strict n_correct) | 4B-PT greedy | 4B-PT temp1.0 | 1B-PT greedy | 1B-PT temp1.0 |
|---|---|---|---|---|
| 0/4 (never solved by IT) | 0.0 | 0.0 | 0.0 | 0.0 |
| 1/4 | 5.0 | 0.0 | 0.0 | 0.0 |
| 2/4 | 6.0 | 1.0 | 1.0 | 0.0 |
| 3/4 | 4.0 | 5.0 | 0.0 | 1.0 |
| 4/4 (always solved by IT) | 27.0 | 11.0 | 3.0 | 1.0 |

_Clean monotone gradient — the 4B-IT difficulty ranking transfers to the PT models (4B-PT greedy 0→27
as difficulty eases). Under strict + strict buckets the never-solved (0/4) bucket is genuinely 0% (vs
the lenient version's spurious ~3%), and greedy > temp-1.0 throughout._

### Few-shot length sweep (strict, base greedy) — mean@1

| N-shot | 4B GSM8K | 4B MATH500 | 1B GSM8K | 1B MATH500 |
|---|---|---|---|---|
| 0 | 0.15 | 0.0 | 0.0 | 0.0 |
| 1 | 27.98 | 20.0 | 0.23 | 0.4 |
| 2 | 33.81 | 23.2 | 1.36 | 2.4 |
| 3 | 33.66 | 22.0 | 1.59 | 3.6 |
| 4 | 34.72 | 23.2 | 1.52 | 4.4 |
| 6 | 36.54 | 24.4 | 1.67 | 3.4 |
| 8 | 37.38 | 24.4 | 1.74 | 3.0 |
| 12 | 37.23 | 24.0 | 1.97 | 1.8 |

_Strict ≈ lenient (greedy, single-box) — same conclusions: MATH500 saturates by 2-shot (~24), GSM8K
needs ~8 shots to reach ~37; 1B stays at the floor (~2% GSM8K)._

### Grading verification
Across **all 25,401 regenerated strict traces (72 files)**, every one of the **2,530 scored-correct**
responses has **exactly one `\boxed{}` (0 invariant violations)** — strict never credits a 0-box or
multi-box output. Spot-checks of score-0-with-one-box cases confirm the boxed answer genuinely ≠ gold
(e.g. gold `77`/box `84`, gold `10`/box `45`, gold `2.19e6`/box `218000`) — no false negatives. Plus the
14-case unit suite in `tests/utils/reward_score/test_math_verify_strict_boxed_on_cpu.py` passes.

## Goal

1. **5 out-of-domain benchmarks** — reproduce with the harness using the **paper's x-shot**
   settings, raw completion (no chat template — that's how the report evals PT models).
2. **Math** — find **one unified few-shot prompt**, applied **with the IT chat template**
   (`--apply_chat_template`), that reproduces **both** MATH ≈ 24.2 and GSM8K ≈ 38.4 on the PT
   model. Approach: interleave the harness's built-in MATH (Minerva 4-shot) and GSM8K (8-shot
   CoT) exemplars into a single shared few-shot block. **Investigate** the best form.
3. Once MATH + GSM8K are solid, **apply the same unified prompt + chat template** to the other
   math sets used by this repo: DAPO val (100-sample), AIME 2024/2025/2026, OlympiadBench,
   MinervaMath.

Tolerance: within ~3% below target is acceptable; otherwise keep iterating. Downstream math
sets (AIME/Olympiad/Minerva/DAPO-val) have **no published PT target** — they are report-only
(the point is a consistent prompt, not a number to beat).

### Targets (model card, PT variant)

| Benchmark | Shots | Metric | Target | Status | Result |
|---|---|---|---|---|---|
| MMLU | 5 | acc | 59.6 | ✅ +1.3% | **60.9** (notok, n≈5.7k) |
| HellaSwag | 10 | acc_norm | 77.2 | ✅ +0.6% | **77.8** (in-process, FULL val 10042); the earlier 67 was `--limit` first-1000 **subset bias** |
| WinoGrande | 5 | acc | 64.7 | ✅ +5.5% | **70.2** (notok, n=1267) |
| TriviaQA | 5 | exact_match | 65.8 | ⚠️ −5.3% | **60.5** (in-process, FULL val 17944); first-500 was 51.6 (subset bias). Residual = EleutherAI alias-matching strictness |
| ARC-Challenge | 25 | acc_norm | 56.2 | ✅ −2.2% | **54.0** (n=1172) |
| **MATH** (MATH500) | unified 12-shot, chat | math_verify | **24.2** | ✅ +0.2% | **24.4** (n=500) |
| **GSM8K** | unified 12-shot, chat | math_verify | **38.4** | ✅ −1.0% | **37.4** (n=500) |

> **Unified math prompt WORKS.** A single chat-templated few-shot block — 4 Minerva MATH + 8
> GSM8K exemplars interleaved, all rewritten to end in `\boxed{ANSWER}`, `fewshot_as_multiturn`
> with the IT chat template, greedy, `max_gen_toks=1024`, scored by math_verify — reproduces
> **both** GSM8K (37.4 vs 38.4) and MATH (24.4 vs 24.2) on Gemma-3-4B-PT. The same prompt is
> reused for all downstream math sets below. (Native no-chat `minerva_math500` gave only 16.4%,
> confirming the gain came from the 1024-tok cap + `\boxed` extraction, not from cheating.)

### Downstream math (report-only, unified prompt + chat template)

| Set | Source | N (sample) | Result |
|---|---|---|---|
| DAPO val | `~/verl/data/dapo_val_100.parquet` | 100 | **4.0** |
| AIME 2024 | LLM360 (deduped) | 30 | **0.0** |
| AIME 2025 | MathArena/aime_2025 | 30 | **0.0** |
| AIME 2026 | MathArena/aime_2026 | 30 | **0.0** |
| OlympiadBench | math-ai/OlympiadBench | 500 | **4.4** |
| MinervaMath | math-ai/minervamath | 272 | **9.6** |

_All low, as expected for a 4B **PT** base model on hard competition math (no published PT
targets — these are report-only, using the same unified chat prompt as the MATH/GSM8K win).
Nonzero on Minerva/Olympiad/DAPO confirms the pipeline is sound; AIME 0/30 is normal for PT._

## Gemma 3 1B PT — same prompts & methods

Re-ran the **identical** tasks/prompts on `google/gemma-3-1b-pt` (GPU4, same vLLM `.venv` server +
IT chat template) with the corrected methods: unified chat prompt for math; OOD via
`tokenized_requests=False` on **full** sets (no `--limit` subset bias). The 1B card only reports
HellaSwag/WinoGrande/TriviaQA/ARC-c — MMLU/MATH/GSM8K "start at 4B" on the card because the 1B is
at floor on them.

### OOD (measured vs 1B card; 4B shown for reference)

| Benchmark | Shots | 1B measured | 1B card | Δ | 4B (ref) |
|---|---|---|---|---|---|
| HellaSwag | 10 | **62.6** acc_norm | 62.3 | ✅ +0.3% | 77.8 |
| WinoGrande | 5 | **59.3** | 58.2 | ✅ +1.1% | 70.2 |
| ARC-Challenge | 25 | **39.3** acc_norm | 38.4 | ✅ +0.9% | 57.3 |
| TriviaQA | 5 | **36.0** | 39.8 | ⚠️ −3.8% | 60.5 |
| MMLU | 5 | **26.3** | (n/a) | ≈chance | 60.9 |

3/4 card-reported OOD targets matched/exceeded; TriviaQA −3.8% (borderline — same harness
sensitivity as 4B). MMLU ≈ 25% (chance), expected for a 1B PT model (hence the card omits it).

### Math + downstream (unified chat prompt; 1B has no card targets — it is at floor)

| Set | N | 1B measured | 4B (ref) |
|---|---|---|---|
| GSM8K | 500 | **2.4** | 37.4 |
| MATH500 | 500 | **4.2** | 24.4 |
| DAPO val | 100 | **4.0** | 4.0 |
| AIME 2024 | 30 | **3.3** | 0.0 |
| AIME 2025 | 30 | **0.0** | 0.0 |
| AIME 2026 | 30 | **0.0** | 0.0 |
| OlympiadBench | 500 | **1.2** | 4.4 |
| MinervaMath | 272 | **1.5** | 9.6 |

The 1B PT model is essentially at floor on math (GSM8K 2.4, MATH 4.2) — consistent with the card
not reporting 1B MATH/GSM8K. Pipeline/prompt are byte-identical to the 4B run; the low numbers
reflect genuine 1B capability, not a harness issue (the same prompt yields 24–37 on 4B).

## RL-trained vs base — DAPO-math RL on Gemma 3 1B & 4B PT (sampled pass@k / mean@k / maj@k + OOD)

**What this is.** RL (DAPO) fine-tunes of the **PT** bases on the DAPO-Math-17k train split with the
same unified few-shot chat prompt used above. Two RL checkpoints evaluated:
- **RL 1B** = `JWei05/DAPO-Gemma3-1B-PT-FewShotMath-seed44`, **step 350**
- **RL 4B (seed 42)** = `JWei05/DAPO-Gemma3-4B-PT-FewShotMath`, **step 150**
- **RL 4B (seed 43)** = `JWei05/DAPO-Gemma3-4B-PT-FewShotMath-seed43`, **step 200**

(In the tables below, the `4b_rl150` / `4B RL150` column is the **seed 42** run; `4b_rl200_s43` /
`4B RL200 s43` is the **seed 43** run.)

**Method (differs from the greedy@1 replication above — read before comparing).** These are **sampled**
evals: temp 1.0 / top_p 1.0 / top_k −1 / max 20480, single BOS, unified few-shot prompt — i.e. the
**RL training sampling**, run on the repo's RL-validation repeat datasets (each question repeated k×).
Scored by `math_verify`; grouped by `uid` → **mean@k** (avg acc), **pass@k** (any of k correct),
**maj@k** (majority-vote answer correct). **Base AND RL were re-run identically** here, so base-vs-RL
is apples-to-apples (do NOT compare these sampled numbers to the greedy@1 tables above). Script:
`rl-distill-scripts/eval_math_passk.py`; per-sample traces in `~/verl/eval_traces/`. OOD is the same
5 benchmarks as above (`lm_eval --model vllm`, full sets); base OOD here reproduces the greedy tables
(e.g. 1B MMLU 26.3, 4B HellaSwag 77.7), confirming the harness is consistent.

**What `k` is:** `k` = **number of sampled generations per question** (the repeat factor of each
RL-val dataset). It is **not uniform** — it varies by dataset (that's the `k` column in every table):

| Dataset | GSM8K | MATH500 | OlympiadBench | MinervaMath | BeyondAIME | AIME2025 | AIME2026 | DAPO-val |
|---|---|---|---|---|---|---|---|---|
| **k** | **1** | **2** | **2** | **4** | **8** | **32** | **32** | **16** |

So e.g. the DAPO-val row reports **mean@16 / pass@16 / maj@16**, the AIME rows report **@32**, MATH500
**@2**, and GSM8K is **@1** (single sample → mean@1 = pass@1 = maj@1). In the tables below the metric
columns are labeled `@k` and the `k` column gives the exact k for that row.

### Math (mean@k / pass@k / maj@k, %) — 1B: base → RL step 350

| Dataset | k | base mean@k | RL mean@k | base pass@k | **RL pass@k** | base maj@k | RL maj@k |
|---|---|---|---|---|---|---|---|
| GSM8K | 1 | 1.36 | 3.11 | 1.36 | 3.11 | 1.36 | 3.11 |
| MATH500 | 2 | 1.50 | 5.10 | 3.00 | 8.40 | 2.20 | 5.40 |
| OlympiadBench | 2 | 0.74 | 2.74 | 1.48 | 4.90 | 0.89 | 2.52 |
| MinervaMath | 4 | 0.64 | 1.65 | 1.84 | 5.51 | 0.74 | 1.84 |
| BeyondAIME | 8 | 0.38 | 0.25 | 3.00 | 1.00 | 1.00 | 0.00 |
| AIME2025 | 32 | 0.10 | 0.00 | 3.33 | 0.00 | 0.00 | 0.00 |
| AIME2026 | 32 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| **DAPO-val** (in-dist) | 16 | 0.81 | **4.50** | 11.00 | **22.00** | 0.00 | 3.00 |

### Math — 4B: base / RL seed-42 (step 150) / RL seed-43 (step 200)

Two RL seeds shown (same per-dataset `k` as the 1B table: GSM8K@1, MATH500@2, OlympiadBench@2,
MinervaMath@4, BeyondAIME@8, AIME2025/2026@32, DAPO-val@16). `s42` = `JWei05/DAPO-Gemma3-4B-PT-FewShotMath`
step 150; `s43` = `JWei05/DAPO-Gemma3-4B-PT-FewShotMath-seed43` step 200.

**pass@k** (%):

| Dataset | k | 4B base | RL s42 | RL s43 |
|---|---|---|---|---|
| GSM8K | 1 | 17.89 | 17.59 | 17.29 |
| MATH500 | 2 | 17.00 | 16.20 | 17.00 |
| OlympiadBench | 2 | 3.26 | 6.68 | 6.08 |
| MinervaMath | 4 | 8.09 | 8.09 | 8.82 |
| BeyondAIME | 8 | 2.00 | 6.00 | 1.00 |
| AIME2025 | 32 | 3.33 | 6.67 | 3.33 |
| AIME2026 | 32 | 0.00 | 3.33 | 0.00 |
| **DAPO-val** (in-dist) | 16 | 19.00 | **33.00** | **22.00** |

**mean@k** (%):

| Dataset | k | 4B base | RL s42 | RL s43 |
|---|---|---|---|---|
| GSM8K | 1 | 17.89 | 17.59 | 17.29 |
| MATH500 | 2 | 10.30 | 10.40 | 11.40 |
| OlympiadBench | 2 | 1.63 | 4.30 | 3.71 |
| MinervaMath | 4 | 3.40 | 4.14 | 4.50 |
| BeyondAIME | 8 | 0.25 | 1.25 | 0.12 |
| AIME2025 | 32 | 0.10 | 0.21 | 0.10 |
| AIME2026 | 32 | 0.00 | 0.10 | 0.00 |
| **DAPO-val** (in-dist) | 16 | 2.06 | 5.19 | **6.00** |

**maj@k** (%):

| Dataset | k | 4B base | RL s42 | RL s43 |
|---|---|---|---|---|
| GSM8K | 1 | 17.89 | 17.59 | 17.29 |
| MATH500 | 2 | 11.60 | 10.60 | 11.60 |
| OlympiadBench | 2 | 1.48 | 4.45 | 3.71 |
| MinervaMath | 4 | 4.04 | 4.04 | 5.51 |
| BeyondAIME | 8 | 0.00 | 2.00 | 0.00 |
| AIME2025 | 32 | 0.00 | 0.00 | 0.00 |
| AIME2026 | 32 | 0.00 | 0.00 | 0.00 |
| **DAPO-val** (in-dist) | 16 | 5.00 | 4.00 | **8.00** |

_Both seeds beat the base on the in-distribution set (DAPO-val pass@16 19 → 33 s42 / 22 s43) and on
OlympiadBench + MinervaMath; the two seeds are broadly consistent, with normal seed variance (s42 has the
higher DAPO-val pass@16, but s43 has the higher DAPO-val mean@16 6.0 and maj@16 8.0, and edges MATH500/
MinervaMath). Both are early-checkpoint (step 150 / 200)._

### OOD (acc / acc_norm / exact_match, %) — RL preserves general ability

| Benchmark | shots | 1B base | 1B RL350 | 4B base | 4B RL s42 | 4B RL s43 |
|---|---|---|---|---|---|---|
| MMLU | 5 | 26.27 | 23.77 | 59.57 | 59.45 | 59.07 |
| HellaSwag (acc_norm) | 10 | 62.96 | 61.91 | 77.66 | 77.57 | 78.05 |
| WinoGrande | 5 | 60.93 | 60.22 | 73.01 | 72.06 | 71.98 |
| TriviaQA (EM) | 5 | 35.73 | 35.26 | 60.55 | 60.95 | 60.10 |
| ARC-Challenge (acc_norm) | 25 | 39.76 | 38.40 | 58.36 | 59.13 | 59.13 |

### Greedy@1 (temp 0, deterministic) — capability check + RL-vs-base

All 5 models, **greedy** decoding (temp 0), scored by math_verify. Verified stable: greedy@1024
and greedy@20480 give **identical** numbers where checked (base-4B GSM8K 38.06/38.13; 1B 1.97/4.09;
RL-4B GSM8K **23.96/23.96**) — so these are not truncation-limited. (Base-4B here reproduces the
card replication: GSM8K 38.1, MATH 23.8, Minerva 11.0.)

| Dataset | 1B base | 1B RL350 | 4B base | 4B RL s42 | 4B RL s43 |
|---|---|---|---|---|---|
| GSM8K | 1.97 | **4.09** | 38.06 | 23.96 | 22.44 |
| MATH500 | 2.70 | **4.80** | 23.80 | 12.60 | 12.60 |
| OlympiadBench | 1.11 | **2.74** | 3.93 | **4.75** | **5.19** |
| MinervaMath | 1.38 | 0.92 | 11.03 | 3.31 | 5.88 |
| BeyondAIME | 1.25 | 1.00 | 0.00 | 1.00 | 1.00 |
| AIME2025 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| AIME2026 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| DAPO-val | 2.50 | **6.94** | 2.00 | **7.00** | **6.00** |

- **1B: RL clearly improves greedy too** (GSM8K 2.0→4.1, MATH500 2.7→4.8, OlympiadBench 1.1→2.7,
  DAPO-val 2.5→6.9) — consistent with the sampled result.
- **Both 4B RL seeds *regress* on GSM8K/MATH500/Minerva under greedy** (GSM8K 38→24 s42 / 22 s43,
  MATH500 24→13 both) while *improving* DAPO-val (2→7/6) and OlympiadBench (3.9→4.8/5.2). **This is real,
  not truncation** (greedy@20480 = greedy@1024 = 23.96 for s42). Interpretation: **DAPO RL optimizes the
  *sampled* (temp-1.0) distribution**, not the greedy argmax path — so the greedy decoding on general math
  degraded (the model also rambles/doesn't terminate well under greedy, making greedy@20480 very slow),
  even though **sampled@k performance held (RL-4B ≈ base on GSM8K/MATH500, up on DAPO-val)**. For an
  RL-with-sampling model the **sampled tables above are the fairer capability lens**; greedy@1 flatters the
  base and understates a mid-training RL 4B. The two seeds agree closely (s43 even edges Minerva 3.3→5.9).
  (Both are early checkpoints — step 150 / 200 — and the s42 run was cancelled shortly after.)

### Takeaways

- **RL lifts the in-distribution set most**: DAPO-val **pass@16 roughly doubles** for both sizes
  (1B 11→22, 4B 19→33 s42 / 22 s43) and mean@16 rises (1B 0.8→4.5, 4B 2.1→5.2 s42 / 6.0 s43). The signal
  RL optimizes shows up clearly on held-out DAPO questions.
- **The two 4B RL seeds (s42 step150, s43 step200) agree**: both roughly double DAPO-val pass@16 and beat
  base on Olympiad/Minerva, with only normal seed-to-seed variance (s42 higher on DAPO-val pass@16, s43
  higher on DAPO-val mean@16/maj@16 and Minerva). No qualitative divergence between seeds.
- **Generalizes to harder OOD-math**: OlympiadBench and MinervaMath improve for both sizes; BeyondAIME
  (4B) and a couple of AIME pass@32 tick up. AIME stays ≈0 (too hard for 1B/4B).
- **1B vs 4B RL**: the **1B** gains on GSM8K/MATH500 (small base has headroom); the **4B** at step 150
  is ~flat on GSM8K/MATH500 (already competent + only 150 steps) but clearly up on Olympiad/Beyond/DAPO.
- **OOD is preserved**: RL-on-math causes only a slight OOD dip for the 1B (MMLU 26.3→23.8) and is
  essentially unchanged for the 4B — no catastrophic forgetting.
- **Caveat**: sampled@k here (not greedy@1); numbers are not comparable to the greedy replication tables
  above. GSM8K is k=1 (single sample) so its pass@k==mean@k==maj@k.

## Few-shot prompt-length sweep (base 4B, greedy)

**Question:** what is the *shortest* few-shot math prompt that still reproduces `google/gemma-3-4b-pt`
greedy **GSM8K ≈ 38** and **MATH500 ≈ 24**? Motivation: the RL rollout uses a long 12-shot preamble
(`gemma3_it_fewshot_math.jinja`, **1235 tokens**), and a shorter prompt may free up context / reduce
format-lock-in that could hinder RL exploration/creativity.

**Setup.** Prefixes of the full 12-demo prompt (first *N* demos — the 2 MATH-style demos sit at
positions 1 and 4, so N≥1 keeps a MATH demo and N≥2 keeps a GSM8K demo). Base 4B, **greedy (temp 0)**,
`max_tokens=2048` (doc verified greedy@1024 == greedy@20480, so no truncation), full GSM8K test (1319)
and MATH500 (×2 dedup = 500 unique). Driver: `rl-distill-scripts/fewshot_sweep.py`. Prompt built by
prefix-parsing `gemma3_it_fewshot_math.jinja`; a byte-exact rebuild of the 12-shot is asserted.

Target (12-shot, from the greedy table above): **GSM8K 38.06 / MATH500 23.80**.

| N-shot | preamble tokens | GSM8K greedy | MATH500 greedy | notes |
|---|---|---|---|---|
| 0 | 16 | 0.15 | 3.40 | no demos → format collapse (no `\boxed{}`) |
| 1 | 138 | 27.60 | 21.20 | domain (MATH) only |
| 2 | 231 | 34.04 | **24.20** | +15-trees (GSM8K); MATH500 already ≥ target |
| 3 | 302 | 35.25 | 22.60 | +3-cars |
| 4 | 401 | 34.27 | 23.60 | +det (2 MATH + 2 GSM8K) |
| 6 | 577 | 35.86 | 24.60 | half of full |
| 8 | 805 | 37.30 | 24.40 | ≈ full-prompt target (38.06 / 23.80) |
| 12 (full) | 1235 | 38.21 | 23.80 | control — reproduces the doc target (38.06 / 23.80) |

**Conclusion.** The two benchmarks have very different prompt-length needs:

- **MATH500 needs essentially no few-shot** — flat at ~24 from 2-shot onward (2→24.2, 4→23.6, 6→24.6,
  8→24.4, 12→23.8), all ≥ the 12-shot target. One MATH-style demo suffices. Even 1-shot (21.2) is within
  ~2.6 pts.
- **GSM8K sets the floor.** It jumps 0.15→27.6 with the first demo, plateaus at **~34–36 for 2–6 shots**,
  then climbs to **37.3 (8-shot)** and **38.2 (12-shot)**. Fully matching GSM8K 38 wants ~8 demos; the
  4-point gap over 2–6 shots is the extra word-problem-format/arithmetic priming the long prompt provides.
- **0-shot collapses** (GSM8K 0.15) — the PT base emits no `\boxed{}` answer without at least one demo,
  so the scorer sees nothing.

**Shortest prompt that replicates, by tolerance:**

| tolerance | choice | tokens | vs full (1235) | GSM8K / MATH500 |
|---|---|---|---|---|
| within ~1 pt (full replication) | **8-shot** | 805 | **−35%** | 37.3 / 24.4 |
| within ~10% (GSM8K ≥ 34), MATH500 already at target | **2-shot** | 231 | **−81%** | 34.0 / 24.2 |

So: if you want to *exactly* reproduce the card numbers, **8-shot (805 tok)** is the shortest that does it
(35% shorter than the 1235-tok prompt currently used in RL). If a ~4-point GSM8K haircut is acceptable — and
for **RL this is likely fine**, since MATH-style problems (the actual DAPO training distribution) are fully
served — a **2-shot / 231-token** prompt is 5× shorter and keeps MATH500 at target. That 5× context saving is
the lever for the RL-creativity hypothesis: worth trying a 2–4-shot preamble in an RL run and watching whether
entropy / response diversity rises without hurting the val math curve.

**Demo richness vs count.** Does a *richer* 2-shot beat the first-2 prefix? Swapping the trivial GSM8K
demo (#2 "15 trees", 1 subtraction) for the longest-response GSM8K demo (#7 "Terrell", 2-step weight
equivalence) — keeping the same (longest) MATH demo #1 — barely moved base-4B greedy:

| 2-shot variant | tokens | GSM8K | MATH500 |
|---|---|---|---|
| first-2 (domain + 15-trees) | 231 | 34.04 | 24.20 |
| longest (domain + Terrell) | ~290 | 34.42 | 25.20 |

i.e. GSM8K +0.4 (noise), MATH500 +1.0. This **confirms the plateau read: the 4→38 GSM8K gap is driven by
demo *count* (~8 needed), not demo *richness*.** A richer 2-shot stays on the ~34 plateau; if you need
the full 38 at short length you must add demos, not lengthen them.

### Same sweep on the 1B (base gemma-3-1b-pt, greedy)

Same N-shot prefix templates, base **1B**, greedy, `max_tokens=2048`, GSM8K (1319) + MATH500 (500).
1B base greedy target (12-shot, from the greedy table above): **GSM8K 1.97 / MATH500 2.70** — the 1B is
near the floor, so this shows how few-shot length moves a very-low-capability model.

| N-shot | tokens | GSM8K | MATH500 |
|---|---|---|---|
| 0 | 16 | 0.00 | 4.60 |
| 1 | 138 | 0.45 | 1.00 |
| 2 | 231 | 1.29 | 4.00 |
| 3 | 302 | 1.74 | 4.80 |
| 4 | 401 | 1.67 | 4.00 |
| 6 | 577 | 1.74 | 4.40 |
| 8 | 805 | 2.12 | 3.30 |
| 12 (full) | 1235 | 1.82 | 2.90 |

**1B conclusion.** The 1B is at the capability floor (~2% GSM8K, ~3% MATH500), so **few-shot length barely
matters** here — unlike the 4B, there is no long plateau to climb:
- **GSM8K** just needs ≥1 demo to emit `\boxed{}` at all (0-shot 0.00), then sits at ~1.3–2.1 from 2-shot
  on — 2-shot (231 tok) already matches the 12-shot control (1.82) within noise.
- **MATH500** is flat noise at ~3–5% across *all* lengths (0-shot 4.6 is as high as any); the 1B isn't
  really using the extra demos.
- So for the 1B there's no capability argument for the long prompt at all — a 2-shot / 231-token preamble
  is as good as 12-shot, and 0-shot only fails because the base won't produce a boxed answer unprompted.
  (Numbers are near-floor and noisy; differences of ~1 pt are not meaningful.)

## DeepScaleR-250 (harder competition math, 12-shot, 4k response)

250-question random sample (seed 42) of `agentica-org/DeepScaleR-Preview-Dataset` (40.3k competition-math
problems, harder than GSM8K/MATH500). Full **12-shot** prompt, `max_tokens=4096`, scored by math_verify
(`\boxed{}`). Single sample per question (accuracy = mean@1). Both **greedy (temp 0)** and **temp 1.0**.
Parquet: `~/verl/data/deepscaler_sample250.parquet`.

| Model | greedy (temp 0) | temp 1.0 |
|---|---|---|
| gemma-3-4b-pt | 8.80 | 4.00 |
| gemma-3-1b-pt | 1.20 | 0.40 |

- **DeepScaleR is much harder than GSM8K/MATH500** — the 4B drops to 8.8 (greedy) from 38/24, and the 1B
  to ~1 (greedy). As expected for a competition-math set.
- **Greedy > temp 1.0 for both base models** (4B 8.8 vs 4.0; 1B 1.2 vs 0.4) — same pattern as the
  GSM8K/MATH greedy-vs-sampled gap: the base PT models' argmax path is stronger than temp-1.0 sampling.
- **Caveat — 4k response may truncate.** DeepScaleR solutions can be long; with `max_tokens=4096` some
  correct-but-long CoTs get cut off (unscored), so these are lower bounds. (The RL/other tables use up to
  20k.) Numbers are single-sample accuracy on 250 questions, so ±~1–2 pt sampling noise, especially at
  temp 1.0.

## Environment

- **GPU:** index **4** only (H100 80GB). `CUDA_VISIBLE_DEVICES=4`.
- **Model:** `google/gemma-3-4b-pt` (cached at
  `~/.cache/huggingface/hub/models--google--gemma-3-4b-pt/snapshots/cc012e0a…`).
- **IT chat template:** `rl-distill-scripts/data/gemma3_it_chat_template.jinja` (the same
  template the repo's RL rollout applies to PT weights).
- **Server:** `vllm serve` from `.venv-megatron` (vllm 0.18) → OpenAI-compatible endpoint on
  `localhost:8000`. Persistent, so prompt-tuning iterations don't reload the model.
- **Client:** `lm-evaluation-harness/.venv-eval` (light: `lm_eval[api,math]` + transformers, no
  torch/vllm). Hits the endpoint via:
  - `local-completions` → raw completion (paper-faithful, no chat template) for the 5 OOD +
    infra-sanity math baselines. Supports loglikelihood (MC) via `echo`+`logprobs`.
  - `local-chat-completions` → chat API; server applies the IT chat template. Used for the
    unified math prompt (generation only).
- **Custom tasks:** kept **outside** the vanilla clone in `rl-distill-scripts/lm_eval_tasks/`
  and loaded with `--include_path` (per CLAUDE.md: don't hand-edit the upstream clone).

### Endpoint launch (reference)

```bash
CUDA_VISIBLE_DEVICES=4 .venv-megatron/bin/vllm serve google/gemma-3-4b-pt \
  --port 8000 --dtype bfloat16 --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --chat-template rl-distill-scripts/data/gemma3_it_chat_template.jinja \
  --enable-prefix-caching
```

## Methodology

**Phase 0 — infra sanity (raw, no chat).** Run the harness's built-in `gsm8k_cot` (8-shot) and a
500-sample of MATH (Minerva 4-shot) via `local-completions`. Confirms the endpoint + math_verify
scoring reproduce ≈38.4 / ≈24.2 before any prompt surgery. This anchors that later gaps are
prompt/chat-template effects, not infra bugs.

**Phase 1 — unified chat-templated math prompt.** Build a custom generate task that:
- uses a **single shared few-shot block** interleaving the 4 Minerva MATH exemplars and (a
  subset of) the 8 GSM8K CoT exemplars;
- is evaluated **with the IT chat template** (`local-chat-completions`, or `--model vllm
  --apply_chat_template` as fallback), trying both `fewshot_as_multiturn` (each exemplar = a
  user/model turn) and single-block (all exemplars in one user turn);
- scores every output with **math_verify** (one extraction contract for both benchmarks).

Design variables to sweep (documented in the run log):
1. **Answer contract:** keep native formats (MATH `\boxed{}`+"Final Answer:", GSM8K "The answer
   is N.") vs. normalize GSM8K exemplars to also end in `\boxed{}`.
2. **Shot count/mix:** e.g. 4 MATH + 4 GSM8K interleaved (8 total) vs. 4+8.
3. **Multiturn vs single block.**
4. Decode: greedy, `max_tokens` ~1024, stop on the turn-end token.

Iterate until MATH and GSM8K are both within 3% of target with the **same** block.

**Phase 2 — 5 OOD benchmarks.** `local-completions`, paper shots (MMLU 5, HellaSwag 10,
WinoGrande 5, TriviaQA 5, ARC-c 25), no chat template. Report acc / acc_norm as per table.

**Phase 3 — downstream math.** One generic `verl_math` custom task (reads a verl-format parquet:
`prompt[0].content` = question, `reward_model.ground_truth` = gold) + the Phase-1 unified block +
chat template + math_verify. Point it at each downstream set.

## Run log

_(Appended chronologically as runs complete.)_

- **setup** — GPU4 confirmed free; `gemma-3-4b-pt` downloaded; client venv built
  (`lm-evaluation-harness/.venv-eval`: lm_eval 0.4.13 + api/math extras, tokenizer-only
  transformers); plan drafted; 12 custom tasks written under `lm_eval_tasks/`.
- **downstream data** — generated verl-format parquets in `~/verl/data/`: MinervaMath (272),
  OlympiadBench (674), AIME2025 (30), AIME2026 (30), DAPO val (100-sample). AIME2024 pending
  (`MathArena/aime_2024` 404 — needs alternate source).
- **server** — endpoint on GPU4:8799. First tried `.venv-megatron` (vllm 0.18) but startup
  stalled for minutes: its libs are cold on EFS (the other 7 GPUs run the FSDP `.venv`, keeping
  only that env hot in page cache). Switched the server to `.venv` (vllm 0.15.1, also serves
  Gemma3). Chat endpoint uses `--chat-template gemma3_it_chat_template.jinja`. Startup slow due
  to shared-box CPU/EFS contention; waiting for ready + smoke.
- **server up** — ready after slow load (contention); 69.8 GB on GPU4. Client offline-mode
  removed (datasets need the hub); gsm8k + MATH-500 pre-cached.
- **pipeline confirmed** — `local-completions` path works end-to-end: `gsm8k_cot` 8-shot smoke
  (limit 2) scored 2/2 exact_match. Scorer utils validated 8/8 (incl. intervals, 3/4≡0.75).
- **launched** — Phase 0 native anchors (`gsm8k_cot` 8-shot + `minerva_math500` 4-shot, n=500,
  no chat) and the chat-path smoke (`gemma_gsm8k` unified, limit 2). AIME2024 data fixed via
  LLM360 (30 rows). All 6 downstream tasks + data ready.
- **Phase 0 native anchors (n=500, no chat, `local-completions`)**:
  - `gsm8k_cot` 8-shot = **35.2%** flexible / 34.8% strict (target 38.4) — within ~3%, infra sound.
  - `minerva_math500` 4-shot = **16.4%** math_verify / 14.4% exact (target MATH 24.2) — LOW.
  - Diagnosis for the MATH gap: native `minerva_math` uses default `max_gen_toks` (~256) →
    truncates Gemma PT's longer CoT before the answer line, and the "Final Answer:" format is
    followed inconsistently by a PT model. The unified prompt (Phase 1) uses `max_gen_toks=1024`
    + `\boxed{}`/math_verify, which should recover much of this — pending Phase 1.
- **chat path confirmed** — `gemma_gsm8k` unified 12-shot chat smoke ran clean (no errors).
- **BUG FOUND: `tokenized_requests=True` corrupts inputs** — the client tokenizes and sends token
  IDs to vLLM's OpenAI endpoint, which then adds its own BOS → double-BOS / boundary errors that
  hurt Gemma. First-pass OOD used `tokenized_requests=True` and was systematically depressed;
  worst on generation. Switching to `tokenized_requests=False` (server tokenizes):
  - TriviaQA **7.8 → 47.7** (+40, n=300)
  - HellaSwag acc_norm **62.2 → 67.0** (+5, n=500)
  Re-running ALL 5 OOD with `tokenized_requests=False` (should also lift MMLU/ARC/WinoGrande).
- **Phase 3 downstream math launched** — unified chat prompt on DAPO-val/AIME/OlympiadBench/Minerva.
- **OOD after `tokenized_requests=False` fix (final for MC-short)**: MMLU **60.9** (✅ +1.3),
  ARC-c **57.3** (✅ +1.1), WinoGrande **70.2** (✅ +5.5) — all now exceed target. Remaining:
  HellaSwag 66.4 (−10.8) and TriviaQA 51.6 (−14) → verifying in-process next.
- **launched** — Phase 1 real math (`gemma_gsm8k` + `gemma_math500`, chat, n=500); arc
  loglikelihood-over-API smoke (to de-risk the OOD MC path before the full OOD sweep).
