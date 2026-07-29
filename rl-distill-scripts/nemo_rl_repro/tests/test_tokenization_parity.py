#!/usr/bin/env python3
"""CPU tokenization parity gate: verl reference vs NeMo-RL prompt pipeline.

For 5 questions from the strict-4/4 train parquet, compare
  (a) the verl-side reference — AutoTokenizer('google/gemma-4-E2B')
      .apply_chat_template([user], chat_template=<12-shot jinja>, tokenize=False,
      add_generation_prompt=True), then encode with add_special_tokens=False —
  (b) the token ids NeMo-RL's pipeline produces: get_tokenizer() with
      policy.tokenizer.chat_template=<jinja path> feeding math_hf_data_processor
      (prompt_file=null). If nemo_rl is not importable in this venv, its
      3-line processor path is replicated exactly and marked as such.

Both must be token-identical, with exactly one BOS, and prompt length < 4096.

Run (CPU, no GPU; HF_TOKEN for the gated model comes from the repo .env):
    /mnt/efs/jasonwei/rl-distill/.venv-gemma4/bin/python \
        rl-distill-scripts/nemo_rl_repro/tests/test_tokenization_parity.py
"""

import os
import sys

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPRO_DIR = os.path.dirname(TESTS_DIR)
REPO_ROOT = os.path.dirname(os.path.dirname(REPRO_DIR))
NEMO_RL_ROOT = os.path.join(REPO_ROOT, "third_party", "nemo-rl")
JINJA_PATH = os.path.join(
    REPO_ROOT, "rl-distill-scripts", "data", "gemma3_it_fewshot_math.jinja"
)
TRAIN_PARQUET = "/mnt/efs/jasonwei/verl/data/deepscaler_4of4strict_rl_train.parquet"
MODEL = "google/gemma-4-E2B"
NUM_QUESTIONS = 5
MAX_INPUT_SEQ_LENGTH = 4096


def load_dotenv() -> None:
    """Export HF_TOKEN etc. from the repo-root .env (untracked)."""
    env_path = os.path.join(REPO_ROOT, ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def main() -> int:
    load_dotenv()

    import pandas as pd
    from transformers import AutoTokenizer

    df = pd.read_parquet(TRAIN_PARQUET).head(NUM_QUESTIONS)
    questions = [row["prompt"][0]["content"] for _, row in df.iterrows()]
    ground_truths = [row["reward_model"]["ground_truth"] for _, row in df.iterrows()]
    with open(JINJA_PATH) as f:
        template = f.read()

    # --- (a) verl-side reference ---
    tok_ref = AutoTokenizer.from_pretrained(MODEL)
    ref_texts = [
        tok_ref.apply_chat_template(
            [{"role": "user", "content": q}],
            chat_template=template,
            tokenize=False,
            add_generation_prompt=True,
        )
        for q in questions
    ]
    ref_ids = [
        tok_ref(text, add_special_tokens=False)["input_ids"] for text in ref_texts
    ]

    # --- (b) NeMo-RL pipeline ---
    sys.path.insert(0, NEMO_RL_ROOT)
    try:
        from nemo_rl.algorithms.utils import get_tokenizer
        from nemo_rl.data.interfaces import TaskDataSpec
        from nemo_rl.data.processors import math_hf_data_processor

        nemo_importable = True
        print("Using the real nemo_rl math_hf_data_processor pipeline")
    except Exception as e:  # heavy import chain may be missing deps in this venv
        nemo_importable = False
        print(
            f"REPLICATED PATH: nemo_rl not importable here ({type(e).__name__}: {e});\n"
            "  replicating get_tokenizer's .jinja branch + math_hf_data_processor's "
            "apply_chat_template/encode lines exactly (nemo_rl/data/processors.py "
            "math_hf_data_processor)."
        )

    nemo_texts = []
    nemo_ids = []
    if nemo_importable:
        tok = get_tokenizer({"name": MODEL, "chat_template": JINJA_PATH})
        task_spec = TaskDataSpec(
            task_name="deepscaler_strict", prompt_file=None, system_prompt_file=None
        )
        for i, (q, gt) in enumerate(zip(questions, ground_truths)):
            datum = {
                "messages": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": gt},
                ],
                "task_name": "deepscaler_strict",
            }
            out = math_hf_data_processor(datum, task_spec, tok, MAX_INPUT_SEQ_LENGTH, i)
            nemo_texts.append(out["message_log"][0]["content"])
            nemo_ids.append(out["message_log"][0]["token_ids"].tolist())
            assert out["loss_multiplier"] == 1.0, (
                f"question {i} was masked (prompt >= {MAX_INPUT_SEQ_LENGTH} tokens)"
            )
    else:
        # Exact replication of get_tokenizer (algorithms/utils.py, .jinja branch)
        # + math_hf_data_processor (data/processors.py) with prompt_file=null:
        tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.chat_template = template
        for q in questions:
            message = tok.apply_chat_template(
                [{"role": "user", "content": q}],
                tokenize=False,
                add_generation_prompt=True,
                add_special_tokens=False,
            )
            token_ids = tok(message, return_tensors="pt", add_special_tokens=False)[
                "input_ids"
            ][0]
            nemo_texts.append(message)
            nemo_ids.append(token_ids.tolist())

    # --- compare ---
    bos_id = tok_ref.bos_token_id
    failures = 0
    for i, q in enumerate(questions):
        ok = True
        if ref_texts[i] != nemo_texts[i]:
            ok = False
            print(f"[FAIL] question {i}: rendered strings differ")
        if ref_ids[i] != nemo_ids[i]:
            ok = False
            print(
                f"[FAIL] question {i}: token ids differ "
                f"(ref {len(ref_ids[i])} toks vs nemo {len(nemo_ids[i])} toks)"
            )
        n_bos = sum(1 for t in nemo_ids[i] if t == bos_id)
        if not (nemo_ids[i][0] == bos_id and n_bos == 1):
            ok = False
            print(f"[FAIL] question {i}: expected exactly one leading BOS, got {n_bos}")
        if len(nemo_ids[i]) >= MAX_INPUT_SEQ_LENGTH:
            ok = False
            print(f"[FAIL] question {i}: prompt length {len(nemo_ids[i])} >= 4096")
        if ok:
            print(
                f"[PASS] question {i}: {len(nemo_ids[i])} tokens, byte- and "
                f"token-identical, single BOS ({q[:60]!r}...)"
            )
        else:
            failures += 1

    if failures:
        print(f"\nTOKENIZATION PARITY: FAIL ({failures}/{len(questions)} mismatched)")
        return 1
    print(f"\nTOKENIZATION PARITY: PASS ({len(questions)}/{len(questions)} identical)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
