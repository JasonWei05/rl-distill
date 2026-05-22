#!/usr/bin/env python3
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build a 9k subset of nvidia/Nemotron-Cascade-2-SFT-Data and push to HF.

Streams 3 configs (chat, instruction_following, science). For each, collects
up to POOL_SIZE filtered samples, then random-samples PICK_PER_SUBSET. Splits
the combined 9k into 8k train / 1k validation and pushes to a public repo.

Filter (gemma-3 tokenizer, content-only, every turn):
  - system content    <= 256 tokens
  - user content      <= 2048 tokens
  - assistant content <= 16384 tokens

Output schema: {domain, source, messages (system removed), generator}.
"""

from __future__ import annotations

import os
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_env() -> None:
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip().strip('"').strip("'")
        os.environ.setdefault(k.strip(), v)


load_env()
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY")
if not HF_TOKEN:
    sys.exit("Missing HF_TOKEN / HUGGINGFACE_API_KEY in .env")
os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN

from datasets import Dataset, DatasetDict, load_dataset  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

SOURCE = "nvidia/Nemotron-Cascade-2-SFT-Data"
TARGET_REPO = "JWei05/Nemotron-Cascade-2-SFT-Data-9k-subset"
SUBSETS = ["chat", "instruction_following", "science"]
TOKENIZER_NAME = "google/gemma-3-4b-it"

POOL_SIZE = 20_000
PICK_PER_SUBSET = 3_000
TRAIN_SIZE = 8_000
SHUFFLE_BUFFER = 20_000
MAX_SEEN_PER_SUBSET = 400_000  # safety cap; abort if filter rate is too low
SEED = 42

LIMITS = {"system": 256, "user": 2048, "assistant": 16 * 1024}


def main() -> None:
    random.seed(SEED)

    print(f"[load] tokenizer={TOKENIZER_NAME}")
    tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME, token=HF_TOKEN)

    def n_tok(s: str) -> int:
        return len(tok.encode(s or "", add_special_tokens=False))

    def passes(ex: dict) -> bool:
        msgs = ex.get("messages") or []
        if not msgs:
            return False
        roles_seen = set()
        for m in msgs:
            role = m.get("role")
            if role not in LIMITS:
                return False
            if n_tok(m.get("content") or "") > LIMITS[role]:
                return False
            roles_seen.add(role)
        return "user" in roles_seen and "assistant" in roles_seen

    def project(ex: dict) -> dict:
        msgs = [
            {"role": m["role"], "content": m.get("content") or ""} for m in ex["messages"] if m.get("role") != "system"
        ]
        return {
            "domain": ex.get("domain"),
            "source": ex.get("source"),
            "messages": msgs,
            "generator": ex.get("generator"),
        }

    all_picked: list[dict] = []
    for subset in SUBSETS:
        print(f"\n=== {subset} ===")
        t0 = time.time()
        ds = load_dataset(SOURCE, subset, split="train", streaming=True, token=HF_TOKEN).shuffle(
            seed=SEED, buffer_size=SHUFFLE_BUFFER
        )

        pool: list[dict] = []
        seen = 0
        for ex in ds:
            seen += 1
            if seen % 2000 == 0:
                rate = len(pool) / max(seen, 1)
                dt = time.time() - t0
                print(f"  seen={seen:>7d} kept={len(pool):>5d} keep_rate={rate:.3f} elapsed={dt:.1f}s")
            if passes(ex):
                pool.append(project(ex))
                if len(pool) >= POOL_SIZE:
                    break
            if seen >= MAX_SEEN_PER_SUBSET:
                print(f"  hit MAX_SEEN_PER_SUBSET={MAX_SEEN_PER_SUBSET}; stopping")
                break

        print(f"  pool={len(pool)} (seen={seen}, {time.time() - t0:.1f}s)")
        if len(pool) < PICK_PER_SUBSET:
            sys.exit(
                f"FATAL: only {len(pool)} samples passed filter for '{subset}', "
                f"need {PICK_PER_SUBSET}. Raise MAX_SEEN_PER_SUBSET or relax limits."
            )
        picks = random.sample(pool, PICK_PER_SUBSET)
        all_picked.extend(picks)

    print(f"\n[combine] total={len(all_picked)}")
    random.shuffle(all_picked)
    train = all_picked[:TRAIN_SIZE]
    val = all_picked[TRAIN_SIZE:]
    print(f"[split] train={len(train)} validation={len(val)}")

    ds_dict = DatasetDict(
        {
            "train": Dataset.from_list(train),
            "validation": Dataset.from_list(val),
        }
    )

    print(f"\n[push] {TARGET_REPO} (public)")
    ds_dict.push_to_hub(TARGET_REPO, token=HF_TOKEN, private=False)
    print("[done]")


if __name__ == "__main__":
    main()
