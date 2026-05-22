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

"""Compute response length stats for the generated teacher dataset."""

from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq
from transformers import AutoConfig, AutoTokenizer

BASE = Path("/home/tiger/verl/data/dapo_gemma3_4b_pt_teacher_v7_independent")
MODEL = "google/gemma-3-4b-pt"


def stats_for(path: Path) -> dict[str, float]:
    total_rows = 0
    total_tokens = 0
    min_len = None
    max_len = None

    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(columns=["teacher_token_ids"], batch_size=2048):
        lengths = pc.list_value_length(batch.column(0))
        rows = len(lengths)
        lengths_sum = pc.sum(lengths).as_py()
        lengths_min = pc.min(lengths).as_py()
        lengths_max = pc.max(lengths).as_py()

        total_rows += rows
        total_tokens += lengths_sum
        min_len = lengths_min if min_len is None else min(min_len, lengths_min)
        max_len = lengths_max if max_len is None else max(max_len, lengths_max)

    return {
        "rows": total_rows,
        "tokens": total_tokens,
        "avg": total_tokens / total_rows,
        "min": min_len,
        "max": max_len,
    }


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    config = AutoConfig.from_pretrained(MODEL)
    tokenizer_vocab_size = len(tokenizer)
    model_vocab_size = getattr(getattr(config, "text_config", None), "vocab_size", None)
    vocab_size = int(model_vocab_size or tokenizer_vocab_size)
    print(f"tokenizer_vocab_size={tokenizer_vocab_size}")
    print(f"model_logits_vocab_size={vocab_size}")

    combined_rows = 0
    combined_tokens = 0
    for split in ("train", "validation"):
        stats = stats_for(BASE / f"{split}.parquet")
        combined_rows += int(stats["rows"])
        combined_tokens += int(stats["tokens"])
        print(
            f"{split}: rows={stats['rows']} tokens={stats['tokens']} "
            f"avg={stats['avg']:.6f} min={stats['min']} max={stats['max']}"
        )

    combined_avg = combined_tokens / combined_rows
    print(f"combined: rows={combined_rows} tokens={combined_tokens} avg={combined_avg:.6f}")

    for dtype_name, bytes_per_value in (("fp32", 4), ("bf16/fp16", 2)):
        bytes_per_response = combined_avg * vocab_size * bytes_per_value
        print(
            f"avg_response_full_vocab_logits_{dtype_name}: "
            f"{bytes_per_response:.0f} bytes "
            f"({bytes_per_response / 1024**2:.2f} MiB, {bytes_per_response / 1e6:.2f} MB)"
        )


if __name__ == "__main__":
    main()
