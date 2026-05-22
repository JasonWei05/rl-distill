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

"""Upload the merged teacher-generated SFT dataset to HF Hub as a public dataset."""

import os
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, create_repo

REPO_ID = os.environ.get("REPO_ID", "JWei05/DAPO-Gemma3-27B-IT-RL-SFT-Data")
MERGED = os.environ.get(
    "MERGED_PARQUET",
    str(Path.home() / "verl/data/teacher_27b_step40_n4.parquet"),
)

README = """---
license: gemma
task_categories:
- text-generation
- question-answering
language:
- en
tags:
- math
- distillation
- gemma3
- sft
- rl
size_categories:
- 10K<n<100K
---

# DAPO-Gemma3-27B-IT-RL-SFT-Data

Teacher-generated SFT/distillation dataset. Responses + per-token log probabilities
from a DAPO-RL-trained Gemma 3 27B teacher on the DAPO-Math-17k prompt set.

## Source

- **Teacher**: [`JWei05/dapo-gemma3-27b-it`](https://huggingface.co/JWei05/dapo-gemma3-27b-it),
  `step_000040` — Gemma 3 27B IT after RL training with DAPO on math.
- **Prompts**: [`BytedTsinghua-SIA/DAPO-Math-17k`](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k)
  (17,391 math problems).
- **Responses per prompt**: 4.
- **Sampling**: temperature=1.0, top_p=1.0, max_tokens=20480.

## Columns

| Column | Type | Description |
|---|---|---|
| `messages` | `list[dict]` | `[{"role":"user","content":...},{"role":"assistant","content":...}]` |
| `teacher_log_probs` | `list[float]` | Teacher log-probability of the sampled token at each generated position |
| `teacher_token_ids` | `list[int]` | Generated token IDs, 1:1 aligned with `teacher_log_probs` |
| `prompt_idx` | `int` | Row index in the original DAPO-Math-17k dataset |

## Intended use — forward-KL distillation

For each response token `t`:
```
loss_t = teacher_log_probs[t] - student_log_prob(teacher_token_ids[t])
```

Averaged over response tokens. Gradient is equivalent to cross-entropy on teacher
tokens; the loss value itself is a proper KL divergence (≥ 0, = 0 when the student
matches the teacher at that token). No top-k or full distribution is required —
only the teacher's log-prob at its own sampled token.

## License

Gemma derivative — subject to the [Gemma terms](https://ai.google.dev/gemma/terms).
"""


def main() -> int:
    if not os.path.exists(MERGED):
        print(f"ERROR: merged parquet not found at {MERGED}", file=sys.stderr)
        return 1

    api = HfApi()
    print(f"creating/ensuring repo: datasets/{REPO_ID}")
    create_repo(
        repo_id=REPO_ID,
        repo_type="dataset",
        private=False,
        exist_ok=True,
    )

    print(f"uploading {MERGED} ({os.path.getsize(MERGED) / 1e6:.1f} MB) ...")
    api.upload_file(
        path_or_fileobj=MERGED,
        path_in_repo=os.path.basename(MERGED),
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message="add teacher-generated SFT parquet (27B@step40, n=4)",
    )

    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        f.write(README)
        readme_path = f.name
    print("uploading README.md ...")
    api.upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message="add dataset card",
    )

    print(f"\nDONE -> https://huggingface.co/datasets/{REPO_ID}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
