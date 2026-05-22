---
dataset_info:
  features:
  - name: messages
    sequence:
    - name: role
      dtype: string
    - name: content
      dtype: string
  - name: input_ids
    sequence: int64
  - name: response_mask
    sequence: int64
  - name: teacher_log_probs
    sequence: float64
  - name: teacher_token_ids
    sequence: int64
  - name: prompt_idx
    dtype: int64
  splits:
  - name: train
    num_examples: 262368
  - name: validation
    num_examples: 1000
---

# DAPO Gemma3 4B PT Teacher Data for DAPO-17.4k

Teacher generations for `JWei05/DAPO-17.4k` using `google/gemma-3-4b-pt`.

- Train split: 16 responses per training question, 262,368 rows total.
- Validation split: 1 response per validation question, 1,000 rows total.
- Sampling: default teacher generation sampling parameters from `generate_teacher_data.py`.
- Prompting: Gemma 3 IT chat template with the boxed-answer instruction.

Each row includes:

- `messages`: OpenAI-style chat messages including the prompt and assistant response.
- `input_ids`: full chat-tokenized prompt plus response token ids.
- `response_mask`: token-level mask aligned to `input_ids`, with response tokens marked as 1.
- `teacher_token_ids`: response-only token ids.
- `teacher_log_probs`: response-only token log probabilities aligned to `teacher_token_ids`.
- `prompt_idx`: original prompt index from the source split.
