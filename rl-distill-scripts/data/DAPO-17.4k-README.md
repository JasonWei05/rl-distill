---
dataset_info:
  features:
    - name: data_source
      dtype: string
    - name: prompt
      sequence:
        - name: content
          dtype: string
        - name: role
          dtype: string
    - name: reward_model
      struct:
        - name: ground_truth
          dtype: string
        - name: style
          dtype: string
    - name: extra_info
      struct:
        - name: index
          dtype: string
        - name: original_question
          dtype: string
        - name: split
          dtype: string
  splits:
    - name: train
      num_examples: 16398
    - name: validation
      num_examples: 1000
---

# DAPO-17.4k

This dataset contains the 17,398-row DAPO math set in the RL parquet format used by verl.

Each prompt asks the model to return the final answer in `\boxed{}`:

```text
Please output the final answer within \boxed{}.
```

## Splits

- `train`: 16,398 examples
- `validation`: 1,000 examples

The validation set was sampled uniformly at random from `dapo-math-17k.parquet` with seed `42`; the train split contains the remaining examples. The splits have no overlap by `extra_info.index`.

## Schema

- `data_source`: source label, currently `math`
- `prompt`: single-turn user prompt with the boxed-answer instruction
- `reward_model`: rule reward metadata with `ground_truth`
- `extra_info`: original question, stable index, and split label
