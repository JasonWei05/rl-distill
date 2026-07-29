"""verl-schema parquet -> NeMo-RL response dataset (DeepScaleR strict-4/4 repro).

Mirrors nemo_rl DAPOMath17KDataset (response_datasets/dapo_math.py) but loads our
verl RL parquets (columns: data_source / prompt / reward_model / ...) from a local
path instead of HuggingFace. The row mapping is 1:1 with what DAPOMath17KDataset
produces, so the stock ``math_hf_data_processor`` consumes it unchanged:

    {"messages": [{"role": "user", "content": prompt[0].content},
                  {"role": "assistant", "content": reward_model.ground_truth}],
     "task_name": "deepscaler_strict"}

The user prompt content already carries the "Please output the final answer
within \\boxed{}." instruction (baked in by build_deepscaler_rl_data.py), and the
12-shot chat template is installed on the tokenizer via
policy.tokenizer.chat_template — this class does no prompt formatting.

Referenced from the config by dotted path
(``rl_distill_nemo.deepscaler_dataset.DeepScalerStrictParquet``) — no nemo_rl
registry edits needed.
"""

from typing import Any

from nemo_rl.data.datasets.raw_dataset import RawDataset
from nemo_rl.data.datasets.utils import load_dataset_from_path


class DeepScalerStrictParquet(RawDataset):
    """Local verl parquet wrapper; used for both train and validation configs."""

    def __init__(
        self,
        data_path: str,
        split_validation_size: float | int = 0,
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        self.task_name = "deepscaler_strict"

        self.dataset = load_dataset_from_path(data_path, None, "train")
        self.dataset = self.dataset.map(
            self.format_data,
            remove_columns=self.dataset.column_names,
        )

        # Only used when this dataset serves as both train and validation
        # (split_validation_size > 0). We pass separate parquets, so keep 0.
        self.val_dataset = None
        self.split_train_validation(split_validation_size, seed)

    def format_data(self, data: dict[str, Any]) -> dict[str, Any]:
        return {
            "messages": [
                {
                    "role": "user",
                    "content": data["prompt"][0]["content"],
                },
                {
                    "role": "assistant",
                    "content": data["reward_model"]["ground_truth"],
                },
            ],
            "task_name": self.task_name,
        }
