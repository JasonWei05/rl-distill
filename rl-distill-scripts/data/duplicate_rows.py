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

from argparse import ArgumentParser, ArgumentTypeError

import pandas as pd


def positive_int(value: str) -> int:
    try:
        intval = int(value)
    except ValueError as exc:
        raise ArgumentTypeError(f"Invalid integer value: {value}") from exc
    if intval <= 0:
        raise ArgumentTypeError("repeat_times must be a positive integer.")
    return intval


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Duplicate rows in a parquet dataset.")
    parser.add_argument(
        "--input_path",
        required=True,
        help="Path to the source parquet file.",
    )
    parser.add_argument(
        "--save_path",
        required=True,
        help="Destination path to write the duplicated parquet file.",
    )
    parser.add_argument(
        "--repeat_times",
        type=positive_int,
        required=True,
        help="Number of times to duplicate the dataset.",
    )
    return parser


def duplicate_dataset(input_path: str, save_path: str, repeat_times: int) -> None:
    df = pd.read_parquet(input_path)
    duplicated_df = pd.concat([df] * repeat_times, ignore_index=True)
    duplicated_df.to_parquet(save_path, index=False)


def main() -> None:
    parser = parse_args()
    args = parser.parse_args()
    duplicate_dataset(args.input_path, args.save_path, args.repeat_times)


if __name__ == "__main__":
    main()
