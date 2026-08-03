# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""
Preprocess the Flow-GRPO OCR prompt dataset to parquet format (for Wan2.2 DanceGRPO training).

The raw dataset is a plain-text file (``train.txt`` / ``test.txt``) with one prompt per line;
the target text the model must render is enclosed in the first pair of double quotes. Unlike
``wan22_hpsv3.py``, lines containing Chinese characters are kept: the OCR ground truth is the
quoted text itself, not the prompt.
"""

import argparse
import os

import pandas as pd
from verl.utils.hdfs_io import copy, makedirs


def extract_solution(solution_str: str) -> str:
    # The solution is stored in the format: 'The image displays "xxx".'
    return solution_str.split('"')[1]


def _load_prompts(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument(
        "--input_dir",
        default="flow_grpo/dataset/ocr",
        help="Directory holding the raw train.txt / test.txt OCR prompt files.",
    )
    parser.add_argument(
        "--output_dir",
        default="~/data/ocr/wan22",
        help="Directory to save the preprocessed parquet files.",
    )
    parser.add_argument(
        "--frame_interval",
        type=int,
        default=2,
        help="Video frame subsampling interval forwarded to the OCR reward via extra_info.",
    )
    parser.add_argument(
        "--smoke_val_size",
        type=int,
        default=16,
        help="Also write a small test_smoke.parquet validation subset (0 disables). Validation "
        "iterates the whole val set per pass, so smoke runs need a tiny val file.",
    )

    args = parser.parse_args()
    input_dir = os.path.expanduser(args.input_dir)
    output_dir = os.path.expanduser(args.output_dir)

    train_prompts = _load_prompts(os.path.join(input_dir, "train.txt"))
    test_prompts = _load_prompts(os.path.join(input_dir, "test.txt"))
    print(f"Loaded {len(train_prompts)} train prompts, {len(test_prompts)} test prompts")

    data_source = "dance_grpo/ocr"
    negative_user_prompt = " "

    def make_record(prompt: str, split: str, idx: int) -> dict:
        return {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": prompt}],
            "negative_prompt": [{"role": "user", "content": negative_user_prompt}],
            "ability": "t2v",
            "reward_model": {"style": "model", "ground_truth": extract_solution(prompt)},
            "extra_info": {"split": split, "index": idx, "frame_interval": args.frame_interval},
        }

    train_records = [make_record(p, "train", i) for i, p in enumerate(train_prompts)]
    test_records = [make_record(p, "test", i) for i, p in enumerate(test_prompts)]

    os.makedirs(output_dir, exist_ok=True)

    train_parquet_path = os.path.join(output_dir, "train.parquet")
    test_parquet_path = os.path.join(output_dir, "test.parquet")
    pd.DataFrame(train_records).to_parquet(train_parquet_path)
    pd.DataFrame(test_records).to_parquet(test_parquet_path)

    print(f"Train: {len(train_records)} records -> {train_parquet_path}")
    print(f"Test:  {len(test_records)} records -> {test_parquet_path}")

    if args.smoke_val_size > 0:
        smoke_records = test_records[: args.smoke_val_size]
        smoke_parquet_path = os.path.join(output_dir, "test_smoke.parquet")
        pd.DataFrame(smoke_records).to_parquet(smoke_parquet_path)
        print(f"Smoke val: {len(smoke_records)} records -> {smoke_parquet_path}")

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=output_dir, dst=args.hdfs_dir)
