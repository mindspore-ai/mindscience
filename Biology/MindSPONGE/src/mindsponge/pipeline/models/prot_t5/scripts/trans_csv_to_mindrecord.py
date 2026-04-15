# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Read the csv files; transform to mindrecord files to train model."""
import argparse
import os
import csv
import re

import numpy as np

from mindspore.mindrecord import FileWriter
from mindformers import T5Tokenizer


# mindrecord schema; data format
SCHEMA = {"raw_ids": {"type": "int32", "shape": [-1]}}
MAX_TOKENS_PER_FILE = 400 * 1024 * 1024 // 4
STEP_PRINT_NUM = 10000
STEP_SAVE_NUM = 1000
file_index = 0


def process_df(row, tokenizer):
    """process df"""
    text = re.sub(r"[UZOB]", "X", row['text'])
    tokens = tokenizer(" ".join(text), truncation=True, add_special_tokens=True, max_length=512)
    token_ids = tokens['input_ids']

    sample = {
        "raw_ids": np.array(token_ids, dtype=np.int32)
    }
    return sample, len(token_ids)


def get_writer(output_dir):
    "writer"
    global file_index
    file_name = os.path.join(output_dir, f"data_{file_index}.mindrecord")
    writer = FileWriter(file_name, shard_num=1, overwrite=True)
    writer.add_schema(SCHEMA, "mindrecord_schema")
    file_index += 1
    return writer


def converse_file(csv_file_path, num_samples, output_dir, tokenizer):
    """read CSV file and transform to mindRecord files."""
    data = []
    current_file_size = 0
    with open(csv_file_path, newline='') as csvfile:
        index = 0
        writer = get_writer(output_dir)
        reader = csv.DictReader(csvfile)
        for row in reader:
            index += 1
            if 0 < num_samples < index:
                break

            if current_file_size > MAX_TOKENS_PER_FILE:
                writer.commit()
                writer = get_writer(output_dir)
                current_file_size = 0

            sample, token_length = process_df(row, tokenizer)

            # compute current file size
            current_file_size += 4 * token_length
            data.append(sample)

            if index % STEP_PRINT_NUM == 0:
                print(f"Samples {index} Done")

            if index % STEP_SAVE_NUM == 0:
                writer.write_raw_data(data)
                data = []

        if data:
            writer.write_raw_data(data)

        writer.commit()


def run(file_path, num_samples, output_dir):
    """run"""
    tokenizer = T5Tokenizer.from_pretrained(args.t5_config_path)
    if os.path.isfile(file_path) and file_path.endswith('csv'):
        converse_file(file_path, num_samples, output_dir, tokenizer)
    else:
        csv_files = [os.path.join(file_path, filename) for filename in os.listdir(file_path) \
                    if filename.endswith('.csv')]
        for cfile in csv_files:
            converse_file(cfile, num_samples, output_dir, tokenizer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--number_samples", default=-1, type=int,
                        help="Choose maximum process data sample number.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Data path to converse to mindrecords; it can file or dir.")
    parser.add_argument("--output_dir", type=str, required=True, help="Data path of output.")
    parser.add_argument('--t5_config_path', type=str, required=True, help='model name or t5 config path')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    run(args.data_dir, args.number_samples, args.output_dir)
