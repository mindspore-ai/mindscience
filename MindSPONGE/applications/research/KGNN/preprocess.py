# Copyright 2022 Huawei Technologies Co., Ltd
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
"""
preprocess script
"""
import os
import argparse
import numpy as np
from src.utils import format_filename

DRUG_EXAMPLE = '{dataset}_{type}_examples.npy'


def generate_bin():
    """Generate bin files."""
    parser = argparse.ArgumentParser(description='preprocess')
    parser.add_argument('--data_dir', type=str, default='./data/', help='Dataset directory')
    parser.add_argument('--dataset', type=str, default='kegg', help='Dataset directory')
    parser.add_argument('--result_path', type=str, default='./preprocess_Result/', help='Result path')
    config = parser.parse_args()
    test_data = np.load(format_filename(config.data_dir, DRUG_EXAMPLE, dataset=config.dataset, type="test"))

    test_data_path = os.path.join(config.result_path, "00_data")
    test_label_path = os.path.join(config.result_path, "01_data")
    os.makedirs(test_data_path)
    os.makedirs(test_label_path)
    data_path = config.dataset + ".bin"
    label_path = config.dataset + "_label" + ".bin"
    input_data = test_data[:, :2]
    input_label = test_data[:, 2:3]
    input_data.tofile(os.path.join(test_data_path, data_path))
    input_label.tofile(os.path.join(test_label_path, label_path))


if __name__ == '__main__':
    generate_bin()
