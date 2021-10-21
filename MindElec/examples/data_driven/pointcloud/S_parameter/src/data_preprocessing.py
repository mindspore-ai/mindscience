# Copyright 2021 Huawei Technologies Co., Ltd
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
"""data preprocessing utilities"""

import os
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str)
parser.add_argument('--label_path', type=str)
parser.add_argument('--data_config_path', type=str, default="./data_config.npz")
parser.add_argument('--save_data_path', default='./', help='checkpoint directory')
opt = parser.parse_args()

def custom_normalize(dataset, mean=None, std=None):
    """ custom normalization """

    ori_shape = dataset.shape
    dataset = dataset.reshape(ori_shape[0], -1)
    dataset = np.transpose(dataset)
    if mean is None:
        mean = np.mean(dataset, axis=1)
        std = np.std(dataset, axis=1)
        std += (np.abs(std) < 0.0000001)
    dataset = dataset - mean[:, None]
    dataset = dataset / std[:, None]
    dataset = np.transpose(dataset)
    dataset = dataset.reshape(ori_shape)
    return dataset, mean, std

def generate_data():
    """generate dataset for s11 parameter prediction"""

    data_input = np.load(opt.input_path)
    if os.path.exists(opt.data_config_path):
        data_config = np.load(opt.data_config_path)
        mean = data_config["mean"]
        std = data_config["std"]
        data_input, mean, std = custom_normalize(data_input, mean, std)
    else:
        data_input, mean, std = custom_normalize(data_input)
    data_label = np.load(opt.label_path)

    print(data_input.shape)
    print(data_label.shape)

    data_input = data_input.transpose((0, 4, 1, 2, 3))
    data_label[:, :] = np.log10(-data_label[:, :] + 1.0)
    if os.path.exists(opt.data_config_path):
        scale_s11 = data_config['scale_s11']
    else:
        scale_s11 = 0.5 * np.max(np.abs(data_label[:, :]))
    data_label[:, :] = data_label[:, :] / scale_s11

    np.savez(opt.data_config_path, scale_s11=scale_s11, mean=mean, std=std)
    np.save(os.path.join(opt.save_data_path, 'data_input.npy'), data_input)
    np.save(os.path.join(opt.save_data_path, 'data_label.npy'), data_label)
    print("data saved in target path")

if __name__ == "__main__":
    generate_data()
