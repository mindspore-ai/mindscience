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
# ==============================================================================
"""
sample fake data for train and test
"""
import os
import argparse
import numpy as np

TIME_SOLUTION = 162
X_SOLUTION = 50
Y_SOLUTION = 50
Z_SOLUTION = 8
TRAIN_NUMBERS = 1
TEST_NUMBERS = 1
IN_CHANNELS = 37
OUT_CHANNELS = 6


def generate_data(train_path, test_path):
    """generate fake data"""
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    inputs = np.random.rand(TRAIN_NUMBERS*TIME_SOLUTION, X_SOLUTION, Y_SOLUTION, Z_SOLUTION,
                            IN_CHANNELS).astype(np.float32)
    label = np.random.rand(TRAIN_NUMBERS*TIME_SOLUTION, X_SOLUTION, Y_SOLUTION, Z_SOLUTION,
                           OUT_CHANNELS).astype(np.float32)
    np.save(train_path + "inputs.npy", inputs)
    np.save(train_path + "label.npy", label)

    inputs = np.random.rand(TEST_NUMBERS*TIME_SOLUTION, X_SOLUTION, Y_SOLUTION, Z_SOLUTION,
                            IN_CHANNELS).astype(np.float32)
    label = np.random.rand(TEST_NUMBERS*TIME_SOLUTION, X_SOLUTION, Y_SOLUTION, Z_SOLUTION,
                           OUT_CHANNELS).astype(np.float32)
    scale = np.ones((OUT_CHANNELS, TIME_SOLUTION), dtype=np.float32)
    np.save(test_path + "inputs.npy", inputs)
    np.save(test_path + "label.npy", label)
    np.save(test_path + "scale.npy", scale)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate data')
    parser.add_argument('--train_path', default='./train/', help='train dataset path')
    parser.add_argument('--test_path', default='./test/', help='test dataset path')
    opt = parser.parse_args()
    generate_data(opt.train_path, opt.test_path)
