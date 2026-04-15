# Copyright 2023 Huawei Technologies Co., Ltd
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

"""process"""
import os

import yaml
import mindspore as ms
import numpy as np
import scipy.io

from sciai.utils import parse_arg


def prepare():
    abs_dir = os.path.abspath(os.path.dirname(__file__))
    with open(f"{abs_dir}/../config.yaml") as f:
        config_dict = yaml.safe_load(f)
    args_ = parse_arg(config_dict)
    return (args_,)


def fun_x(x_list):
    fun = np.zeros(len(x_list))
    fun = np.reshape(fun, (-1, 1))
    for i, x in enumerate(x_list):
        if x <= 0:
            fun[i] = 0.2 * np.sin(6 * x)
        else:
            fun[i] = 0.1 * x * np.cos(18 * x) + 1
    return fun


def get_data(num_grid, dtype):
    x = np.linspace(-3, 3, num_grid + 1)
    x = np.reshape(x, (-1, 1))
    y = fun_x(x)
    x = ms.Tensor(x, dtype)
    y = ms.Tensor(y, dtype)
    return x, y


def save_mse_hist(data_path, mse_hist):
    with open(f'{data_path}/History_NN.mat', mode='wb') as f:
        scipy.io.savemat(f, {'MSE_hist': mse_hist})
