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
"""process for pinn helmholtz"""
import os

import yaml
import numpy as np
from mindspore import Tensor

from sciai.utils import parse_arg


def prepare():
    abs_dir = os.path.abspath(os.path.dirname(__file__))
    with open(f"{abs_dir}/../config.yaml") as f:
        config_dict = yaml.safe_load(f)
    args_ = parse_arg(config_dict)
    return (args_,)


class HelmholtzPMLDataPool:
    """data pool"""

    def __init__(self, data):
        self.ps = data['Ps']
        self.x = data['x_star']
        self.z = data['z_star']
        self.m = data['m']
        self.a = data['A']
        self.b = data['B']
        self.c = data['C']
        self.pos = np.concatenate([self.x, self.z], 1)
        self.total_size = self.x.shape[0]

    def generate_training_data(self, size):
        """generate training data randomly"""
        idx = np.random.choice(self.total_size, size, replace=False)
        return self.a[idx, :], self.b[idx, :], self.c[idx, :], self.ps[idx, :], self.m[idx, :], \
               self.x[idx, :], self.z[idx, :]


def generate_data(args, data, dtype):
    """generate data from pool"""
    hhz_data_pool = HelmholtzPMLDataPool(data)
    n_train = round(hhz_data_pool.total_size / args.num_batch)
    a_train, b_train, c_train, ps_train, m_train, x_train, z_train = hhz_data_pool.generate_training_data(n_train)

    train_param = generate_train_param(a_train, b_train, c_train, m_train, ps_train, dtype=dtype)
    train_tensor = generate_train_data(x_train, z_train, dtype)
    bounds = generate_bounds(hhz_data_pool.pos, dtype)

    return train_param, train_tensor, bounds


def generate_bounds(x, dtype):
    """convert bound data"""
    lb_tensor = Tensor(x.min(0).tolist(), dtype=dtype)
    ub_tensor = Tensor(x.max(0).tolist(), dtype=dtype)
    return lb_tensor, ub_tensor


def generate_train_data(x_train, z_train, dtype):
    """convert train data"""
    x_tensor = Tensor(x_train.tolist(), dtype=dtype)
    z_tensor = Tensor(z_train.tolist(), dtype=dtype)
    return x_tensor, z_tensor


def generate_train_param(*inputs, dtype):
    """convert train parameters"""
    a_train, b_train, c_train, m_train, ps_train = inputs
    a_tensor_real = Tensor(np.real(a_train), dtype=dtype)
    a_tensor_imag = Tensor(np.imag(a_train), dtype=dtype)
    b_tensor_real = Tensor(np.real(b_train), dtype=dtype)
    b_tensor_imag = Tensor(np.imag(b_train), dtype=dtype)
    c_tensor_real = Tensor(np.real(c_train), dtype=dtype)
    c_tensor_imag = Tensor(np.imag(c_train), dtype=dtype)
    ps_tensor_real = Tensor(np.real(ps_train), dtype=dtype)
    ps_tensor_imag = Tensor(np.imag(ps_train), dtype=dtype)
    m_tensor = Tensor(m_train.tolist(), dtype=dtype)
    fre = 3.0
    pi = 3.1415926
    omega = 2.0 * pi * fre
    return a_tensor_real, a_tensor_imag, b_tensor_real, b_tensor_imag, c_tensor_real, c_tensor_imag, \
           ps_tensor_real, ps_tensor_imag, m_tensor, omega
