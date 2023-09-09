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
"""process for xpinns"""
import os

import yaml
import numpy as np
import scipy.io as scio

from sciai.utils import to_tensor, parse_arg


def prepare():
    abs_dir = os.path.abspath(os.path.dirname(__file__))
    with open(f"{abs_dir}/../config.yaml") as f:
        config_dict = yaml.safe_load(f)
    args_ = parse_arg(config_dict)
    return (args_,)


def generate_inputs(dtype, *inputs):
    """generate model inputs"""
    x_ub, ub_, x_f1, x_f2, x_f3, x_fi1, x_fi2 = inputs
    ub, x_ub, y_ub, x_f1, y_f1, x_f2, y_f2, x_f3, y_f3, x_fi1, y_fi1, x_fi2, y_fi2 \
        = to_tensor((ub_, x_ub[:, 0:1], x_ub[:, 1:2], x_f1[:, 0:1], x_f1[:, 1:2], x_f2[:, 0:1], x_f2[:, 1:2],
                     x_f3[:, 0:1], x_f3[:, 1:2], x_fi1[:, 0:1], x_fi1[:, 1:2], x_fi2[:, 0:1], x_fi2[:, 1:2]),
                    dtype)

    return ub, x_ub, y_ub, x_f1, y_f1, x_f2, y_f2, x_f3, y_f3, x_fi1, y_fi1, x_fi2, y_fi2


def generate_data(data_path, dtype):
    """generate data"""

    # Boundary points from subdomain 1
    n_ub = 200
    # Residual points in three subdomains
    n_f1 = 5000
    n_f2 = 1800
    n_f3 = 1200
    # Interface points along the two interfaces
    n_i1 = 100
    n_i2 = 100
    # Load training data (boundary points), residual and interface points from .mat file
    # All points are generated in Matlab
    data = scio.loadmat(f'{data_path}/XPINN_2D_PoissonEqn.mat')
    u_exact, u_exact2, u_exact3, ub_train, x_f1, x_f2, x_f3, xb, xi1, xi2, y_f1, y_f2, y_f3, yb, yi1, yi2 \
        = get_data(data)
    x_f1_train = np.hstack((x_f1.flatten()[:, None], y_f1.flatten()[:, None]))
    x_f2_train = np.hstack((x_f2.flatten()[:, None], y_f2.flatten()[:, None]))
    x_f3_train = np.hstack((x_f3.flatten()[:, None], y_f3.flatten()[:, None]))
    x_fi1_train = np.hstack((xi1.flatten()[:, None], yi1.flatten()[:, None]))
    x_fi2_train = np.hstack((xi2.flatten()[:, None], yi2.flatten()[:, None]))
    x_ub_train = np.hstack((xb.flatten()[:, None], yb.flatten()[:, None]))
    # Points in the whole  domain
    x_star1 = np.hstack((x_f1.flatten()[:, None], y_f1.flatten()[:, None]))
    x_star2 = np.hstack((x_f2.flatten()[:, None], y_f2.flatten()[:, None]))
    x_star3 = np.hstack((x_f3.flatten()[:, None], y_f3.flatten()[:, None]))
    # Randomly select the residual points from subdomains
    idx1 = np.random.choice(x_f1_train.shape[0], n_f1, replace=False)
    x_f1_train = x_f1_train[idx1, :]
    idx2 = np.random.choice(x_f2_train.shape[0], n_f2, replace=False)
    x_f2_train = x_f2_train[idx2, :]
    idx3 = np.random.choice(x_f3_train.shape[0], n_f3, replace=False)
    x_f3_train = x_f3_train[idx3, :]
    # Randomly select boundary points
    idx4 = np.random.choice(x_ub_train.shape[0], n_ub, replace=False)
    x_ub_train = x_ub_train[idx4, :]
    ub_train = ub_train[idx4, :]
    # Randomly select the interface points along two interfaces
    idxi1 = np.random.choice(x_fi1_train.shape[0], n_i1, replace=False)
    x_fi1_train = x_fi1_train[idxi1, :]
    idxi2 = np.random.choice(x_fi2_train.shape[0], n_i2, replace=False)
    x_fi2_train = x_fi2_train[idxi2, :]
    model_inputs = generate_inputs(dtype, x_ub_train, ub_train, x_f1_train, x_f2_train, x_f3_train, x_fi1_train,
                                   x_fi2_train)
    return x_f1_train, x_f2_train, x_f3_train, x_fi1_train, x_fi2_train, x_star1, x_star2, x_star3, x_ub_train, \
           model_inputs, u_exact, u_exact2, u_exact3, xb, xi1, xi2, yb, yi1, yi2


def get_data(data):
    """obtain data"""
    x_f1 = data['x_f1'].flatten()[:, None]
    y_f1 = data['y_f1'].flatten()[:, None]
    x_f2 = data['x_f2'].flatten()[:, None]
    y_f2 = data['y_f2'].flatten()[:, None]
    x_f3 = data['x_f3'].flatten()[:, None]
    y_f3 = data['y_f3'].flatten()[:, None]
    xi1 = data['xi1'].flatten()[:, None]
    yi1 = data['yi1'].flatten()[:, None]
    xi2 = data['xi2'].flatten()[:, None]
    yi2 = data['yi2'].flatten()[:, None]
    xb = data['xb'].flatten()[:, None]
    yb = data['yb'].flatten()[:, None]
    ub_train = data['ub'].flatten()[:, None]
    u_exact = data['u_exact'].flatten()[:, None]
    u_exact2 = data['u_exact2'].flatten()[:, None]
    u_exact3 = data['u_exact3'].flatten()[:, None]
    return u_exact, u_exact2, u_exact3, ub_train, x_f1, x_f2, x_f3, xb, xi1, xi2, y_f1, y_f2, y_f3, yb, yi1, yi2
