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
"""nsf_net dataset"""
import numpy as np
import mindspore as ms
from sciai.utils.ms_utils import to_tensor


def generate_data(args, dtype):
    """generate data"""
    n_train = args.n_train
    n_bound = args.n_bound
    # Load Data
    re = 40
    lam = 0.5 * re - np.sqrt(0.25 * (re ** 2) + 4 * (np.pi ** 2))
    x = np.linspace(-0.5, 1.0, n_bound + 1)
    y = np.linspace(-0.5, 1.5, n_bound + 1)
    yb1 = np.array([-0.5] * n_bound)
    yb2 = np.array([1] * n_bound)
    xb1 = np.array([-0.5] * n_bound)
    xb2 = np.array([1.5] * n_bound)
    y_train1 = np.concatenate([y[1:n_bound + 1], y[0:n_bound], xb1, xb2], 0)
    x_train1 = np.concatenate([yb1, yb2, x[0:n_bound], x[1:n_bound + 1]], 0)
    xb_train = x_train1.reshape(x_train1.shape[0], 1)
    yb_train = y_train1.reshape(y_train1.shape[0], 1)
    ub_train = 1 - np.exp(lam * xb_train) * np.cos(2 * np.pi * yb_train)
    vb_train = lam / (2 * np.pi) * np.exp(lam * xb_train) * np.sin(2 * np.pi * yb_train)
    x_train = (np.random.rand(n_train, 1) - 1 / 3) * 3 / 2
    y_train = (np.random.rand(n_train, 1) - 1 / 4) * 2
    xb_train, yb_train, ub_train, vb_train, x_train, y_train = to_tensor(
        (xb_train, yb_train, ub_train, vb_train, x_train, y_train), dtype=dtype)
    return lam, ub_train, vb_train, x_train, xb_train, y_train, yb_train


def generate_test_data(lam):
    x_star = (np.random.rand(1000, 1) - 1 / 3) * 3 / 2
    y_star = (np.random.rand(1000, 1) - 1 / 4) * 2
    u_star = 1 - np.exp(lam * x_star) * np.cos(2 * np.pi * y_star)
    v_star = (lam / (2 * np.pi)) * np.exp(lam * x_star) * np.sin(2 * np.pi * y_star)
    p_star = 0.5 * (1 - np.exp(2 * lam * x_star))
    # Prediction
    x_star, y_star = to_tensor((x_star, y_star), dtype=ms.float32)
    return p_star, u_star, v_star, x_star, y_star
