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
"""NSFNet dataset"""
import numpy as np


def read_training_data():
    """prepare data"""
    # prepare the training data
    n_train = 2601
    n_bound = 4*101
    re = 40
    lam = 0.5 * re - np.sqrt(0.25 * (re ** 2) + 4 * (np.pi ** 2))
    x = np.linspace(-0.5, 1.0, n_bound + 1)
    y = np.linspace(-0.5, 1.5, n_bound + 1)
    yb1 = np.array([-0.5] * n_bound)
    yb2 = np.array([1] * n_bound)
    xb1 = np.array([-0.5] * n_bound)
    xb2 = np.array([1.5] * n_bound)

    # stack the datapoints for generate the data
    y_train1 = np.concatenate([y[1:n_bound + 1], y[0:n_bound], xb1, xb2], 0)
    x_train1 = np.concatenate([yb1, yb2, x[0:n_bound], x[1:n_bound + 1]], 0)
    xb_train = x_train1.reshape(x_train1.shape[0], 1)
    yb_train = y_train1.reshape(y_train1.shape[0], 1)

    ub_train = 1 - np.exp(lam * xb_train) * np.cos(2 * np.pi * yb_train)
    vb_train = lam / (2 * np.pi) * np.exp(lam * xb_train) * np.sin(2 * np.pi * yb_train)

    x_train = (np.random.rand(n_train, 1) - 1 / 3) * 3 / 2 # ([0 1)-1/3))*3/2 = [-1/3 2/3)*3/2 = [-0.5 1)
    y_train = (np.random.rand(n_train, 1) - 1 / 4) * 2     # ([0 1) - 1/4) *2 = [-1/4 3/4)*2 = [-0.5 1.5)
    return xb_train, yb_train, ub_train, vb_train, x_train, y_train
