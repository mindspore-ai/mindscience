
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

"""Data processing"""
import os
from math import sqrt

import yaml
import numpy as np
from mindspore import dataset as ds

from sciai.common.dataset import DatasetGenerator
from sciai.utils import parse_arg


def prepare():
    abs_dir = os.path.abspath(os.path.dirname(__file__))
    with open(f"{abs_dir}/../config.yaml") as f:
        config_dict = yaml.safe_load(f)
    args_ = parse_arg(config_dict)
    return (args_,)


def get_data(args):
    """Get data"""
    l = 1
    x_start = 0
    x_end = x_start + l
    r_inlet = 0.05
    n_pt = 100
    unique_x = np.linspace(x_start, x_end, n_pt)
    mu = 0.5 * (x_end - x_start)
    n_y = 20
    x_2d = np.tile(unique_x, n_y)
    x_2d = np.reshape(x_2d, (len(x_2d), 1))

    def three_d_mesh(x_2d, tmp_1d):
        tmp_3d = np.expand_dims(np.tile(tmp_1d, len(x_2d)), 1).astype('float')
        x = []
        for x0 in x_2d:
            tmpx = np.tile(x0, len(tmp_1d))
            x.append(tmpx)
        x = np.reshape(x, (len(tmp_3d), 1))
        return x, tmp_3d

    sigma = 0.1
    # negative means aneurysm
    scale_start = -0.02
    scale_end = 0
    ng = 50
    scale_1d = np.linspace(scale_start, scale_end, ng, endpoint=True)
    x, scale = three_d_mesh(x_2d, scale_1d)
    # axisymetric boundary
    r = scale * 1 / sqrt(2 * np.pi * sigma ** 2) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
    # Generate stenosis
    y_up = (r_inlet - r) * np.ones_like(x)
    y = np.zeros([len(x), 1])
    for x0 in unique_x:
        index = np.where(x[:, 0] == x0)[0]
        rsec = max(y_up[index])
        tmpy = np.linspace(-rsec, rsec, len(index)).reshape(len(index), -1)
        y[index] = tmpy
    d_p = 0.1
    rho = 1
    dataset_generator = DatasetGenerator(x, y, scale)
    dataset = ds.GeneratorDataset(dataset_generator, ["data", "label", "scale"], shuffle=True)
    dataset = dataset.batch(batch_size=args.batch_size, drop_remainder=True)
    dataset_iter = dataset.create_dict_iterator()
    return d_p, dataset_iter, l, mu, r_inlet, rho, sigma, x_end, x_start
