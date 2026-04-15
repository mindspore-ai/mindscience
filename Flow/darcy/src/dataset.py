# ============================================================================
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
"""create dataset"""
import numpy as np
from scipy.constants import pi as PI

from mindflow.data import Dataset
from mindflow.geometry import Rectangle
from mindflow.geometry import generate_sampling_config


def create_test_dataset(config):
    """load labeled data for evaluation"""
    # acquire config
    coord_min = config["geometry"]["coord_min"]
    coord_max = config["geometry"]["coord_max"]
    axis_size = config["geometry"]["axis_size"]

    # set mesh
    axis_x = np.linspace(coord_min[0], coord_max[0], num=axis_size, endpoint=True)
    axis_y = np.linspace(coord_min[1], coord_max[1], num=axis_size, endpoint=True)

    mesh_x, mesh_y = np.meshgrid(axis_x, axis_y)

    input_data = np.hstack(
        (mesh_y.flatten()[:, None], mesh_x.flatten()[:, None])
    ).astype(np.float32)

    label = np.zeros((axis_size, axis_size, 3))
    for i in range(axis_size):
        for j in range(axis_size):
            in_x = axis_x[i]
            in_y = axis_y[j]
            label[i, j, 0] = -2 * PI * np.cos(2 * PI * in_x) * np.cos(2 * PI * in_y)
            label[i, j, 1] = 2 * PI * np.sin(2 * PI * in_x) * np.sin(2 * PI * in_y)
            label[i, j, 2] = np.sin(2 * PI * in_x) * np.cos(2 * PI * in_y)

    label = label.reshape(-1, 3).astype(np.float32)
    return input_data, label


def create_training_dataset(config, name):
    """create training dataset by online sampling"""
    # define geometry
    coord_min = config["geometry"]["coord_min"]
    coord_max = config["geometry"]["coord_max"]
    data_config = config["data"]

    flow_region = Rectangle(
        name,
        coord_min=coord_min,
        coord_max=coord_max,
        sampling_config=generate_sampling_config(data_config),
    )
    geom_dict = {flow_region: ["domain", "BC"]}

    # create dataset for train
    dataset = Dataset(geom_dict)
    return dataset
