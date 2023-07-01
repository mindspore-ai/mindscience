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
"""Create dataset."""
import math
import numpy as np

import mindspore as ms
from mindflow.data import Dataset
from mindflow.geometry import Rectangle, generate_sampling_config


def create_train_dataset(config, shuffle=True):
    """Create training dataset."""
    sampling_config = generate_sampling_config(config['data'])

    # The feasible region
    region = Rectangle("rectangle", **config['rectangle'], sampling_config=sampling_config)

    # The region where the  point source is located
    region_src = Rectangle(
        "rectangle_src", **config['rectangle_src'],
        sampling_config=sampling_config)

    dataset = Dataset({region: ['domain', 'BC'], region_src: ["domain"]})

    ds_train = dataset.create_dataset(
        batch_size=config['batch_size'], shuffle=shuffle, prebatched_data=True, drop_remainder=True
    )

    return ds_train


def create_test_dataset(config, n_samps_per_axis=100):
    """Create testing dataset."""
    axis_x = np.linspace(
        config['rectangle']['coord_min'][0],
        config['rectangle']['coord_max'][0],
        n_samps_per_axis,
        endpoint=True)
    axis_y = np.linspace(
        config['rectangle']['coord_min'][1],
        config['rectangle']['coord_max'][1],
        n_samps_per_axis,
        endpoint=True)
    mesh_x, mesh_y = np.meshgrid(axis_x, axis_y)
    mesh = np.stack((mesh_x.flatten(), mesh_y.flatten()), axis=-1)

    label = np.zeros(mesh.shape[0], dtype=np.float32)  # Calculate the analytical solution
    truncation_number = 100
    x_src, y_src = math.pi / 2, math.pi / 2  # Coordinate of the point source
    for i in range(1, truncation_number + 1):
        for j in range(1, truncation_number + 1):
            label += np.sin(i * mesh[:, 0]) * math.sin(i * x_src) * \
                np.sin(j * mesh[:, 1]) * math.sin(j * y_src) / (i**2 + j**2)

    label = label * 4.0 / (math.pi**2)

    return (ms.Tensor(mesh, dtype=ms.float32), ms.Tensor(label, dtype=ms.float32))
