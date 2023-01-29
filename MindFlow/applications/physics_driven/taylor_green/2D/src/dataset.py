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

from mindflow.data import Dataset
from mindflow.geometry import Rectangle, TimeDomain, GeometryWithTime, generate_sampling_config


def create_test_dataset(config):
    """ generate evaluation dataset by analytical solution"""
    coord_min = np.array(config["geometry"]["coord_min"] + [config["geometry"]["time_min"]]).astype(np.float32)
    coord_max = np.array(config["geometry"]["coord_max"] + [config["geometry"]["time_max"]]).astype(np.float32)

    axis_x = np.linspace(coord_min[0], coord_max[0], num=100, endpoint=True)
    axis_y = np.linspace(coord_min[1], coord_max[1], num=100, endpoint=True)
    axis_t = np.linspace(coord_min[2], coord_max[2], num=10, endpoint=True)

    mesh_x, mesh_t, mesh_y = np.meshgrid(axis_x, axis_t, axis_y)

    inputs = np.hstack(
        (mesh_x.flatten()[:, None], mesh_y.flatten()[:, None], mesh_t.flatten()[:, None])
    ).astype(np.float32)

    label = []
    for p in inputs:
        x = p[0]
        y = p[1]
        t = p[2]

        u = - np.cos(x) * np.sin(y) * np.exp(-2 * t)
        v = np.sin(x) * np.cos(y) * np.exp(-2 * t)
        p = -0.25 * (np.cos(2*x) + np.cos(2*y)) * np.exp(-4*t)
        label.append(np.float32([u, v, p]))
    label = np.array(label)

    inputs = inputs.reshape((10, 100, 100, 3))
    label = label.reshape((10, 100, 100, 3))

    return inputs, label


def create_training_dataset(config):
    """create training dataset by sampling"""
    geom_config = config["geometry"]
    data_config = config["data"]

    time_interval = TimeDomain("time", geom_config["time_min"], geom_config["time_max"])
    spatial_region = Rectangle("rect", geom_config["coord_min"], geom_config["coord_max"])
    domain_region = GeometryWithTime(spatial_region, time_interval)
    domain_region.set_sampling_config(generate_sampling_config(data_config))

    geom_dict = {domain_region: ["domain", "IC", "BC"]}
    dataset = Dataset(geom_dict)
    return dataset
