# Copyright 2022 Huawei Technologies Co., Ltd
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
import os
import numpy as np

from mindspore import Tensor
from mindspore import dtype as mstype

from mindflow.data import Dataset
from mindflow.geometry import Interval, TimeDomain, GeometryWithTime
from mindflow.geometry import generate_sampling_config


def create_training_dataset(config):
    """create training dataset by online sampling"""
    geom_config = config["geometry"]
    data_config = config["data"]

    time_interval = TimeDomain("time", geom_config["time_min"], geom_config["time_max"])
    spatial_region = Interval("domain", geom_config["coord_min"], geom_config["coord_max"])
    region = GeometryWithTime(spatial_region, time_interval)
    region.set_sampling_config(generate_sampling_config(data_config))

    geom_dict = {region: ["domain", "IC", "BC"]}
    dataset = Dataset(geom_dict)

    return dataset


def create_test_dataset(test_dataset_path):
    test_data = np.load(os.path.join(test_dataset_path, "Burgers.npz"))
    x, t, u = test_data["x"], test_data["t"], test_data["usol"].T
    xx, tt = np.meshgrid(x, t)

    test_data = Tensor(np.vstack((np.ravel(xx), np.ravel(tt))).T, mstype.float32)
    test_label = u.flatten()[:, None]
    return test_data, test_label
