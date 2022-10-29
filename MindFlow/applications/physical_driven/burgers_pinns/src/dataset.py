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
from mindflow.data import Dataset
from mindflow.geometry import FixedPoint, Interval, TimeDomain, GeometryWithTime
from mindflow.geometry import create_config_from_edict

from .sampling_config import src_sampling_config, bc_sampling_config


def create_random_dataset(config):
    """create training dataset by online sampling"""
    coord_min = config["coord_min"]
    coord_max = config["coord_max"]

    time_interval = TimeDomain("time", 0.0, config["range_t"])
    spatial_region = Interval("flow_region", coord_min, coord_max)
    region = GeometryWithTime(spatial_region, time_interval)
    region.set_sampling_config(create_config_from_edict(src_sampling_config))

    point1 = FixedPoint("point1", coord_min)
    boundary_1 = GeometryWithTime(point1, time_interval)
    boundary_1.set_name("bc1")
    boundary_1.set_sampling_config(create_config_from_edict(bc_sampling_config))

    point2 = FixedPoint("point2", coord_max)
    boundary_2 = GeometryWithTime(point2, time_interval)
    boundary_2.set_name("bc2")
    boundary_2.set_sampling_config(create_config_from_edict(bc_sampling_config))

    geom_dict = {region: ["domain", "IC"],
                 boundary_1: ["BC"],
                 boundary_2: ["BC"]}

    dataset = Dataset(geom_dict)

    return dataset
