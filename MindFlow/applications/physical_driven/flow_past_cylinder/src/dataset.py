# Copyright 2021 Huawei Technologies Co., Ltd
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

from mindflow.data import Dataset, ExistedDataConfig
from mindflow.geometry import Rectangle, TimeDomain, GeometryWithTime
from mindflow.geometry import create_config_from_edict
from .sampling_config import domain_sampling_config


def create_evaluation_dataset(test_data_path):
    """load labeled data for evaluation"""
    # check data
    print("get dataset path: {}".format(test_data_path))
    paths = [test_data_path + '/eval_points.npy', test_data_path + '/eval_label.npy']
    inputs = np.load(paths[0])
    label = np.load(paths[1])
    print("check eval dataset length: {}".format(inputs.shape))
    return inputs, label


def create_training_dataset(config):
    """create training dataset by online sampling"""
    coord_min = config["coord_min"]
    coord_max = config["coord_max"]
    rectangle = Rectangle("rect", coord_min, coord_max)

    time_interval = TimeDomain("time", 0.0, config["range_t"])
    domain_region = GeometryWithTime(rectangle, time_interval)
    domain_region.set_name("domain")
    domain_region.set_sampling_config(create_config_from_edict(domain_sampling_config))

    geom_dict = {domain_region: ["domain"]}

    data_path = config["train_data_path"]
    config_bc = ExistedDataConfig(name="bc",
                                  data_dir=[data_path + "/bc_points.npy", data_path + "/bc_label.npy"],
                                  columns_list=["points", "label"],
                                  constraint_type="BC",
                                  data_format="npy")
    config_ic = ExistedDataConfig(name="ic",
                                  data_dir=[data_path + "/ic_points.npy", data_path + "/ic_label.npy"],
                                  columns_list=["points", "label"],
                                  constraint_type="IC",
                                  data_format="npy")
    dataset = Dataset(geom_dict, existed_data_list=[config_bc, config_ic])
    return dataset
