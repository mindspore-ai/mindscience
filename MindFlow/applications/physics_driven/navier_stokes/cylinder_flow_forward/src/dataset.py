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
import os
import numpy as np

from mindflow.data import Dataset, ExistedDataConfig
from mindflow.geometry import Rectangle, TimeDomain, GeometryWithTime, generate_sampling_config
from mindflow.utils import print_log


def create_test_dataset(test_data_path):
    """load labeled data for evaluation"""
    # check data
    print_log("get dataset path: {}".format(test_data_path))
    paths = [test_data_path + '/eval_points.npy',
             test_data_path + '/eval_label.npy']
    inputs = np.load(paths[0])
    label = np.load(paths[1])
    print_log("check eval dataset length: {}".format(inputs.shape))
    return inputs, label


def create_training_dataset(config):
    """create training dataset by online sampling"""
    geom_config = config["geometry"]
    data_config = config["data"]

    time_interval = TimeDomain(
        "time", geom_config["time_min"], geom_config["time_max"])
    spatial_region = Rectangle(
        "rect", geom_config["coord_min"], geom_config["coord_max"])
    domain_region = GeometryWithTime(spatial_region, time_interval)
    domain_region.set_sampling_config(generate_sampling_config(data_config))

    geom_dict = {domain_region: ["domain"]}

    data_dir = data_config["root_dir"]
    print_log(data_dir)
    config_bc = ExistedDataConfig(name="bc",
                                  data_dir=[os.path.join(
                                      data_dir, "bc_points.npy"), os.path.join(data_dir, "bc_label.npy")],
                                  columns_list=["points", "label"],
                                  constraint_type="BC",
                                  data_format="npy")
    config_ic = ExistedDataConfig(name="ic",
                                  data_dir=[os.path.join(
                                      data_dir, "ic_points.npy"), os.path.join(data_dir, "ic_label.npy")],
                                  columns_list=["points", "label"],
                                  constraint_type="IC",
                                  data_format="npy")
    dataset = Dataset(geom_dict, existed_data_list=[config_bc, config_ic])
    return dataset
