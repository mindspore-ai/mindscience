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
from mindflow.data import Dataset, ExistedDataConfig
from mindflow.geometry import Rectangle, TimeDomain, GeometryWithTime, generate_sampling_config


def create_test_dataset(config):
    """load labeled data for evaluation"""
    # check data
    test_data_path = config["data"]["root_dir"]
    print("get dataset path: {}".format(test_data_path))
    paths = [test_data_path + '/eval_points.npy', test_data_path + '/eval_label.npy']
    inputs = np.load(paths[0])
    label = np.load(paths[1])
    print("check eval dataset length: {}".format(inputs.shape))
    return inputs, label


def create_training_dataset(config):
    """create training dataset by online sampling"""
    geom_config = config["geometry"]
    data_config = config["data"]

    time_interval = TimeDomain("time", geom_config["time_min"], geom_config["time_max"])
    spatial_region = Rectangle("rect", geom_config["coord_min"], geom_config["coord_max"])
    domain_region = GeometryWithTime(spatial_region, time_interval)
    domain_region.set_sampling_config(generate_sampling_config(data_config))

    geom_dict = {domain_region: ["domain"]}

    data_path = config["data"]["root_dir"]
    print(data_path)
    train_data = ExistedDataConfig(name="train",
                                   data_dir=[data_path + "/train_points.npy", data_path + "/train_label.npy"],
                                   columns_list=["points", "label"],
                                   data_format="npy")

    dataset = Dataset(geom_dict, existed_data_list=[train_data])
    return dataset
