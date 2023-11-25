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
import os
import numpy as np

from mindspore import Tensor, dataset
from mindspore import dtype as mstype

from mindflow.data import Dataset, ExistedDataConfig
from mindflow.geometry import Rectangle, Interval, TimeDomain, GeometryWithTime
from mindflow.geometry import generate_sampling_config
from mindflow.utils import print_log


def create_burgers_train_dataset(config):
    """create training dataset by online sampling"""
    geom_config = config["geometry"]
    data_config = config["data"]

    time_interval = TimeDomain(
        "time", geom_config["time_min"], geom_config["time_max"])
    spatial_region = Interval(
        "domain", geom_config["coord_min"], geom_config["coord_max"])
    region = GeometryWithTime(spatial_region, time_interval)
    region.set_sampling_config(generate_sampling_config(data_config))

    geom_dict = {region: ["domain", "IC", "BC"]}
    burgers_dataset = Dataset(geom_dict)

    return burgers_dataset


def create_burgers_test_dataset(test_data_path):
    test_data = np.load(os.path.join(test_data_path, "Burgers.npz"))
    x, t, u = test_data["x"], test_data["t"], test_data["usol"].T
    xx, tt = np.meshgrid(x, t)

    test_data = Tensor(
        np.vstack((np.ravel(xx), np.ravel(tt))).T, mstype.float32)
    test_label = u.flatten()[:, None]
    return test_data, test_label


def create_cylinder_flow_test_dataset(test_data_path):
    """load labeled data for evaluation"""
    # check data
    print_log("get dataset path: {}".format(test_data_path))
    paths = [os.path.join(test_data_path, 'eval_points.npy'),
             os.path.join(test_data_path, 'eval_label.npy')]
    inputs = np.load(paths[0])
    label = np.load(paths[1])
    print_log("check eval dataset length: {}".format(inputs.shape))
    return inputs, label


def create_cylinder_flow_train_dataset(config):
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

    data_path = data_config["train_data_path"]
    print_log(data_path)
    config_bc = ExistedDataConfig(name="bc",
                                  data_dir=[os.path.join(data_path, "bc_points.npy"),
                                            os.path.join(data_path, "bc_label.npy")],
                                  columns_list=["points", "label"],
                                  constraint_type="BC",
                                  data_format="npy")
    config_ic = ExistedDataConfig(name="ic",
                                  data_dir=[os.path.join(data_path, "ic_points.npy"),
                                            os.path.join(data_path, "ic_label.npy")],
                                  columns_list=["points", "label"],
                                  constraint_type="IC",
                                  data_format="npy")
    cylinder_flow_dataset = Dataset(
        geom_dict, existed_data_list=[config_bc, config_ic])
    return cylinder_flow_dataset


class PeriodicHillDataset():
    """convert raw data into train dataset"""

    def __init__(self, bc_data, pde_data):
        self.coord = bc_data[:, :2]
        self.label = bc_data[:, 2:]
        self.pde_coord = pde_data[:, :2]
        self.bc_len = self.coord.shape[0]

    def __getitem__(self, index):
        return self.pde_coord[index], self.coord[index % self.bc_len], self.label[index % self.bc_len]

    def __len__(self):
        return self.pde_coord.shape[0]


def create_periodic_hill_test_dataset(test_data_path):
    """load labeled data for evaluation"""
    data = np.load(os.path.join(test_data_path, "periodic_hill.npy")
                   )  # shape=(700*300, 10)  x, y, u, v, p, uu, uv, vv, rho, nu
    data = data.reshape((700, 300, 10)).astype(np.float32)
    data = data[:, :, :8]
    test_data = data.reshape((-1, 8))
    test_coord = test_data[:, :2]
    test_label = test_data[:, 2:]
    return test_coord, test_label


def create_periodic_hill_train_dataset(config):
    """create training dataset by online sampling"""
    train_data_path = config["data"]["train_data_path"]
    batch_size = config['data']["train_batch_size"]
    data = np.load(os.path.join(train_data_path, "periodic_hill.npy"))
    data = np.reshape(data, (300, 700, 10)).astype(np.float32)
    data = data[:, :, :8]

    bc_data = data[:5].reshape((-1, 8))
    bc_data = np.concatenate((bc_data, data[-5:].reshape((-1, 8))), axis=0)
    bc_data = np.concatenate(
        (bc_data, data[5:-5, :5].reshape((-1, 8))), axis=0)
    bc_data = np.concatenate(
        (bc_data, data[5:-5, -5:].reshape((-1, 8))), axis=0)

    pde_data = data[5:-5, 5:-5].reshape((-1, 8))
    dataset_generator = PeriodicHillDataset(bc_data, pde_data)
    periodic_hill_dataset = dataset.GeneratorDataset(
        dataset_generator, ["pde_coord", "coord", "label"], shuffle=True
    )

    periodic_hill_dataset = periodic_hill_dataset.batch(
        batch_size, drop_remainder=True)
    return periodic_hill_dataset
