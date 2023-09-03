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
"""for dataset"""
from collections import namedtuple
import os

import numpy as np

from mindspore import ops, set_seed, dataset, Tensor

from .utils import random_data, random_cylinder_flow_data, random_periodic_hill_data

set_seed(0)


class CreateDataset():
    """convert raw data into train dataset"""

    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        return (self.data[index], self.label[index])

    def __len__(self):
        return len(self.data)


def create_burgers_dataset(dataset_config, dataset_type="nn"):
    """create dataset with noise for burgers"""
    # load data
    data = np.load(dataset_config["path"])
    x, t, u = data["x"], data["t"], data["usol"].T
    u = u.T

    x_num = x.shape[0]
    t_num = t.shape[0]
    total = x_num * t_num

    # choose number of training data
    choose_train = dataset_config["choose_train"]
    # choose number of validation data
    choose_validate = dataset_config["choose_validate"]
    noise_level = dataset_config["noise_level"]

    # set data
    data = ops.zeros(2)

    # produce noise
    for j in range(x_num):
        for i in range(t_num):
            u[j, i] = u[j, i]*(1+0.01*noise_level*np.random.uniform(-1, 1))

    # create noise_data dir
    noise_save_path = dataset_config["noise_save_path"]
    if not os.path.exists(os.path.abspath(noise_save_path)):
        os.makedirs(os.path.abspath(noise_save_path))

    # save noise data
    np.save(os.path.join(
        noise_save_path, "burgers_u_%dnoise.npy" % (noise_level)), u)

    DataInfo = namedtuple('DataInfo',
                          ['total', 'choose_train', 'choose_validate',
                           'x', 't', 'u', 'x_num', 't_num'])
    data_info = DataInfo(total, choose_train,
                         choose_validate, x, t, u, x_num, t_num)

    # produce random data for training and validating
    h_data_choose, h_data_validate, database_choose, database_validate = random_data(
        data_info)

    # create datasets
    h_data_choose = h_data_choose.astype(np.float32)
    h_data_validate = h_data_validate.astype(np.float32)
    database_choose = database_choose.astype(np.float32)
    database_validate = database_validate.astype(np.float32)
    if dataset_type == "pinn":
        DatasetResult = namedtuple('DatasetResult',
                                   ['database_choose', 'h_data_choose',
                                    'database_validate', 'h_data_validate'])
        return DatasetResult(Tensor(database_choose), Tensor(h_data_choose),
                             Tensor(database_validate), h_data_validate)
    train_dataset_generator = CreateDataset(database_choose, h_data_choose)
    train_dataset = dataset.GeneratorDataset(
        train_dataset_generator, ["data", "label"], shuffle=True)
    train_dataset = train_dataset.batch(
        batch_size=dataset_config["train_batch"], drop_remainder=True)
    return train_dataset, database_validate, h_data_validate


def create_cylinder_flow_dataset(dataset_config, dataset_type="nn"):
    """create dataset with noise for cylinder flow"""
    # load data
    points = np.load(dataset_config["points_path"])
    label = np.load(dataset_config["label_path"])

    # choose number of training data
    choose_train = dataset_config["choose_train"]
    # choose number of validation data
    choose_validate = dataset_config["choose_validate"]
    noise_level = dataset_config["noise_level"]

    # produce random data for training and validating
    random_data_result = random_cylinder_flow_data(
        choose_train, choose_validate, points, label)
    h_data_choose, h_data_validate, database_choose, database_validate = random_data_result

    # for chosen data create noise
    for i in range(choose_train):
        h_data_choose[i] = h_data_choose[i]*(1+0.01*noise_level *
                                             np.random.uniform(-1, 1))
        h_data_validate[i] = h_data_validate[i]*(1+0.01*noise_level *
                                                 np.random.uniform(-1, 1))

    # create noise_data dir
    noise_save_path = dataset_config["noise_save_path"]
    if not os.path.exists(os.path.abspath(noise_save_path)):
        os.makedirs(os.path.abspath(noise_save_path))

    # save noise data
    np.save(os.path.join(
        noise_save_path, "cylinder_flow_h_data_%dnoise.npy" % (noise_level)), h_data_choose)

    # create datasets
    h_data_choose = h_data_choose.astype(np.float32)
    h_data_validate = h_data_validate.astype(np.float32)
    database_choose = database_choose.astype(np.float32)
    database_validate = database_validate.astype(np.float32)

    if dataset_type == "pinn":
        DatasetResult = namedtuple('DatasetResult',
                                   ['database_choose', 'h_data_choose',
                                    'database_validate', 'h_data_validate'])
        return DatasetResult(Tensor(database_choose), Tensor(h_data_choose),
                             Tensor(database_validate), h_data_validate)

    train_dataset_generator = CreateDataset(database_choose, h_data_choose)
    train_dataset = dataset.GeneratorDataset(
        train_dataset_generator, ["data", "label"], shuffle=True)
    train_dataset = train_dataset.batch(
        batch_size=dataset_config["train_batch"], drop_remainder=True)
    return train_dataset, database_validate, h_data_validate


def create_periodic_hill_dataset(dataset_config, dataset_type="nn"):
    """create dataset with noise for periodic hill flow"""
    # load data
    data = np.load(dataset_config["data_path"])
    data = np.reshape(data, (300, 700, 10)).astype(np.float32)
    data = data[:, :, :8]

    points = data[:, :, :2]
    label = data[:, :, 2:5]

    # choose number of training data
    choose_train = dataset_config["choose_train"]
    # choose number of validation data
    choose_validate = dataset_config["choose_validate"]
    noise_level = dataset_config["noise_level"]

    # produce random data for training and validating
    randam_data_result = random_periodic_hill_data(
        choose_train, choose_validate, points, label)
    h_data_choose, h_data_validate, database_choose, database_validate = randam_data_result

    # for chosen data create noise
    for i in range(choose_train):
        h_data_choose[i] = h_data_choose[i]*(1+0.01*noise_level *
                                             np.random.uniform(-1, 1))
        h_data_validate[i] = h_data_validate[i]*(1+0.01*noise_level *
                                                 np.random.uniform(-1, 1))

    # create noise_data dir
    noise_save_path = dataset_config["noise_save_path"]
    if not os.path.exists(os.path.abspath(noise_save_path)):
        os.makedirs(os.path.abspath(noise_save_path))

    # save noise data
    np.save(os.path.join(
        noise_save_path, "periodic_hill_h_data_%dnoise.npy" % (noise_level)), h_data_choose)

    # create datasets
    h_data_choose = h_data_choose.astype(np.float32)
    h_data_validate = h_data_validate.astype(np.float32)
    database_choose = database_choose.astype(np.float32)
    database_validate = database_validate.astype(np.float32)

    if dataset_type == "pinn":
        DatasetResult = namedtuple('DatasetResult',
                                   ['database_choose', 'h_data_choose',
                                    'database_validate', 'h_data_validate'])
        return DatasetResult(Tensor(database_choose), Tensor(h_data_choose),
                             Tensor(database_validate), h_data_validate)
    train_dataset_generator = CreateDataset(database_choose, h_data_choose)
    train_dataset = dataset.GeneratorDataset(
        train_dataset_generator, ["data", "label"], shuffle=True)
    train_dataset = train_dataset.batch(
        batch_size=dataset_config["train_batch"], drop_remainder=True)
    return train_dataset, database_validate, h_data_validate


def create_dataset(case_name, dataset_config):
    """create dataset for different equations"""
    if case_name == "burgers":
        train_dataset, inputs, label = create_burgers_dataset(
            dataset_config)
    elif case_name == "cylinder_flow":
        train_dataset, inputs, label = create_cylinder_flow_dataset(
            dataset_config)
    else:
        train_dataset, inputs, label = create_periodic_hill_dataset(
            dataset_config)
    return train_dataset, inputs, label


def create_pinn_dataset(case_name, dataset_config):
    """pinn create dataset for different equations"""
    dataset_type = "pinn"
    if case_name == "burgers":
        pinn_dataset = create_burgers_dataset(
            dataset_config, dataset_type)
    elif case_name == "cylinder_flow":
        pinn_dataset = create_cylinder_flow_dataset(
            dataset_config, dataset_type)
    else:
        pinn_dataset = create_periodic_hill_dataset(
            dataset_config, dataset_type)
    return pinn_dataset
