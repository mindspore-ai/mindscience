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
"""Loading data and create dataset"""
from collections import namedtuple

import numpy as np
import h5py

import mindspore.dataset as ds


class TrainDatasetSource:
    """
    Indexing the dataset based on data_dir and dataset_list, processing the dataset and returning train_dataset,
    train_velocity, train_ur, valid_dataset, valid_velocity and valid_ur.
    Parameters:
        data_dir: Path address of the dataset
        dataset_list: The train data list:['5.0', '5.5', '6.0', '6.5']
    """

    def __init__(self, data_dir, dataset_list, ratio=0.8):
        self.data_dir = data_dir
        self.dataset_list = dataset_list
        self.ratio = ratio

    def train_data(self):
        """ data for train"""
        train_dataset = []
        valid_dataset = []
        train_velocity = []
        valid_velocity = []
        train_ur = []
        valid_ur = []

        for i in self.dataset_list:
            data_source = h5py.File(f"{self.data_dir}/Ur{i}/total_puv.mat")
            data_sample = data_source['total_puv'][:, :, :, 2:]
            data_sample = np.array(data_sample.transpose([0, 3, 1, 2]), np.float32)

            data_length = data_sample.shape[0]
            train_dataset.append(data_sample[0:int(data_length * self.ratio)])
            valid_dataset.append(data_sample[int(data_length * self.ratio):])

            data_source = h5py.File(f"{self.data_dir}/ur{i}/velocity.mat")
            data_velocity = data_source['velocity'][:]
            data_velocity = np.array(data_velocity, np.float32)

            train_velocity.append(data_velocity[0:int(data_length * self.ratio)])
            valid_velocity.append(data_velocity[int(data_length * self.ratio):])

            ur = np.array(float(i), np.float32)
            ur_01 = ur / 10.0
            data_ur = ur_01 * np.ones(data_length, dtype=np.float32)

            train_ur.append(data_ur[0:int(data_length * self.ratio)])
            valid_ur.append(data_ur[int(data_length * self.ratio):])

        DatasetResult = namedtuple('DatasetResult',
                                   ['train_dataset', 'train_velocity', 'train_ur', 'valid_dataset', 'valid_velocity',
                                    'valid_ur'])

        return DatasetResult(train_dataset, train_velocity, train_ur, valid_dataset, valid_velocity, valid_ur)


class TrainDatasetMake:
    """
    According dataset, ur and time_steps to make train dataset so that retrieve data based on index.
    Parameters:
        dataset: Train data and valid data
        velocity: The speed of the moving structure
        ur: Calculation conditions used as frequency
        time_steps: The number of time steps to predict
    """

    def __init__(self, dataset, velocity, ur, time_steps, dataset_list):
        self.dataset = dataset
        self.velocity = velocity
        self.ur = ur
        self.time_steps = time_steps
        self.dataset_numbers = len(dataset_list)

    def __len__(self):
        return (len(self.dataset[0]) - 2 * self.time_steps) * self.dataset_numbers

    def __getitem__(self, idx):
        idx_dataset = idx // (len(self.dataset[0]) - 2 * self.time_steps)
        idx = idx % (len(self.dataset[0]) - 2 * self.time_steps)

        train_input = self.dataset[idx_dataset][idx:idx + 2 * self.time_steps:2]
        train_velocity = self.velocity[idx_dataset][idx + 2 * (self.time_steps - 1)]
        train_ur = self.ur[idx_dataset][idx + 2 * (self.time_steps - 1)]
        train_label = self.dataset[idx_dataset][idx + 2 * self.time_steps]

        TrainDatasetResult = namedtuple('TrainDatasetResult',
                                        ['train_input', 'train_velocity', 'train_ur', 'train_label'])

        return TrainDatasetResult(train_input, train_velocity, train_ur, train_label)


def generate_dataset(data_dir, time_steps, dataset_list):
    """According data_dir, time_steps and dataset_list to process and generate train_dataset, valid_dataset"""
    train_data, train_velocity, train_ur, valid_data, valid_velocity, valid_ur = TrainDatasetSource \
        (data_dir, dataset_list).train_data()

    train_dataset = TrainDatasetMake(train_data, train_velocity, train_ur, time_steps, dataset_list)
    train_dataset = ds.GeneratorDataset(train_dataset, ["inputs", "v", "ur", "labels"], shuffle=True)
    train_dataset = train_dataset.batch(batch_size=16, drop_remainder=True)

    valid_dataset = TrainDatasetMake(valid_data, valid_velocity, valid_ur, time_steps, dataset_list)
    valid_dataset = ds.GeneratorDataset(valid_dataset, ["inputs", "v", "ur", "labels"], shuffle=False)
    valid_dataset = valid_dataset.batch(batch_size=16, drop_remainder=True)

    return train_dataset, valid_dataset


class TestDatasetMake:
    """
    According dataset, velocity, ur and time_steps to make dataset so that retrieve data based on index.
    Parameters:
        dataset: Train data and valid data
        velocity: The speed of the moving structure
        ur: Calculation conditions used as frequency
        time_steps: The number of time steps to predict
    """

    def __init__(self, dataset, velocity, ur, time_steps):
        self.dataset = dataset
        self.velocity = velocity
        self.ur = ur
        self.time_steps = time_steps

    def __len__(self):
        return (len(self.dataset) - 2 * self.time_steps) // 2

    def __getitem__(self, idx):
        test_input = self.dataset[2 * idx:2 * idx + 2 * self.time_steps:2]
        test_velocity = self.velocity[2 * idx + 2 * (self.time_steps - 1)]
        test_ur = self.ur[2 * idx + 2 * (self.time_steps - 1)]
        test_label = self.dataset[2 * idx + 2 * self.time_steps]

        TestDatasetResult = namedtuple('TestDatasetResult',
                                       ['test_input', 'test_velocity', 'test_ur', 'test_label'])

        return TestDatasetResult(test_input, test_velocity, test_ur, test_label)


def my_test_dataset(data_dir, time_steps, dataset_list):
    """According data_dir, time_steps and time_steps to process and generate test_dataset"""
    data_source = h5py.File(f"{data_dir}/ur{dataset_list[0]}/total_puv.mat")
    data_sample = data_source['total_puv'][800:2000, :, :, :]
    test_data = np.array(data_sample[:, :, :, 2:].transpose([0, 3, 1, 2]), np.float32)

    surf_xy = data_sample[:, :, 0, 0:2]

    data_source = h5py.File(f"{data_dir}/ur{dataset_list[0]}/velocity.mat")
    data_sample = data_source['velocity'][800:2000]
    test_velocity = np.array(data_sample, np.float32)

    data_length = test_data.shape[0]

    # normalize in the 0-10 range
    ur = float(dataset_list[0])
    ur_01 = ur / 10.0
    test_ur = ur_01 * np.ones(data_length, dtype=np.float32)

    test_dataset = TestDatasetMake(test_data, test_velocity, test_ur, time_steps)

    return test_dataset, surf_xy
