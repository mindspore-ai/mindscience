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
    train_velocity, valid_dataset, valid_velocity
    Parameters:
        data_dir: Path address of the dataset
        dataset_list: The train data list:['0.00', '0.25', '0.35', '0.45']
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
        for i in self.dataset_list:
            data_source = h5py.File(f"{self.data_dir}/f0.90h{i}/project/total_puv_project.mat")
            data_sample = data_source['total_puv'][:]
            data_sample = np.array(data_sample.transpose([0, 3, 1, 2]), np.float32)

            data_length = data_sample.shape[0]
            train_dataset.append(data_sample[0:int(data_length * self.ratio)])
            valid_dataset.append(data_sample[int(data_length * self.ratio):])

            data_source = h5py.File(f"{self.data_dir}/f0.90h{i}/project/velocity.mat")
            data_velocity = data_source['velocity'][:]
            data_velocity = np.array(data_velocity, np.float32)

            train_velocity.append(data_velocity[0:int(data_length * self.ratio)])
            valid_velocity.append(data_velocity[int(data_length * self.ratio):])

        DatasetResult = namedtuple('DatasetResult',
                                   ['train_dataset', 'train_velocity', 'valid_dataset', 'valid_velocity'])

        return DatasetResult(train_dataset, train_velocity, valid_dataset, valid_velocity)


class TrainDatasetMake:
    """
    According dataset, velocity, time_steps and dataset_list to make train dataset so that retrieve data based on index.
    Parameters:
        dataset: Train data and valid data
        velocity: The speed of the moving structure
        time_steps: The number of time steps to predict
        dataset_list: The data list
    """

    def __init__(self, dataset, velocity, time_steps, dataset_list):
        self.dataset = dataset
        self.velocity = velocity
        self.time_steps = time_steps
        self.dataset_numbers = len(dataset_list)

    def __len__(self):
        return (len(self.dataset[0]) - self.time_steps) * self.dataset_numbers

    def __getitem__(self, idx):
        idx_dataset = idx // (len(self.dataset[0]) - self.time_steps)
        idx = idx % (len(self.dataset[0]) - self.time_steps)

        return self.dataset[idx_dataset][idx:idx + self.time_steps], \
               self.velocity[idx_dataset][idx:idx + self.time_steps], \
               self.dataset[idx_dataset][idx + self.time_steps]


def my_train_dataset(data_dir, time_steps, dataset_list):
    """According data_dir, time_steps and dataset_list to process and generate train_dataset, valid_dataset"""
    train_data, train_velocity, valid_data, valid_velocity = TrainDatasetSource(data_dir, dataset_list).train_data()

    train_dataset = TrainDatasetMake(train_data, train_velocity, time_steps, dataset_list)
    train_dataset = ds.GeneratorDataset(train_dataset, ["inputs", "v", "labels"], shuffle=True)
    train_dataset = train_dataset.batch(batch_size=16, drop_remainder=True)

    valid_dataset = TrainDatasetMake(valid_data, valid_velocity, time_steps, dataset_list)
    valid_dataset = ds.GeneratorDataset(valid_dataset, ["inputs", "v", "labels"], shuffle=False)
    valid_dataset = valid_dataset.batch(batch_size=16, drop_remainder=True)

    return train_dataset, valid_dataset


class TestDatasetMake:
    """
    According dataset, velocity, matrix_01 and time_steps to make dataset so that retrieve data based on index.
    Parameters:
        dataset: Train data and valid data
        velocity: The speed of the moving structure
        matrix_01: The matrix of test data, 4-D logical. Each element is a Boolean value
        time_steps: The number of time steps to predict
    """

    def __init__(self, dataset, velocity, matrix_01, time_steps):
        self.dataset = dataset
        self.velocity = velocity
        self.matrix_01 = matrix_01
        self.time_steps = time_steps

    def __len__(self):
        return len(self.dataset) - self.time_steps

    def __getitem__(self, idx):
        test_input = self.dataset[idx:idx + self.time_steps]
        test_velocity = self.velocity[idx:idx + self.time_steps]
        test_label = self.dataset[idx + self.time_steps]
        test_matrix_01 = self.matrix_01[idx + self.time_steps]

        TestDatasetResult = namedtuple('TestDatasetResult',
                                       ['test_input', 'test_velocity', 'test_label', 'test_matrix_01'])

        return TestDatasetResult(test_input, test_velocity, test_label, test_matrix_01)


def my_test_dataset(data_dir, time_steps):
    """According data_dir, time_steps and time_steps to process and generate test_dataset"""
    data_source = h5py.File(f"{data_dir}/project/total_puv_project.mat")
    data_sample = data_source['total_puv'][0:10]
    test_data = np.array(data_sample.transpose([0, 3, 1, 2]), np.float32)

    data_source = h5py.File(f"{data_dir}/project/velocity.mat")
    data_sample = data_source['velocity'][0:10]
    test_velocity = np.array(data_sample, np.float32)

    data_source = h5py.File(f"{data_dir}/project/Matrix_01.mat")
    data_sample = data_source['Matrix'][0:10]
    data_matrix_01 = np.array(data_sample.transpose([0, 3, 1, 2]), np.float32)

    test_dataset = TestDatasetMake(test_data, test_velocity, data_matrix_01, time_steps)
    test_dataset = ds.GeneratorDataset(test_dataset, ["input", "velocity", "label", "matrix_01"], shuffle=False)
    test_dataset = test_dataset.batch(batch_size=1, drop_remainder=True)

    return test_dataset
