# Copyright 2024 Huawei Technologies Co., Ltd
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
"""
dataset
"""
import os

import numpy as np
import mindspore.dataset as ds


def init_dataset(data_params):
    """initial_dataset"""
    dataset = DataSet(data_params)
    print("initialing dataset")
    train_dataset, test_dataset, means, stds = dataset.create_dataset()
    return train_dataset, test_dataset, means, stds


class HeatConductionData:
    """HeatConductionData"""

    def __init__(self, data_path, t_in, t_out):
        self.t_in = t_in
        self.t_out = t_out
        input_list = []
        target_list = []
        files = os.listdir(data_path)
        for file_name in files:
            if file_name.endswith(".npz"):
                file_path = os.path.join(data_path, file_name)
                data = np.load(file_path)
                input_data = data['a'][0:2].astype(np.float32)
                target_data = data['a'][2:3].astype(np.float32)
                input_transposed = np.transpose(input_data, (1, 2, 0))
                target_transposed = np.transpose(target_data, (1, 2, 0))
                input_list.append(input_transposed)
                target_list.append(target_transposed)
        self.inputs = np.array(input_list).astype(np.float32)
        self.labels = np.array(target_list).astype(np.float32)

        print("input size", self.inputs.shape)
        print("label size", self.labels.shape)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]


class DataSet:
    """DataSet"""

    def __init__(self, data_params):
        self.data_path = data_params['root_dir']
        self.dataset_name = data_params['dataset_name']
        self.t_in = data_params['T_in']
        self.t_out = data_params['T_out']
        self.train_batch_size = data_params['train_batch_size']
        self.test_batch_size = data_params['test_batch_size']

        self.dataset_generator = HeatConductionData(self.data_path, t_in=self.t_in, t_out=self.t_out)
        self.mean_inputs = np.array(data_params['means']).astype('float32')
        self.std_inputs = np.array(data_params['stds']).astype('float32')
        self.data_size = data_params['data_size']
        self.train_size = data_params['train_size']

    def create_dataset(self, drop_remainder=True):
        """create dataset"""
        dataset = ds.GeneratorDataset(self.dataset_generator, ["inputs", "labels"], shuffle=False)
        train_ds, test_ds = dataset.split([self.train_size / self.data_size, 1 - self.train_size / self.data_size],
                                          randomize=False)

        print("train_batch_size : {}".format(self.train_batch_size))

        data_set_batch_train = train_ds.batch(self.train_batch_size, drop_remainder=drop_remainder)
        data_set_batch_test = test_ds.batch(self.test_batch_size, drop_remainder=drop_remainder)
        print("train batch dataset size: {}".format(data_set_batch_train.get_dataset_size()))
        print("test batch dataset size: {}".format(data_set_batch_test.get_dataset_size()))
        return data_set_batch_train, data_set_batch_test, self.mean_inputs, self.std_inputs

    def process_fn(self, inputs):
        return (inputs - self.mean_inputs) / (self.std_inputs + 1e-10)
