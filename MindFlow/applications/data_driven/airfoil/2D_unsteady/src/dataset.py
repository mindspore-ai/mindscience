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
"""
dataset
"""
import os

import numpy as np
import mindspore.dataset as ds


def init_dataset(data_params):
    """initial_dataset"""
    dataset = DataSet(data_params)
    train_dataset, test_dataset, means, stds = dataset.create_dataset()
    return train_dataset, test_dataset, means, stds


class UnsteadyAirfoilData:
    """UnsteadyAirfoilData"""

    def __init__(self, data, t_in, t_out):
        xx = []
        yy = []
        for i in range(len(data) - t_in - t_out):
            x_temp = data[i:i + t_in, ...]
            y_temp = data[i + t_in:(i + t_in + t_out), ...]
            xx.append(x_temp)
            yy.append(y_temp)
        self.inputs = np.squeeze(np.stack(xx, axis=0))  # (train_size, t_in, H, W, C)
        self.labels = np.squeeze(np.stack(yy, axis=0))  # (train_size, t_out, H, W, C)

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
        self.airfoil_data = np.load(os.path.join(self.data_path, self.dataset_name))['arr_0'][..., -3:].astype(
            'float32')[:data_params['data_size'], ...]  # (t,h,w,c )
        self.dataset_generator = UnsteadyAirfoilData(self.airfoil_data[:, ::1, :, :], t_in=self.t_in, t_out=self.t_out)
        self.mean_inputs = np.array(data_params['means']).astype('float32')
        self.std_inputs = np.array(data_params['stds']).astype('float32')
        self.data_size = data_params['data_size']
        self.train_size = data_params['train_size']

    def create_dataset(self, drop_remainder=True):
        """create dataset"""
        dataset = ds.GeneratorDataset(self.dataset_generator, ["inputs", "labels"], shuffle=False)
        train_ds, test_ds = dataset.split([self.train_size / self.data_size, 1 - self.train_size / self.data_size],
                                          randomize=False)
        data_set_norm_train = train_ds.shuffle(self.train_batch_size * 4).map(operations=self.process_fn,
                                                                              input_columns=["inputs"])
        data_set_norm_test = test_ds.map(operations=self.process_fn, input_columns=["inputs"])
        print("train_batch_size : {}".format(self.train_batch_size))
        print("train dataset size: {}".format(data_set_norm_train.get_dataset_size()))
        print("test dataset size: {}".format(data_set_norm_test.get_dataset_size()))

        data_set_batch_train = data_set_norm_train.batch(self.train_batch_size, drop_remainder=drop_remainder)
        data_set_batch_test = data_set_norm_test.batch(self.test_batch_size, drop_remainder=drop_remainder)
        print("train batch dataset size: {}".format(data_set_batch_train.get_dataset_size()))
        print("test batch dataset size: {}".format(data_set_batch_test.get_dataset_size()))
        return data_set_batch_train, data_set_batch_test, self.mean_inputs, self.std_inputs

    def process_fn(self, inputs):
        return (inputs - self.mean_inputs) / (self.std_inputs + 1e-10)
