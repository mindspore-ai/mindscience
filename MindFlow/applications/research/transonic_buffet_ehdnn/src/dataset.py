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
import numpy as np
import h5py

import mindspore.dataset as ds


class DataSource:
    """
    Load process of data samples for dataset generation

    actual angle of attack : 3.3 degrees / 3.4 degrees / 3.5 degrees /
                      3.6 degrees / 3.7 degrees / 3.8 degrees / 3.9 degrees

    Args:
        data_url (str): The download address of the data sample
        data_length (int): The length of the intercepted data
        aoa_list (list): The value of angle of attack (the value is 10 times the actual angle of attack)
                   (33 ,34 , 35 , 36 , 37 , 38 , 39)

    Outputs:
        data_sample_list (list), the list of array of data sample for train and the array of data sample for prediction
            the array of data sample shape=(t, C, H, W) :(data_length,3,200,200)

    Examples:
        >>> import numpy as np
        >>> import h5py
        >>> data_dir = "./dataset"
        >>> data_length = 2000
        >>> aoa_list = [35]
        >>> datasource = DataSource(data_url,data_length,aoa_list)
        >>> data_sample_list = datasource.train_data()
        >>> print(data_sample_list[0].shape)
        (2000,3,200,200)

    """

    def __init__(self,
                 data_dir,
                 data_length,
                 aoa_list):
        self.data_dir = data_dir
        self.data_length = data_length
        self.aoa_list = aoa_list

    def train_data(self):
        """ data for train"""
        data_sample_list = []
        for aoa in self.aoa_list:
            data_source = h5py.File(f"{self.data_dir}/total{aoa}_puv.mat")
            data_sample = data_source['total'][:self.data_length]
            data_sample = np.array(data_sample.transpose([0, 3, 1, 2]), np.float32)
            data_sample_list.append(data_sample)
        return data_sample_list

    def prediction_data(self):
        """data for prediction"""
        data_source = h5py.File(f"{self.data_dir}/total{self.aoa_list[0]}_puv.mat")
        data_sample = data_source['total'][self.data_length:self.data_length + 100]
        data_sample = data_sample.transpose([0, 3, 1, 2])
        return np.array(data_sample, np.float32)


class DatasetMake:
    """Data set making"""

    def __init__(self, data_sample_list, x_length=16, y_length=1):
        xx = []
        yy = []
        for step in range(len(data_sample_list[0]) - x_length - y_length):
            for data in data_sample_list:
                x_temp = data[step: step + x_length, :, :, :]
                y_temp = data[step + x_length: step + x_length + y_length, :, :, :]
                xx.append(x_temp)
                yy.append(y_temp)
        self.input = np.squeeze(np.stack(xx, axis=0))
        self.label = np.expand_dims(np.squeeze(np.stack(yy, axis=0)), 1)
        if y_length > 1:
            self.label = np.squeeze(np.stack(yy, axis=0))

        print("input shape", self.input.shape)
        print("label shape", self.label.shape)

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        return self.input[idx], self.label[idx]


def create_dataset(data_dir, data_length, train_ratio, aoa_list):
    """Create dataset for train and evaluation"""
    data_set = DatasetMake(DataSource(data_dir, data_length, aoa_list).train_data())
    dataset = ds.GeneratorDataset(data_set, ["inputs", "labels"], shuffle=False)
    dataset_train, dataset_eval = dataset.split([train_ratio, 1 - train_ratio], randomize=False)
    return dataset_train, dataset_eval
