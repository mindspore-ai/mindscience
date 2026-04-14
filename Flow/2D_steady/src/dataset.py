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
# ==============================================================================
"""
dataset
"""
import os
import math
import numpy as np
import mindspore.dataset as ds

from mindflow.utils import print_log

np.random.seed(0)
ds.config.set_seed(0)


class AirfoilDataset:
    """
    airfoil 2D-steady problem based on ViT
    Supported Platforms:
        ``Ascend`` ``GPU``

    """

    def __init__(self, max_value, min_value):
        self.max_value_list = max_value
        self.min_value_list = min_value

    def create_dataset(self,
                       dataset_dir,
                       train_file_name,
                       test_file_name,
                       finetune_file_name,
                       finetune_size=0.2,
                       mode="train",
                       batch_size=8,
                       shuffle=False,
                       drop_remainder=False):
        """
        create dataset
        """
        train_dataset_path = os.path.join(dataset_dir, train_file_name)
        test_dataset_path = os.path.join(dataset_dir, test_file_name)
        finetune_dataset_path = os.path.join(dataset_dir, finetune_file_name)
        if mode == "finetune":
            dataset = ds.MindDataset(dataset_files=finetune_dataset_path, shuffle=shuffle)
            train_dataset, test_dataset = dataset.split([finetune_size, 1 - finetune_size])
        else:
            train_dataset = ds.MindDataset(dataset_files=train_dataset_path, shuffle=shuffle)
            test_dataset = ds.MindDataset(dataset_files=test_dataset_path, shuffle=False)
        train_dataset = train_dataset.shuffle(batch_size * 4)

        train_dataset_norm = train_dataset.map(operations=self._process_fn,
                                               input_columns=["inputs"])
        test_dataset_norm = test_dataset.map(operations=self._process_fn,
                                             input_columns=["inputs"])

        print_log("train dataset size: {}".format(train_dataset_norm.get_dataset_size()))
        print_log("test dataset size: {}".format(test_dataset_norm.get_dataset_size()))

        train_dataset_batch = train_dataset_norm.batch(batch_size, drop_remainder)
        test_dataset_batch = test_dataset_norm.batch(batch_size, drop_remainder)
        print_log("train batch : {}".format(train_dataset_batch.get_dataset_size()))
        print_log("test batch : {}".format(test_dataset_batch.get_dataset_size()))
        return train_dataset_batch, test_dataset_batch

    def _process_fn(self, data):
        """
        preprocess data
        """
        _, h, w = data.shape
        x = np.linspace(0, 5, h)
        scale_fn = np.array([math.exp(-x0) for x0 in x])
        scale = np.repeat(scale_fn[:, np.newaxis], w, axis=1)

        aoa = data[0:1, ...]
        x = data[1:2, ...]
        y = data[2:3, ...]

        scaled_x = x * scale
        scaled_y = y * scale
        data = np.vstack((aoa, scaled_x, scaled_y))
        eps = 1e-8

        for i in range(0, data.shape[0]):
            max_value, min_value = self.max_value_list[i], self.min_value_list[i]
            data[i, :, :] = (data[i, :, :] - min_value) / (max_value - min_value + eps)
        return data.astype('float32')
