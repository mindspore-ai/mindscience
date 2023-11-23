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
"""dataset"""
import os
import numpy as np

from mindflow.data import Dataset, ExistedDataConfig
from mindflow.utils import print_log

EPS = 1e-8


def create_npy(data_params, step=8):
    '''create inputs and label data for trainset and testset'''
    data_path = data_params["root_dir"]
    train_size = data_params["train"]["num_samples"]
    test_size = data_params["test"]["num_samples"]
    s = 2 ** 13 // data_params["step"]

    train_path = os.path.join(data_path, "train")
    test_path = os.path.join(data_path, "test")

    if not os.path.exists(train_path):
        os.makedirs(train_path)
    else:
        print_log("Data preparation finished")
        return
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    inputs = np.load(os.path.join(train_path, "inputs.npy"))
    label = np.load(os.path.join(train_path, "label.npy"))

    x_train = inputs[:train_size, :][:, ::step]
    y_train = label[:train_size, :][:, ::step]
    x_test = inputs[-test_size:, :][:, ::step]
    y_test = label[-test_size:, :][:, ::step]
    x_train = x_train.reshape(train_size, s, 1)
    x_test = x_test.reshape(test_size, s, 1)

    train_input_path = os.path.join(train_path, "inputs.npy")
    train_label_path = os.path.join(train_path, "label.npy")
    test_input_path = os.path.join(test_path, "inputs.npy")
    test_label_path = os.path.join(test_path, "label.npy")

    with os.fdopen(train_input_path, "wb") as f:
        np.save(f, x_train)
    with os.fdopen(train_label_path, "wb") as f:
        np.save(f, y_train)
    with os.fdopen(test_input_path, "wb") as f:
        np.save(f, x_test)
    with os.fdopen(test_label_path, "wb") as f:
        np.save(f, y_test)


def create_training_dataset(data_params, model_params, shuffle=True, drop_remainder=True, is_train=True):
    """create dataset"""
    create_npy(data_params)
    data_path = data_params["root_dir"]
    if is_train:
        train_path = os.path.abspath(os.path.join(data_path, "train"))
        input_path = os.path.join(train_path, "inputs.npy")
        print_log('input_path: ', np.load(input_path).shape)
        label_path = os.path.join(train_path, "label.npy")
        print_log('label_path: ', np.load(label_path).shape)
    else:
        test_path = os.path.abspath(os.path.join(data_path, "test"))
        input_path = os.path.join(test_path, "inputs.npy")
        label_path = os.path.join(test_path, "label.npy")
    burgers_1d_data = ExistedDataConfig(name=data_params["name"],
                                        data_dir=[input_path, label_path],
                                        columns_list=["inputs", "label"],
                                        data_format="npy")
    dataset = Dataset(existed_data_list=[burgers_1d_data])
    data_loader = dataset.create_dataset(batch_size=data_params["batch_size"],
                                         shuffle=shuffle,
                                         drop_remainder=drop_remainder)

    operations = [lambda x, y: (x.reshape(-1, data_params["resolution"],
                                          data_params["t_in"], model_params["out_channels"]),
                                y.reshape(-1, data_params["resolution"],
                                          data_params["t_out"], model_params["out_channels"]))]
    data_loader = data_loader.map(operations, input_columns=["burgers1d_inputs", "burgers1d_label"])
    return data_loader
