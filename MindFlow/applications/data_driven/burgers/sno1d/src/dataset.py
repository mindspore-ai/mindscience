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
"""dataset"""
import os

import numpy as np
from mindflow.data import Dataset, ExistedDataConfig
from mindflow.cell import interpolate_1d_dataset


def load_interp_data(data_config, dataset_type='train', kind='cubic'):
    """loads and interpolates data on Gauss grid"""
    data_path = data_config['root_dir']
    data_path = os.path.join(data_path, dataset_type)
    inputs = np.load(os.path.join(data_path, "inputs.npy"))
    labels = np.load(os.path.join(data_path, "label.npy"))

    shape = list(inputs.shape)
    new_shape = shape[:-1]
    inputs = inputs.reshape(new_shape)

    poly_type = data_config['poly_type']
    inputs = interpolate_1d_dataset(inputs, poly_type, kind)
    labels = interpolate_1d_dataset(labels, poly_type, kind)

    if dataset_type == 'test':
        inputs = np.expand_dims(inputs, 1)
        labels = np.expand_dims(labels, 1)
        return {'test_inputs': inputs, 'test_labels': labels}

    inputs = np.expand_dims(np.expand_dims(inputs, 1), 1)
    labels = np.expand_dims(np.expand_dims(labels, 1), 1)

    np.save(os.path.join(data_path, "inputs_interp.npy"), inputs)
    np.save(os.path.join(data_path, "labels_interp.npy"), labels)
    return None


def create_training_dataset(data_config,
                            shuffle=True,
                            drop_remainder=True):
    """creates training dataset"""
    data_path = data_config["root_dir"]

    train_path = os.path.join(data_path, "train")
    input_path = os.path.join(train_path, "inputs_interp.npy")
    label_path = os.path.join(train_path, "labels_interp.npy")

    burgers_1d_data = ExistedDataConfig(name=data_config["name"],
                                        data_dir=[input_path, label_path],
                                        columns_list=["inputs", "label"],
                                        data_format="npy")
    dataset = Dataset(existed_data_list=[burgers_1d_data])
    data_loader = dataset.create_dataset(batch_size=data_config["batch_size"],
                                         shuffle=shuffle,
                                         drop_remainder=drop_remainder)

    return data_loader
