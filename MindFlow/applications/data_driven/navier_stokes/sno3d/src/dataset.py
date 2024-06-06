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

from .sno_utils import interpolate_2d_dataset


def load_interp_data(data_config, dataset_type='train', kind='cubic'):
    """loads and interpolates data on Gauss grid"""
    data_path = data_config['root_dir']
    data_path = os.path.join(data_path, dataset_type)
    inputs = np.load(os.path.join(data_path, dataset_type +"_a.npy"))
    labels = np.load(os.path.join(data_path, dataset_type + "_u.npy"))

    res = data_config['resolution']
    poly_type = data_config['poly_type']

    n_samples = inputs.shape[0]
    inp_channels = inputs.shape[-1]
    out_channels = labels.shape[-1]
    inputs_2d = np.transpose(inputs, (0, 3, 1, 2)).reshape((-1, res, res))
    labels_2d = np.transpose(labels, (0, 3, 1, 2)).reshape((-1, res, res))

    inputs_2d = interpolate_2d_dataset(inputs_2d, poly_type, kind)
    labels_2d = interpolate_2d_dataset(labels_2d, poly_type, kind)

    inputs_2d = inputs_2d.reshape((n_samples, inp_channels, res, res))
    labels_2d = labels_2d.reshape((n_samples, out_channels, res, res))

    if dataset_type == 'test':
        return {'a': inputs_2d, 'u': labels_2d}

    np.save(os.path.join(data_path, "train_a_interp.npy"), inputs_2d)
    np.save(os.path.join(data_path, "train_u_interp.npy"), labels_2d)
    return None


def create_training_dataset(config,
                            shuffle=True,
                            drop_remainder=True):
    """create dataset"""
    data_path = os.path.join(config["root_dir"], 'train')
    input_path = os.path.join(data_path, "train_a_interp.npy")
    label_path = os.path.join(data_path, "train_u_interp.npy")

    ns_3d_data = ExistedDataConfig(name=config["name"],
                                   data_dir=[input_path, label_path],
                                   columns_list=["inputs", "label"],
                                   data_format="npy")
    dataset = Dataset(existed_data_list=[ns_3d_data])
    data_loader = dataset.create_dataset(batch_size=config["batch_size"],
                                         shuffle=shuffle,
                                         drop_remainder=drop_remainder)

    return data_loader
