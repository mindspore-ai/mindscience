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
"""get dataset"""
import os

import numpy as np
import scipy.io as sio
from omegaconf import DictConfig
from mindspore import Tensor
from mindspore.dataset import GeneratorDataset


def get_origin(quanlity_list=None, realpath='./data'):
    """get original data from files"""
    if not quanlity_list:
        quanlity_list = [
            'Static Pressure', 'Static Temperature',
            'Density', 'Vx', 'Vy', 'Vz',
            'Relative Total Temperature',
            'Absolute Total Temperature',
        ]
    field_reader = MatLoader(os.path.join(realpath, 'sampleStruct_128_64_6000'))
    grid = field_reader.read_field('grid')
    fields_list = []
    for quanlity in quanlity_list:
        fields_list.append(field_reader.read_field(quanlity)[..., np.newaxis])
    fields = np.concatenate(fields_list, axis=-1)
    design_reader = MatLoader(os.path.join(realpath, 'designStruct_100_6000'))
    design = design_reader.read_field('design')
    return design, fields, grid


def create_dataloader(inputs, outputs, batch_size=32, shuffle=True):
    """create dataloader with inputs and outputs"""
    dataset = GvrbDataset(inputs, outputs)
    dataloader = GeneratorDataset(
        source=dataset,
        column_names=['design_param', 'field_param'],
        shuffle=shuffle,
        python_multiprocessing=False,
    )
    return dataloader.batch(batch_size)


def load_dataset(config: DictConfig, shuffle=True):
    """Loads a dataset based on the configuration."""
    num_train = config.data.num_samples.train
    num_test = config.data.num_samples.test
    design, fields, _ = get_origin(realpath=config.data.path)
    x_norm = DataNormer(data_type='x_norm')
    input_tensor_train = x_norm.norm(design[:num_train, ...])
    input_tensor_test = x_norm.norm(design[-num_test:, ...])
    y_norm = DataNormer(data_type='y_norm')
    output_tensor_train = y_norm.norm(fields[:num_train, ...])
    output_tensor_test = y_norm.norm(fields[-num_test:, ...])
    data_loader_train = create_dataloader(
        input_tensor_train,
        output_tensor_train,
        batch_size=config.train.batch_size,
        shuffle=shuffle,
    )
    data_loader_test = create_dataloader(
        input_tensor_test,
        output_tensor_test,
        batch_size=config.train.batch_size,
        shuffle=shuffle,
    )
    return data_loader_train, data_loader_test


class MatLoader:
    """load data from .mat file"""
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        self.data = None
        self._load_file()

    def read_field(self, field):
        x = self.data[field]
        return x.astype(np.float32)

    def _load_file(self):
        self.data = sio.loadmat(self.file_path)


class DataNormer:
    """data normalization at last dimension"""
    def __init__(self, data_type, load_path='./data'):
        norm_data = np.load(os.path.join(load_path, 'normalization.npz'))
        self.mean = norm_data[f'{data_type}_mean']
        self.std = norm_data[f'{data_type}_std']
        if data_type == 'x_norm':
            self.mean = np.concatenate((np.full(96, 0.500), self.mean), axis=0)
            self.std = np.concatenate((np.full(96, 0.144), self.std), axis=0)

    def norm(self, x):
        if isinstance(x, Tensor):
            x = (x - Tensor(self.mean, dtype=x.dtype)) / Tensor(self.std + 1e-10, dtype=x.dtype)
        else:
            x = (x - self.mean) / (self.std + 1e-10)
        return x

    def un_norm(self, x):
        if isinstance(x, Tensor):
            x = x * Tensor(self.std + 1e-10, dtype=x.dtype) + Tensor(self.mean, dtype=x.dtype)
        else:
            x = x * (self.std + 1e-10) + self.mean
        return x


class GvrbDataset:
    """Basic Dataset for the gvrb turbine"""
    def __init__(self,
                 inputs,
                 outputs,
                 ) -> None:
        self.len_samples = inputs.shape[0]
        self.inputs = inputs
        self.outputs = outputs

    def __getitem__(self, idx: int):
        inputs, outputs = self.inputs[idx, ...], self.outputs[idx, ...]
        return inputs, outputs

    def __len__(self) -> int:
        return self.len_samples
