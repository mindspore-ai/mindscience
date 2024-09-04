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
"""Dataset setting"""
import os

import numpy as np
import scipy.io as sio
from omegaconf import DictConfig
from mindspore import Tensor
from mindspore.dataset import GeneratorDataset


def get_origin(data_type: str = 'train', hole_num: int = 1, realpath: str = './data'):
    """get original data from files"""
    sample_file = os.path.join(realpath, 'pakb_' + str(hole_num) + '_hole_' + data_type)
    design = MatLoader(sample_file).read_field("sdf")
    fields = MatLoader(sample_file).read_field("Temperature")
    return design, fields


def get_grid(realpath: str = './data'):
    """get_grid"""
    sample_file = os.path.join(realpath, 'pakb_1_hole_test')
    grids_x = MatLoader(sample_file).read_field("Grids_x")
    grids_y = MatLoader(sample_file).read_field("Grids_y")
    return np.concatenate((grids_x[:, :, np.newaxis], grids_y[:, :, np.newaxis]), axis=-1)


def create_dataloader(inputs, outputs, batch_size=32, shuffle=True):
    """create dataloader with inputs and outputs"""
    dataset = PakbDataset(inputs, outputs)
    dataloader = GeneratorDataset(
        source=dataset,
        column_names=['design_param', 'field_param'],
        shuffle=shuffle,
        python_multiprocessing=False,
        )
    return dataloader.batch(batch_size)


def load_dataset(config: DictConfig):
    """Loads a dataset based on the configuration."""
    train_input_list, train_output_list, test_input_list, test_output_list = [], [], [], []
    num_train = config.data.num_samples.train
    for hole_num in config.train.hole_num_set:
        design, fields = get_origin(data_type='train', hole_num=hole_num, realpath=config.data.path)
        train_input_list.append(design[:num_train].copy())
        train_output_list.append(fields[:num_train].copy())
    num_test = config.data.num_samples.test
    for hole_num in config.test.hole_num_set:
        design, fields = get_origin(data_type='test', hole_num=hole_num, realpath=config.data.path)
        test_input_list.append(design[:num_test].copy())
        test_output_list.append(fields[:num_test].copy())
    train_loader = combine_dataset(train_input_list, train_output_list,
                                   batch_size=config.train.batch_size,
                                   combine_list=True)
    test_loader_list = combine_dataset(test_input_list, test_output_list,
                                       batch_size=config.test.batch_size,
                                       combine_list=False,
                                       padding=False)
    return train_loader, test_loader_list


def combine_dataset(data_x_list, data_y_list,
                    combine_list=True, batch_size=32,
                    channel_num=10, padding=True):
    """Generates a data loader from input data lists with optional normalization and padding."""
    data_x_all = []
    x_norm = DataNormer(data_type='x_norm')
    y_norm = DataNormer(data_type='y_norm')
    if padding:
        for data_x in data_x_list:
            data_x_all.append(padding_data(data_x, channel_num=channel_num))
        data_x_list = data_x_all
    if combine_list:
        data_x = x_norm.norm(np.concatenate(data_x_list, axis=0))
        data_y = y_norm.norm(np.concatenate(data_y_list, axis=0))
        data_loader = create_dataloader(data_x, data_y, batch_size=batch_size)
    else:
        data_loader = [create_dataloader(x_norm.norm(train_x), y_norm.norm(train_y), batch_size=batch_size)
                       for train_x, train_y in zip(data_x_list, data_y_list)]
    return data_loader


def padding_data(data: np.ndarray, const=350, channel_num=10, shuffle=True) -> np.ndarray:
    """data_padding"""
    current_channel_num = data.shape[-1]
    inputs = np.zeros([*data.shape[:-1], channel_num], dtype=data.dtype) + const
    inputs[..., :current_channel_num] = data
    if shuffle:
        for i in range(inputs.shape[0]):
            idx = np.random.permutation(inputs.shape[-1])
            inputs[i] = inputs[i, :, :, idx].transpose(1, 2, 0)
    return inputs.astype(data.dtype)


class MatLoader:
    """load data from .mat file"""
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        self.data = None
        self._load_file()

    def read_field(self, field):
        xx_inputs = self.data[field]
        return xx_inputs.astype(np.float32)

    def _load_file(self):
        self.data = sio.loadmat(self.file_path)


class DataNormer:
    """data normalization at last dimension"""
    def __init__(self, data_type):
        if data_type == 'x_norm':
            mean, std = 280, 116
        elif data_type == 'y_norm':
            mean, std = 321, 1.9
        self.mean = np.array(mean)
        self.std = np.array(std)

    def norm(self, xx):
        if isinstance(xx, Tensor):
            return (xx - Tensor(self.mean, dtype=xx.dtype)) / Tensor(self.std + 1e-10, dtype=xx.dtype)
        return (xx - self.mean) / (self.std + 1e-10)

    def un_norm(self, xx):
        if isinstance(xx, Tensor):
            return xx * Tensor(self.std + 1e-10, dtype=xx.dtype) + Tensor(self.mean, dtype=xx.dtype)
        return xx * (self.std + 1e-10) + self.mean


class PakbDataset:
    """Basic_Dataset"""
    def __init__(self, inputs, outputs) -> None:
        self.len_samples = inputs.shape[0]
        self.inputs = inputs
        self.outputs = outputs

    def __getitem__(self, idx: int):
        inputs, outputs = self.inputs[idx], self.outputs[idx]
        return inputs, outputs

    def __len__(self) -> int:
        return self.len_samples
