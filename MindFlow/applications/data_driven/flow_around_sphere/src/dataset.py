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
"""load data and create dataset"""
import os
import numpy as np

import mindspore.dataset as ds

from .utils import max_min_normalize


class FlowDataset:
    """
    The iterable Python object for GeneratorDataset.

    Args:
        data(ndarray): flow snapshots. shape: [T, C, D, H, W].
        t_in(int): Number of historical time steps as input. Default: 1.
        t_out(int): Number of historical time steps as out. Default: 1.
        skip(int): Number of sampling intervals for flow field snapshots. Default: 1.
        scale(float): The magnification of the flow field label. Default: 1000.
        residual(bool): If true, the target of the prediction is the change between the future and the current moment.
                        Default: ``True``.
    """

    def __init__(self, data, t_in=1, t_out=1, skip=1, residual=True, scale=1000.0):
        self.data = data
        self.t_in = t_in
        self.t_out = t_out
        self.skip = skip
        self.scale = scale
        self.residual = residual

    def __len__(self):
        return len(self.data) - (self.t_in + self.t_out - 1) * self.skip

    def __getitem__(self, index):
        inputs = self.data[index:index + self.t_in * self.skip:self.skip]
        labels = self.scale * (self.data[index + self.t_in * self.skip:index + (self.t_in + self.t_out) * self.skip:
                                         self.skip] -
                               self.residual * self.data[index:index + self.t_out * self.skip:self.skip])
        return inputs, labels


def create_dataset(data_params, is_train=False, is_eval=False, norm=False, is_infer=False, residual=True, scale=1000.0):
    """
    Create dataset for train, eval, or infer according to the config file.
    Check if the dataset exists, and if it does not, divide and shuffle original ndarray according to requirements
    such as normalization and proportion

    Args:
        data_params(dict): para dictionary includes data_path, norm, train_radio, and so on.
        is_train(bool): If True, generate training dataset. Default: `False`.
        is_eval(bool): If True, generate eval dataset. Default: `False`.
        is_infer(bool): If True, generate infer dataset. Default: `False`.
        norm(bool): If True, the data needs to be normalized. Default: `False`.
        residual(bool): Whether to perform indirect prediction. Default: `True`. If True, the label data is the
                        difference between the future flow field and the current flow field.
        scale(float): The zoom factor of the label data. Default: 1000.0.

    Return:
        data_loader(GeneratorDataset): A source dataset that generates data.
    """
    file_name = is_train * 'train_data' + is_eval * 'eval_data' + is_infer * 'infer_data' + norm * f'_norm' + '.npy'
    file = os.path.join(data_params['data_path'], file_name)
    if os.path.exists(file):
        data = np.load(file).astype(np.float32)
    else:
        original_data = np.load(os.path.join(data_params['data_path'],
                                             f'original_data.npy')).transpose((0, 1, 4, 2, 3)).astype(np.float32)

        length = original_data.shape[0]
        train_radio = data_params['train_radio']
        eval_radio = data_params['eval_radio']

        if norm:
            # Time averaging and maximum normalization
            train_original_data = original_data[:int(train_radio * length), ...]
            original_data -= np.mean(train_original_data, axis=0)
            original_data = max_min_normalize(train_original_data, original_data)

        if is_train:
            data = original_data[:int(train_radio * length), ...]
        elif is_eval:
            data = original_data[int(train_radio * length):int((train_radio + eval_radio) * length), ...]
        else:
            data = original_data[int((train_radio + eval_radio) * length):, ...]
        np.save(file, data)

    data_loader = ds.GeneratorDataset(
        source=FlowDataset(data, t_in=data_params['t_in'], t_out=data_params['t_out'], skip=data_params['skip'],
                           residual=residual, scale=scale),
        column_names=['prev', 'target'], num_parallel_workers=1, shuffle=is_train)
    return data_loader
