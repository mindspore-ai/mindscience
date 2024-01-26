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
"""Module providing precipitation dataset class"""
import os

import numpy as np

from mindearth.utils import get_datapath_from_date
from mindearth.data.dataset import Era5Data


class Era5DataTp(Era5Data):
    """
    Self-defined class for processing ERA5 dataset with total precipitation label, inherited from `Era5Data`.

    Args:
        data_params (dict): dataset-related configuration of the model.
        run_mode (str, optional): whether the dataset is used for training, evaluation or testing. Supports [“train”,
            “test”, “valid”]. Default: 'train'.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """
    def __init__(self, data_params, run_mode='train'):
        super().__init__(data_params, run_mode=run_mode)
        self.pressure_level = data_params.get("pressure_level_num", 13)
        self.tp_dir = data_params.get("tp_dir", "")
        if run_mode == 'train':
            self.train_len = self._get_file_count(self.train_dir, self.train_period)
        elif run_mode == 'valid':
            self.valid_len = self._get_file_count(self.valid_dir, self.valid_period)
        else:
            self.test_len = self._get_file_count(self.test_dir, self.test_period)

    def __len__(self):
        if self.run_mode == 'train':
            length = (self.train_len * self.data_frequency -
                      (self.t_out + self.t_in) * self.pred_lead_time) // self.train_interval

        elif self.run_mode == 'valid':
            length = (self.valid_len * self.data_frequency -
                      (self.t_out + self.t_in) * self.pred_lead_time) // self.valid_interval

        else:
            length = (self.test_len * self.data_frequency -
                      (self.t_out + self.t_in) * self.pred_lead_time) // self.test_interval
        return length

    def __getitem__(self, idx):
        inputs_lst = []
        inputs_surface_lst = []
        label_lst = []
        idx = idx * self.interval

        for t in range(self.t_in):
            cur_input_data_idx = idx + t * self.pred_lead_time
            input_date, year_name = get_datapath_from_date(self.start_date, cur_input_data_idx.item())
            x = np.load(os.path.join(self.path, input_date)).astype(np.float32)
            x_surface = np.load(os.path.join(self.surface_path, input_date)).astype(np.float32)
            x_static = np.load(os.path.join(self.static_path, year_name)).astype(np.float32)
            x_surface_static = np.load(os.path.join(self.static_surface_path, year_name)).astype(np.float32)
            x = self._get_origin_data(x, x_static)
            x_surface = self._get_origin_data(x_surface, x_surface_static)
            x, x_surface = self.normalize(x, x_surface)
            inputs_lst.append(x)
            inputs_surface_lst.append(x_surface)

        for t in range(self.t_out):
            cur_label_data_idx = idx + (self.t_in + t) * self.pred_lead_time
            label_date, year_name = get_datapath_from_date(self.start_date, cur_label_data_idx.item())
            label = np.load(os.path.join(self.tp_dir, label_date)).astype(np.float32)
            label_lst.append(label)

        x = np.squeeze(np.stack(inputs_lst, axis=0), axis=1).astype(np.float32)
        x_surface = np.squeeze(np.stack(inputs_surface_lst, axis=0), axis=1).astype(np.float32)
        label = np.stack(label_lst, axis=0).astype(np.float32)
        _, _, _, _, feature_size = x.shape
        surface_size = x_surface.shape[-1]
        x = x.transpose((0, 2, 3, 4, 1)).reshape(self.t_in,
                                                 self.h_size * self.w_size,
                                                 self.pressure_level * feature_size
                                                 )
        x_surface = x_surface.reshape(self.t_in, self.h_size * self.w_size, surface_size)
        inputs = np.concatenate([x, x_surface], axis=-1)
        inputs = inputs.transpose((1, 0, 2)).reshape(self.h_size * self.w_size,
                                                     self.t_in * (self.pressure_level * feature_size + surface_size)
                                                     )
        return inputs, label

    def normalize(self, x, x_surface):
        try:
            x = (x - self.mean_pressure_level) / self.std_pressure_level
            x_surface = (x_surface - self.mean_surface) / self.std_surface
        except ZeroDivisionError:
            return x, x_surface
        return x, x_surface
