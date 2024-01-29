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
"""Module providing dataset functions"""
import os

import numpy as np
import pandas as pd
import mindspore as ms
from mindspore import Tensor

class CMIP5Data:
    """
        This class is used to process CMIP5 data, and is used to generate the dataset generator supported by
        MindSpore.

        Args:
            root_dir (str): The root dir of input data.
            data_period (list): Dataset time period.
            obs_time (int): The length of time the data can be observed.
            pred_time (int): The length of time model predited.

        Supported Platforms:
            ``Ascend`` ``GPU``
    """
    def __init__(self, root_dir, data_period, obs_time, pred_time):
        self.obs_time = obs_time
        self.pred_time = pred_time
        self.t_begin = data_period[0]
        self.t_end = data_period[1]

        self.var_dir = os.path.join(root_dir, 'CMIP5_sst_ssh_slp.npy')
        self.index_dir = os.path.join(root_dir, 'CMIP5_nino34.npy')
        time_range = pd.date_range(start="18610101", end="20041201", freq="MS")
        time_need = (time_range.year >= self.t_begin) & (time_range.year <= self.t_end)

        self.var = np.load(self.var_dir)[:, time_need]
        self.index = np.load(self.index_dir)[:, time_need]
        self.num_module = self.var.shape[0]
        self.num_mon = self.var.shape[1]
        self.module_len = self.num_mon - self.obs_time - self.pred_time

    def __len__(self):
        return self.num_module * self.module_len

    def __getitem__(self, index):
        try:
            module = int(index / self.module_len)
            mon = int(index % self.module_len)
        except ZeroDivisionError:
            return 0
        datax = Tensor(self.var[module, mon: mon + self.obs_time], dtype=ms.float32)
        datay = Tensor(self.index[module, mon: mon + self.obs_time + self.pred_time], dtype=ms.float32)
        return datax, datay


class ReanalysisData:
    """
        This class is used to process SODA Reanalysis data, and is used to generate the dataset generator supported by
        MindSpore.

        Args:
            root_dir (str): The root dir of input data.
            data_period (list): Dataset time period.
            obs_time (int): The length of time the data can be observed.
            pred_time (int): The length of time model predited.

        Supported Platforms:
            ``Ascend`` ``GPU``
    """
    def __init__(self, root_dir, data_period, obs_time, pred_time):
        self.var_dir = os.path.join(root_dir, 'SODA_sst_ssh_slp.npy')
        self.index_dir = os.path.join(root_dir, 'SODA_nino34.npy')

        self.obs_time = obs_time
        self.pred_time = pred_time

        self.t_begin = data_period[0]
        self.t_end = data_period[1]
        time_range = pd.date_range(start="19410101", end="20081201", freq="MS")
        time_need = (time_range.year >= self.t_begin) & (time_range.year <= self.t_end)

        self.var = np.load(self.var_dir)[time_need]
        self.index = np.load(self.index_dir)[time_need]
        self.num_mon = self.var.shape[0]

    def __len__(self):
        return self.num_mon - self.obs_time - self.pred_time

    def __getitem__(self, index):
        datax = Tensor(self.var[index: index + self.obs_time], dtype=ms.float32)
        datay = Tensor(self.index[index: index + self.obs_time + self.pred_time], dtype=ms.float32)
        return datax, datay
