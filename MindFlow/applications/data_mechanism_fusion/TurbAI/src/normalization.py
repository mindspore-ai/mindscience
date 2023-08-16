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
"""
Normalization
"""
import numpy as np

from mindspore import Tensor
from mindspore.common import dtype as mstype


class Normalization:
    """Normalization"""
    def __init__(self, feature_method, label_method, file_path):
        self.feature_method = feature_method
        self.label_method = label_method
        self.file_path = file_path

    def normalize_feature(self, data):
        """normalize_feature"""
        if self.feature_method == "MinMax":
            return self.feature_min_max_norm(data)
        if self.feature_method == "Std":
            return self.feature_std_norm(data)
        return self.feature_min_max_norm(data)

    def normalize_label(self, mut_value, uy_value, vx_value):
        """normalize_label"""
        if self.label_method == "MinMax":
            return self.label_min_max_norm(mut_value, uy_value, vx_value)
        if self.label_method == "Std":
            return self.label_std_norm(mut_value, uy_value, vx_value)
        return self.label_min_max_norm(mut_value, uy_value, vx_value)

    def feature_min_max_norm(self, data):
        '''
        特征最大最小归一化
        '''
        df_max, df_min = get_min_max_data_from_txt(self.file_path)
        fenmu = (df_max[:, :-1] - df_min[:, :-1])
        fenzi = (data - df_min[:, :-1])
        df_data = fenzi / fenmu
        return df_data

    def feature_std_norm(self, data):
        '''
        特征均值方差标准化
        '''
        df_mean, df_std = get_mean_std_data_from_txt(self.file_path)
        fenmu = df_std[:, :-1]
        fenzi = (data - df_mean[:, :-1])
        df_data = fenzi / fenmu
        return df_data

    def label_min_max_norm(self, mut_value, uy_value, vx_value):
        '''
        损失函数特征及标签
        最大最小归一化
        '''
        df_max, df_min = get_min_max_data_from_txt(self.file_path)
        norm_mut = (mut_value - df_min[:, -1]) / (df_max[:, -1] - df_min[:, -1])

        sij = 0.5 * (uy_value + vx_value) / 6500
        sij = (sij - sij.min()) / (sij.max() - sij.min())

        rs_value = 2 * sij * norm_mut
        return norm_mut, sij, rs_value

    def label_std_norm(self, mut_value, uv_value, vx_value):
        '''
        损失函数特征及标签
        均值方差标准化
        '''
        df_mean, df_std = get_mean_std_data_from_txt(self.file_path)
        norm_mut = (mut_value - df_mean[:, -1]) / df_std[:, -1]

        sij = 0.5 * (uv_value + vx_value) / 6500
        sij = (sij - sij.mean()) / sij.std()

        rs_value = 2 * sij * norm_mut
        return norm_mut, rs_value, sij


def get_min_max_data_from_txt(data_path):
    """get_min_max_data_from_txt"""
    df_max = np.loadtxt(data_path + '/2d_max.txt')
    df_min = np.loadtxt(data_path + '/2d_min.txt')
    df_max = Tensor(df_max, mstype.float32).reshape(1, -1)
    df_min = Tensor(df_min, mstype.float32).reshape(1, -1)
    return df_max, df_min


def get_mean_std_data_from_txt(data_path):
    """get_mean_std_data_from_txt"""
    df_mean = np.loadtxt(data_path + '/mean.txt')
    df_std = np.loadtxt(data_path + '/std.txt')
    df_mean = Tensor(df_mean, mstype.float32).reshape(1, -1)
    df_std = Tensor(df_std, mstype.float32).reshape(1, -1)
    return df_mean, df_std
