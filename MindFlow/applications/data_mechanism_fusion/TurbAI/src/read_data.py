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
read_data
"""
import os

import pandas as pd

from mindspore import Tensor
from mindspore.common import dtype as mstype

from .build_feature import BuildFeature
from .normalization import Normalization


def get_datalist_from_txt(data_file):
    """get_datalist_from_txt"""
    std_data_path = 'dataset/data_std'
    file_list = []
    with open(data_file, encoding="utf-8") as dat_file:
        for line in dat_file:
            file_list.append(line.strip())
    all_data = []
    for file in file_list:
        data_path = os.path.join(std_data_path, file)
        df_data = pd.read_csv(data_path)
        all_data.append(df_data)
    df_data = pd.concat(all_data)
    df_data = df_data[df_data['dis'] > 0]
    return df_data


def get_tensor_data(df_data, feature_method, label_method, file_path):
    """get_tensor_data"""
    aoa = Tensor(df_data['AoA'].values, mstype.float32)
    reynolds = Tensor(df_data['Re'].values, mstype.float32)
    dis = Tensor(df_data['dis'].values, mstype.float32)
    ux_value = Tensor(df_data['Ux'].values, mstype.float32)
    u_value = Tensor(df_data['U'].values, mstype.float32)
    v_value = Tensor(df_data['V'].values, mstype.float32)
    uy_value = Tensor(df_data['Uy'].values, mstype.float32)
    vx_value = Tensor(df_data['Vx'].values, mstype.float32)
    vy_value = Tensor(df_data['Vy'].values, mstype.float32)
    mut = Tensor(df_data['Mut'].values, mstype.float32)
    y_value = Tensor(df_data['Y'].values, mstype.float32)
    p_value = Tensor(df_data['P'].values, mstype.float32)
    ru_value = Tensor(df_data['Ru'].values, mstype.float32)

    bf_op = BuildFeature()
    data, mut = bf_op(mut, aoa, reynolds, dis, p_value, ru_value,
                      y_value, u_value, v_value, ux_value, uy_value, vx_value, vy_value)

    norm_op = Normalization(feature_method, label_method, file_path)
    data = norm_op.normalize_feature(data)
    label, sij, rs_value = norm_op.normalize_label(mut, uy_value, vx_value)
    return data, label, sij, rs_value
