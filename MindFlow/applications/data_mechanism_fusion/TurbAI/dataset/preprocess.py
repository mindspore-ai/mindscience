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
preprocess
"""
import os
from datetime import datetime
from time import time

import pandas as pd


def process_mid(mid_path):
    """process_mid"""
    mid_columns = ['X', 'Y', 'U/Uinf', 'V/Vinf', 'P/Pinf', 'T/Tinf', 'MACH', 'cp',
                   'tur. vis', 'Dis', 'rho', 'dUx', 'dUy', 'dVx', 'dVy']
    mid_data = pd.DataFrame(columns=mid_columns)
    flag = False
    with open(mid_path, encoding="utf-8") as dat_file:
        for line in dat_file:
            if flag:
                data = line.split()
                mid_data.loc[len(mid_data)] = data
            if line.lstrip().startswith('DT='):
                flag = True
    return mid_data


def process_ori(ori_path):
    """process_ori"""
    ori_columns = ['Mach', 'alpha', 'beta', 'ReUe', 'Tinf,dR', 'time']
    ori_data = pd.DataFrame(columns=ori_columns)
    flag = False
    with open(ori_path, encoding="utf-8") as dat_file:
        for line in dat_file:
            if flag:
                data = line.split()
                ori_data.loc[len(ori_data)] = data
                break
            if line.lstrip().startswith('Mach'):
                flag = True
    return ori_data


def get_std_data(mid_data, ori_data):
    """get_std_data"""
    std_columns = ['X', 'Y', 'U', 'V', 'P', 'Mut', 'dis', 'Ru', 'Ux',
                   'Uy', 'Vx', 'Vy', 'Re', 'Ma', 'AoA']
    std2mid = {'X': 'X', 'Y': 'Y', 'U': 'U/Uinf', 'V': 'V/Vinf', 'P': 'P/Pinf',
               'Mut': 'tur. vis', 'dis': 'Dis', 'Ru': 'rho', 'Ux': 'dUx',
               'Uy': 'dUy', 'Vx': 'dVx', 'Vy': 'dVy'}
    std_data = pd.DataFrame(columns=std_columns)
    mach = ori_data.loc[0, 'Mach']
    reynolds = ori_data.loc[0, 'ReUe']
    aoa = ori_data.loc[0, 'alpha']
    for col, value in std2mid.items():
        std_data[col] = mid_data[value]
    std_data['Re'] = reynolds
    std_data['Ma'] = mach
    std_data['AoA'] = aoa
    return std_data


def transdata(file_name):
    """transdata"""
    start = time()
    mid_path = '/data_ori/' + file_name
    ori_path = './data_mid/' + file_name
    std_path = './data_std/' + file_name.replace('dat', 'csv')

    mid_data = process_mid(mid_path)
    ori_data = process_ori(ori_path)
    std_data = get_std_data(mid_data, ori_data)
    std_data.to_csv(std_path, index=None)
    end = time()
    print(f"finish  cost time is {end - start}, now is {datetime.now()}")


if __name__ == '__main__':
    for root, folders, files in os.walk('./data_mid/'):
        for file in files:
            transdata(file)
