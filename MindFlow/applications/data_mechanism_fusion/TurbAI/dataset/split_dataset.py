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
split_dataset
"""
import datetime
import os
import random
import re

import numpy as np

from src.read_data import get_datalist_from_txt


def build_features_2d(df_data, cur_path):
    """build_features_2d"""
    mach = df_data['Ma']
    aoa = df_data['AoA']
    reynolds = df_data['Re']
    u_value = df_data['U'] / mach
    v_value = df_data['V'] / mach
    ux_value = df_data['Ux'] / mach
    uy_value = df_data['Uy'] / mach
    vx_value = df_data['Vx'] / mach
    vy_value = df_data['Vy'] / mach

    r_norm = np.sqrt(0.5*(uy_value - vx_value)**2)
    s_norm = np.sqrt(ux_value**2 + vy_value**2 + 0.5*(uy_value + vx_value)**2)
    da_r = df_data['dis'] * df_data['dis'] * r_norm * (1 - np.tanh(df_data['dis']))
    entropy = (1.4*df_data['P'] / df_data['Ru']**1.4) - 1

    dref0 = 1.0 / np.sqrt(reynolds)
    dref1 = np.min([df_data['dis'], dref0], axis=0)
    dref2 = np.max([df_data['dis'], dref0], axis=0)

    dsturb_min = df_data['dis'].min()
    expfunc = 2.71828**(np.sqrt(dref1/dsturb_min))
    expfunc = expfunc * np.sqrt(dref0/dref2) - 2

    sig = np.sign(df_data['Y']).values
    v_direct = np.arctan(sig*v_value/u_value)

    proj_stream = sig * (-v_value + u_value * np.tan(np.pi*aoa/180.0))

    mut = df_data['Mut'].values
    mut = mut / (df_data['Re'].values / 1e6)
    trans = 1.0 / (df_data['dis'].values ** 0.6)
    mut = mut * trans

    data = np.vstack((u_value, r_norm, s_norm, entropy, da_r,
                      v_direct, proj_stream, expfunc, mut)).T

    df_min = np.min(data, 0)
    df_max = np.max(data, 0)
    np.savetxt(cur_path + '/2d_max.txt', df_max)
    np.savetxt(cur_path + '/2d_min.txt', df_min)

    df_mean = np.mean(data, 0)
    df_std = np.std(data, 0)
    np.savetxt(cur_path + '/2d_std.txt', df_std)
    np.savetxt(cur_path + '/2d_mean.txt', df_mean)


def build_features_3d(df_data, cur_path, task='train'):
    """build_features_3d"""
    mach = df_data['Ma']
    reynolds = df_data['Re']
    u_value = df_data['U']
    v_value = df_data['V']
    ux_value = df_data['Ux']
    uy_value = df_data['Uy']
    uz_value = 0
    vx_value = df_data['Vx']
    vy_value = df_data['Vy']
    vz_value = 0
    wx_value = 0
    wy_value = 0
    wz_value = 0
    y_value = df_data['Y'].values
    dis = df_data['dis'].values

    p_value = df_data['P'].values * mach ** 2
    ru_value = df_data['Ru'].values

    q1_1 = u_value

    w1_value = 0.5 * (wy_value - vz_value)
    w2_value = 0.5 * (uz_value - wx_value)
    w3_value = 0.5 * (vx_value - uy_value)
    rf_value = np.sqrt(w1_value ** 2 + w2_value ** 2 + w3_value ** 2)
    q3_value = (p_value / ru_value ** 1.4) - 1.0

    sig_y = np.sign(y_value)
    q5_value = np.arctan(sig_y * v_value / u_value)

    s11 = ux_value
    s12 = 0.5 * (uy_value + vx_value)
    s13 = 0.5 * (uz_value + wx_value)
    s22 = vy_value
    s23 = 0.5 * (vz_value + wy_value)
    s33 = wz_value
    sf_value = np.sqrt(s11 ** 2 + s12 ** 2 + s13 ** 2 + s22 ** 2 + s23 ** 2 + s33 ** 2)

    q8_value = dis ** 2 * rf_value * (1 - np.tanh(dis))
    q9_value = dis ** 2 * sf_value * (1 - np.tanh(dis))

    label = np.where(dis < 0.02, 1, 2)

    dref0 = 1.0 / np.sqrt(reynolds)
    dref1 = np.min([df_data['dis'], dref0], axis=0)
    dref2 = np.max([df_data['dis'], dref0], axis=0)
    expfunc = 2.71828 ** (np.sqrt(dref1 / (dis)))
    q10 = expfunc * np.sqrt(dref0 / dref2)

    q11 = (rf_value ** 2 - sf_value ** 2) / (rf_value ** 2 + sf_value ** 2)

    mut = df_data['Mut'].values
    dis = df_data['dis'].values

    trans = np.ones_like(dis)
    trans = np.where(dis < 0.02, 10.0, 0.02)
    mut = mut / (reynolds / 1e6)
    mut_min = mut.min()
    mut_max = mut.max()
    mut = (mut - mut.min()) / (mut.max() - mut.min())
    data = np.vstack((q1_1, label, rf_value, q3_value, q5_value,
                      sf_value, q8_value, q9_value, q10, q11)).T
    sca_min = np.min(data, 0)
    sca_max = np.max(data, 0)
    data_sca = (data - sca_min) / (sca_max - sca_min)
    sca_min = np.append(sca_min, mut_min)
    sca_max = np.append(sca_max, mut_max)

    factor = 6500.0
    sij = 0.5 * (uy_value + vx_value + uz_value + wx_value + vz_value + wy_value) / factor
    sij = (sij - sij.min()) / (sij.max() - sij.min())
    length = len(df_data)
    data2 = np.ones((length, 14))
    data2[:, 0:10] = data_sca[:, :]
    data2[:, 10] = mut
    data2[:, 11] = trans
    data2[:, 12] = sij
    data2[:, 13] = 2 * data2[:, 10] * data2[:, 12]
    if task == 'train':
        np.savetxt(cur_path + '/3d_max.dat', sca_max)
        np.savetxt(cur_path + '/3d_min.dat', sca_min)
        np.save(cur_path + '/train_data_3d.npy', data2)
    elif task == 'val':
        np.save(cur_path + '/val_data_3d.npy', data2)
    else:
        np.save(cur_path + '/test_data_3d.npy', data2)


def save_minmax_data(cur_path):
    """save_minmax_data"""
    train_file = cur_path + '/train.txt'
    val_file = cur_path + '/val.txt'
    test_file = cur_path + '/test.txt'
    df_train = get_datalist_from_txt(train_file)
    df_val = get_datalist_from_txt(val_file)
    df_test = get_datalist_from_txt(test_file)

    # 2d网络最大最小值归一化及标准化
    build_features_2d(df_train, cur_path)

    # 3d网络最大最小值归一化
    build_features_3d(df_train, cur_path, 'train')
    build_features_3d(df_val, cur_path, 'val')
    build_features_3d(df_test, cur_path, 'test')


def write_to_txt(target_file_path, train_list, val_list, test_list, exp_name):
    """write_to_txt"""
    cur_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    cur_path = target_file_path + exp_name + cur_time
    train_file = cur_path + '/train.txt'
    val_file = cur_path + '/val.txt'
    test_file = cur_path + '/test.txt'

    if not os.path.exists(cur_path):
        os.mkdir(cur_path)
    with open(train_file, "w", encoding="utf-8") as f:
        for data in train_list:
            f.write(data + '\n')
    with open(val_file, "w", encoding="utf-8") as f:
        for data in val_list:
            f.write(data + '\n')
    with open(test_file, "w", encoding="utf-8") as f:
        for data in test_list:
            f.write(data + '\n')
    return cur_path


def get_train_val_test(file_list):
    """get_train_val_test"""
    length = len(file_list)
    train = int(length * 0.6)
    val = int(length * 0.2)
    test_start = train + val
    random.shuffle(file_list)
    return file_list[:train], file_list[train:test_start], file_list[test_start:]


def get_change_work_condition(source_path, target_file_path):
    """get_change_work_condition"""
    # 按照6：2：2变工况
    train_list = []
    val_list = []
    test_list = []
    file_dict = {}
    for _, _, files in os.walk(source_path):
        print(f"files nums total {len(files)}")
        for file in files:
            swing_type = re.split('ffr06p|m073', file)[1]
            if swing_type in file_dict:
                file_dict[swing_type].append(file)
            else:
                file_dict[swing_type] = [file]

    print(f"swing_type total {len(file_dict.keys())}")
    for item in file_dict.values():
        train, val, test = get_train_val_test(item)
        train_list.extend(train)
        val_list.extend(val)
        test_list.extend(test)
    cur_path = write_to_txt(target_file_path, train_list, val_list, test_list, 'changeCondition_')
    save_minmax_data(cur_path)
    print("finish change condition")


if __name__ == '__main__':
    get_change_work_condition('dataset/data_std', 'dataset/experiment/')
