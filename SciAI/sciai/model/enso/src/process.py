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
process
"""
import os

import yaml
import numpy as np
from sklearn.utils import shuffle

from sciai.utils import print_log, parse_arg


def prepare():
    abs_dir = os.path.abspath(os.path.dirname(__file__))
    with open(f"{abs_dir}/../config.yaml") as f:
        config_dict = yaml.safe_load(f)
    args_ = parse_arg(config_dict)
    return (args_,)


def load_var(path, io_len=(3, 20), out_put_one=False):
    """ to load data from a path.
    Args:
        path (str): data path.
        io_len (int, int): input and output data length.
        out_put_one (bool): return full nino34 data or last column only.
    Returns:
        np.ndarray
        sst, ssh and nino3.4 index
    """
    ip_data_ls_, op_data_ls_, ip_data_ls1_ = [], [], []
    ip_len, op_len = io_len

    sst, ssh, nino34 = np.load(path + "/sst.npy"), np.load(path + "/ssh.npy"), np.load(path + "/nino34.npy")
    case = path.split("/")[-1]
    print_log(f"{case}, data_shape: {sst.shape}, {ssh.shape}, {nino34.shape}")

    for i in range(ip_len):
        # end index of each group
        idr = -ip_len + i + 1 - op_len if -ip_len + i + 1 - op_len != 0 else None
        # index data
        ip_data_sst, ip_data_ssh = sst[i:idr][:, :, :, np.newaxis], ssh[i:idr][:, :, :, np.newaxis]
        # append
        ip_data_ls_.append(ip_data_sst)
        ip_data_ls1_.append(ip_data_ssh)
    for j in range(op_len):
        # start indx
        idl = j + ip_len
        # end index
        idr = -op_len + j + 1 if -op_len + j + 1 != 0 else None
        # indx data
        op_data = nino34[idl:idr][:, np.newaxis]
        # append
        op_data_ls_.append(op_data)

    ip_data_ls = np.concatenate(ip_data_ls_, axis=3)
    ip_data_ls1 = np.concatenate(ip_data_ls1_, axis=3)
    op_data_ls = np.concatenate(op_data_ls_, axis=1)

    return ip_data_ls, ip_data_ls1, op_data_ls[:, -1] if out_put_one else op_data_ls


def load_spmonth(path, io_len=(3, 13), nino_month=1, label="noVar", out_put_one=True):
    """ load sp month data"""
    print_log(path)
    ip_len, op_len = io_len
    sst, ssh, nino34 = np.load(path + "/sst.npy"), np.load(path + "/ssh.npy"), np.load(path + "/nino34.npy")
    print_log({path.split("/")[-1], "Ini_data_shape:", sst.shape, ssh.shape, nino34.shape})
    bg_month = 2 if label == "noVar" else 1
    ip_st_month = (nino_month - op_len - ip_len) % 12 + 1
    idx_begin = ip_st_month - bg_month
    nino_idx_begin = idx_begin + ip_len + op_len - 1

    op_data_ls = nino34[nino_idx_begin::12]
    ip_data_ls, ip_data_ls1 = [], []

    for eh in range(ip_len):
        ip_data, ip_data1 = sst[idx_begin + eh::12], ssh[idx_begin + eh::12]
        diff_len = ip_data.shape[0] - op_data_ls.shape[0]
        if diff_len:
            ip_data, ip_data1 = ip_data[:-diff_len], ip_data1[:-diff_len]
        ip_data_ls.append(ip_data[..., np.newaxis])
        ip_data_ls1.append(ip_data1[..., np.newaxis])
    ip_data_ls = np.concatenate(ip_data_ls, axis=-1)
    ip_data_ls1 = np.concatenate(ip_data_ls1, axis=-1)
    return ip_data_ls, ip_data_ls1, op_data_ls[:, -1] if out_put_one else op_data_ls


def load_train(path, io_len=(3, 20), with_obs=True, out_put_one=False, load0_func=load_var):
    """ load train data
    Args:
        path (str): train data path
        io_len ((int, int), optional): input and output length. Defaults to (3, 20).
        with_obs (bool , optional): use Obs data for train if True
        out_put_one (bool, optional): return full data or last column only
        load0_func (func): the wanted data loading function
    Returns:
        sst ssh nino3.4 (Numpy.ndarray)
    """
    # get Model list
    fn_ls = os.listdir(path)
    # build ls for save
    ip_data_ls_ls, ip_data_ls1_ls, op_data_ls_ls = [], [], []
    # read each model data
    for fn in fn_ls:
        if with_obs or fn != "obs":
            ip_data_ls, ip_data_ls1, op_data_ls = load0_func(path + "/" + fn, io_len, out_put_one=out_put_one)
            ip_data_ls_ls.append(ip_data_ls)
            ip_data_ls1_ls.append(ip_data_ls1)
            op_data_ls_ls.append(op_data_ls)

    ip_data_ls_ls = np.concatenate(ip_data_ls_ls, axis=0)
    ip_data_ls1_ls = np.concatenate(ip_data_ls1_ls, axis=0)
    op_data_ls_ls = np.concatenate(op_data_ls_ls, axis=0)

    print_log("=" * 80)
    print_log(f"All Data Shape: {ip_data_ls_ls.shape}, {ip_data_ls1_ls.shape}, {op_data_ls_ls.shape}")
    print_log("=" * 80)
    return ip_data_ls_ls, ip_data_ls1_ls, op_data_ls_ls


def fetch_dataset_nino34(data_dir):
    """fetch dataset nino34"""
    sst_train, ssh_train, nino34_train = load_train(f"{data_dir}/train_data", io_len=(3, 17), with_obs=True)
    sst_var, ssh_var, nino34_var = load_var(f"{data_dir}/var_data", io_len=(3, 17))
    obs_sst_train, obs_ssh_train, obs_nino34_train = load_var(f"{data_dir}/train_data/obs", io_len=(3, 17))
    sst_std, ssh_std, nino34_std = sst_train.std(), ssh_train.std(), nino34_train.std()
    print_log(f"sst_std: {sst_std}, ssh_std: {ssh_std}, nino34_std: {nino34_std}")

    sst_train, ssh_train, nino34_train = sst_train / sst_std, ssh_train / ssh_std, nino34_train / nino34_std
    sst_var, ssh_var, nino34_var = sst_var / sst_std, ssh_var / ssh_std, nino34_var / nino34_std

    obs_sst_train, obs_ssh_train = obs_sst_train / sst_std, obs_ssh_train / ssh_std
    obs_nino34_train = obs_nino34_train / nino34_std

    ip_var = np.concatenate([sst_var, ssh_var], axis=3)
    ip_train = np.concatenate([sst_train, ssh_train], axis=3)
    obs_ip_train = np.concatenate([obs_sst_train, obs_ssh_train], axis=3)

    ip_var, ip_train, obs_ip_train = np.transpose(ip_var, (0, 3, 1, 2)), \
                                     np.transpose(ip_train, (0, 3, 1, 2)), np.transpose(obs_ip_train, (0, 3, 1, 2))

    print_log(f"ip_train shape: {ip_train.shape}, ip_var shape: {ip_var.shape}")
    print_log(f"Nan Elements in train data: {True in np.isnan(ip_train)}")
    print_log(f"nino34_train shape: {nino34_train.shape}")

    print_log(f"obs_ip_train shape: {obs_ip_train.shape}")
    print_log(f"Nan Elements in obs_ip_train data: {True in np.isnan(obs_ip_train)}")
    print_log(f"obs_nino34_train shape: {obs_nino34_train.shape}")

    ip_train, nino34_train = shuffle(ip_train, nino34_train)
    obs_ip_train, obs_nino34_train = shuffle(obs_ip_train, obs_nino34_train)
    return ip_train, nino34_train, ip_var, nino34_var, obs_ip_train, obs_nino34_train
