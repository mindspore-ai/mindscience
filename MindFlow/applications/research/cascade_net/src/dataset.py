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
"""Dataset Load"""
import scipy.io as sio
import numpy as np


class AccesstrainDataset:
    """Create Train Dataset Loader"""
    def __init__(self, path):
        self.u_r10_train = sio.loadmat(path + 'ur_10.mat')['ur_10_train'].astype(np.float32)  # (10792, 128, 128, 2)
        self.u_r5_train = sio.loadmat(path + 'ur_5.mat')['ur_5_train'].astype(np.float32)  # (10792, 128, 128, 2)
        self.u_r3_train = sio.loadmat(path + 'ur_3.mat')['ur_3_train'].astype(np.float32)  # (10792, 128, 128, 2)
        self.u_r1_train = sio.loadmat(path + 'ur_1.mat')['ur_1_train'].astype(np.float32)  # (10792, 128, 128, 2)
        self.cp_fluc_train = sio.loadmat(path + 'Input_flucReal_Cp_pd.mat')['cp_flucReal_train_dp'].astype(
            np.float32)  # (10792, 128, 128, 3)
        self.re_c_train = sio.loadmat(path + 'conditions_Re_pd.mat')['Re_c_train'].astype(np.float32)  # (10792, 1)
        self.scaling_input_train = sio.loadmat(path + 'Input_scaling_u_pd.mat')['fine_input_train'].astype(
            np.float32)  # (10792,20)

    def __getitem__(self, index):
        return (self.u_r10_train[index], self.u_r5_train[index], self.u_r3_train[index], self.u_r1_train[index],
                self.cp_fluc_train[index], self.re_c_train[index], self.scaling_input_train[index],)

    def __len__(self):
        return len(self.u_r10_train)


def validation_test_dataset(path):
    """Load Validation and Test Dataset"""
    u_r10_validation = sio.loadmat(path + 'ur_10.mat')['ur_10_validation'].astype(np.float32)  # (1704, 128, 128, 2)
    u_r5_validation = sio.loadmat(path + 'ur_5.mat')['ur_5_validation'].astype(np.float32)  # (1704, 128, 128, 2)
    u_r3_validation = sio.loadmat(path + 'ur_3.mat')['ur_3_validation'].astype(np.float32)  # (1704, 128, 128, 2)
    u_r1_validation = sio.loadmat(path + 'ur_1.mat')['ur_1_validation'].astype(np.float32)  # (1704, 128, 128, 2)
    cp_fluc_validation = sio.loadmat(path + 'Input_flucReal_Cp_pd.mat')['cp_flucReal_validation_dp'].astype(
        np.float32)  # (1704, 128, 128, 3)
    re_c_validation = sio.loadmat(path + 'conditions_Re_pd.mat')['Re_c_validation'].astype(np.float32)  # (1704,1)
    scaling_input_validation = sio.loadmat(path + 'Input_scaling_u_pd.mat')['fine_input_validation'].astype(
        np.float32)  # (1704, 20)

    u_r10_test = sio.loadmat(path + 'ur_10.mat')['ur_10_test'].astype(np.float32)  # (3408, 128, 128, 2)
    u_r5_test = sio.loadmat(path + 'ur_5.mat')['ur_5_test'].astype(np.float32)  # (3408, 128, 128, 2)
    u_r3_test = sio.loadmat(path + 'ur_3.mat')['ur_3_test'].astype(np.float32)  # (3408, 128, 128, 2)
    u_r1_test = sio.loadmat(path + 'ur_1.mat')['ur_1_test'].astype(np.float32)  # (3408, 128, 128, 2)
    cp_fluc_test = sio.loadmat(path + 'Input_flucReal_Cp_pd.mat')['cp_flucReal_test_dp'].astype(
        np.float32)  # (3408, 128, 128, 3)
    re_c_test = sio.loadmat(path + 'conditions_Re_pd.mat')['Re_c_test'].astype(np.float32)  # (3408, 1)
    scaling_input_test = sio.loadmat(path + 'Input_scaling_u_pd.mat')['fine_input_test'].astype(np.float32)  # (3408,20)

    return (u_r10_validation, u_r5_validation, u_r3_validation, u_r1_validation,
            cp_fluc_validation, re_c_validation, scaling_input_validation,
            u_r10_test, u_r5_test, u_r3_test, u_r1_test, cp_fluc_test, re_c_test, scaling_input_test)
