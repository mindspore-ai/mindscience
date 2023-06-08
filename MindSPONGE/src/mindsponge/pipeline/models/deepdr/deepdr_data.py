# Copyright 2023 @ Huawei Technologies Co., Ltd
#
# This code is a part of MindSPONGE:
# MindSpore Simulation Package tOwards Next Generation molecular modelling.
#
# MindSPONGE is open-source software based on the AI-framework:
# MindSpore (https://www.mindspore.cn/)
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
"""deepdr data"""
import numpy as np
from sklearn.model_selection import train_test_split


def build_dataset(x, nf=0.5, std=1.0):
    """"build_dataset"""
    noise_factor = nf
    if isinstance(x, list):
        xs = train_test_split(*x, test_size=0.2)
        x_train = []
        x_test = []
        for jj in range(0, len(xs), 2):
            x_train.append(xs[jj])
            x_test.append(xs[jj + 1])
            x_train_noisy = list(x_train)
            x_test_noisy = list(x_test)
        for ii, _ in enumerate(x_train):
            x_train_noisy[ii] = x_train_noisy[ii] + noise_factor * np.random.normal(loc=0.0, scale=std,
                                                                                    size=x_train[ii].shape)
            x_test_noisy[ii] = x_test_noisy[ii] + noise_factor * np.random.normal(loc=0.0, scale=std,
                                                                                  size=x_test[ii].shape)
            x_train_noisy[ii] = np.clip(x_train_noisy[ii], 0, 1)
            x_test_noisy[ii] = np.clip(x_test_noisy[ii], 0, 1)
    else:
        x_train, x_test = train_test_split(x, test_size=0.2)
        x_train_noisy = x_train.copy()
        x_test_noisy = x_test.copy()
        x_train_noisy = x_train_noisy + noise_factor * np.random.normal(loc=0.0, scale=std, size=x_train.shape)
        x_test_noisy = x_test_noisy + noise_factor * np.random.normal(loc=0.0, scale=std, size=x_test.shape)
        x_train_noisy = np.clip(x_train_noisy, 0, 1)
        x_test_noisy = np.clip(x_test_noisy, 0, 1)
    output = (x_train_noisy, x_train, x_test_noisy, x_test)
    return output
