# Copyright 2025 Huawei Technologies Co., Ltd
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
"""get dataset for DAE-PINN network"""

import os
import numpy as np
from mindspore import dataset as ds
from mindspore import Tensor


def get_dataset(data_params, shuffle=False, num_val=100):
    data = np.load(os.path.join(data_params['data_dir'], 'data.npz'))
    x_train, x_test = data['X_train'], data['X_test']
    batch_size = data_params['batch_size']

    train_dataset = ds.NumpySlicesDataset(x_train, shuffle=shuffle)
    train_dataset = train_dataset.batch(batch_size)
    val_dataset = Tensor(x_train[:num_val])
    test_dataset = Tensor(x_test)
    return train_dataset, test_dataset, val_dataset
