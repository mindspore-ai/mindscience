# # Copyright 2023 Huawei Technologies Co., Ltd
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# # http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# # ============================================================================
"""create dataset"""
import numpy as np

import mindspore as ms


class PeriodicHillDataset():
    """convert raw data into train dataset"""
    def __init__(self, bc_data, pde_data):
        self.coord = bc_data[:, :2]
        self.label = bc_data[:, 2:]
        self.pde_coord = pde_data[:, :2]
        self.bc_len = self.coord.shape[0]

    def __getitem__(self, index):
        return self.pde_coord[index], self.coord[index%self.bc_len], self.label[index%self.bc_len]

    def __len__(self):
        return self.pde_coord.shape[0]


def create_test_dataset(data_path):
    """load labeled data for evaluation"""
    data = np.load(data_path)  # shape=(700*300, 10)  x, y, u, v, p, uu, uv, vv, rho, nu
    data = data.reshape((700, 300, 10)).astype(np.float32)
    data = data[:, :, :8]
    test_data = data.reshape((-1, 8))
    test_coord = test_data[:, :2]
    test_label = test_data[:, 2:]
    return test_coord, test_label


def create_train_dataset(data_path, batch_size):
    """create training dataset by online sampling"""
    data = np.load(data_path)  # shape=(700*300, 10)  x, y, u, v, p, uu, uv, vv, rho, nu
    data = np.reshape(data, (300, 700, 10)).astype(np.float32)
    data = data[:, :, :8]

    bc_data = data[:5].reshape((-1, 8))
    bc_data = np.concatenate((bc_data, data[-5:].reshape((-1, 8))), axis=0)
    bc_data = np.concatenate((bc_data, data[5:-5, :5].reshape((-1, 8))), axis=0)
    bc_data = np.concatenate((bc_data, data[5:-5, -5:].reshape((-1, 8))), axis=0)

    pde_data = data[5:-5, 5:-5].reshape((-1, 8))
    dataset_generator = PeriodicHillDataset(bc_data, pde_data)
    dataset = ms.dataset.GeneratorDataset(
        dataset_generator, ["pde_coord", "coord", "label"], shuffle=True
    )
    dataset = dataset.batch(batch_size, drop_remainder=True)

    return dataset
