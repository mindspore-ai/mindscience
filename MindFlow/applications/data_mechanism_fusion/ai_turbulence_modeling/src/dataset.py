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
DatasetGenerator
"""
import numpy as np

import mindspore.dataset as ds

from .read_data import get_tensor_data, get_datalist_from_txt


class DatasetGenerator2D:
    """DatasetGenerator2D"""
    def __init__(self, data, label, sij, rs_value):
        self.data = data
        self.label = label
        self.sij = sij
        self.rs_value = rs_value

    def __getitem__(self, index):
        return self.data[index], self.label[index], self.sij[index], self.rs_value[index]

    def __len__(self):
        return len(self.data)


def data_parallel_2d(config, data_path, rank_id, rank_size, is_train=True):
    """data_parallel_2d"""
    train_data_csv = get_datalist_from_txt(data_path)
    data, label, sij, re_stress = get_tensor_data(train_data_csv, config["feature_norm"],
                                                  config["label_norm"], config["data_path"])
    data = data.asnumpy().astype("float32")
    label = label.asnumpy().astype("float32").reshape(-1, 1)
    sij = sij.asnumpy().astype("float32").reshape(-1, 1)
    re_stress = re_stress.asnumpy().astype("float32").reshape(-1, 1)
    dataset_generator = DatasetGenerator2D(data, label, sij, re_stress)
    dataset = ds.GeneratorDataset(dataset_generator, ["data", "label", "Sij", "Rs"],
                                  shuffle=True, num_shards=rank_size, shard_id=rank_id)
    if is_train:
        return dataset.batch(config["batch_size"])
    return dataset.batch(len(data))


class DatasetGenerator3D:
    """DatasetGenerator3D"""
    def __init__(self, data, label, dis, sij, rs_value):
        self.data = data
        self.label = label
        self.dis = dis
        self.sij = sij
        self.rs_value = rs_value

    def __getitem__(self, index):
        return self.data[index], self.label[index], self.dis[index], \
                self.sij[index], self.rs_value[index]

    def __len__(self):
        return len(self.data)


def data_parallel_3d(data_path, rank_id, rank_size, batch_size, is_train=True):
    """data_parallel_3d"""
    all_data = np.load(data_path).astype(np.float32)
    data, label = all_data[:, 0:10], all_data[:, 10].reshape(-1, 1)
    dis, sij = all_data[:, 11].reshape(-1, 1), all_data[:, 12].reshape(-1, 1)
    re_stress = all_data[:, 13].reshape(-1, 1)
    generator = DatasetGenerator3D(data, label, dis, sij, re_stress)
    columns = ["data", "label", "dis", "sij", "rs"]
    dataset = ds.GeneratorDataset(generator, columns, shuffle=True,
                                  num_shards=rank_size, shard_id=rank_id)
    if is_train:
        return dataset.batch(batch_size)
    return dataset.batch(len(data))
