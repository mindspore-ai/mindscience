# Copyright 2022 Huawei Technologies Co., Ltd
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
construct dataset based on saved mindrecord
"""
import mindspore.dataset as ds


class DataPrepare():
    """Obtain dataset for train or test from mindrecord."""

    def __init__(self, config, data_file):
        self.mesh_size = config["mesh_size"]
        self.batch_size = config["batch_size"]
        self.data_file = data_file

    def test_data_prepare(self, step):
        dataset = ds.MindDataset(dataset_files=self.data_file, shuffle=True,
                                 columns_list=["u0", "u_step{}".format(step)])
        dataset = dataset.batch(batch_size=1)
        return dataset

    def train_data_prepare(self):
        dataset = ds.MindDataset(dataset_files=self.data_file, shuffle=True, columns_list=["u0", "uT"])
        dataset = dataset.batch(batch_size=self.batch_size, drop_remainder=True)
        operations = [lambda x, y: (
            x.reshape(-1, 1, self.mesh_size, self.mesh_size), y.reshape(-1, 1, self.mesh_size, self.mesh_size))]
        dataset = dataset.map(operations, input_columns=["u0", "uT"])
        dataset_train, dataset_eval = dataset.split([0.5, 0.5])
        return dataset_train, dataset_eval
