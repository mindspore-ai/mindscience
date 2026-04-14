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
dataset
"""
import os

from mindflow.data import Dataset, ExistedDataConfig


def create_training_dataset(config,
                            shuffle=True,
                            drop_remainder=True,
                            is_train=True):
    """create dataset"""
    data_path = config["root_dir"]
    if is_train:
        input_path = os.path.join(data_path, "train_a.npy")
        label_path = os.path.join(data_path, "train_u.npy")
    else:
        input_path = os.path.join(data_path, "test_a.npy")
        label_path = os.path.join(data_path, "test_a.npy")
    ns_3d_data = ExistedDataConfig(name=config["name"],
                                   data_dir=[input_path, label_path],
                                   columns_list=["inputs", "label"],
                                   data_format="npy")
    dataset = Dataset(existed_data_list=[ns_3d_data])
    data_loader = dataset.create_dataset(batch_size=config["batch_size"],
                                         shuffle=shuffle,
                                         drop_remainder=drop_remainder)

    return data_loader
