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
"""dataset"""
import mindspore.dataset as ds


class Dataset:
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        return data[:2], data[2]


def create_dataset(data, batch_size, rank_size=1, rank_id=0, num_parallel_workers=8, is_training=True):
    train_ds = Dataset(data)
    train_loader = ds.GeneratorDataset(train_ds, column_names=["data", "label"],
                                       num_parallel_workers=num_parallel_workers,
                                       shuffle=is_training, num_shards=rank_size, shard_id=rank_id)
    train_loader = train_loader.batch(batch_size=batch_size, drop_remainder=True)
    return train_loader
