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
"""dataset"""
import numpy as np
import mindspore as ms


class CAETransformerDataset:
    """convert raw data into train dataset"""

    def __init__(self, data, seq_len, pred_len):
        self.time_size = data.shape[0] - seq_len - pred_len
        self.full_time_size = data.shape[0]
        self.raynold = data.shape[1]
        self.input_dim = data.shape[-1]
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.data = np.reshape(data, (-1, 1, self.input_dim, self.input_dim))

    def __getitem__(self, index):
        time_index = index % self.time_size
        raynold_index = (index // self.time_size) * self.full_time_size
        return self.data[raynold_index+time_index:raynold_index+time_index+self.seq_len], \
               self.data[raynold_index+time_index+self.seq_len:raynold_index+time_index+self.seq_len+self.pred_len]

    def __len__(self):
        return self.time_size * self.raynold


def create_caetransformer_dataset(data_path, batch_size, seq_len, pred_len, train_ratio=0.8):
    """create Transformer dataset"""
    true_data = np.load(data_path)
    train_split = int(true_data.shape[1] * train_ratio)
    train_data = true_data[:, :train_split]
    eval_data = true_data[:, train_split:]

    data = np.expand_dims(train_data, 2).astype(np.float32)
    dataset_generator = CAETransformerDataset(data, seq_len, pred_len)
    dataset = ms.dataset.GeneratorDataset(
        dataset_generator, ["data", "label"], shuffle=True
    )
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset, eval_data
