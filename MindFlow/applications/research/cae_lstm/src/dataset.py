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
"""dataset"""
import numpy as np
import mindspore as ms
from scipy.ndimage import gaussian_filter1d


class CreateDataset:
    """convert raw data into train dataset"""
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)


def create_cae_dataset(data_path, batch_size, multiple):
    """create cae dataset"""
    true_data = np.load(data_path)
    true_data_multiple = true_data * multiple
    true_data_multiple = np.expand_dims(true_data_multiple, 1).astype(np.float32)

    dataset_generator = CreateDataset(true_data_multiple, true_data_multiple)
    dataset = ms.dataset.GeneratorDataset(dataset_generator, ["data", "label"], shuffle=True)
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
    return dataset, true_data


def create_lstm_dataset(latent_true, batch_size, time_size, latent_size, time_window, gaussian_filter_sigma):
    """create lstm dataset"""
    latent_true = np.squeeze(latent_true.asnumpy())
    latent_true = latent_true.astype(np.float32)
    encoded_f = np.copy(latent_true).astype(np.float32)

    for i in range(latent_size):
        encoded_f[:, i] = gaussian_filter1d(encoded_f[:, i], sigma=gaussian_filter_sigma)

    input_seq = np.zeros(shape=(time_size - time_window, time_window, latent_size)).astype(np.float32)
    output_seq = np.zeros(shape=(time_size - time_window, 1, latent_size)).astype(np.float32)

    sample = 0
    for t in range(time_window, time_size):
        input_seq[sample, :, :] = encoded_f[t - time_window:t, :]
        output_seq[sample, 0, :] = encoded_f[t, :]
        sample = sample + 1

    dataset_generator = CreateDataset(input_seq, output_seq)
    dataset = ms.dataset.GeneratorDataset(dataset_generator, ["data", "label"], shuffle=True)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset, input_seq
