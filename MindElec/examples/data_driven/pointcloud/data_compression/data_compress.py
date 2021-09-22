# Copyright 2021 Huawei Technologies Co., Ltd
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
"""data compression"""

import os
import argparse
import datetime

import numpy as np
import mindspore.dataset as ds
from mindspore.common import set_seed
from mindspore import context
from mindspore.train.serialization import load_checkpoint

from src.model import EncoderDecoder

set_seed(0)
np.random.seed(0)

print("pid:", os.getpid())
print(datetime.datetime.now())

parser = argparse.ArgumentParser()
parser.add_argument('--device_num', type=int, default=1)
parser.add_argument('--input_path', type=str)
parser.add_argument('--model_path', type=str)
parser.add_argument('--data_config_path', type=str, default="./src/data_config.npy")
parser.add_argument('--output_save_path', type=str, default="./")


opt = parser.parse_args()
context.set_context(mode=context.GRAPH_MODE, save_graphs=False, device_target="Ascend", device_id=opt.device_num)


def data_compress():
    """data compression"""

    #######   load model  #############
    encoder = EncoderDecoder(config["input_channels"], config["patch_shape"], decoding=False)

    load_checkpoint(opt.model_path, encoder)
    print("model loaded")

    #######   load data  #############
    space_temp = np.load(opt.input_path)
    patch_num = [space_temp.shape[i] // config["patch_shape"][i] for i in range(3)]
    train_data_space = np.zeros((patch_num[0]*patch_num[1]*patch_num[2],
                                 config["patch_shape"][0],
                                 config["patch_shape"][1],
                                 config["patch_shape"][2],
                                 space_temp.shape[-1])).astype(np.float32)

    encodings = np.zeros((patch_num[0]*patch_num[1]*patch_num[2]//config['batch_size'],
                          config['batch_size'],
                          config['encoding_size'])).astype(np.float32)

    for i in range(patch_num[0]):
        for j in range(patch_num[1]):
            for k in range(patch_num[2]):
                idx = i*patch_num[1]*patch_num[2] + j*patch_num[2] + k
                train_data_space[idx, :, :, :, :] = space_temp[25*i : 25*(i + 1),
                                                               50*j : 50*(j + 1),
                                                               25*k : 25*(k + 1),
                                                               :]

    train_data_space[:, :, :, :, 2] = np.log10(train_data_space[:, :, :, :, 2] + 1.0)
    data_config = np.load(opt.data_config_path)
    for i in range(4):
        train_data_space[:, :, :, :, i] = train_data_space[:, :, :, :, i] / data_config[i]

    train_data_space = np.transpose(train_data_space, (0, 4, 1, 2, 3))
    num_workers = min([os.cpu_count(), config['batch_size'], 4])

    def create_dataset(net_input, batch_size=16, num_workers=4, shuffle=False):
        data = (net_input,)
        dataset = ds.NumpySlicesDataset(data, column_names=["data",], shuffle=shuffle)
        dataset = dataset.batch(batch_size=batch_size, num_parallel_workers=num_workers)
        return dataset

    train_loader = create_dataset(train_data_space,
                                  batch_size=config['batch_size'],
                                  num_workers=num_workers,
                                  shuffle=False)

    # data compression
    encoder.set_train(False)
    for iter_data in train_loader:
        test_input_space = iter_data[0]
        encoding = encoder(test_input_space)
        encoding = encoding.asnumpy().squeeze()

    encodings = np.reshape(encodings, (20, 40, 3, 496))

    np.save(os.path.join(opt.output_save_path, "compress_output.npy"), encodings)
    print("compressed file saved")


if __name__ == '__main__':
    # data compression configurations
    config = {
        "batch_size": 200,
        'patch_shape': [25, 50, 25],
        'encoding_size': 496,
        'input_channels': 4,
    }

    data_compress()
