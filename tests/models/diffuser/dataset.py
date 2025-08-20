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
"""dataset api"""
import os
import struct

import numpy as np
from mindspore import jit_class
import mindspore.dataset as ds

FILE_DIR = '/home/workspace/mindspore_dataset/mindscience/mindflow/diffusion'
CKPT_PATH = os.path.join(FILE_DIR, 'ae.ckpt')


def load_data(filename):
    """load data from `FILE_DIR`"""
    return np.load(os.path.join(FILE_DIR, filename))


def load_mnist(kind='train'):
    """Load MNIST data from `path`"""
    path = os.path.join(FILE_DIR, 'MNIST')
    labels_path = os.path.join(path, f'{kind}-labels.idx1-ubyte')
    images_path = os.path.join(path, f'{kind}-images.idx3-ubyte')
    with open(labels_path, 'rb') as lbpath:
        _, _ = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        _, _, _, _ = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

    return images.astype(np.float32), labels


class TrainData:
    """MNIST dataset class"""

    def __init__(self, x):
        self.length = x.shape[0]
        self.x = x

    def __getitem__(self, index):
        return self.x[index]

    def __len__(self):
        return self.length


class LatentData:
    """MNIST latent dataset class"""

    def __init__(self, x, y):
        self.length = x.shape[0]
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.length


@jit_class
class SampleScaler:
    """MNIST scale class"""

    def __init__(self):
        pass

    @staticmethod
    def scale(x):
        """scale image"""
        return x/255.

    @staticmethod
    def unscale(x):
        """unscale image"""
        return x*255.


def get_dataset(batch_size):
    """build MNIST dataset"""
    x, _ = load_mnist()
    # scale MNIST dataset
    x = SampleScaler.scale(x)
    x = x.reshape(x.shape[0], x.shape[1])

    train_itr = TrainData(x)
    train_dataset = ds.GeneratorDataset(train_itr,
                                        column_names=['images'],
                                        shuffle=True,
                                        num_parallel_workers=1)

    train_dataset = train_dataset.batch(
        batch_size=batch_size, drop_remainder=True)
    print('train_dataset:', len(train_dataset))
    return train_dataset


def get_latent_dataset(latent_path, cond_path, batch_size):
    """build MNIST latent dataset"""
    x = load_data(latent_path)
    x = x.reshape(x.shape[0], 1, x.shape[1])
    cond = load_data(cond_path)
    cond = cond.reshape(cond.shape[0], 1).astype(np.float32)
    train_itr = LatentData(x, cond)
    train_dataset = ds.GeneratorDataset(train_itr,
                                        column_names=['images', 'cond'],
                                        shuffle=True,
                                        num_parallel_workers=1)

    train_dataset = train_dataset.batch(
        batch_size=batch_size, drop_remainder=True)
    print('train_dataset:', len(train_dataset))
    return train_dataset
