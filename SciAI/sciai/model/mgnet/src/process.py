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

"""process for mgnet"""
import os

import yaml
from mindspore import int32
from mindspore.dataset.transforms import Compose
from mindspore.dataset.transforms import transforms as trans
from mindspore.dataset.vision import RandomCrop, RandomHorizontalFlip, ToTensor, Normalize
from mindspore.dataset import Cifar10Dataset, Cifar100Dataset, MnistDataset

from sciai.utils import parse_arg


def prepare():
    abs_dir = os.path.abspath(os.path.dirname(__file__))
    with open(f"{abs_dir}/../config.yaml") as f:
        config_dict = yaml.safe_load(f)
    args_ = parse_arg(config_dict)
    return (args_,)


def load_data(data_path, batch_size, dataset):
    """load the data from dataset"""
    def data_process(datasets, transforms):
        type_cast_op = trans.TypeCast(int32)
        train, test = datasets
        if dataset == 'cifar100':
            train = train.project(['image', 'fine_label']).map(operations=type_cast_op, input_columns='fine_label')
            test = test.project(['image', 'fine_label']).map(operations=type_cast_op, input_columns='fine_label')
        else:
            train = train.map(operations=type_cast_op, input_columns='label')
            test = test.map(operations=type_cast_op, input_columns='label')
        train = train.map(operations=transforms[0], input_columns='image').batch(batch_size, drop_remainder=True)
        test = test.map(operations=transforms[1], input_columns='image').batch(batch_size, drop_remainder=True)
        return train, test

    if dataset == 'mnist':
        transform_train = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
        transform_test = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
        train_set = MnistDataset(dataset_dir=data_path, usage='train', shuffle=True, num_parallel_workers=4)
        test_set = MnistDataset(dataset_dir=data_path, usage='test', shuffle=True, num_parallel_workers=4)
        train_set, test_set = data_process(datasets=(train_set, test_set), transforms=(transform_train, transform_test))
        num_classes = 10
        return train_set, test_set, num_classes

    if dataset == 'cifar10':
        normalize = Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010), is_hwc=False)
        transform_train = Compose([RandomCrop(32, padding=4), RandomHorizontalFlip(), ToTensor(), normalize])
        transform_test = Compose([ToTensor(), normalize])
        train_set = Cifar10Dataset(dataset_dir=data_path, usage='train', shuffle=True, num_parallel_workers=4)
        test_set = Cifar10Dataset(dataset_dir=data_path, usage='test', shuffle=True, num_parallel_workers=4)
        train_set, test_set = data_process(datasets=(train_set, test_set), transforms=(transform_train, transform_test))
        num_classes = 10
        return train_set, test_set, num_classes

    if dataset == 'cifar100':
        normalize = Normalize(mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2762), is_hwc=False)
        transform_train = Compose([RandomCrop(32, padding=4), RandomHorizontalFlip(), ToTensor(), normalize])
        transform_test = Compose([ToTensor(), normalize])
        train_set = Cifar100Dataset(dataset_dir=data_path, usage='train', shuffle=True, num_parallel_workers=4)
        test_set = Cifar100Dataset(dataset_dir=data_path, usage='test', shuffle=True, num_parallel_workers=4)
        train_set, test_set = data_process(datasets=(train_set, test_set), transforms=(transform_train, transform_test))
        num_classes = 100
        return train_set, test_set, num_classes

    raise Exception(f"data set: '{dataset}' isn't supported in this model")
