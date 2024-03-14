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
"""Module providing Radar dataset class"""
import os

import numpy as np
from mindearth.data import Dataset
import mindspore.dataset as ds


class RadarData:
    """
    Self-defined class for processing USA-MRMS Radar dataset.

    Args:
        data_params (dict): dataset-related configuration of the model.
        run_mode (str, optional): whether the dataset is used for training, evaluation or testing. Supports [“train”,
            “test”, “valid”]. Default: 'train'.
        module_name(str, optional): generation or evolution

    Supported Platforms:
        ``Ascend`` ``GPU``
    """
    def __init__(self, data_params, run_mode='train', module_name='generation'):
        self.data_params = data_params
        self.module_name = module_name
        if run_mode == 'train':
            case_list = os.listdir(os.path.join(self.data_params.get("root_dir"), "train"))
            self.case_list = [os.path.join(self.data_params.get("root_dir"), "train", x) for x in case_list]
        elif run_mode == 'valid':
            case_list = os.listdir(os.path.join(self.data_params.get("root_dir"), "valid"))
            self.case_list = [os.path.join(self.data_params.get("root_dir"), "valid", x) for x in case_list]
        elif run_mode == 'test':
            case_list = os.listdir(os.path.join(self.data_params.get("root_dir"), "test"))
            self.case_list = [os.path.join(self.data_params.get("root_dir"), "test", x) for x in case_list]

    def __len__(self):
        return len(self.case_list)

    def __getitem__(self, item):
        data = np.load(self.case_list[item])
        if self.module_name == 'generation':
            inp, evo = data['ori'], data['evo'] / 128
            return inp[0, :self.data_params.get("t_in", 9)], evo[0], inp[0, self.data_params.get("t_in", 9):]
        return data['ori'][0, :self.data_params.get("t_in", 9) + self.data_params.get("t_out", 20)]


class NowcastDataset(Dataset):
    """
    Create the dataset for training, validation and testing,
    and output an instance of class mindspore.dataset.GeneratorDataset.

    Args:
        dataset_generator (Data): the data generator of weather dataset.
        module_name(str, optional): generation or evolution
        distribute (bool, optional): whether or not to perform parallel training. Default: False.
        num_workers (int, optional): number of workers(threads) to process the dataset in parallel. Default: 1.
        shuffle (bool, optional): whether or not to perform shuffle on the dataset. Random accessible input is
                required. Default: True, expected order behavior shown in the table.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """
    def __init__(self, dataset_generator, module_name='generation', distribute=False, num_workers=1, shuffle=True):
        super(NowcastDataset, self).__init__(dataset_generator, distribute, num_workers, shuffle)
        self.module_name = module_name

    def create_dataset(self, batch_size):
        """
        create dataset.

        Args:
            batch_size (int, optional): An int number of rows each batch is created with.

        Returns:
            BatchDataset, dataset batched.
        """
        ds.config.set_prefetch_size(1)
        if self.module_name == 'generation':
            dataset = ds.GeneratorDataset(self.dataset_generator,
                                          ['inputs', 'evo', 'labels'],
                                          shuffle=self.shuffle,
                                          num_parallel_workers=self.num_workers)
        else:
            dataset = ds.GeneratorDataset(self.dataset_generator,
                                          ['inputs'],
                                          shuffle=self.shuffle,
                                          num_parallel_workers=self.num_workers)
        if self.distribute:
            distributed_sampler_train = ds.DistributedSampler(self.rank_size, self.rank_id)
            dataset.use_sampler(distributed_sampler_train)
        dataset_batch = dataset.batch(batch_size=batch_size, drop_remainder=True,
                                      num_parallel_workers=self.num_workers)
        return dataset_batch
