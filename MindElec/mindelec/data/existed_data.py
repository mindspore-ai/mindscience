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
# ==============================================================================
#pylint: disable=W0223
#pylint: disable=W0221
"""
This dataset module supports npy type of datasets. Some of the operations that are
provided to users to preprocess data include shuffle, batch, repeat, map, and zip.
"""
from __future__ import absolute_import

import numpy as np
from mindspore import log as logger

from .data_base import Data, ExistedDataConfig


class ExistedDataset(Data):
    r"""
    Creates a dataset with given data path.

    Note:
        The `npy` data format is supported now.

    Parameters:
        name (str, optional): specifies the name of dataset (default=None). If `data_config` is None, the `name` should
            not be None.
        data_dir (Union[str, list, tuple], optional): the path of existed data files (default=None). If `data_config`
            is None, the `data_dir` should not be None.
        columns_list (Union[str, list, tuple], optional): list of column names of the dataset (default=None). If
            `data_config` is None, the `columns_list` should not be None.
        data_format (str, optional): the format of existed data files (default='npy').
        constraint_type (str, optional): specifies the constraint type of the created dataset (default="Label").
        random_merge (bool, optional): specifies whether randomly merge the given datasets (default=True).
        data_config (ExistedDataConfig, optional): Instance of ExistedDataConfig which collect the info
            described above (default=None). If it's not None, the dataset class will be create by using it for
            simplified. If it's None, the info of (name, data_dir, columns_list, data_format, constraint_type,
            random_merge) will be used for replacement.

    Raises:
        ValueError: Argument name/data_dir/columns_list is None when data_config is None.
        TypeError: If data_config is not a instance of ExistedDataConfig.
        ValueError: If data_format is not 'npy'.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> from mindelec.data import ExistedDataConfig, ExistedDataset
        >>> data_config = ExistedDataConfig(name='exist',
        ...                                 data_dir=['./data.npy'],
        ...                                 columns_list=['input_data'], data_format="npy", constraint_type="Equation")
        >>> dataset = ExistedDataset(data_config=data_config)
    """
    def __init__(self,
                 name=None,
                 data_dir=None,
                 columns_list=None,
                 data_format='npy',
                 constraint_type="Label",
                 random_merge=True,
                 data_config=None):
        if data_config is None:
            if name is None or data_dir is None or columns_list is None:
                raise ValueError("If data_config is None, argument: name, data_dir and columns_list should not be"
                                 " None, but got name: {}, data_dir: {}, columns_list: {}"
                                 .format(name, data_dir, columns_list))
            data_config = ExistedDataConfig(name, data_dir, columns_list, data_format, constraint_type, random_merge)
        elif not isinstance(data_config, ExistedDataConfig):
            raise TypeError("data_config should be instance of ExistedDataConfig but got {}"
                            .format(type(data_config)))

        name = data_config.name
        columns_list = [name + "_" + column_name for column_name in data_config.columns_list]
        constraint_type = data_config.constraint_type

        self.data_dir = data_config.data_dir
        self._data_format = data_config.data_format
        self._random_merge = data_config.random_merge

        self.data = None
        self.data_size = None
        self.batch_size = 1
        self.shuffle = False
        self.batched_data_size = None
        self._index = None

        super(ExistedDataset, self).__init__(name, columns_list, constraint_type)

    def _initialization(self, batch_size=1, shuffle=False):
        """load datasets from given paths"""
        if self._data_format.lower() == "npy":
            data = self._load_npy_data()
        else:
            raise ValueError("`data_format` should be `npy`.")
        if not isinstance(data, tuple):
            data = (data,)
        self.data = data
        self.data_size = len(data[0])
        self.batch_size = batch_size
        if batch_size > self.data_size:
            raise ValueError("If prebatch data, batch_size: {} should not be larger than data size: {}.".format(
                batch_size, self.data_size
            ))
        self.batched_data_size = self.data_size // batch_size
        self.shuffle = shuffle
        self._index = np.arange(self.data_size)
        logger.info("Get existed dataset: {}, columns: {}, size: {}, batched_size: {}, shuffle: {}".format(
            self.name, self.columns_list, self.data_size, self.batched_data_size, self.shuffle))
        return data

    def _load_npy_data(self):
        """
        Load npy-type data from exited file. For every column the data shape should be 2D, i.e. (batch_size, dim)
        """
        data = tuple()
        for path in self.data_dir:
            logger.info("Read data from file: {}".format(path))
            data_tmp = np.load(path)
            data += (data_tmp.astype(np.float32),)
        logger.info("Load npy data size: {}".format(len(data[0])))
        return data

    def __getitem__(self, index):
        if not self.data:
            self._initialization()
        if self._random_merge:
            index = np.random.randint(0, self.batched_data_size) if index >= self.batched_data_size else index
        else:
            index = index % self.batched_data_size if index >= self.batched_data_size else index

        if self.shuffle and index % self.batched_data_size == 0:
            self._index = np.random.permutation(self.data_size)

        col_data = None
        for i in range(len(self.columns_list)):
            temp_data = self.data[i][self._index[index]] if self.batch_size == 1 else \
                        self.data[i][self._index[index * self.batch_size : (index + 1) * self.batch_size]]
            col_data = (temp_data,) if col_data is None else col_data + (temp_data,)
        return col_data

    def __len__(self):
        if not self.data:
            self._initialization()
        return self.batched_data_size
