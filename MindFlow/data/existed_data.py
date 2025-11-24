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

from mindscience.utils.check_func import check_param_type
from .data_base import Data, ExistedDataConfig



class ExistedDataset(Data):
    """
    Load existing dataset (currently supports npy format only).
    """

    def __init__(
        self,
        name=None,
        data_dir=None,
        columns_list=None,
        data_format="npy",
        constraint_type="Label",
        random_merge=True,
        data_config=None
    ):
        if data_config is None:
            if not name or not data_dir or not columns_list:
                raise ValueError(
                    f"If data_config is None, name/data_dir/columns_list must not be None, "
                    f"but got name={name}, data_dir={data_dir}, columns_list={columns_list}"
                )
            data_config = ExistedDataConfig(
                name=name,
                data_dir=data_dir,
                columns_list=columns_list,
                data_format=data_format,
                constraint_type=constraint_type,
                random_merge=random_merge
            )

        check_param_type(data_config, "data_config", data_type=ExistedDataConfig)

        name = data_config.name
        columns_list = [f"{name}_{col}" for col in data_config.columns_list]
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

        self.load_data = {
            "npy": self._load_npy_data
        }

        super().__init__(name=name, columns_list=columns_list, constraint_type=constraint_type)

    def _initialization(self, batch_size=1, shuffle=False):
        """Load data once before training starts."""
        loader = self.load_data.get(self._data_format.lower())
        if loader is None:
            raise ValueError(f"Unsupported data format: {self._data_format}")

        data = loader()
        if not isinstance(data, tuple):
            data = (data,)

        self.data = data
        self.data_size = len(data[0])
        self.batch_size = batch_size

        if batch_size > self.data_size:
            raise ValueError(
                f"If prebatch data, batch_size={batch_size} cannot exceed data_size={self.data_size}"
            )

        self.batched_data_size = self.data_size // batch_size
        self.shuffle = shuffle
        self._index = np.arange(self.data_size)

        logger.info(
            f"Loaded existed dataset: {self.name}, columns={self.columns_list}, "
            f"size={self.data_size}, batched_size={self.batched_data_size}, shuffle={self.shuffle}"
        )

    def __getitem__(self, index):
        if self.data is None:
            self._initialization()

        if self._random_merge:
            index = (
                np.random.randint(0, self.batched_data_size)
                if index >= self.batched_data_size
                else index
            )
        else:
            index = index % self.batched_data_size

        if self.shuffle and index % self.batched_data_size == 0:
            self._index = np.random.permutation(self.data_size)

        col_data = None
        for i in range(len(self.columns_list)):

            if self.batch_size == 1:
                idx = self._index[index]
            else:
                idx = self._index[
                    index * self.batch_size : (index + 1) * self.batch_size
                ]

            temp = self.data[i][idx]
            col_data = (temp,) if col_data is None else col_data + (temp,)

        return col_data

    def _load_npy_data(self):
        """Load data from npy files."""
        results = []
        for path in self.data_dir:
            logger.info(f"Loading npy data from: {path}")
            arr = np.load(path).astype(np.float32)

            if arr.ndim < 1:
                raise ValueError(f"Loaded npy file must have at least 1 dimension: {path}")

            results.append(arr)

        logger.info(f"Loaded npy dataset size: {len(results[0])}")
        return tuple(results)

    def __len__(self):
        if self.data is None:
            self._initialization()
        return self.batched_data_size
