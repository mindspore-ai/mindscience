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
Sampling data of equation domain.
"""
from __future__ import absolute_import
import numpy as np

from mindspore import log as logger

from .data_base import Data
from ..geometry import Geometry


class Equation(Data):
    """
    Sampling data of equation domain.

    Args:
        geometry (Geometry): specifies geometry information of equation domain.

    Raises:
        TypeError: if geometry is not instance of class Geometry.
        ValueError: if sampling_config of geometry is None.
        KeyError: if sampling_config.domain of geometry is None.

    Supported Platforms:
        ``Ascend``
    """
    def __init__(self, geometry):
        if not isinstance(geometry, Geometry):
            raise TypeError("geometry: {} should be instance of class Geometry".format(geometry))
        self.geometry = geometry

        if self.geometry.sampling_config is None:
            raise ValueError("sampling_config of geometry: {} should not be none when create dataset, please set it"
                             .format(geometry.name))
        if self.geometry.sampling_config.domain is None:
            raise KeyError("Domain info for the current geometry: {} was not found".format(geometry.name))
        self.sampling_config = self.geometry.sampling_config.domain

        self.data = None
        self.data_size = None
        self.batch_size = 1
        self.shuffle = False
        self.batched_data_size = None
        self.columns_list = None

        self._domain_index = None
        self._domain_index_num = 0
        self._random_merge = self.sampling_config.random_merge

        name = geometry.name + "_domain"
        columns_list = [geometry.name + "_domain_points"]
        constraint_type = "Equation"
        super(Equation, self).__init__(name, columns_list, constraint_type)

    def _get_sampling_data(self):
        sample_data = self.geometry.sampling(geom_type="domain")
        return sample_data, self.geometry.columns_dict["domain"]

    def _initialization(self, batch_size=1, shuffle=False):
        """initialization: sampling and set attrs."""
        data, self.columns_list = self._get_sampling_data()
        if not isinstance(data, tuple):
            data = (data,)
        self.data = data
        self.data_size = len(self.data[0])

        self.batch_size = batch_size
        if batch_size > self.data_size:
            raise ValueError("If prebatch data, batch_size: {} should not be larger than data size: {}.".format(
                batch_size, self.data_size
            ))
        self.batched_data_size = self.data_size // batch_size
        self.shuffle = shuffle
        self._domain_index = np.arange(self.data_size)
        logger.info("Get domain dataset: {}, columns: {}, size: {}, batched_size: {}, shuffle: {}".format(
            self.name, self.columns_list, self.data_size, self.batched_data_size, self.shuffle))
        return data

    def _get_index_when_sample_iter(self, index):
        if self._domain_index_num == self.batched_data_size:
            self.data = self._initialization(self.batch_size, self.shuffle)
            self._domain_index_num = 0
        index = self._domain_index_num
        self._domain_index_num += 1
        return index

    def _get_index_when_sample_all(self, index):
        data_size = self.__len__()
        if (self._random_merge or self.shuffle) and index % data_size == 0:
            self._domain_index = np.random.permutation(self.data_size)
        index = index % data_size if index >= data_size else index
        return index

    def __getitem__(self, index):
        if not self.data:
            self._initialization()
        if self.sampling_config.random_sampling:
            index = self._get_index_when_sample_iter(index)
        else:
            index = self._get_index_when_sample_all(index)

        col_data = None
        for i in range(len(self.columns_list)):
            temp_data = self.data[i][self._domain_index[index]] if self.batch_size == 1 else \
                        self.data[i][self._domain_index[index * self.batch_size : (index + 1) * self.batch_size]]
            col_data = (temp_data,) if col_data is None else col_data + (temp_data,)
        return col_data

    def __len__(self):
        if not self.data:
            self._initialization()
        return self.batched_data_size
