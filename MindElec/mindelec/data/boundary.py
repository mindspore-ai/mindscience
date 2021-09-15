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
Sampling data of boundary condition and initial condition.
"""
from __future__ import absolute_import
import numpy as np

from mindspore import log as logger
from .data_base import Data
from ..geometry.geometry_base import Geometry


class Boundary(Data):
    """
    Base class of boundary condition and initial condition.

    Args:
        geometry (Geometry): specifies geometry information of boundary condition and initial condition.

    Raises:
        ValueError: if sampling_config of geometry is None.
        TypeError: if geometry is not an instance of Geometry.

    Supported Platforms:
        ``Ascend``
    """
    def __init__(self, geometry):
        if not isinstance(geometry, Geometry):
            raise TypeError("geometry should be an instance of Geometry")
        self.geometry = geometry
        if self.geometry.sampling_config is None:
            raise ValueError("sampling_config of geometry: {} should not be none when create dataset, please set it"
                             .format(geometry.name))

        self.data = None
        self.data_size = None
        self.batch_size = 1
        self.shuffle = False
        self.batched_data_size = None
        self.columns_list = None
        self._bound_index = None
        self._bound_index_num = 0
        super(Boundary, self).__init__()

    def _get_sampling_data(self, geom_type="BC"):
        """get sampling data"""
        sample_data = self.geometry.sampling(geom_type=geom_type)
        if not isinstance(sample_data, tuple):
            sample_data = (sample_data,)
        logger.info("Get {} sample_data size: {}".format(geom_type, len(sample_data[0])))
        return sample_data, self.geometry.columns_dict[geom_type]

    def _initialize(self, batch_size=1, shuffle=False, geom_type="BC"):
        """initialization: sampling and set attrs."""
        data, self.columns_list = self._get_sampling_data(geom_type=geom_type)
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
        self._bound_index = np.arange(self.data_size)
        logger.info("Get {} dataset: {}, columns: {}, size: {}, batched_size: {}, shuffle: {}".format(
            geom_type, self.name, self.columns_list, self.data_size, self.batched_data_size, self.shuffle))
        return data

    def _get_index_when_sample_iter(self, index, geom_type="BC"):
        if self._bound_index_num == self.batched_data_size:
            self.data = self._initialize(self.batch_size, self.shuffle, geom_type=geom_type)
            self._bound_index_num = 0
        index = self._bound_index_num
        self._bound_index_num += 1
        return index

    def _get_index_when_sample_all(self, index):
        data_size = self.__len__()
        if (self._random_merge or self.shuffle) and index % data_size == 0:
            self._bound_index = np.random.permutation(self.data_size)
        index = index % data_size if index >= data_size else index
        return index

    def _get_item(self, index):
        col_data = None
        for i in range(len(self.columns_list)):
            temp_data = self.data[i][self._bound_index[index]] if self.batch_size == 1 else \
                        self.data[i][self._bound_index[index * self.batch_size : (index + 1) * self.batch_size]]
            col_data = (temp_data,) if col_data is None else col_data + (temp_data,)
        return col_data

    def __len__(self):
        return self.batched_data_size


class BoundaryBC(Boundary):
    """
    Sampling data of boundary condition.

    Args:
        geometry (Geometry): specifies geometry information of boundary condition.

    Raises:
        ValueError: if sampling_config.bc of geometry is None.

    Supported Platforms:
        ``Ascend``
    """
    def __init__(self, geometry):
        super(BoundaryBC, self).__init__(geometry)
        if geometry.sampling_config.bc is None:
            raise ValueError("BC info for the current geometry: {} was not found".format(geometry.name))
        self.sampling_config = self.geometry.sampling_config.bc
        self.name = geometry.name + "_BC"
        self.constraint_type = "BC"
        self._random_merge = self.sampling_config.random_merge

    def _initialization(self, batch_size=1, shuffle=False):
        """initialization: sampling and set attrs."""
        return self._initialize(batch_size=batch_size, shuffle=shuffle, geom_type="BC")

    def __getitem__(self, bc_index):
        if not self.data:
            self._initialization()
        if self.sampling_config.random_sampling:
            bc_index = self._get_index_when_sample_iter(bc_index, geom_type="BC")
        else:
            bc_index = self._get_index_when_sample_all(bc_index)
        return self._get_item(bc_index)


class BoundaryIC(Boundary):
    """
    Sampling data of initial condition.

    Args:
        geometry (Geometry): specifies geometry information of initial condition.

    Raises:
        ValueError: if sampling_config.ic of geometry is None.

    Supported Platforms:
        ``Ascend``
    """
    def __init__(self, geometry):
        super(BoundaryIC, self).__init__(geometry)
        if geometry.sampling_config.ic is None:
            raise ValueError("IC info for the current geometry: {} was not found".format(geometry.name))
        self.sampling_config = self.geometry.sampling_config.ic
        self.name = geometry.name + "_IC"
        self.constraint_type = "IC"
        self._random_merge = self.sampling_config.random_merge

    def _initialization(self, batch_size=1, shuffle=False):
        """initialization: sampling and set attrs."""
        return self._initialize(batch_size=batch_size, shuffle=shuffle, geom_type="IC")

    def __getitem__(self, ic_index):
        if not self.data:
            self._initialization()
        if self.sampling_config.random_sampling:
            ic_index = self._get_index_when_sample_iter(ic_index, geom_type="IC")
        else:
            ic_index = self._get_index_when_sample_all(ic_index)
        return self._get_item(ic_index)
