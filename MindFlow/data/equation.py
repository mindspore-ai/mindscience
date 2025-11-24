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

from mindscience.data.flow.geometry import Geometry, SamplingConfig
from mindscience.utils.check_func import check_param_type
from .data_base import Data


_SPACE = ""


class Equation(Data):
    """
    Sampling data of equation domain.
    """

    def __init__(self, geometry):
        check_param_type(geometry, "geometry", data_type=Geometry)
        check_param_type(
            geometry.sampling_config,
            f"sampling_config of geometry:{geometry.name}",
            data_type=SamplingConfig,
        )

        self.sampling_config = geometry.sampling_config.domain
        if not self.sampling_config:
            raise ValueError(f"domain info of geometry: {geometry.name} should not be None")

        self.geometry = geometry

        self.data = None
        self.data_size = None
        self.batch_size = 1
        self.shuffle = False
        self.batched_data_size = None
        self.columns_list = None

        self._domain_index = None
        self._domain_index_num = 0
        self._random_merge = self.sampling_config.random_merge

        name = f"{geometry.name}_domain"
        columns_list = [f"{geometry.name}_domain_points"]
        constraint_type = "Equation"

        super().__init__(name=name, columns_list=columns_list, constraint_type=constraint_type)

    def _get_sampling_data(self):
        """Sampling domain points."""
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
            raise ValueError(
                f"If prebatch data, batch_size: {batch_size} should not be larger than "
                f"data size: {self.data_size}."
            )

        self.batched_data_size = self.data_size // batch_size
        self.shuffle = shuffle
        self._domain_index = np.arange(self.data_size)

        logger.info(
            f"Get domain dataset: {self.name}, columns: {self.columns_list}, "
            f"size: {self.data_size}, batched_size: {self.batched_data_size}, shuffle: {self.shuffle}"
        )

    def _get_index_when_sample_iter(self, index):
        """Index logic for random_sampling=True."""
        if self._domain_index_num == self.batched_data_size:
            self._initialization(self.batch_size, self.shuffle)
            self._domain_index_num = 0
        index = self._domain_index_num
        self._domain_index_num += 1
        return index

    def _get_index_when_sample_all(self, index):
        """Index logic for random_sampling=False."""
        data_size = len(self)
        if (self._random_merge or self.shuffle) and index % data_size == 0:
            self._domain_index = np.random.permutation(self.data_size)
        return index % data_size

    def __getitem__(self, index):
        if self.data is None:
            self._initialization()

        if self.sampling_config.random_sampling:
            index = self._get_index_when_sample_iter(index)
        else:
            index = self._get_index_when_sample_all(index)

        col_data = None
        for i in range(len(self.columns_list)):
            if self.batch_size == 1:
                idx = self._domain_index[index]
            else:
                idx = self._domain_index[
                    index * self.batch_size : (index + 1) * self.batch_size
                ]

            temp_data = self.data[i][idx]
            col_data = (temp_data,) if col_data is None else col_data + (temp_data,)

        return col_data

    def __len__(self):
        if self.data is None:
            self._initialization()
        return self.batched_data_size
