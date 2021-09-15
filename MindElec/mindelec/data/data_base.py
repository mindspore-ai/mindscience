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
"""
This dataset module supports various type of datasets, including .... Some of the operations that are
provided to users to preprocess data include shuffle, batch, repeat, map, and zip.
"""
from __future__ import absolute_import
import os
import abc

CONSTRAINT_TYPES = ["equation", "bc", "ic", "label", "function"]


class Data:
    """
    This class is the base class of Dataset, Equation, Boundary and ExistedDataset. It represents
    a node in the data flow graph.

    Args:
        name (str): distinguished name of specified dataset (default=None).
        columns_list (Union[list, tuple]): list of column names (default=None).
        constraint_type (str, optional): constraint type of the specified dataset to get it's corresponding loss
            function (default=None). The constraint_type can be equation, bc, ic, label or function.

    Raises:
        TypeError: if constraint_type is None or constraint_type.lower() not in ["equation", "bc", "ic", "label",
                   "function"].

    Supported Platforms:
        ``Ascend``
    """
    def __init__(self, name=None, columns_list=None, constraint_type=None):
        if name is not None and not isinstance(name, str):
            raise TypeError("the type of name should be str, but got {}".format(type(name)))
        if columns_list is not None and not isinstance(columns_list, (list, tuple)):
            raise TypeError("the type of columns_list should be list/tuple, but got {}".format(type(columns_list)))
        if constraint_type is not None and not isinstance(constraint_type, str):
            raise TypeError("the type of constraint_type should be str, but got {}".format(type(constraint_type)))

        self.name = name
        self.columns_list = columns_list
        self.constraint_type = constraint_type
        if constraint_type is not None and constraint_type.lower() not in CONSTRAINT_TYPES:
            raise TypeError("Unknown constraint type: {}, only: {} are supported"
                            .format(constraint_type, CONSTRAINT_TYPES))
        self.dataset_type = type(self).__name__

    def set_constraint_type(self, constraint_type="Equation"):
        if constraint_type.lower() not in CONSTRAINT_TYPES:
            raise TypeError("Unknown constraint type: {}, only: {} are supported"
                            .format(constraint_type, CONSTRAINT_TYPES))
        self.constraint_type = constraint_type

    @abc.abstractmethod
    def _initialization(self):
        """Initialize dataset to get data"""
        raise NotImplementedError("{}._initialization not implemented".format(self.dataset_type))

    @abc.abstractmethod
    def create_dataset(self):
        """Return a dataset of the size `batch_size`."""
        raise NotImplementedError("{}.create_dataset not implemented".format(self.dataset_type))

    @abc.abstractmethod
    def __getitem__(self, index):
        """Defines behavior for when an item is accessed. Return the corresponding element for given index."""
        raise NotImplementedError("{}.__getitem__ not implemented".format(self.dataset_type))

    def __len__(self):
        """Return length of dataset"""
        raise NotImplementedError("{}.__len__ not implemented".format(self.dataset_type))


class ExistedDataConfig:
    """
    Set arguments of ExistedDataset.

    Args:
        name (str): specifies the name of dataset.
        data_dir (Union[str, list, tuple]): the path of existed data files.
        columns_list (Union[str, list, tuple]): list of column names of the dataset.
        data_format (str, optional): the format of existed data files (default='npy'). The format of 'npy'
            is supported now.
        constraint_type (str, optional): specifies the constraint type of the created dataset (default="Label").
        random_merge (bool, optional): specifies whether randomly merge the given datasets (default=True).

    Supported Platforms:
        ``Ascend``
    """
    def __init__(self, name, data_dir, columns_list, data_format="npy", constraint_type="Label", random_merge=True):
        if not isinstance(name, str):
            raise TypeError("the type of name should be str, but got {}".format(type(name)))
        self.name = name
        if isinstance(data_dir, str):
            data_dir = [data_dir]
        elif not isinstance(data_dir, (list, tuple)):
            raise TypeError("data_dir should be str/list/tuple, but got: {}, type: {}".format(
                data_dir, type(data_dir)))
        for path in data_dir:
            if not os.path.exists(path):
                raise ValueError('ExistedDataset file: {} does not exist'.format(path))
        self.data_dir = data_dir

        if isinstance(columns_list, str):
            columns_list = [columns_list]
        elif not isinstance(columns_list, (list, tuple)):
            raise TypeError("columns_list should be str/list/tuple, but got: {}, type: {}".format(
                columns_list, type(columns_list)))
        self.columns_list = columns_list

        if not isinstance(constraint_type, str):
            raise TypeError("the type of constraint_type should be str, but got {}".format(type(constraint_type)))
        if constraint_type.lower() not in CONSTRAINT_TYPES:
            raise TypeError("Unknown constraint type: {}, only: {} are supported"
                            .format(constraint_type, CONSTRAINT_TYPES))
        self.constraint_type = constraint_type
        if not isinstance(data_format, str):
            raise TypeError("the type of data_format should be str, but got {}".format(type(data_format)))
        if data_format != "npy":
            raise ValueError("`data_format` should be `npy`, but got {}".format(data_format))
        self.data_format = data_format
        if not isinstance(random_merge, bool):
            raise TypeError("the type of random_merge should be bool, but got {}".format(type(random_merge)))
        self.random_merge = random_merge
