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

from ..utils.check_func import check_param_type, check_param_value, check_param_type_value

CONSTRAINT_TYPES = ["equation", "bc", "ic", "label", "function", "custom"]
DATA_FORMATS = ["npy"]


class Data:
    """
    This class is the base class of Dataset, Equation, Boundary and ExistedDataset. It represents
    a node in the data flow graph.

    Args:
        name (str): distinguished name of specified dataset. Default: ``None``.
        columns_list (Union[list, tuple]): list of column names. Default: ``None``.
        constraint_type (str, optional): constraint type of the specified dataset to get it's corresponding loss
            function. Default: ``None``. The `constraint_type` can be ``"equation"``, ``"bc"``, ``"ic"``,
            ``"label"`` or ``"function"``.

    Raises:
        TypeError: if `constraint_type` is ``None`` or `constraint_type.lower()` is not in
            [``"equation"``, ``"bc"``, ``"ic"``, ``"label"``, ``"function"``].

    Supported Platforms:
        ``Ascend``
    """
    def __init__(self, name=None, columns_list=None, constraint_type=None):
        none_type = type(None)
        check_param_type(name, "name", data_type=[str, none_type])
        check_param_type(columns_list, "columns_list", data_type=[list, tuple, none_type])
        check_param_type(constraint_type, "constraint_type", data_type=[str, none_type])
        check_param_type(constraint_type, "constraint_type", data_type=(str, none_type))
        if constraint_type:
            check_param_value(constraint_type.lower(), "constraint_type", CONSTRAINT_TYPES)
        self.name = name
        self.columns_list = columns_list
        self.constraint_type = constraint_type
        self.dataset_type = type(self).__name__

    def set_constraint_type(self, constraint_type="Equation"):
        check_param_type(constraint_type, "constraint_type", data_type=str)
        check_param_value(constraint_type.lower(), "constraint_type", CONSTRAINT_TYPES)
        self.constraint_type = constraint_type

    @abc.abstractmethod
    def create_dataset(self):
        """Return a dataset of the size `batch_size`."""
        raise NotImplementedError("{}.create_dataset not implemented".format(self.dataset_type))

    @abc.abstractmethod
    def _initialization(self):
        """Initialize dataset to get data"""
        raise NotImplementedError("{}._initialization not implemented".format(self.dataset_type))

    @abc.abstractmethod
    def __getitem__(self, index):
        """Defines behavior for when an item is accessed. Return the corresponding element for given index."""
        raise NotImplementedError("{}.__getitem__ not implemented".format(self.dataset_type))

    @abc.abstractmethod
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
        data_format (str, optional): the format of existed data files. Default: ``'npy'``.
            The format of ``'npy'`` is supported now.
        constraint_type (str, optional): specifies the constraint type of the created dataset.
            Default: ``"Label"``.
        random_merge (bool, optional): specifies whether randomly merge the given datasets.
            Default: ``True``.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """
    def __init__(self, name, data_dir, columns_list, data_format="npy", constraint_type="Label", random_merge=True):
        check_param_type(name, "name", data_type=str)
        self.name = name
        if isinstance(data_dir, str):
            data_dir = [data_dir]

        check_param_type(data_dir, "data_dir", data_type=[str, list, tuple])
        for path in data_dir:
            if not os.path.exists(path):
                raise ValueError('ExistedDataset file: {} does not exist'.format(path))
        self.data_dir = data_dir

        if isinstance(columns_list, str):
            columns_list = [columns_list]
        check_param_type(columns_list, "columns_list", data_type=[str, tuple, list])
        self.columns_list = columns_list

        check_param_type(constraint_type, "constraint_type", data_type=str)
        check_param_value(constraint_type.lower(), "constraint_type", CONSTRAINT_TYPES)
        self.constraint_type = constraint_type

        check_param_type_value(data_format, "data_format", DATA_FORMATS, data_type=str)
        self.data_format = data_format

        check_param_type(random_merge, "random_merge", data_type=bool)
        self.random_merge = random_merge
