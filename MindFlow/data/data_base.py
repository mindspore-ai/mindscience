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

from mindscience.utils.check_func import (
    check_param_type,
    check_param_value,
    check_param_type_value
)

CONSTRAINT_TYPES = ["equation", "bc", "ic", "label", "function", "custom"]
DATA_FORMATS = ["npy"]


class Data:
    """
    Base class of Dataset, Equation, Boundary and ExistedDataset.

    Args:
        name (str): Dataset name.
        columns_list (list/tuple): Column names.
        constraint_type (str): Type of constraint.

    Supported Platforms:
        ``Ascend``
    """

    def __init__(self, name=None, columns_list=None, constraint_type=None):
        none_type = type(None)

        check_param_type(name, "name", data_type=[str, none_type])
        check_param_type(columns_list, "columns_list", data_type=[list, tuple, none_type])
        check_param_type(constraint_type, "constraint_type", data_type=[str, none_type])

        if constraint_type:
            check_param_value(constraint_type.lower(), "constraint_type", CONSTRAINT_TYPES)

        self.name = name
        self.columns_list = columns_list
        self.constraint_type = constraint_type
        self.dataset_type = type(self).__name__

    def set_constraint_type(self, constraint_type="Equation"):
        """Set dataset constraint type."""
        check_param_type(constraint_type, "constraint_type", data_type=str)
        check_param_value(constraint_type.lower(), "constraint_type", CONSTRAINT_TYPES)
        self.constraint_type = constraint_type

    @abc.abstractmethod
    def create_dataset(self):
        """Return a dataset (abstract)."""
        raise NotImplementedError(f"{self.dataset_type}.create_dataset not implemented")

    @abc.abstractmethod
    def _initialization(self):
        """Initialize dataset (abstract)."""
        raise NotImplementedError(f"{self.dataset_type}._initialization not implemented")

    @abc.abstractmethod
    def __getitem__(self, index):
        """Return item by index (abstract)."""
        raise NotImplementedError(f"{self.dataset_type}.__getitem__ not implemented")

    @abc.abstractmethod
    def __len__(self):
        """Return dataset length (abstract)."""
        raise NotImplementedError(f"{self.dataset_type}.__len__ not implemented")


class ExistedDataConfig:
    """
    Configuration of ExistedDataset.

    Args:
        name (str): Dataset name.
        data_dir (str/list): Path(s) to existing data files.
        columns_list (str/list): Column names.
        data_format (str): File format (supports 'npy').
        constraint_type (str): Constraint type.
        random_merge (bool): Whether to randomly merge datasets.
    """

    def __init__(
        self,
        name,
        data_dir,
        columns_list,
        data_format="npy",
        constraint_type="Label",
        random_merge=True
    ):
        check_param_type(name, "name", data_type=str)
        self.name = name

        if isinstance(data_dir, str):
            data_dir = [data_dir]

        check_param_type(data_dir, "data_dir", data_type=[str, list, tuple])
        for path in data_dir:
            if not os.path.exists(path):
                raise ValueError(f"ExistedDataset file: {path} does not exist")

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
