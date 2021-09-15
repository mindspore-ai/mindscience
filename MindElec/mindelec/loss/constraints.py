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
get constraints function map.
"""
from __future__ import absolute_import

from mindspore import log as logger
from mindspore import nn

from ..data.dataset import Dataset
from ..architecture.util import check_mode

_CONSTRAINT_TYPES = ["Equation", "BC", "IC", "Label", "Function"]


class _Instance(nn.Cell):
    """
    Obtain the corresponding constraint function based on different conditions.

    Args:
        problem (Problem): The pde problem.
        constraint_type (str): The pde equation and constraint type of pde problem. Defaults: "Equation"

    Returns:
        Function, the constraint function of the problem.

    Supported Platforms:
        ``Ascend``
    """
    def __init__(self, problem, constraint_type="Equation"):
        super(_Instance, self).__init__()
        self.type = constraint_type
        self.problem = problem
        self.governing_equation = self.problem.governing_equation
        self.boundary_condition = self.problem.boundary_condition
        self.initial_condition = self.problem.initial_condition
        self.constraint_function = self.problem.constraint_function

        if self.type not in _CONSTRAINT_TYPES:
            raise ValueError("Unknown constrain type: {}, only: {} are supported".format(self.type, _CONSTRAINT_TYPES))

    def construct(self, *output, **kwargs):
        """Defines the computation to be performed"""
        if self.type == "Equation":
            return self.governing_equation(*output, **kwargs)
        if self.type == "BC":
            return self.boundary_condition(*output, **kwargs)
        if self.type == "IC":
            return self.initial_condition(*output, **kwargs)
        if self.type == "Function":
            return self.constraint_function(*output, **kwargs)
        if self.type == "Label":
            return self.constraint_function(*output, **kwargs)
        return None


class Constraints:
    """
    Definition of the loss for all sub-dataset.

    Args:
        dataset (Dataset): The dataset including pde equation, boundary condition and initial condition.
        pde_dict(dict): The dict of pde problem.

    Supported Platforms:
        ``Ascend``
    """
    def __init__(self, dataset, pde_dict):
        check_mode("Constraints")
        if not isinstance(dataset, Dataset):
            raise TypeError("dataset should be an instance of Dataset")
        if not isinstance(pde_dict, dict):
            raise TypeError("pde_dict: {} should be dictionary, but got: {}".format(pde_dict, type(pde_dict)))
        self.dataset_columns_map = dataset.dataset_columns_map
        self.column_index_map = dataset.column_index_map
        self.dataset_constraint_map = dataset.dataset_constraint_map
        if not self.dataset_columns_map or not self.column_index_map or not self.dataset_constraint_map:
            raise ValueError("Maps info for dataset should not be none, please call create_dataset() first to "
                             "avoid unexpected error")
        self.pde_dict = pde_dict
        self.dataset_cell_index_map = {}
        self.fn_cell_list = nn.CellList()
        self._get_loss_func()
        logger.info("check dataset_columns_map: {}".format(self.dataset_columns_map))
        logger.info("check column_index_map: {}".format(self.column_index_map))
        logger.info("check dataset_constraint_map: {}".format(self.dataset_constraint_map))
        logger.info("check dataset_cell_index_map: {}".format(self.dataset_cell_index_map))
        logger.info("check fn_cell_list: {}".format(self.fn_cell_list))

    def _get_loss_func(self):
        """Get the loss fn map."""
        index = -1
        for dataset_name in self.dataset_columns_map.keys():
            for name in self.pde_dict.keys():
                if name is None:
                    raise ValueError("pde_dict should not be None")
                if dataset_name in [name, name + "_BC", name + "_IC", name + "_domain"]:
                    index += 1
                    self.dataset_cell_index_map[dataset_name] = index
                    constraint_type = self.dataset_constraint_map[dataset_name]
                    problem = _Instance(self.pde_dict[name], constraint_type)
                    self.fn_cell_list.append(problem)
