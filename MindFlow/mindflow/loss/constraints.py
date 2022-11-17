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
from ..data import Dataset, CONSTRAINT_TYPES
from ..pde import Problem
from ..utils.check_func import check_mode


class _Instance(nn.Cell):
    """
    Obtain the corresponding constraint function based on different conditions.

    Args:
        problem (Problem): The pde problem.
        constraint_type (str): The pde equation and constraint type of pde problem. Defaults: "Equation"

    Returns:
        Function, the constraint function of the problem.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """
    def __init__(self, problem, constraint_type="Equation"):
        super(_Instance, self).__init__()
        self.type = constraint_type.lower()
        self.problem = problem
        self.governing_equation = self.problem.governing_equation
        self.boundary_condition = self.problem.boundary_condition
        self.initial_condition = self.problem.initial_condition
        self.constraint_function = self.problem.constraint_function
        self.custom_function = self.problem.custom_function

        if self.type not in CONSTRAINT_TYPES:
            raise ValueError("Only constraint types {} are supported, but got {}".format(CONSTRAINT_TYPES, self.type))

    def construct(self, *output, **kwargs):
        """Defines the computation to be performed"""
        if self.type == "equation":
            return self.governing_equation(*output, **kwargs)
        if self.type == "bc":
            return self.boundary_condition(*output, **kwargs)
        if self.type == "ic":
            return self.initial_condition(*output, **kwargs)
        if self.type == "function":
            return self.constraint_function(*output, **kwargs)
        if self.type == "label":
            return self.constraint_function(*output, **kwargs)
        if self.type == "custom":
            return self.custom_function(*output, **kwargs)
        return None


class Constraints:
    """
    Definition of the loss for all sub-dataset.

    Args:
        dataset (Dataset): The dataset includes partial differential equation, boundary condition and initial condition.
        problem_list (list): A list of problems related to domain, BC, IC and others.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """
    def __init__(self, dataset, problem_list):
        check_mode("Constraints")
        if not isinstance(dataset, Dataset):
            raise TypeError("dataset should be an instance of Dataset")
        for problem in problem_list:
            if not isinstance(problem, Problem):
                raise TypeError("{} should be Problem, but got: {}".format(problem, type(problem)))
        self.dataset_columns_map = dataset.dataset_columns_map
        self.column_index_map = dataset.column_index_map
        self.dataset_constraint_map = dataset.dataset_constraint_map
        if not self.dataset_columns_map or not self.column_index_map or not self.dataset_constraint_map:
            raise ValueError("Maps info for dataset should not be none, please call create_dataset() first to "
                             "avoid unexpected error")
        self.pde_dict = {}
        index = 0
        for dataset in dataset.all_datasets:
            problem_list[index].set_name("domain", "{}_points".format(dataset.name))
            problem_list[index].set_name("ic", "{}_points".format(dataset.name))
            problem_list[index].set_name("bc", "{}_points".format(dataset.name))
            problem_list[index].set_name("ic_label", "{}_label".format(dataset.name))
            problem_list[index].set_name("bc_label", "{}_label".format(dataset.name))
            self.pde_dict[dataset.name] = problem_list[index]
            index += 1

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
            for name in self.pde_dict:
                if name is None:
                    raise ValueError("pde_dict should not be None")
                if dataset_name in [name, name + "_BC", name + "_IC", name + "_domain"]:
                    index += 1
                    self.dataset_cell_index_map[dataset_name] = index
                    constraint_type = self.dataset_constraint_map[dataset_name]
                    problem = _Instance(self.pde_dict[name], constraint_type)
                    self.fn_cell_list.append(problem)
