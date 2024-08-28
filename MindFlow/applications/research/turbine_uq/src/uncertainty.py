# Copyright 2024 Huawei Technologies Co., Ltd
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
# ============================================================================
"""calculate uncertainty"""
import os
from typing import List

import numpy as np
from numpy import mean, var
from mindflow import load_yaml_config


class InputTransformer:
    """InputTransformer"""
    def __init__(self, var_name):
        self.problem = get_problem_set(var_name)
        self.num_var = self.problem['num_vars']
        self.range_var = self.problem['bounds']
        self.evaluater = {
            'input_shape': [100,],
            'names': [f'x{i}' for i in range(100)],
            'output_shape': [64, 128, 1],
            'reference': [0.5] * (96) + [0, 719, 344740, 8279],
        }
        self.names_input = self.evaluater['names']
        self.reference = self.evaluater['reference']
        self.opt_slice_index = self.get_problem_slice(self.problem)
        self.uq_list = None
        self.slice_index = None

    def get_problem_slice(self, problem):
        """get_problem_slice"""
        slice_index = []
        for i in range(problem['num_vars']):
            slice_index.append(self.names_input.index(problem['names'][i]))
        return slice_index

    def transfor_inputs(self, x_var):
        """transfor_inputs"""
        var_range = np.array(self.range_var)
        x_var = x_var * (var_range[np.newaxis, :, 1] - var_range[np.newaxis, :, 0]) + var_range[np.newaxis, :, 0]
        x_input = np.tile(self.reference, [x_var.shape[0], 1])
        x_input[:, self.opt_slice_index] = x_var
        return x_input

    def transfor_outputs(self, output):
        """transfor_outputs"""
        return output


class UQTransformer(InputTransformer):
    """UQTransformer"""
    def __init__(self, var_name, uq_name=None, uq_number=None, uq_list=('mean', 'var')):
        super().__init__(var_name)
        self.uq_number = uq_number
        uq_problem = get_problem_set(uq_name)
        self.uq_slice_index = self.get_problem_slice(uq_problem)
        self.uq_problem = uq_problem
        self.uq_list = uq_list

    def transfor_inputs(self, x_var):
        """transfor_inputs"""
        x_input = super().transfor_inputs(x_var)
        uq_input = self.generate_sample(self.uq_number, paradict={'mu': 0.5, 'sigma': 0.1})
        design_range = np.array(self.uq_problem['bounds'])
        uq_input = uq_input * (design_range[np.newaxis, :, 1] - \
                               design_range[np.newaxis, :, 0]) + design_range[np.newaxis, :, 0]
        x_input = np.tile(x_input, [uq_input.shape[0], 1])
        x_input[:, self.uq_slice_index] = np.repeat(uq_input, x_var.shape[0], axis=0)
        return x_input

    def transfor_outputs(self, output):
        """transfor_outputs"""
        output = output.reshape([self.uq_number, -1, output.shape[-1]])
        rst = [self.calculate_moment(output, moment_type=uq_name) for uq_name in self.uq_list]
        return np.concatenate(rst, axis=1)

    def generate_sample(self, number, paradict=None):
        """generate_sample"""
        samples_1d_list = []
        for _ in range(len(self.uq_slice_index)):
            samples_1d_list.append(
                np.random.normal(loc=paradict['mu'], scale=paradict['sigma'], size=number).reshape(-1, 1))
        return np.concatenate(samples_1d_list, axis=1)

    def calculate_moment(self, data, moment_type='mean'):
        """calculate_moment"""
        if moment_type == 'mean':
            rst = mean(data, axis=0, keepdims=False)
        elif moment_type == 'var':
            rst = var(data, axis=0, keepdims=False)
        return rst


def get_problem_set(name_list: List, load_path='./configs'):
    """get_problem_set"""
    optimize_const = load_yaml_config(os.path.join(load_path, 'optimization.yaml'))
    match_dict = optimize_const['optimize_match']
    range_dict = optimize_const['optimize_range']
    temp = []
    for name in name_list:
        temp.extend([x for x in match_dict.keys() if x.startswith(name)])
    name_list = temp
    var_idx = sum([match_dict.get(name, match_dict['default']) for name in name_list], [])
    problem = {'num_vars': len(var_idx), 'names': [f'x{i}' for i in var_idx]}
    bounds = [range_dict.get(name, range_dict['default']) for name in problem['names']]
    problem.update({'bounds': bounds})
    return problem
