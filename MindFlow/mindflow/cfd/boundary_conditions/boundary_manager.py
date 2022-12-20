# Copyright 2022 Huawei Technologies Co., Ltd
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
"""boundary manager"""
import mindspore.numpy as mnp
from mindspore import jit_class, ops

from .neumann import Neumann
from .wall import Wall
from .periodic import Periodic
from .symmetry import Symmetry

_boundary_dict = {
    'Neumann': Neumann,
    'Wall': Wall,
    'Periodic': Periodic,
    'Symmetry': Symmetry,
}


def define_boundary(config):
    """Define boundary according to boundary configuration"""
    ret = _boundary_dict.get(config['type'])
    if ret is None:
        err = "boundary {} has not been implied".format(config['type'])
        raise NameError(err)
    return ret(config)


@jit_class
class BoundaryManager():
    """Container of boundaries for all active axis."""

    def __init__(self, config, mesh_info):
        self.mesh_info = mesh_info
        self.head_list = []
        self.tail_list = []

        if 0 in self.mesh_info.active_axis:
            self.head_list.append(define_boundary(config['x_min']))
            self.tail_list.append(define_boundary(config['x_max']))

        if 1 in self.mesh_info.active_axis:
            self.head_list.append(define_boundary(config['y_min']))
            self.tail_list.append(define_boundary(config['y_max']))

        if 2 in self.mesh_info.active_axis:
            self.head_list.append(define_boundary(config['z_min']))
            self.tail_list.append(define_boundary(config['z_max']))

    def fill_boundarys(self, pri_var):
        """Fill pad values according to boundaries."""
        pad_size = self.mesh_info.pad
        for i in self.mesh_info.active_axis:
            permute = list(range(4))
            permute[i + 1] = 1
            permute[1] = i + 1
            permute_tuple = tuple(permute)

            pri_var = ops.Transpose()(pri_var, permute_tuple)

            head_val = self.head_list[i].fill_values_head(pri_var, i, pad_size)
            tail_val = self.tail_list[i].fill_values_tail(pri_var, i, pad_size)

            pri_var = mnp.concatenate((head_val, pri_var, tail_val), axis=1)

            pri_var = ops.Transpose()(pri_var, permute_tuple)

        return pri_var
