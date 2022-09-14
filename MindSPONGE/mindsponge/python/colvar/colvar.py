# Copyright 2021-2022 @ Shenzhen Bay Laboratory &
#                       Peking University &
#                       Huawei Technologies Co., Ltd
#
# This code is a part of MindSPONGE:
# MindSpore Simulation Package tOwards Next Generation molecular modelling.
#
# MindSPONGE is open-source software based on the AI-framework:
# MindSpore (https://www.mindspore.cn/)
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
"""
Collective variables
"""

import mindspore as ms
from mindspore import ops
from mindspore.ops import functional as F
from mindspore.nn import Cell
from mindspore.common import Tensor

from ..function import functions as func
from ..function.operations import GetVector
from ..function.units import Units, global_units

class Colvar(Cell):
    r"""Base class for collective variables.

        The function "construct" of Colvar must has the argument "coordinates"

    Args:
        dim_output (int):   The output dimension, i.e., the last dimension of output Tensor.

        periodic (bool):    Whether the CV is periodic or not. Default: False

        use_pbc (bool):     Whether to calculate the CV at periodic boundary condition (PBC).
                            If "None" is given, it will be determined at runtime based on
                            whether the "pbc_box" is given or not. Default: None

        length_unit (str):  Length unit for position coordinates.
                            If "None" is given, it will use the global units. Default: None

   """

    def __init__(self,
                 dim_output: int = 1,
                 periodic: bool = False,
                 use_pbc: bool = None,
                 length_unit: str = None,
                 ):

        super().__init__()

        self.dim_output = dim_output

        self.get_vector = GetVector(use_pbc)
        self.use_pbc = use_pbc

        if length_unit is not None:
            self.use_global_units = False
            self.units = Units(length_unit)
        else:
            self.use_global_units = True
            self.units = global_units

        # the CV is periodic or not
        if isinstance(periodic, bool):
            periodic = Tensor([periodic]*self.dim_output, ms.bool_)
        elif isinstance(periodic, (list, tuple)):
            if len(periodic) != self.dim_output:
                if len(periodic) == 1:
                    periodic = Tensor(periodic*self.dim_output, ms.bool_)
                else:
                    raise ValueError("The number of periodic mismatch")
        else:
            raise TypeError("Unsupported type for periodic:" +
                            str(type(periodic)))

        self.periodic = F.reshape(periodic, (1, 1, self.dim_output))

        self.any_periodic = self.periodic.any()
        self.all_periodic = self.periodic.all()

        self.identity = ops.Identity()

    @property
    def length_unit(self):
        """length unit"""
        return self.units.length_unit

    def vector_in_box(self, vector: Tensor, pbc_box: Tensor) -> Tensor:
        """Make the difference of vecters at the range from -0.5 box to 0.5 box"""
        return func.vector_in_box(vector, pbc_box)

    def set_pbc(self, use_pbc: bool):
        """set periodic boundary condition"""
        self.use_pbc = use_pbc
        self.get_vector.set_pbc(use_pbc)
        return self

    def construct(self, coordinate, pbc_box=None):
        raise NotImplementedError
