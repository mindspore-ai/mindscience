# Copyright 2021-2023 @ Shenzhen Bay Laboratory &
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
"""Base cell for bais potential"""

from inspect import signature

import mindspore as ms
from mindspore import Tensor
from mindspore import nn, ops
from mindspore.ops import functional as F
try:
    # MindSpore 2.X
    from mindspore import jit
except ImportError:
    # MindSpore 1.X
    from mindspore import ms_function as jit

from .bias import Bias
from ...function import get_arguments
from ...function.units import Length, Energy
from ...function import functions as func


class SphericalRestrict(Bias):
    r"""Basic cell for bias potential

    Math:

    .. Math::

        V(R) = k \log{\left ( 1 + e^{\frac{|R - R_0| - r_0}{\sigma}} \right )}

    Args:

        radius (float):         Radius of sphere (r_0).

        center (Tensor):        Coordinate of the center of sphere (R_0).

        force_constant (float): Force constant of the bias potential(k). Default: Energy(500, 'kj/mol')

        depth (float):          Wall depth of the restriction (\sigma). Default: Length(0.01, 'nm')

        length_unit (str):      Length unit. If None is given, it will be assigned with the global length unit.
                                Default: None

        energy_unit (str):      Energy unit. If None is given, it will be assigned with the global energy unit.
                                Default: None

        use_pbc (bool):         Whether to use periodic boundary condition.

        name (str):             Name of the bias potential. Default: 'spherical'

    Supported Platforms:

        ``Ascend`` ``GPU``

    """
    def __init__(self,
                 radius: float,
                 center: Tensor = 0,
                 force_constant: float = Energy(500, 'kj/mol'),
                 depth: float = Length(0.01, 'nm'),
                 length_unit: str = None,
                 energy_unit: str = None,
                 use_pbc: bool = None,
                 name: str = 'spherical',
                 **kwargs,
                 ):

        super().__init__(
            name=name,
            length_unit=length_unit,
            energy_unit=energy_unit,
            use_pbc=use_pbc,
        )
        self._kwargs = get_arguments(locals(), kwargs)

        self.radius = Tensor(radius, ms.float32)
        self.center = Tensor(center, ms.float32)

        if isinstance(force_constant, Energy):
            force_constant = force_constant(self.units)
        self.force_constant = Tensor(force_constant, ms.float32)

        if isinstance(depth, Length):
            depth = depth(self.units)
        self.depth = Tensor(depth, ms.float32)

        self.norm_last_dim = None
        # MindSpore < 2.0.0-rc1
        if 'ord' not in signature(ops.norm).parameters.keys():
            self.norm_last_dim = nn.Norm(-1, False)

    @jit
    def calc_colvar(self,
                    coordinate: Tensor,
                    neighbour_index: Tensor = None,
                    neighbour_mask: Tensor = None,
                    neighbour_vector: Tensor = None,
                    neighbour_distance: Tensor = None,
                    pbc_box: Tensor = None
                    ):
        """calculate the value of collective variables"""
        # (B, A, D) - (D)
        vector = coordinate - self.center
        # (B, A) <- (B, A, D)
        if self.norm_last_dim is None:
            distance = ops.norm(vector, None, -1)
        else:
            distance = self.norm_last_dim(vector)
        return distance

    def calc_bias(self, colvar: Tensor) -> Tensor:
        """calculate the value of bias potential as the function of collective variables"""
        diff = colvar - self.radius
        bias = self.force_constant * F.log1p(F.exp(diff/self.depth))
        # (B, 1) <- (B, A)
        return func.keepdims_sum(bias, -1)
