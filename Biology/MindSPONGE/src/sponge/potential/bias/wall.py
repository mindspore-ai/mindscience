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
"""Wall bais"""

from typing import Union
from numpy import ndarray

import mindspore as ms
from mindspore import Tensor
from mindspore import ops
from mindspore.ops import functional as F

from .bias import Bias
from ...colvar import Colvar
from ...function import get_arguments
from ...function.units import Energy
from ...function import get_ms_array

__all__ = [
    'WallBias',
    'UpperWall',
    'LowerWall',
]


class WallBias(Bias):
    r"""Bias potential to limit the values of the collective variables (CVs) to a certain range.

    Args:

        colvar (Colvar):    Collective variables (CVs) :math:`s(R)` to be limited.

        depth (Union[float, Tensor, ndarray]):
                            Wall depth of the restriction :math:`\sigma`. Default: 0.1

        energy_constant (Union[float, Energy, Tensor, ndarray]):
                            Force constant of the bias potential :math:`k`. Default: Energy(100, 'kj/mol')

        length_unit (str):  Length unit. If None is given, it will be assigned with the global length unit.
                            Default: None

        energy_unit (str):  Energy unit. If None is given, it will be assigned with the global energy unit.
                            Default: None

        use_pbc (bool):     Whether to use periodic boundary condition.

        name (str):         Name of the bias potential. Default: 'wall'

    Supported Platforms:

        ``Ascend`` ``GPU``

    """

    def __init__(self,
                 colvar: Colvar,
                 depth: Union[float, Tensor, ndarray] = 0.1,
                 energy_constant: Union[float, Energy, Tensor, ndarray] = Energy(100, 'kj/mol'),
                 length_unit: str = None,
                 energy_unit: str = None,
                 use_pbc: bool = None,
                 name: str = 'wall',
                 **kwargs,
                 ):

        super().__init__(
            name=name,
            colvar=colvar,
            update_pace=0,
            length_unit=length_unit,
            energy_unit=energy_unit,
            use_pbc=use_pbc,
        )
        self._kwargs = get_arguments(locals(), kwargs)

        if self.colvar.ndim == 1:
            self.reduce_sum = ops.ReduceSum(True)
            self.axis = -1
        else:
            self.reduce_sum = ops.ReduceSum(False)
            self.axis = tuple(range(1, self.colvar.ndim))

        if isinstance(energy_constant, Energy):
            energy_constant = energy_constant(self.units)
        self.energy_constant = self._check_ndim(energy_constant, 'energy_constant')

        self.depth = self._check_ndim(depth, 'depth')

    def calc_diff(self, colvar: Tensor) -> Tensor:
        """calculate different between colvar and limit"""
        raise NotImplementedError

    def calc_bias(self, colvar: Tensor) -> Tensor:
        """calculate the value of bias potential as the function of collective variables"""
        diff = self.calc_diff(colvar)
        bias = self.energy_constant * F.log1p(F.exp(diff/self.depth))
        bias = self.reduce_sum(bias, self.axis)

        if self.axis == -1:
            return bias
        return F.expand_dims(bias, -1)

    def _check_ndim(self, value: Tensor, name: str):
        """check ndim of tensor"""
        tensor = get_ms_array(value, ms.float32)
        if tensor.ndim > self.colvar.ndim + 1:
            raise ValueError(f'The rank (ndim) of {name} ({tensor.ndim}) cannot be larger than '
                             f'the rank of colvar ({self.colvar.ndim}) +1')
        return tensor


class UpperWall(WallBias):
    r"""Bias potential to limit the maximum values of the collectiva variables (CVs).

    Math:

    .. math:

        V(R) = k \log{\left [ 1 + e^{\frac{s(R) - s_0}{\sigma}} \right ]}

    Args:

        colvar (Colvar):    Collective variables (CVs) :math:`s(R)` to be limited.

        boundary (Union[float, Tensor, ndarray]):
                            Upper boundary values :math:`s_0` of the CVs.

        depth (Union[float, Tensor, ndarray]):
                            Wall depth of the restriction :math:`\sigma`. Default: 0.1

        energy_constant (Union[float, Energy, Tensor, ndarray]):
                            Force constant of the bias potential :math:`k`. Default: Energy(100, 'kj/mol')

        length_unit (str):  Length unit. If None is given, it will be assigned with the global length unit.
                            Default: None

        energy_unit (str):  Energy unit. If None is given, it will be assigned with the global energy unit.
                            Default: None

        use_pbc (bool):     Whether to use periodic boundary condition.

        name (str):         Name of the bias potential. Default: 'upper_wall'

    Supported Platforms:

        ``Ascend`` ``GPU``

    """

    def __init__(self,
                 colvar: Colvar,
                 boundary: Union[float, Tensor, ndarray],
                 depth: Union[float, Tensor, ndarray] = 0.1,
                 energy_constant: Union[float, Energy, Tensor, ndarray] = Energy(100, 'kj/mol'),
                 length_unit: str = None,
                 energy_unit: str = None,
                 use_pbc: bool = None,
                 name: str = 'upper_wall',
                 **kwargs,
                 ):

        super().__init__(
            colvar=colvar,
            depth=depth,
            energy_constant=energy_constant,
            length_unit=length_unit,
            energy_unit=energy_unit,
            use_pbc=use_pbc,
            name=name,
        )
        self._kwargs = get_arguments(locals(), kwargs)

        self.boundary = self._check_ndim(boundary, 'limit')

    def calc_diff(self, colvar: Tensor) -> Tensor:
        return colvar - self.boundary


class LowerWall(WallBias):
    r"""Bias potential to limit the minimum values of the collectiva variables (CVs).

    Math:

    .. math:

        V(R) = k \log{\left [ 1 + e^{\frac{s_0 - s(R)}{\sigma}} \right ]}

    Args:

        colvar (Colvar):    Collective variables (CVs) :math:`s(R)` to be limited.

        boundary (Union[float, Tensor, ndarray]):
                            Lower boundary values :math:`s_0` of the CVs.

        depth (Union[float, Tensor, ndarray]):
                            Wall depth of the restriction :math:`\sigma`. Default: 0.1

        energy_constant (Union[float, Energy, Tensor, ndarray]):
                            Force constant of the bias potential :math:`k`. Default: Energy(100, 'kj/mol')

        length_unit (str):  Length unit. If None is given, it will be assigned with the global length unit.
                            Default: None

        energy_unit (str):  Energy unit. If None is given, it will be assigned with the global energy unit.
                            Default: None

        use_pbc (bool):     Whether to use periodic boundary condition.

        name (str):         Name of the bias potential. Default: 'lower_wall'

    Supported Platforms:

        ``Ascend`` ``GPU``

    """

    def __init__(self,
                 colvar: Colvar,
                 boundary: Union[float, Tensor, ndarray],
                 depth: Union[float, Tensor, ndarray] = 0.1,
                 energy_constant: Union[float, Energy, Tensor, ndarray] = Energy(100, 'kj/mol'),
                 length_unit: str = None,
                 energy_unit: str = None,
                 use_pbc: bool = None,
                 name: str = 'lower_wall',
                 **kwargs,
                 ):

        super().__init__(
            colvar=colvar,
            depth=depth,
            energy_constant=energy_constant,
            length_unit=length_unit,
            energy_unit=energy_unit,
            use_pbc=use_pbc,
            name=name,
        )
        self._kwargs = get_arguments(locals(), kwargs)

        self.boundary = self._check_ndim(boundary, 'limit')

    def calc_diff(self, colvar: Tensor) -> Tensor:
        return self.boundary - colvar
