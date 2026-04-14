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
"""Harmonic oscillator potential"""

from typing import Union
from numpy import ndarray

import mindspore as ms
from mindspore import Tensor
from mindspore import ops
from mindspore.ops import functional as F

from .bias import Bias
from ...colvar import Colvar
from ...function import get_arguments
from ...function import get_ms_array
from ...function.units import Energy


class HarmonicOscillator(Bias):
    r"""A bias potential in the form of a harmonic oscillator to limit the values of
        the collective variables (CVs).

    Math:

    .. math:

        V[s(R)] = 1 / 2 * k * [s(R) - s_0]^2

    Args:

        colvar (Colvar):    Collective variables (CVs) :math:`s(R)` to be limited.

        offset (Union[float, Tensor, ndarray]):
                            Offset value :math:`s_0` of the CVs.

        spring_constant (Union[float, Energy, Tensor, ndarray]):
                            Spring constant of the bias potential :math:`k`. Default: 1

        length_unit (str):  Length unit. If None is given, it will be assigned with the global length unit.
                            Default: None

        energy_unit (str):  Energy unit. If None is given, it will be assigned with the global energy unit.
                            Default: None

        use_pbc (bool):     Whether to use periodic boundary condition.

        name (str):         Name of the bias potential. Default: 'harmonic_oscillator'

    Supported Platforms:

        ``Ascend`` ``GPU``

    """

    def __init__(self,
                 colvar: Colvar,
                 offset: Union[float, Tensor, ndarray] = 0,
                 spring_constant: Union[float, Energy, Tensor, ndarray] = 1,
                 length_unit: str = None,
                 energy_unit: str = None,
                 use_pbc: bool = None,
                 name: str = 'harmonic_oscillator',
                 **kwargs,
                 ):
        # pylint: disable=unexpected-keyword-arg
        super().__init__(
            name=name,
            covlar=colvar,
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

        if isinstance(spring_constant, Energy):
            spring_constant = spring_constant(self.units)
        self.spring_constant = get_ms_array(spring_constant, ms.float32)
        if self.spring_constant.ndim > self.colvar.ndim + 1:
            raise ValueError(f'The rank (ndim) of spring_constant ({self.spring_constant.ndim}) '
                             f'cannot be larger than the rank of colvar ({self.colvar.ndim}) + 1')
        self.offset = get_ms_array(offset, ms.float32)
        if self.offset.ndim > self.colvar.ndim + 1:
            raise ValueError(f'The rank (ndim) of offset ({self.offset.ndim}) '
                             f'cannot be larger than the rank of colvar ({self.colvar.ndim}) + 1')

    def calc_bias(self, colvar: Tensor) -> Tensor:
        """calculate the value of bias potential as the function of collective variables"""
        bias = 0.5 * self.spring_constant * F.square(colvar - self.offset)
        bias = self.reduce_sum(bias, axis=self.axis)
        if self.axis == -1:
            return bias
        return F.expand_dims(bias, -1)
