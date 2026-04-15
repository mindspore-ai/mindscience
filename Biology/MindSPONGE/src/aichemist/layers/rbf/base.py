# Copyright 2021-2023 @ Shenzhen Bay Laboratory &
#                       Peking University &
#                       Huawei Technologies Co., Ltd
#
# This code is a part of AIchemist package.
#
# The AIchemist is open-source software based on the AI-framework:
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
Radical basis functions (RBF)
"""

from typing import Union
from numpy import ndarray

import mindspore as ms
from mindspore.nn import Cell
from mindspore import Tensor

from ...utils.units import Units, GLOBAL_UNITS, Length, get_length


class RadicalBasisFunctions(Cell):
    r"""Network of radical basis functions.

    Args:
        r_max (Length):         Maximum distance. Defatul: 1 nm

        r_min (Length):         Minimum distance. Default: 0 nm

        sigma (float):          Simga. Default: 0

        delta (float):          Space interval. Default: ``None``.

        num_basis (int):        Number of basis functions. Defatul: None

        rescale (bool):         Whether to rescale the output of RBF from -1 to 1. Default: ``False``.

        clip_distance (bool):   Whether to clip the value of distance. Default: ``False``.

        length_unit (str):      Unit for distance. Default: = 'nm',

        hyper_param (dict):     Hyperparameter. Default: ``None``.

    """

    def __init__(self,
                 num_basis: int,
                 r_max: Union[Length, float, Tensor, ndarray],
                 r_min: Union[Length, float, Tensor, ndarray] = 0,
                 clip_distance: bool = False,
                 length_unit: Union[str, Units] = 'nm',
                 **kwargs,
                 ):

        super().__init__()
        self._kwargs = kwargs

        if length_unit is None:
            length_unit = GLOBAL_UNITS.length_unit
        self.units = Units(length_unit)

        self.num_basis = num_basis
        self.r_max = ms.Tensor(get_length(r_max, self.units), ms.float32)
        self.r_min = ms.Tensor(get_length(r_min, self.units), ms.float32)
        self.clip_distance = Tensor(clip_distance, ms.bool_)

        self.length_unit = self.units.length_unit

        if self.r_max <= self.r_min:
            raise ValueError(f'In RBF, r_max ({self.r_max}) must be larger than r_min ({self.r_min})!')

        self.r_range = self.r_max - self.r_min

    def __str__(self):
        return 'RadicalBasisFunctions<>'

    def print_info(self, num_retraction: int = 6, num_gap: int = 3, char: str = ' '):
        """print the information of RBF"""
        ret = char * num_retraction
        gap = char * num_gap
        print(ret+gap+f' Minimum distance: {self.r_min} {self.units.length_unit}')
        print(ret+gap+f' Maximum distance: {self.r_max} {self.units.length_unit}')
        print(ret+gap+f' Number of basis functions: {self.num_basis}')
        if self.clip_distance:
            print(ret+gap+f' Clip the range of distance to ({self.r_min}, {self.r_max}).')
        return self

    def construct(self, distance: Tensor) -> Tensor:
        """Compute gaussian type RBF.

        Args:
            distance (Tensor): Tensor of shape `(...)`. Data type is float.

        Returns:
            rbf (Tensor): Tensor of shape `(..., K)`. Data type is float.

        Symbol:
            K: Number of basis functions.

        """
        raise NotImplementedError
