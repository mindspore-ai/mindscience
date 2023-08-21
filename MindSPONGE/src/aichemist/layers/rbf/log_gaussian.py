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
import mindspore.numpy as msnp
from mindspore import Tensor
from mindspore.ops import functional as F
from mindspore.ops import composite as C

from ...configs import Config
from ...utils.units import Length, get_length

from .base import RadicalBasisFunctions
from ...configs import Registry as R


@R.register('rbf.log_gaussian')
class LogGaussianBasis(RadicalBasisFunctions):
    r"""Log Gaussian type RBF.

    Args:
        r_max (Length):         Maximum distance. Defatul: 1 nm

        r_min (Length):         Minimum distance. Default: 0.04 nm

        sigma (float):          Simga. Default: 0.3

        delta (float):          Space interval. Default: 0.0512

        num_basis (int):        Number of basis functions. Defatul: None

        rescale (bool):         Whether to rescale the output of RBF from -1 to 1. Default: ``True``.

        clip_distance (bool):   Whether to clip the value of distance. Default: ``False``.

        length_unit (str):      Unit for distance. Default: = 'nm',

        hyper_param (dict):     Hyperparameter. Default: ``None``.

        r_ref (Length):         Reference distance. Default: 1 nm

    """

    def __init__(self,
                 r_max: Union[Length, float, Tensor, ndarray] = Length(1, 'nm'),
                 r_min: Union[Length, float, Tensor, ndarray] = Length(0.04, 'nm'),
                 sigma: Union[Length, float, Tensor, ndarray] = 0.3,
                 delta: Union[Length, float, Tensor, ndarray] = 0.0512,
                 num_basis: int = None,
                 rescale: bool = True,
                 clip_distance: bool = False,
                 length_unit: str = 'nm',
                 r_ref: Length = Length(1, 'nm'),
                 **kwargs,
                 ):

        super().__init__(
            num_basis=num_basis,
            r_max=r_max,
            r_min=r_min,
            clip_distance=clip_distance,
            length_unit=length_unit,
        )
        self._kwargs = Config.get_arguments(locals(), kwargs)

        if num_basis is None and delta is None:
            raise TypeError('"num_basis" and "delta" cannot both be "None".')
        if num_basis is not None and num_basis <= 0:
            raise ValueError('"num_basis" must be larger than 0.')
        if delta is not None and delta <= 0:
            raise ValueError('"delta" must be larger than 0.')
        if isinstance(delta, Length):
            raise TypeError(
                '"delta" in Log Gaussian RBF is dimensionless, so its type should not be "Length"')

        self.delta = ms.Tensor(delta, ms.float32)

        if isinstance(sigma, Length):
            raise TypeError('"sigma" in Log Gaussian RBF is dimensionless, so its type should not be "Length"')

        self.sigma = ms.Tensor(sigma, ms.float32)
        self.rescale = rescale

        self.r_ref = ms.Tensor(get_length(r_ref, self.units), ms.float32)

        log_rmin = msnp.log(self.r_min/self.r_ref, dtype=ms.float32)
        log_rmax = msnp.log(self.r_max/self.r_ref, dtype=ms.float32)
        log_range = log_rmax-log_rmin
        if self.delta is None:
            self.offsets = msnp.linspace(
                log_rmin, log_rmax, self.num_basis, dtype=ms.float32)
            self.delta = Tensor(log_range/(self.num_basis-1), ms.float32)
        else:
            if self.num_basis is None:
                num_basis = msnp.ceil(log_range/self.delta, ms.int32) + 1
                self.num_basis = int(num_basis)
            self.offsets = msnp.log(self.r_min/self.r_ref) + \
                msnp.arange(0, self.num_basis) * self.delta

        self.coeff = -0.5 * msnp.reciprocal(msnp.square(self.sigma))
        self.inv_ref = msnp.reciprocal(self.r_ref)

    def __str__(self):
        return 'LogGaussianBasis<>'

    def print_info(self, num_retraction: int = 6, num_gap: int = 3, char: str = ' '):
        ret = char * num_retraction
        gap = char * num_gap
        print(ret+gap+f' Minimum distance: {self.r_min} {self.units.length_unit}')
        print(ret+gap+f' Maximum distance: {self.r_max} {self.units.length_unit}')
        print(ret+gap+f' Reference distance: {self.r_ref} {self.units.length_unit}')
        print(ret+gap+f' Log Gaussian begin: {self.offsets[0]}')
        print(ret+gap+f' Log Gaussian end: {self.offsets[-1]}')
        print(ret+gap+f' Interval for log Gaussian: {self.delta}')
        print(ret+gap+f' Sigma for log gaussian: {self.sigma}')
        print(ret+gap+f' Number of basis functions: {self.num_basis}')
        if self.clip_distance:
            print(ret+gap+f' Clip the range of distance to ({self.r_min}, {self.r_max}).')
        if self.rescale:
            print(ret+gap+' Rescale the range of RBF to (-1, 1).')
        return self

    def construct(self, distance: Tensor) -> Tensor:
        """Compute gaussian type RBF.

        Args:
            distance (Tensor):  Tensor of shape `(...)`. Data type is float.

        Returns:
            rbf (Tensor):       Tensor of shape `(..., K)`. Data type is float.

        """
        if self.clip_distance:
            distance = C.clip_by_value(distance, self.r_min, self.r_max)

        # The shape looks like (...)
        log_r = F.log(distance * self.inv_ref)
        # The shape looks like (..., 1)
        log_r = F.expand_dims(log_r, -1)

        # The shape is (..., K) from (..., 1) - (K)
        log_diff = log_r - self.offsets
        rbf = F.exp(self.coeff*F.square(log_diff))

        if self.rescale:
            rbf = rbf * 2 - 1.0

        return rbf
