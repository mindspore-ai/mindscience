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
Cutoff functions
"""

from typing import Union, Tuple
from numpy import ndarray

import mindspore as ms
from mindspore import ops
from mindspore import Tensor

from ...configs import Config
from ...utils.units import Length

from .base import Cutoff
from ...configs import Registry as R


@R.register('cutoff.gaussian')
class GaussianCutoff(Cutoff):
    r"""Gaussian-type cutoff network.

    Args:
        cutoff (float): Cutoff distance.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self,
                 cutoff: Union[Length, float, Tensor, ndarray] = None,
                 sigma: Union[Length, float, Tensor, ndarray] = None,
                 **kwargs
                 ):
        super().__init__(cutoff=cutoff)
        self._kwargs = Config.get_arguments(locals(), kwargs)

        self.name = 'Gaussian Cutoff'

        self.sigma = ms.Tensor(sigma, ms.float32)

    def construct(self,
                  distance: Tensor,
                  mask: Tensor = None,
                  cutoff: Tensor = None
                  ) -> Tuple[Tensor, Tensor]:
        """Compute cutoff.

        Args:
            distance (Tensor):  Tensor of shape (..., K). Data type is float.
            mask (Tensor):      Tensor of shape (..., K). Data type is bool.
            cutoff (Tensor):    Tensor of shape (), (1,) or (..., K). Data type is float.

        Returns:
            decay (Tensor):     Tensor of shape (..., K). Data type is float.
            mask (Tensor):      Tensor of shape (..., K). Data type is bool.

        """
        if cutoff is None:
            cutoff = self.cutoff

        sigma = self.sigma
        if self.sigma is None:
            sigma = cutoff

        dis = distance - cutoff
        dis2 = dis * dis
        decay = 1. - ops.exp(-0.5 * dis2 * ops.reciprocal(ops.square(sigma)))

        if mask is None:
            mask = distance < cutoff
        else:
            mask &= distance < cutoff

        decay *= mask

        return decay, mask
