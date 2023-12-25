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
"""
Constant value
"""

from typing import Union
from numpy import ndarray

import mindspore as ms
from mindspore import Tensor

from ...function import get_ms_array
from ..colvar import Colvar


class ConstantValue(Colvar):
    r"""Constant value.

    Args:
        value (Union[Tensor, ndarray, list, tuple]): Constant value.

        name (str): Name of the Colvar. Default: 'constant'

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    """

    def __init__(self,
                 value: Union[Tensor, ndarray, list, tuple],
                 name: str = 'constant'
                 ):

        super().__init__(
            periodic=False,
            name=name,
            unit=None,
        )

        self.value = get_ms_array(value, ms.float32)
        self._set_shape(self.value.shape)

    def construct(self, coordinate: Tensor, pbc_box: bool = None):
        r"""return constant value.

        Args:
            coordinate (Tensor):    Tensor of shape (B, A, D). Data type is float.
            pbc_box (Tensor):       Tensor of shape (B, D). Data type is float.
                                    Default: ``None``.

        Returns:
            constant value (Tensor):         Tensor of shape (B, ...) or (B, ..., 1). Data type is float.

        """
        #pylint: disable=unused-argument

        return self.value
