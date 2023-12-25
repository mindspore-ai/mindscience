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
Volume of simulation system.
"""

from mindspore import Tensor

from ...function import keepdims_prod
from ..colvar import Colvar


class Volume(Colvar):
    r"""Volume of simulation system.

    Args:
        name (str): Name of the Colvar. Default: 'volume'.

    Supported Platforms:
        ``Ascend`` ``GPU``

    """

    def __init__(self,
                 name: str = 'volume'
                 ):

        super().__init__(
            shape=(1,),
            periodic=False,
            name=name,
            unit=None,
        )

    def set_pbc(self, use_pbc: bool):
        if use_pbc is False:
            raise ValueError('The Volume cannot be used without periodic boundary condition.')
        return super().set_pbc(use_pbc)

    def construct(self, coordinate: Tensor, pbc_box: bool = None):
        r"""return constant value.

        Args:
            coordinate (Tensor):    Tensor of shape (B, A, D). Data type is float.
            pbc_box (Tensor):       Tensor of shape (B, D). Data type is float.
                                    Default: ``None``.

        Returns:
            volume (Tensor):         Tensor of shape (B, ...) or (B, ..., 1). Data type is float.
        """
        #pylint: disable=unused-argument

        return keepdims_prod(pbc_box, -1)
