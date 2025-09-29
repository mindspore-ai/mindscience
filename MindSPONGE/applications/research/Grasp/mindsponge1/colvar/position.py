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
Position
"""

import mindspore as ms
from mindspore.common import Tensor

from .colvar import Colvar


class Position(Colvar):
    r"""Position coordinate

    Args:
        dim_output (str):   Output dimension. Default: 3
        use_pbc (bool):     Whether to use periodic boundary condition. Default: False

    """
    def __init__(self,
                 dim_output: int = 3,
                 use_pbc: bool = None
                 ):

        super().__init__(
            dim_output=dim_output,
            periodic=False,
            use_pbc=use_pbc
        )

    def construct(self, coordinate, pbc_box=None):
        raise NotImplementedError


class Atom(Position):
    r"""Atom position

    Args:
        index (int):    index of atoms

    """
    def __init__(self, index: int):
        super().__init__()
        self.index = Tensor(index, ms.int32)

    def construct(self, coordinate, pbc_box=None):
        return coordinate[..., self.index, :]
