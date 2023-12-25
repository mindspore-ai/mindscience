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
Transformation of Colvar
"""

from typing import Tuple, Callable

from mindspore import Tensor

from ..get import get_colvar
from ..colvar import Colvar


class TransformCV(Colvar):
    r"""
    Transformation of the values of the a collective variable :math:`s(R)` using a specific functions :math:`f(x)`.

    .. math::

        s' = f[s(R)]

    Args:
        colvar (Colvar): Collective variables (CVs) :math:`s(R)`.

        function (Callable): Transformation function :math:`f(x)`.

        periodic (bool): Whether the transformed collective variables is periodic. Default: ``False``.

        shape (Tuple[int]): Shape of the transformed collective variables. If None is given,
            then it will be assigned to the shape of the original `colvar`. Default: ``None``.

        unit (str): Unit of the collective variables. Default: ``None``.
            NOTE: This is not the `Units` Cell that wraps length and energy.

        name (str): Name of the collective variables. Default: 'transform'.

    Supported Platforms:
        ``Ascend`` ``GPU``

    """
    def __init__(self,
                 colvar: Colvar,
                 function: Callable,
                 periodic: bool = False,
                 shape: Tuple[int] = None,
                 unit: str = None,
                 name: str = 'transform',
                 ):

        super().__init__(
            periodic=periodic,
            unit=unit,
            name=name,
        )

        self.colvar = get_colvar(colvar)
        self.function = function

        self.set_pbc(self.colvar.use_pbc)
        if shape is None:
            shape = self.colvar.shape
        self._set_shape(shape)

        self._dtype = self.colvar.dtype

    def set_pbc(self, use_pbc: bool):
        """set whether to use periodic boundary condition"""
        super().set_pbc(use_pbc)
        self.colvar.set_pbc(use_pbc)
        return self

    def construct(self, coordinate: Tensor, pbc_box: Tensor = None):
        r"""return the cosine value of the collective variables (CVs).

        Args:
            coordinate (Tensor):    Tensor of shape `(B, A, D)`. Data type is float.
                                    Position coordinate of colvar in system.
                                    `B` means batchsize, i.e. number of walkers in simulation.
                                    `A` means number of colvar in system.
                                    `D` means dimension of the simulation system. Usually is 3.
            pbc_box (Tensor):       Tensor of shape `(B, D)`. Data type is float.
                                    Tensor of PBC box. Default: ``None``.

        Returns:
            cos_cv (Tensor):        Tensor of shape `(B, S_1, S_2, ..., S_n)`. Data type is float.
                                    `{S_i}` means dimensions of collective variables.

        """
        colvar = self.colvar(coordinate, pbc_box)

        return self.function(colvar)
