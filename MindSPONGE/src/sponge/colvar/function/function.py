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
Function Colvar
"""

from typing import Union, List, Tuple, Callable
from mindspore import Tensor
from mindspore.nn import CellList

from ..colvar import Colvar


class FunctionCV(Colvar):
    r"""
    A composite Colvar that combines a set of collective variables (CVs) :math:`{s_i(R)}`
    using a custom function :math:`f(s_1(R), s_2(R), ..., s_i(R))`.

    .. math::

        S = f(s_1(R), s_2(R), ... s_i(R))

    Args:
        colvar (Union[Colvar, List[Colvar], Tuple[Colvar]]): Collective variables to be combined :math:`{s_i(R)}`.

        function (callable): Custom function :math:`f(s_1(R), s_2(R), ... s_i(R))`.

        periodic (bool): Whether the custom collective variables is periodic.

        shape (tuple): Shape of custom collective variables. If None is given and all CVs in the `colvar`
            have the same shape, then it will be assigned the shape. If the shape of each CV in `colvar`
            is not exactly the same, the `shape` must be set. Default: ``None``.

        unit (str): Unit of the collective variables. Default: ``None``.
            NOTE: This is not the `Units` Cell that wraps length and energy.

        name (str): Name of the collective variables. Default: 'combine'.

    Supported Platforms:
        ``Ascend`` ``GPU``

    """
    def __init__(self,
                 colvar: Union[Colvar, List[Colvar], Tuple[Colvar]],
                 function: Callable,
                 periodic: bool,
                 shape: Tuple[int] = None,
                 unit: str = None,
                 use_pbc: bool = None,
                 name: str = 'function',
                 ):

        super().__init__(
            periodic=periodic,
            unit=unit,
            use_pbc=use_pbc,
            name=name,
        )

        if isinstance(colvar, Colvar):
            colvar = [colvar]
        elif not isinstance(colvar, (list, tuple)):
            raise TypeError(f'The type of "colvar" must be Colvar or list of Colvar '
                            f'but got: {type(colvar)}')

        self.num_colvar = len(colvar)

        self.function = function

        shape_ = []
        colvar_ = []
        for i, cv in enumerate(colvar):
            if not isinstance(cv, Colvar):
                raise TypeError(f'The type of the elements in `colvar` must be `Colvar, '
                                f'but the type of the {i}-th element is {type(cv)}')
            if use_pbc is not None:
                cv.set_pbc(use_pbc)
            colvar_.append(cv)
            shape_.append(cv.shape)

        self.colvar: List[Colvar] = CellList(colvar_)

        if shape is None:
            shape_ = set(shape_)
            if len(shape_) != 1:
                raise ValueError('The `shape` cannot be `None` when the shape of each CV '
                                 'in `colvar` is not exactly the same')
            shape = list(shape_)[0]

        self._set_shape(shape)

    def set_pbc(self, use_pbc: bool):
        """set whether to use periodic boundary condition"""
        self._use_pbc = use_pbc
        self.get_vector.set_pbc(use_pbc)
        for i in range(self.num_colvar):
            self.colvar[i].set_pbc(use_pbc)
        return self

    def construct(self, coordinate: Tensor, pbc_box: Tensor = None):
        r"""get position coordinates of colvar group

        Args:
            coordinate (Tensor):    Tensor of shape `(B, A, D)`. Data type is float.
                                    Position coordinate of colvar in system.
                                    `B` means batchsize, i.e. number of walkers in simulation.
                                    `A` means number of colvar in system.
                                    `D` means dimension of the simulation system. Usually is 3.
            pbc_box (Tensor):       Tensor of shape `(B, D)`. Data type is float.
                                    Tensor of PBC box. Default: ``None``.

        Returns:
            combine (Tensor):       Tensor of shape `(B, S_1, S_2, ..., S_n)`. Data type is float.
                                    `{S_i}` means dimensions of collective variables.

        """
        colvar = ()
        for i in range(self.num_colvar):
            colvar += (self.colvar[i](coordinate, pbc_box),)

        return self.function(*colvar)
