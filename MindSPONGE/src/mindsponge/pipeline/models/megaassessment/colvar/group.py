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
Atom group
"""

from typing import Union, List, Tuple
from mindspore import ops
from mindspore.ops import functional as F
from mindspore import Tensor
from mindspore.nn import CellList

from .colvar import Colvar
from ..function import get_integer


class ColvarGroup(Colvar):
    r"""Concatenate a group of `Colvar` classes into one `Colvar` class

    Args:
        colvar (list or tuple):
                    Array of `Colvar` to be concatenated.

        axis (int): Axis to be concatenated. NOTE: This refers to the axis of the output Tensor
                    with the shape `(B, S_1, S_2, ..., S_n)`. Default: -1

        name (str): Name of the collective variables. Default: 'colvar_group'

    Supported Platforms:
        ``Ascend`` ``GPU``

    """
    def __init__(self,
                 colvar: Union[List[Colvar], Tuple[Colvar]],
                 axis: int = -1,
                 use_pbc: bool = None,
                 name: str = 'colvar_group',
                 ):

        super().__init__(name=name)

        if isinstance(colvar, Colvar):
            colvar = [colvar]
        elif not isinstance(colvar, (list, tuple)):
            raise TypeError(f'The type of "colvar" must be list of Colvar but got: {type(colvar)}')

        self.num_colvar = len(colvar)
        axis = get_integer(axis)
        if axis == 0:
            raise ValueError(f'The axis ({axis}) cannot be 0 (the dimension of batch size)!')

        shape = None
        dim = 0
        periodic = ()
        colvar_ = []
        for i, cv in enumerate(colvar):
            shape_ = (1,) + cv.shape
            dim += shape_[axis]

            if axis == -1:
                shape_ = shape_[:-1] + (None,)
            else:
                shape_ = shape_[:axis] + (None,) + shape_[axis+1:]
            if i > 0 and shape_ != shape:
                raise ValueError(f'The shape of the No.{i} colvar {cv.shape} cannot be '
                                 f'concatenate with the shape of the colvar group: {shape}')
            shape = shape_

            if use_pbc is not None:
                cv.set_pbc(use_pbc)
            colvar_.append(cv)

            periodic += (F.expand_dims(cv.periodic, 0),)

        self.colvar = CellList(colvar_)

        if axis == -1:
            shape = shape[:-1] + (dim,)
        else:
            shape = shape[:axis] + (dim,) + shape[axis+1:]

        self._shape = shape[1:]
        self._ndim = len(self._shape)

        self.concat = ops.Concat(axis)

        self._periodic = F.squeeze(self.concat(periodic), 0)

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
                                    Position coordinate of colvar in system
            pbc_box (Tensor):       Tensor of shape `(B, D)`. Data type is float.
                                    Tensor of PBC box. Default: ``None``.

        Returns:
            position (Tensor):  Tensor of shape `(B, S_1, S_2, ..., S_n)`. Data type is float.

        Note:
            B:      Batchsize, i.e. number of walkers in simulation
            A:      Number of colvar in system.
            {S_i}:  Dimensions of collective variables.
            D:      Dimension of the simulation system. Usually is 3.

        """
        colvar = ()
        for i in range(self.num_colvar):
            # (B, a_1'(i), a_2, ..., a_n, D)
            colvar += (self.colvar[i](coordinate, pbc_box),)

        # (B, a_1, a_2, ..., a_n, D) <- (B, a_1'(i), a_2, ..., a_n, D)
        colvar = self.concat(colvar)

        if self.do_reshape:
            new_shape = coordinate.shape[0] + self._shape
            colvar = F.reshape(colvar, new_shape)

        return colvar
