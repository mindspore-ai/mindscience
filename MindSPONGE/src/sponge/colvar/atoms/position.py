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

from typing import Union
from numpy import ndarray
import mindspore as ms
import mindspore.numpy as msnp
from mindspore import ops
from mindspore.ops import functional as F
from mindspore import Tensor, Parameter

from .atoms import AtomsBase
from ...function import get_ms_array

__all__ = ['Position', 'BatchedPosition']


class Position(AtomsBase):
    r"""
    Virtual atom(s) in fixed position(s).

    Args:
        coordinate (Union[Tensor, Parameter, ndarray]):
                            Array of th position coordinate(s) of specific virtual atom(s).
                            The shape of Tensor is (a_1, a_2, ..., a_n, D), and the data type is float.
                            `a_{i}` means dimension of specific atoms.
                            `D` means dimension of the simulation system. Usually is 3.

        batched (bool):     Whether the first dimension of coordinate is the batch size.
                            Default: ``False``.

        keep_in_box (bool): Whether to displace the coordinate in PBC box.
                            Default: ``False``.

        dimension (int):    Space dimension of system. Default: 3.

        name (str):         Name of the Colvar. Default: 'position'.

    Supported Platforms:
        ``Ascend`` ``GPU``

    """
    def __init__(self,
                 coordinate: Union[Tensor, Parameter, ndarray],
                 batched: bool = False,
                 keep_in_box: bool = False,
                 name: str = 'position',
                 ):

        super().__init__(
            keep_in_box=keep_in_box,
            name=name,
        )

        self.coordinate = get_ms_array(coordinate, ms.float32)
        if batched:
            if self.coordinate.ndim < 2:
                raise ValueError('The rank of coordinate must be larger than 1 '
                                 'when using batched coordinate.')
        else:
            self.coordinate = F.expand_dims(self.coordinate, 0)

        if self.coordinate.ndim == 2:
            self.coordinate = F.expand_dims(self.coordinate, -2)

        self._set_shape(self.coordinate.shape[1:])

        self.identity = ops.Identity()

    def update(self, coordinate: Tensor):
        self.coordinate = coordinate
        return self

    def reshape(self, input_shape: tuple):
        """rearranges the shape of atoms"""
        shape = self.coordinate.shape[0] + input_shape
        self.coordinate = F.reshape(self.coordinate, shape)
        self._ndim = self.coordinate.ndim - 1
        self._shape = self.coordinate.shape[1:]
        self._periodic = msnp.full(self._shape, False)
        return self

    def construct(self, coordinate: Tensor, pbc_box: Tensor = None):
        r"""get position coordinate(s) of virtual atom(s)

        Args:
            coordinate (Tensor):    Tensor of shape (B, A, D). Data type is float.
                                    Position coordinate of atoms in system.
                                    `B` means batchsize, i.e. number of walkers in simulation.
                                    `A` means number of atoms in system.
            pbc_box (Tensor):       Tensor of shape (B, D). Data type is float.
                                    Tensor of PBC box. Default: ``None``.

        Returns:
            position (Tensor):  Tensor of shape (B, a_1, a_2, ..., a_n, D). Data type is float.
        """

        if coordinate.shape[0] > 1:
            shape = (coordinate.shape[0],) + self.coordinate.shape[1:]
            position = msnp.broadcast_to(self.coordinate, shape)
        else:
            position = self.identity(self.coordinate)

        if self.keep_in_box:
            return self.coordinate_in_pbc(position, pbc_box)
        return position


class BatchedPosition(Position):
    r"""
    Virtual atom(s) in fixed position(s) with batched coordinate

    Args:
        coordinate (Tensor):    Tensor of shape (B, a_1, a_2, ..., a_n, D). Data type is float.
                                Position coordinate(s) of virtual atom(s).

        keep_in_box (bool):     Whether to displace the coordinate in PBC box.
                                Default: ``False``.

        dimension (int):        Space dimension of system. Default: 3

        name (str):             Name of the Colvar. Default: 'position'

    Supported Platforms:
        ``Ascend`` ``GPU``

    """
    def __init__(self,
                 coordinate: Union[Tensor, Parameter, ndarray],
                 keep_in_box: bool = False,
                 dimension: int = 3,
                 name: str = 'position',
                 ):

        super().__init__(
            coordinate=coordinate,
            batched=True,
            keep_in_box=keep_in_box,
            dimension=dimension,
            name=name,
        )
