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
Center of atoms
"""

from typing import Union
from numpy import ndarray
import mindspore as ms
from mindspore import ops
from mindspore import Tensor
from mindspore.ops import functional as F

from .atoms import AtomsBase
from .get import get_atoms
from ...function import get_ms_array, get_integer


class Center(AtomsBase):
    r"""
    Center of specific atoms

    Args:
        atoms (Union[AtomsBase, Tensor, ndarray, list]):
                            Specific atoms or virtual atoms of shape (..., G, D).
                            `G` means number of the group of atoms to be averaged.
                            `D` means spatial dimension of the simulation system. Usually is 3.
        mass (Union[Tensor, ndarray, list]):
                            Array of the mass of the atoms to calculate the center of mass (COM).
                            The shape of Tensor is (..., G) or (B, ..., G), and the data type is float.
                            If it is None, the geometric center of coordinate will be calculated.
                            Default: ``None``. `B` means Batchsize, i.e. number of walkers in simulation.

        batched (bool):     Whether the first dimension of index and mass is the batch size.
                            Default: ``False``.

        keep_in_box (bool): Whether to displace the coordinate in PBC box.
                            Default: ``False``.

        keepdims (bool):    If this is set to True, the axis which is reduced will be left,
                            and the shape the center will be (..., 1, D).
                            If this is set to False, the shape of the center will be (..., D).
                            if None, its value will be determined according to the rank (number of dimension) of
                            the input atoms: False if the rank is greater than 2, otherwise True.
                            Default: ``None``.

        axis (int):         Axis along which the average of position are coumputed. Default: -2

        name (str):         Name of the Colvar. Default: 'atoms_center'

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import mindspore as ms
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from sponge.colvar import Center
        >>> crd = Tensor(np.random.random((4, 3)), ms.float32)
        >>> mass = Tensor(np.random.random(4), ms.float32)
        >>> atoms = Tensor([0, 2], ms.int32)
        >>> ct = Center(atoms, mass[atoms])
        >>> ct(crd)
        Tensor(shape=[1, 3], dtype=Float32, value=
        [[ 7.61003494e-01,  6.70868099e-01,  5.67968249e-01]])
    """
    def __init__(self,
                 atoms: Union[AtomsBase, Tensor, ndarray, list],
                 mass: Union[Tensor, ndarray, list] = None,
                 batched: bool = False,
                 keep_in_box: bool = False,
                 keepdims: bool = None,
                 axis: int = -2,
                 name: str = 'atoms_center',
                 ):

        super().__init__(
            keep_in_box=keep_in_box,
            name=name,
        )

        self.axis = get_integer(axis)
        if self.axis == 0 or self.axis == -1:
            raise ValueError(f'The axis ({self.axis}) cannot be 0 or -1!')

        self.atoms = get_atoms(atoms, batched)

        if keepdims is None:
            if self.atoms.ndim > 2:
                keepdims = False
            else:
                keepdims = True

        # (B, ..., G, D)
        shape = (1,) + self.atoms.shape
        if keepdims:
            # shape of coordinate: (B, ..., 1, D)
            shape = shape[:self.axis] + (1,) + shape[self.axis+1:]
        else:
            # shape of coordinate: (B, ..., D)
            shape = shape[:self.axis] + shape[self.axis+1:]

        # (..., D) or (..., 1, D)
        self._set_shape(shape[1:])

        self.reduce_mean = ops.ReduceMean(keepdims)
        self.reduce_sum = ops.ReduceSum(keepdims)

        self.mass = None
        self.total_mass = None
        self.set_mass(mass, batched)

    def set_mass(self, mass: Tensor, batched: bool = False):
        """set the mass of atoms"""
        self.mass = get_ms_array(mass, ms.float32)
        if self.mass is None:
            self.total_mass = None
        else:
            if batched:
                # shape of mass: (B, ..., G)
                if self.mass.shape[1:] != self.atoms.shape[:-1]:
                    raise ValueError(f'The shape of mass {self.mass.shape} does not match '
                                     f'the shape of atoms ({self.atoms.shape}).')
            else:
                # shape of mass: (..., G)
                if self.mass.shape != self.atoms.shape[:-1]:
                    raise ValueError(f'The shape of mass {self.mass.shape} must be equal to '
                                     f'the shape of atoms {self.atoms.shape}')
            # (B, ..., 1, 1) or (B, ..., 1)
            self.total_mass = self.reduce_sum(F.expand_dims(self.mass, -1), self.axis)
        return self

    def construct(self, coordinate: Tensor, pbc_box: Tensor = None):
        r"""calculate the position of the center of specific atom(s)

        Args:
            coordinate (Tensor):    Tensor of shape (B, A, D). Data type is float.
                                    Position coordinate of atoms in system. A means number of atoms in system.
            pbc_box (Tensor):       Tensor of shape (B, D). Data type is float.
                                    Tensor of PBC box. Default: ``None``.

        Returns:
            center (Tensor):        Tensor of shape (B, ..., D). Data type is float.
                                    Position of the center of the atoms.
        """
        # (B, ..., G, D) <- (B, A, D)
        atoms = self.atoms(coordinate, pbc_box)

        if self.mass is None:
            # (B, ..., 1, D) or (B, ..., D) <- (B, ..., G, D)
            center = self.reduce_mean(atoms, self.axis)
        else:
            # (..., G, 1) or (B, ..., G, 1)
            mass = F.expand_dims(self.mass, -1)
            # 1. (B, ..., G, D) * (..., G, 1) OR (B, ..., G, D) * (B, ..., G, 1)
            # 2. (B, ..., D) or (B, ..., 1, D) <- (B, ..., G, D)
            # 3. (B, ..., D) / (B, ..., 1) OR (B, ..., 1, D) / (B, ..., 1, 1)
            center = self.reduce_sum(atoms * mass, self.axis) / self.total_mass

        if self.do_reshape:
            new_shape = coordinate.shape[0] + self._shape
            center = F.reshape(center, new_shape)

        # (B, ..., D) or (B, ..., 1, D)
        return center
