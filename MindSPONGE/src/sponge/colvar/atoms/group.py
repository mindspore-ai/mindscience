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
from numpy import ndarray
from mindspore import ops
from mindspore.ops import functional as F
from mindspore import Tensor, Parameter
from mindspore.nn import CellList

from .atoms import AtomsBase, Atoms
from ...function import get_integer


class Group(AtomsBase):
    r"""
    Group of atoms.

    Args:
        atoms (Union[List[AtomsBase], Tuple[AtomsBase]]):
                            List of AtomsBase. Member should be the subclass of AtomsBase.

        batched (bool):     Whether the first dimension of index is the batch size.
                            Default: ``False``.

        keep_in_box (bool): Whether to displace the coordinate in PBC box.
                                Default: ``False``.

        axis (int):         Axis to concatenate the coordinates of atoms.

        name (str):         Name of the Colvar. Default: 'atoms_group'.

    Supported Platforms:
        ``Ascend`` ``GPU``

    """
    def __init__(self,
                 atoms: Union[List[AtomsBase], Tuple[AtomsBase]],
                 batched: bool = False,
                 keep_in_box: bool = False,
                 axis: int = 1,
                 name: str = 'atoms_group',
                 ):

        super().__init__(
            keep_in_box=keep_in_box,
            name=name,
        )

        if isinstance(atoms, AtomsBase):
            atoms = [atoms]
        elif isinstance(atoms, (Tensor, Parameter, ndarray)):
            atoms = [Atoms(atoms, batched, keep_in_box)]
        elif isinstance(atoms, (list, tuple)):
            if set(map(type, atoms)) == {int}:
                atoms = [Atoms(atoms, batched, keep_in_box)]
        else:
            raise TypeError(f'The type of "atoms" must be list, tuple or AtomsBase, '
                            f'but got: {type(atoms)}')

        self.num_groups = len(atoms)

        axis = get_integer(axis)
        if axis in (0, -1):
            raise ValueError(f'The axis ({axis}) cannot be 0 or -1!')

        atoms_ = []
        dim = 0
        shape = None
        periodic = ()
        for i, a in enumerate(atoms):
            if isinstance(a, (Tensor, Parameter, ndarray)):
                a = Atoms(a, batched, keep_in_box)
            elif isinstance(atoms, (list, tuple)):
                if set(map(type, atoms)) == {int}:
                    atoms = [Atoms(atoms, batched, keep_in_box)]
            elif not isinstance(a, AtomsBase):
                raise TypeError(f'The type of elements in "atoms" must be AtomsBase, Tensor, Parameter or ndarray '
                                f'but got: {type(a)}')

            shape_ = (1,) + a.shape
            dim += shape_[axis]
            shape_ = shape_[:axis] + shape_[axis+1:]
            periodic += (F.expand_dims(a.periodic, 0),)

            if i > 0 and shape_ != shape:
                raise ValueError(f'The shape of the No.{i} AtomsBase {a.shape} cannot be '
                                 f'concatenate with the shape of previous one: {shape}')
            shape = shape_

            atoms_.append(a)

        self.atoms: List[AtomsBase] = CellList(atoms_)

        shape = shape[:axis] + (dim,) + shape[axis:]
        self._shape = shape[1:]
        self._ndim = len(self._shape)

        self.concat = ops.Concat(axis)

        self._periodic = F.squeeze(self.concat(periodic), 0)

    def construct(self, coordinate: Tensor, pbc_box: Tensor = None):
        r"""get position coordinates of atoms group

        Args:
            coordinate (Tensor):    Tensor of shape (B, A, D). Data type is float.
                                    Position coordinate of atoms in system.
                                    `B` means batchsize, i.e. number of walkers in simulation.
                                    `A` means number of colvar in system.
                                    `D` means dimension of the simulation system. Usually is 3.
            pbc_box (Tensor):       Tensor of shape (B, D). Data type is float.
                                    Tensor of PBC box. Default: ``None``.

        Returns:
            position (Tensor):  Tensor of shape (B, a_1, a_2, ..., a_n, D). Data type is float.
            `a_{i}` means Dimension of specific atoms.
        """
        atoms = ()
        for i in range(self.num_groups):
            # (B, a_1'(i), a_2, ..., a_n, D)
            atoms += (self.atoms[i](coordinate, pbc_box),)

        # (B, a_1, a_2, ..., a_n, D) <- (B, a_1'(i), a_2, ..., a_n, D)
        atoms = self.concat(atoms)

        if pbc_box is not None and self.keep_in_box:
            atoms = self.coordinate_in_pbc(atoms, pbc_box)

        if self.do_reshape:
            new_shape = coordinate.shape[0] + self._shape
            atoms = F.reshape(atoms, new_shape)

        return atoms
