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
Use the distances between atoms to calculate neighbour list
"""

import mindspore as ms
import mindspore.numpy as msnp
from mindspore import Tensor
from mindspore.nn import Cell
from mindspore import ops
from mindspore.ops import functional as F

from ..function.functions import get_integer
from ..function.functions import calc_distance
from ..function.functions import calc_distance_with_pbc
from ..function.functions import calc_distance_without_pbc


class DistanceNeighbours(Cell):
    r"""
    Neighbour list calculated by distance.

    Args:
        cutoff (float):         Cutoff distance.
        num_neighbours (int):   Number of neighbours. If input "None", this value will be calculated by
                                the ratio of the number of neighbouring grids to the total number of grids.
                                Default: None
        atom_mask (Tensor):     Tensor of shape (B, A). Data type is bool\_.
                                Mask of atoms in the system. Default: None
        exclude_index (Tensor): Tensor of shape (B, A, Ex). Data type is int32.
                                Index of neighbour atoms which could be excluded from the neighbour list.
                                Default: None
        use_pbc (bool):         Whether to use periodic boundary condition. Default: None
        cutoff_scale (float):   Factor to scale the cutoff distance. Default: 1.2
        large_dis (float):      A large number of distance to fill the default atoms. Default: 1e4

    Supported Platforms:
        ``Ascend`` ``GPU``

    Symbols:
        B:  Number of simulation walker.
        A:  Number of atoms in system.
        Ex: Maximum number of excluded neighbour atoms.
    """

    def __init__(self,
                 cutoff: float,
                 num_neighbours: int = None,
                 atom_mask: Tensor = None,
                 exclude_index: Tensor = None,
                 use_pbc: bool = None,
                 cutoff_scale: float = 1.2,
                 large_dis: float = 1e4,
                 ):

        super().__init__()

        self.cutoff = Tensor(cutoff, ms.float32)
        self.cutoff_scale = Tensor(cutoff_scale, ms.float32)
        self.scaled_cutoff = self.cutoff * self.cutoff_scale

        self.num_neighbours = get_integer(num_neighbours)

        self.large_dis = Tensor(large_dis, ms.float32)

        self.emtpy_atom_shift = 0
        self.atom_mask = None
        self.has_empty_atom = False
        if atom_mask is not None:
            # (B,A)
            self.atom_mask = Tensor(atom_mask, ms.bool_)
            if self.atom_mask.ndim == 1:
                self.atom_mask = F.expand_dims(self.atom_mask, 0)

            self.has_empty_atom = F.logical_not(self.atom_mask.all())
            if self.has_empty_atom:
                emtpy_atom_mask = F.logical_not(self.atom_mask)
                # (B,1,A)
                self.emtpy_atom_shift = F.expand_dims(emtpy_atom_mask, -2) * self.large_dis

        self.exclude_index = None
        if exclude_index is not None:
            # (B,A,E)
            self.exclude_index = Tensor(exclude_index, ms.int32)
            if self.exclude_index.ndim == 2:
                self.exclude_index = F.expand_dims(self.exclude_index, 0)

        if use_pbc is None:
            self.get_distances = calc_distance
        else:
            if use_pbc:
                self.get_distances = calc_distance_with_pbc
            else:
                self.get_distances = calc_distance_without_pbc

        self.sort = ops.Sort(-1)
        self.reduce_all = ops.ReduceAll()

    def set_exclude_index(self, exclude_index: Tensor):
        # (B,A,Ex)
        self.exclude_index = Tensor(exclude_index, ms.int32)
        if self.exclude_index.ndim == 2:
            self.exclude_index = F.expand_dims(self.exclude_index, 0)
        return self

    def print_info(self):
        return self

    def check_neighbours_number(self, neighbour_mask: Tensor):
        """
        check number of neighbours in neighbour list.

        Args:
            neighbour_mask (Tensor):    The neighbour list mask.
        """
        max_neighbours = F.cast(msnp.max(F.cast(msnp.sum(neighbour_mask, -1), ms.float32)), ms.int32)
        if max_neighbours > self.num_neighbours:
            print(
                '================================================================================')
            print(
                'Warning! Warning! Warning! Warning! Warning! Warning! Warning! Warning! Warning!')
            print(
                '--------------------------------------------------------------------------------')
            print('The max number of neighbour atoms is larger than that in neighbour list!')
            print('The max number of neighbour atoms:')
            print(max_neighbours)
            print('The number of neighbour atoms in neighbour list:')
            print(self.num_neighbours)
            print('Please increase the value of grid_num_scale or num_neighbours!')
            print(
                '================================================================================')
        return self

    def construct(self,
                  coordinate: Tensor,
                  pbc_box: Tensor = None,
                  atom_mask: Tensor = None,
                  exclude_index: Tensor = None
                  ):
        r"""
        Calculate distances and neighbours.

        Args:
            coordinate (Tensor):    Tensor of (B, A, D). Data type is float.
                                    Position coordinates of atoms.
            pbc_box (Tensor):       Tensor of (B, D). Data type is bool.
                                    Periodic boundary condition box. Default: None
            atom_mask (Tensor):     Tensor of (B, A). Data type is bool.
                                    Atomic mask.
            exclude_index (Tensor): Tensor of (B, A, Ex). Data type is int.
                                    Index of the atoms that should be excluded from the neighbour list.
                                    Default: None

        Returns:
            - distances (Tensor), Tensor of (B, A, N). Data type is float.
            - neighbours (Tensor), Tensor of (B, A, N). Data type is int.
            - neighbour_mask (Tensor), Tensor of (B, A, N). Data type is bool.

        Symbols:
            B:  Batch size.
            A:  Number of atoms in system.
            N:  Number of neighbour atoms.
            D:  Dimension of position coordinates.
            Ex: Maximum number of excluded neighbour atoms.
        """

        # A
        num_atoms = coordinate.shape[-2]
        # (B,A,A) <- (B,A,1,3) - (B,1,A,3)
        distances = self.get_distances(F.expand_dims(
            coordinate, -2), F.expand_dims(coordinate, -3), pbc_box).squeeze(-1)

        if atom_mask is None:
            atom_mask = self.atom_mask
            if self.has_empty_atom:
                # (B,A,A) + (B,1,A)
                distances += self.emtpy_atom_shift
        else:
            if not atom_mask.all():
                emtpy_atom_mask = F.logical_not(atom_mask)
                # (B,1,A)
                emtpy_atom_shift = F.expand_dims(
                    emtpy_atom_mask, -2) * self.large_dis
                distances += emtpy_atom_shift

        distances, neighbours = self.sort(distances)
        # (B,A)
        neighbour_mask = distances < self.scaled_cutoff

        if self.num_neighbours is None:
            num_neighbours = num_atoms - 1
        else:
            num_neighbours = self.num_neighbours

        distances = distances[..., 1:num_neighbours+1]
        neighbours = neighbours[..., 1:num_neighbours+1]
        neighbour_mask = neighbour_mask[..., 1:num_neighbours+1]
        if self.num_neighbours is not None:
            self.check_neighbours_number(neighbour_mask)

        if exclude_index is None:
            exclude_index = self.exclude_index
        if exclude_index is not None:
            # (B,A,n,E) <- (B,A,n,1) != (B,A,1,E)
            exc_mask = F.expand_dims(
                neighbours, -1) != F.expand_dims(exclude_index, -2)
            # (B,A,n)
            exc_mask = self.reduce_all(exc_mask, -1)
            neighbour_mask = F.logical_and(neighbour_mask, exc_mask)

        if atom_mask is not None:
            # (B,A,n) <- (B,A,n) && (B,A,1)
            neighbour_mask = F.logical_and(
                neighbour_mask, F.expand_dims(atom_mask, -1))

        # (B,A,n)
        no_idx = msnp.arange(num_atoms).reshape(1, -1, 1)
        neighbours = msnp.where(neighbour_mask, neighbours, no_idx)

        return distances, neighbours, neighbour_mask
