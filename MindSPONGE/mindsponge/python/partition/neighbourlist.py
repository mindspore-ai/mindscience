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
Neighbour list
"""

import mindspore as ms
import mindspore.numpy as msnp
from mindspore import Tensor, jit
from mindspore import Parameter
from mindspore import ops
from mindspore.ops import functional as F
from mindspore.nn import Cell

from . import FullConnectNeighbours, DistanceNeighbours, GridNeighbours
from ..system import Molecule


class NeighbourList(Cell):
    r"""
    Neighbour list.

    Args:
        system (Molecule):      Simulation system.
        cutoff (float):         Cutoff distance. Default: None
        update_steps (int):     Steps of update frequency. Default: 20
        exclude_index (Tensor): Tensor of shape (B, A, Ex). Data type is int.
                                Index of neighbour atoms which could be excluded from the neighbour list.
                                Default: None
        num_neighbours (int):   Number of neighbours. If input "None", this value will be calculated by
                                the ratio of the number of neighbouring grids to the total number of grids.
                                Default: None
        num_cell_cut (int):     Number of subdivision of grid cells according to cutoff. Default: 1
        cutoff_scale (float):   Factor to scale cutoff distance. Default: 1.2
        cell_cap_scale (float): Scale factor for "cell_capacity". Default: 1.25
        grid_num_scale (float): Scale factor to calculate "num_neighbours" by ratio of grids.
                                If "num_neighbours" is not None, it will not be used. Default: 1.5
        large_dis (float):      A large number of distance to fill the default atoms. Default: 1e4
        use_grids (bool):       Whether to use grids to calculate the neighbour list. Default: None

    Supported Platforms:
        ``Ascend`` ``GPU``

    Symbols:
        B:  Number of simulation walker.
        A:  Number of atoms in system.
        Ex: Maximum number of excluded neighbour atoms.
    """

    def __init__(self,
                 system: Molecule,
                 cutoff: float = None,
                 update_steps: int = 20,
                 exclude_index: Tensor = None,
                 num_neighbours: int = None,
                 num_cell_cut: int = 1,
                 cutoff_scale: float = 1.2,
                 cell_cap_scale: float = 1.25,
                 grid_num_scale: float = 2,
                 large_dis: float = 1e4,
                 use_grids: bool = None,
                 ):

        super().__init__()

        self.num_walker = system.num_walker
        self.coordinate = system.get_coordinate()
        self.num_atoms = self.coordinate.shape[-2]
        self.dim = self.coordinate.shape[-1]

        self.pbc_box = system.get_pbc_box()

        self.atom_mask = system.atom_mask
        self.exclude_index = exclude_index
        if exclude_index is not None:
            self.exclude_index = Tensor(exclude_index, ms.int32)

        large_dis = Tensor(large_dis, ms.float32)
        self.units = system.units
        self.use_grids = use_grids

        self.no_mask = False
        if cutoff is None:
            self.cutoff = None
            self.neighbour_list = FullConnectNeighbours(self.num_atoms)
            if self.exclude_index is None:
                self.no_mask = True
        else:
            self.cutoff = Tensor(cutoff, ms.float32)
            if self.use_grids or self.use_grids is None:
                self.neighbour_list = GridNeighbours(
                    cutoff=self.cutoff,
                    coordinate=self.coordinate,
                    pbc_box=self.pbc_box,
                    atom_mask=self.atom_mask,
                    exclude_index=self.exclude_index,
                    num_neighbours=num_neighbours,
                    num_cell_cut=num_cell_cut,
                    cutoff_scale=cutoff_scale,
                    cell_cap_scale=cell_cap_scale,
                    grid_num_scale=grid_num_scale,
                )
                if self.neighbour_list.neigh_capacity >= self.num_atoms:
                    if self.use_grids:
                        print('Warning! The number of neighbour atoms in GridNeighbours (' +
                              str(self.neighbour_list.neigh_capacity) +
                              ') is not less than the number of atoms ('+str(self.num_atoms) +
                              '. It would be more efficient to use "DistanceNeighbours"'
                              ' (set "use_grids" to False or None).')
                    else:
                        self.use_grids = False
                else:
                    self.use_grids = True

            if not self.use_grids:
                if num_neighbours is None and self.pbc_box is not None:
                    op = ms.ops.ReduceProd(keep_dims=True)
                    volume = msnp.min(op(self.pbc_box, -1))
                    num_neighbours = grid_num_scale * self.num_atoms * \
                        msnp.power(cutoff*cutoff_scale, self.dim) / volume
                    num_neighbours = num_neighbours.astype(ms.int32)

                self.neighbour_list = DistanceNeighbours(
                    cutoff=self.cutoff,
                    num_neighbours=num_neighbours,
                    atom_mask=self.atom_mask,
                    exclude_index=self.exclude_index,
                    use_pbc=True,
                    cutoff_scale=cutoff_scale,
                    large_dis=large_dis
                )

        self.num_neighbours = self.neighbour_list.num_neighbours

        self.update_steps = update_steps
        if update_steps <= 0:
            raise ValueError('update_steps must be larger than 0!')

        index, mask = self.calcaulate(self.coordinate, self.pbc_box)

        self.neighbours = Parameter(
            index, name='neighbours', requires_grad=False)
        if self.cutoff is None and self.exclude_index is None:
            self.neighbour_mask = None
        else:
            self.neighbour_mask = Parameter(
                mask, name='neighbour_mask', requires_grad=False)

        self.identity = ops.Identity()

    def set_exclude_index(self, exclude_index: Tensor):
        """
        set exclude index.

        Args:
            exclude_index (Tensor): Tensor of exclude indexes.

        Returns:
            bool.
        """
        if exclude_index is None:
            return True
        self.exclude_index = Tensor(exclude_index, ms.int32)
        self.neighbour_list.set_exclude_index(exclude_index)
        index, mask = self.calcaulate(self.coordinate, self.pbc_box)
        success = True
        success = F.depend(success, F.assign(self.neighbours, index))
        if self.neighbour_mask is None:
            self.neighbour_mask = Parameter(
                mask, name='neighbour_mask', requires_grad=False)
        else:
            success = F.depend(success, F.assign(self.neighbour_mask, mask))
        return success

    def print_info(self):
        """print information of neighbour list"""
        self.neighbour_list.print_info()
        return self

    @jit
    def calcaulate(self, coordinate: Tensor, pbc_box: Tensor = None):
        """
        calculate neighbour list.

        Args:
            coordinate (Tensor):    Tensor of coordinates.
            pbc_box (Tensor):       Tensor of PBC box.

        Returns:
            - index(Tensor).
            - mask(Tensor).
        """
        coordinate = F.stop_gradient(coordinate)
        pbc_box = F.stop_gradient(pbc_box)
        if self.cutoff is None:
            return self.neighbour_list(self.atom_mask, self.exclude_index)

        if self.use_grids:
            return self.neighbour_list(coordinate, pbc_box)

        _, index, mask = self.neighbour_list(
            coordinate, pbc_box, self.atom_mask, self.exclude_index)

        return index, mask

    def get_neighbour_list(self):
        """
        get neighbour list.

        Returns:
            - index(Tensor).
            - mask(Tensor).
        """
        index = self.identity(self.neighbours)
        mask = None
        if self.neighbour_mask is not None:
            mask = self.identity(self.neighbour_mask)
        return index, mask

    def construct(self, coordinate: Tensor, pbc_box: Tensor = None) -> bool:
        r"""
        Gather coordinate of neighbours atoms.

        Args:
            coordinate (Tensor):    Tensor of shape (B,A,D). Data type is float.
            pbc_box (Tensor):       Tensor of shape (B,D). Data type is float.

        Returns:
            - neighbours (Tensor), Tensor of shape (B,A,N). Data type is int.
            - neighbour_mask (Tensor or None), Tensor of shape (B,A,N). Data type is bool.

        Symbols:
            B:  Number of simulation walker.
            A:  Number of atoms in system.
            N:  Number of neighbour atoms.
            D:  Dimension of position coordinates.
            Ex: Maximum number of excluded neighbour atoms.
        """

        neighbours, neighbour_mask = self.calcaulate(coordinate, pbc_box)
        success = True
        success = F.depend(success, F.assign(self.neighbours, neighbours))
        if self.neighbour_mask is not None:
            success = F.depend(success, F.assign(self.neighbour_mask, neighbour_mask))
        return success
