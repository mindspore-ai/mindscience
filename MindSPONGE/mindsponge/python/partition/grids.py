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
Use grids to calculate neighbour list
"""

import itertools
import numpy as np
import scipy.stats
import mindspore as ms
import mindspore.numpy as msnp
from mindspore.nn import Cell
from mindspore import Tensor
from mindspore import ops
from mindspore.ops import functional as F

from ..function.functions import get_integer, displace_in_box


class GridNeighbours(Cell):
    r"""
    Neighbour list calculated by grids.

    Args:
        cutoff (float):         Cutoff distance.
        coordinate (Tensor):    Tensor of shape (B, A, D). Data type is float32.
                                position coordinates of atoms in the simulation system.
        pbc_box (Tensor):       Tensor of shape (B, A, D). Data type is float32.
                                Box size of periodic boundary condition. Default: None
        atom_mask (Tensor):     Tensor of shape (B, A). Data type is bool\_.
                                Mask of atoms in the system.
                                Default: None
        exclude_index (Tensor): Tensor of shape (B, A, Ex). Data type is int32.
                                Index of neighbour atoms which could be excluded from the neighbour list.
                                Default: None
        num_neighbours (int):   Number of neighbours. If input "None", this value will be calculated by
                                the ratio of the number of neighbouring grids to the total number of grids.
                                Default: None
        cell_capacity (int):    Capacity number of atoms in grid cell. If input "None", this value will be multiplied
                                by a factor of the maximum number of atoms in the grid cell at the initial coordinate.
                                Default: None
        num_cell_cut (int):     Number of subdivision of grid cells according to the cutoff. Default: 1
        cutoff_scale (float):   Factor to scale the cutoff distance. Default: 1.2
        cell_cap_scale (float): Factor to scale "cell_capacity". Default: 1.25
        grid_num_scale (float): Scale factor to calculate "num_neighbours" by the ratio of grids.
                                If "num_neighbours" is not None, it will not be used. Default: 1.5

    Supported Platforms:
        ``Ascend`` ``GPU``

    Symbols:
        B:  Number of simulation walker.
        A:  Number of atoms in system.
        D:  Dimension of position coordinates.
        Ex: Maximum number of excluded neighbour atoms.
    """

    def __init__(self,
                 cutoff: float,
                 coordinate: Tensor,
                 pbc_box: Tensor = None,
                 atom_mask: Tensor = None,
                 exclude_index: Tensor = None,
                 num_neighbours: int = None,
                 cell_capacity: int = None,
                 num_cell_cut: int = 1,
                 cutoff_scale: float = 1.2,
                 cell_cap_scale: float = 1.25,
                 grid_num_scale: float = 1.5,
                 ):

        super().__init__()

        self.num_atoms = coordinate.shape[-2]
        self.dim = coordinate.shape[-1]

        self.cutoff = Tensor(cutoff, ms.float32)

        self.cutoff_scale = Tensor(cutoff_scale, ms.float32)
        self.cell_cap_scale = Tensor(cell_cap_scale, ms.float32)
        self.grid_num_scale = Tensor(grid_num_scale, ms.float32)

        # N_c
        num_cell_cut = get_integer(num_cell_cut)

        self.grid_cutoff = self.cutoff / num_cell_cut
        self.scaled_cutoff = self.cutoff * self.cutoff_scale
        self.scaled_grid_cutoff = self.grid_cutoff * self.cutoff_scale

        if pbc_box is None:
            self.use_pbc = False
            # (B,1,D) <- (B,A,D)
            rmax = msnp.max(coordinate, -2, keepdims=True)
            rmin = msnp.min(coordinate, -2, keepdims=True)
            center = msnp.mean(coordinate, -2, keepdims=True)
            # (B,2,D)
            rhalf = msnp.concatenate((rmax-center, center-rmin))
            # (B,D)
            rhalf = msnp.max(rhalf, -2)
            # (D)
            rhalf = msnp.max(rhalf, 0)
            box = rhalf * 2
            self.origin_grid_dims = msnp.ceil(box/self.scaled_grid_cutoff).astype(np.int32)
            self.grid_dims = self.origin_grid_dims + 2
            box = self.origin_grid_dims * self.scaled_grid_cutoff
            self.half_box = box / 2
        else:
            self.use_pbc = True
            center = None
            # (B,D)
            box = Tensor(pbc_box, ms.float32)
            if box.ndim == 1:
                box = F.expand_dims(pbc_box, 0)
            self.half_box = box / 2
            if (self.cutoff > self.half_box).any():
                raise ValueError(
                    '"cutoff" cannot be greater than half the length of the shortest side of the PBC pbc_box!')
            # (B,D)
            self.origin_grid_dims = msnp.floor(box/self.scaled_grid_cutoff)
            # (D)
            self.origin_grid_dims = Tensor(
                np.min(self.origin_grid_dims.asnumpy(), axis=0).astype(np.int32))
            self.grid_dims = self.origin_grid_dims

        # (D)
        grid_mask = self.grid_dims > 3
        self.grid_dims = msnp.where(grid_mask, self.grid_dims, 1)
        self.max_grid_index = self.origin_grid_dims - 1

        # G
        self.num_grids = int(np.prod(self.grid_dims.asnumpy()))

        # (D)
        self.grid_factor = msnp.cumprod(self.grid_dims[::-1], axis=-1)
        self.grid_factor = msnp.concatenate(
            (self.grid_factor[1::-1], Tensor([1], ms.int32)), axis=-1)

        # (G,D)
        grids = [np.arange(dim).tolist() for dim in self.grid_dims.asnumpy()]
        grids = Tensor(tuple(itertools.product(*grids)), ms.int32)

        # (B,1,D)
        box = F.expand_dims(box, -2)
        if self.use_pbc:
            # (B,1,D) = (B,D) / (D)
            self.cell = box / self.grid_dims
            if (self.cell < self.grid_cutoff).any():
                raise ValueError(
                    'The cell length of cannot be smaller than cutoff!')
            # (B,A,D) = ((B,A,D) - (D)) / (B,1,D)
            atom_grid_idx = msnp.floor(
                (displace_in_box(coordinate, pbc_box))/self.cell).astype(ms.int32)
        else:
            self.cell = msnp.broadcast_to(self.scaled_grid_cutoff, (self.dim,))
            # (B,A,D) = (B,A,D) - (B,1,D) + (D)
            scaled_coord = (coordinate - center +
                            self.half_box) / self.scaled_grid_cutoff
            scaled_coord = msnp.where(scaled_coord < 0, 0, scaled_coord)
            atom_grid_idx = msnp.floor(scaled_coord).astype(ms.int32)
            atom_grid_idx = msnp.where(atom_grid_idx < self.origin_grid_dims,
                                       atom_grid_idx, self.max_grid_index)
            atom_grid_idx += 1

        # (B,A) <- (B,A,D) * (D)
        atom_grid_idx = msnp.sum(atom_grid_idx * self.grid_factor, axis=-1)

        # (D): [n_1,n_2,...n_D]
        num_extend_neigh = np.where(grid_mask.asnumpy(), num_cell_cut, 0)
        dim_neigh_grids = num_extend_neigh * 2 + 1
        self.num_neigh_grids = int(np.prod(dim_neigh_grids))
        self.dim_neigh_grids = Tensor(dim_neigh_grids)

        if cell_capacity is None:
            # (B,1)
            _, max_num_in_cell = scipy.stats.mode(
                atom_grid_idx.asnumpy(), axis=1)
            max_num_in_cell = get_integer(np.max(max_num_in_cell))
            # C
            cell_capacity = get_integer(
                msnp.ceil(max_num_in_cell*self.cell_cap_scale))
            self.cell_capacity = int(min(cell_capacity, self.num_atoms))
        else:
            self.cell_capacity = get_integer(cell_capacity)

        # N_cap = n * C
        self.neigh_capacity = self.num_neigh_grids * self.cell_capacity

        # G*C
        self.grid_cap = self.num_grids * self.cell_capacity
        self.sort_id_factor = msnp.mod(
            msnp.arange(self.num_atoms), self.cell_capacity)

        # (n,D)
        neigh_offsets = [np.arange(-num_extend_neigh[i], num_extend_neigh[i]+1,
                                   dtype=np.int32).tolist() for i in range(self.dim)]
        neigh_offsets = Tensor(
            tuple(itertools.product(*neigh_offsets)), ms.int32)

        if num_neighbours is None:
            if self.use_pbc:
                # N' = ceil(A * n / G * n_scale)
                num_neighbours = msnp.ceil(
                    self.num_atoms*self.num_neigh_grids/self.num_grids*self.grid_num_scale).asnumpy()
                # N = min(N',n*C)
                self.num_neighbours = int(min(num_neighbours, self.num_atoms))
            else:
                self.num_neighbours = int(
                    min(self.neigh_capacity, self.num_atoms))
        else:
            self.num_neighbours = get_integer(num_neighbours)
            if self.num_neighbours > self.num_atoms:
                raise ValueError(
                    'The value of "num_neighbours" cannot be larger than the number of atoms!')

        # (G,n,D)
        neigh_grids = F.expand_dims(grids, -2) + neigh_offsets
        # neigh_grids = msnp.select([neigh_grids<0, neigh_grids>=self.grid_dims],
        #                           [neigh_grids+self.grid_dims, neigh_grids-self.grid_dims], neigh_grids)
        neigh_grids = F.select(
            neigh_grids < 0, neigh_grids+self.grid_dims, neigh_grids)
        neigh_grids = F.select(neigh_grids >= self.grid_dims, neigh_grids-self.grid_dims, neigh_grids)

        # (G*n)
        self.neigh_idx = msnp.sum(
            neigh_grids*self.grid_factor, axis=-1).reshape(-1)
        self.atom_idx = msnp.arange(
            self.num_atoms).reshape(1, self.num_atoms, 1)

        if atom_mask is None:
            self.atom_mask = None
        else:
            # (B,A)
            self.atom_mask = Tensor(atom_mask, ms.bool_)
            if self.atom_mask.shape[-1] != self.num_atoms:
                raise ValueError('The number of atoms in atom_mask ('+str(self.atom_mask.shape[-1]) +
                                 ') is mismatch with that in coordinate ('+str(self.num_atoms)+').')
            if self.atom_mask.ndim == 1:
                self.atom_mask = F.expand_dims(self.atom_mask, 0)

        if exclude_index is None:
            self.exclude_index = None
        else:
            # (B,A,Ex)
            self.exclude_index = Tensor(exclude_index, ms.int32)
            if self.exclude_index.shape[-2] != self.num_atoms:
                raise ValueError('The number of atoms in exclude_index ('+str(self.exclude_index.shape[-2]) +
                                 ') is mismatch with that in coordinate ('+str(self.num_atoms)+').')
            if self.exclude_index.ndim == 2:
                self.exclude_index = F.expand_dims(self.exclude_index, 0)

        self.sort = ops.Sort(-1)
        self.reduce_all = ops.ReduceAll()

    def set_exclude_index(self, exclude_index: Tensor):
        """
        set excluded neighbour index.

        Args:
            exclude_index (Tensor): Tensor of excluded neighbour indexes.
        """
        # (B,A,Ex)
        self.exclude_index = Tensor(exclude_index, ms.int32)
        if self.exclude_index.shape[-2] != self.num_atoms:
            raise ValueError('The number of atoms in exclude_index ('+str(self.exclude_index.shape[-2]) +
                             ') is mismatch with that in coordinate ('+str(self.num_atoms)+').')
        if self.exclude_index.ndim == 2:
            self.exclude_index = F.expand_dims(self.exclude_index, 0)
        return self

    def check_neighbours_number(self, grid_neigh_atoms: Tensor, num_neighbours: int = None):
        """
        check number of neighbours in neighbour list.

        Args:
            grid_neigh_atoms (Tensor):  Tensor of grid of neighbour atoms.
            num_neighbours (int):       Number of neighbours.
        """
        if num_neighbours is None:
            num_neighbours = self.num_neighbours
        max_neighbours = msnp.sum(grid_neigh_atoms != self.num_atoms, axis=-1)
        max_neighbours = F.cast(
            msnp.max(F.cast(max_neighbours, ms.float32)), ms.int32)
        if max_neighbours > num_neighbours:
            print(
                '================================================================================')
            print(
                'Warning! Warning! Warning! Warning! Warning! Warning! Warning! Warning! Warning!')
            print(
                '--------------------------------------------------------------------------------')
            print('The max number of neighbour atoms '
                  'is larger than that in neighbour list!')
            print('The max number of neighbour atoms:')
            print(max_neighbours)
            print('The number of neighbour atoms in neighbour list:')
            print(num_neighbours)
            print('Please increase the value of grid_num_scale or num_neighbours!')
            print(
                '================================================================================')
        return self

    def print_info(self):
        """print information of neighbour list"""
        print('Calculate neighbour list from grids')
        print('   Cutoff distance: '+str(self.cutoff))
        print('   Grid cell length: '+str(self.scaled_grid_cutoff))
        print('   Initial size of grid cell: '+str(F.squeeze(self.cell)))
        print('   Grid dimensions: '+str(self.grid_dims))
        print('   Number of Grids: '+str(self.num_grids))
        print('   Grid cell capacity: '+str(self.cell_capacity))
        print('   Dimension of neighbour cells: '+str(self.dim_neigh_grids))
        print('   Number of atoms: '+str(self.num_atoms))
        print('   Max number of neighbour atoms: '+str(self.num_neighbours))
        return self

    def get_neighbours_from_grids(self, atom_grid_idx: Tensor, num_neighbours: int):
        """
        get neighbour list from grids

        Args:
            atom_grid_idx (Tensor): Tensor of atoms grid indexes.
            num_neighbours (int):   Number of neighbours.

        Returns:
            list, neighbour list from grids.
        """
        #pylint: disable=unused-argument
        # Sorted grid index
        # (B,A)
        sorted_grid_idx, sort_arg = self.sort(F.cast(atom_grid_idx, ms.float32))
        sorted_grid_idx = F.cast(sorted_grid_idx, ms.int32)
        sorted_grid_idx = sorted_grid_idx * self.cell_capacity + self.sort_id_factor

        num_walker = atom_grid_idx.shape[0]
        # Atom index in each grid
        # (B,G*C)
        scatter_shape = (num_walker, self.grid_cap)
        grid_atoms = msnp.full(scatter_shape, self.num_atoms)
        if num_walker == 1:
            grid_atoms[:, sorted_grid_idx[0]] = sort_arg
        else:
            # (B,1,1)
            batch_idx = msnp.arange(num_walker).reshape(num_walker, 1, 1)
            # (B,A,1)
            batch_idx = msnp.broadcast_to(
                batch_idx, (num_walker, self.num_atoms, 1))
            # (B,A,2)
            scatter_idx = msnp.concatenate(
                (batch_idx, F.expand_dims(sorted_grid_idx, -1)), axis=-1)
            grid_atoms = F.tensor_scatter_update(
                grid_atoms, scatter_idx, sort_arg)
        # (B,G,C)
        grid_atoms = F.reshape(
            grid_atoms, (num_walker, self.num_grids, self.cell_capacity))

        # Atom index in neighbour grids for each grid
        # (B,G*n,C)
        grid_neigh_atoms = F.gather(grid_atoms, self.neigh_idx, -2)
        # (B,G,n,C)
        shape = (num_walker, self.num_grids,
                 self.num_neigh_grids, self.cell_capacity)
        grid_neigh_atoms = F.reshape(grid_neigh_atoms, shape)
        # (B,G,n*C)
        shape = (num_walker, self.num_grids,
                 self.num_neigh_grids*self.cell_capacity)
        grid_neigh_atoms = F.reshape(grid_neigh_atoms, shape)
        grid_neigh_atoms, _ = self.sort(F.cast(grid_neigh_atoms, ms.float32))
        grid_neigh_atoms = F.cast(grid_neigh_atoms, ms.int32)

        self.check_neigbours_number(grid_neigh_atoms, num_neighbours)
        grid_neigh_atoms = grid_neigh_atoms[..., :num_neighbours]

        # neighbour atoms for each atom
        # (B,A,N)
        if num_walker == 1:
            return grid_neigh_atoms[:, atom_grid_idx[0], :]
        return msnp.take_along_axis(grid_neigh_atoms, F.expand_dims(atom_grid_idx, -1), -2)

    def construct(self,
                  coordinate: Tensor,
                  pbc_box: Tensor = None,
                  atom_mask: Tensor = None,
                  exclude_index: Tensor = None,
                  ):
        """
        Calculate neighbour list.

        Args:
            coordinate (Tensor):    Tensor of shape (B, A, D). Data type is float.
                                    Atom coordinates.
            pbc_box (Tensor):       Tensor of shape (B, D). Data type is float.
                                    PBC box.Default: None
            atom_mask (Tensor):     Tensor of shape (B, A). Data type is bool.
                                    Mask of atoms. Default: None
            exclude_index (Tensor): Tensor of shape (B, A, Ex). Data type is int.
                                    Index of atoms that should be exclude from neighbour list.
                                    Default: None

        Returns:
            - neighbours(Tensor).
            - mask(Tensor).

        Sysmbols:
            B:  Number of simulation walker.
            A:  Number of atoms in system.
            D:  Dimension of position coordinates.
            Ex: Maximum number of excluded neighbour atoms.
        """

        if self.use_pbc:
            if pbc_box is None:
                cell = self.cell
            else:
                # (B,1,D) = (B,D) / (D)
                cell = F.expand_dims(pbc_box/self.grid_dims, -2)
                if (cell < self.grid_cutoff).any():
                    print('Warning! The cell length is smaller than cutoff')
            # (B,A,D) = ((B,A,D) - (D)) / (B,1,D)
            atom_grid_idx = msnp.floor(
                (displace_in_box(coordinate, pbc_box))/cell).astype(ms.int32)
        else:
            # (B,1,D) <- (B,A,D)
            center = msnp.mean(coordinate, -2, keepdims=True)
            # (B,A,D) = (B,A,D) - (B,1,D) + (D)
            scaled_coord = (coordinate - center +
                            self.half_box) / self.scaled_grid_cutoff
            scaled_coord = msnp.where(scaled_coord < 0, 0, scaled_coord)
            atom_grid_idx = msnp.floor(scaled_coord).astype(ms.int32)
            atom_grid_idx = msnp.where(atom_grid_idx < self.origin_grid_dims,
                                       atom_grid_idx, self.max_grid_index)
            atom_grid_idx += 1

        # Grid index for each atom
        # (B,A) <- (B,A,D) * (D)
        atom_grid_idx = msnp.sum(atom_grid_idx * self.grid_factor, axis=-1)

        neighbours = self.get_neighbours_from_grids(
            atom_grid_idx, self.num_neighbours)

        mask = neighbours != self.num_atoms
        atom_idx = msnp.broadcast_to(self.atom_idx, neighbours.shape)
        neighbours = F.select(mask, neighbours, atom_idx)
        mask = (neighbours != atom_idx)

        if atom_mask is None:
            atom_mask = self.atom_mask

        if exclude_index is None:
            exclude_index = self.exclude_index
        if exclude_index is not None:
            # (B,A,N,Ex) = (B,A,N,1) != (B,1,1,Ex)
            exmask = (F.expand_dims(neighbours, -1) !=
                      F.expand_dims(exclude_index, -2))
            # (B,A,N)
            exmask = self.reduce_all(exmask, -1)
            mask = F.logical_and(mask, exmask)

        return neighbours, mask
