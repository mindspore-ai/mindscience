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
Neighbour list
"""

from inspect import signature
from distutils.version import LooseVersion
from typing import Tuple
import mindspore as ms
import mindspore.numpy as msnp
from mindspore import Tensor
from mindspore import Parameter
from mindspore import ops, nn
from mindspore.ops import functional as F
from mindspore.nn import Cell

from . import FullConnectNeighbours, DistanceNeighbours, GridNeighbours
from ..system import Molecule
from ..function.functions import gather_vector, get_integer, get_ms_array
from ..function.operations import GetVector


class NeighbourList(Cell):
    r"""Neighbour list

    Args:
        system (Molecule):      Simulation system.

        cutoff (float, optional): Cut-off distance. If ``None`` is given under periodic
                                  boundary condition (PBC), the cutoff will be assigned
                                  with the default value of 1 nm.
                                  Default: ``None``.

        pace (int, optional):   Update frequency for neighbour list. Default: ``20``

        exclude_index (Tensor, optional): Tensor of the indices of the neighbouring atoms
                                which could be excluded from the neighbour list. The shape
                                of Tensor is :math:`(B, A, Ex)`, and the data type is int.
                                Default: ``None``.

        num_neighbours (int, optional):   Maximum number of neighbours.
                                If ``None`` is given, this value will be calculated
                                by the ratio of the number of neighbouring grids to the
                                total number of grids. Default: ``None``.

        num_cell_cut (int, optional): Number of subdivision of grid cells according to cutoff. Default: ``1``

        cutoff_scale (float, optional):   Factor to scale cutoff distance. Default: ``1.2``

        cell_cap_scale (float, optional): Scale factor for `cell_capacity`. Default: ``1.25``

        grid_num_scale (float, optional): Scale factor to calculate `num_neighbours` by ratio of grids.
                                If `num_neighbours` is not ``None``, it will not be used. Default: ``2``

        use_grids (bool, optional):       Whether to use grids to calculate the neighbour list. Default: ``None``.

        cast_fp16 (bool, optional):       If this is set to ``True``, the data will be cast to float16 before sort.
                                For use with some devices that only support sorting of float16 data.
                                Default: ``False``.

    Note:
        - B:  Batchsize, i.e. number of walkers of the simulation.

        - A:  Number of the atoms in the simulation system.

        - N:  Number of the maximum neighbouring atoms.

        - D:  Dimension of position coordinates.

        - Ex: Maximum number of excluded neighbour atoms.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import sponge
        >>> from sponge.partition import NeighbourList
        >>> from sponge.system import Molecule
        >>> system = Molecule(template='water.spce.yaml')
        >>> neighbourlist = NeighbourList(system, 0.5)
        >>> neighbourlist(system.coordinate, system.pbc_box)
        (Tensor(shape=[1, 3, 2], dtype=Int64, value=
         [[[1, 2],
         [0, 2],
         [0, 1]]]),
         Tensor(shape=[1, 3, 2, 3], dtype=Float32, value=
         [[[[ 8.16490427e-02,  5.77358976e-02,  0.00000000e+00],
             [-8.16490427e-02,  5.77358976e-02,  0.00000000e+00]],
         [[-8.16490427e-02, -5.77358976e-02,  0.00000000e+00],
             [-1.63298085e-01,  0.00000000e+00,  0.00000000e+00]],
         [[ 8.16490427e-02, -5.77358976e-02,  0.00000000e+00],
             [ 1.63298085e-01,  0.00000000e+00,  0.00000000e+00]]]]),
         Tensor(shape=[1, 3, 2], dtype=Float32, value=
         [[[ 1.00000001e-01,  1.00000001e-01],
         [ 1.00000001e-01,  1.63298085e-01],
         [ 1.00000001e-01,  1.63298085e-01]]]),
         Tensor(shape=[1, 3, 2], dtype=Bool, value=
         [[[ True,  True],
         [ True,  True],
         [ True,  True]]]))
        >>> neighbourlist.calculate(system.coordinate, system.pbc_box)
        (Tensor(shape=[1, 3, 2], dtype=Int64, value=
         [[[1, 2],
         [0, 2],
         [0, 1]]]),
         Tensor(shape=[1, 3, 2], dtype=Bool, value=
         [[[ True,  True],
         [ True,  True],
         [ True,  True]]]))
        >>> neighbourlist.pace
        20

    """

    def __init__(self,
                 system: Molecule,
                 cutoff: float = None,
                 pace: int = 20,
                 exclude_index: Tensor = None,
                 num_neighbours: int = None,
                 num_cell_cut: int = 1,
                 cutoff_scale: float = 1.2,
                 cell_cap_scale: float = 1.25,
                 grid_num_scale: float = 2,
                 use_grids: bool = False,
                 cast_fp16: bool = False,
                 ):

        super().__init__()

        if LooseVersion(ms.__version__) < LooseVersion('2.1.0'):
            cast_fp16 = True

        self.num_walker = system.num_walker
        self.coordinate = system.get_coordinate()
        self.num_atoms = self.coordinate.shape[-2]
        self.dim = self.coordinate.shape[-1]

        self.pbc_box = system.get_pbc_box()
        use_pbc = self.pbc_box is not None

        self.atom_mask = system.atom_mask
        self.exclude_index = exclude_index
        if exclude_index is not None:
            self.exclude_index = Tensor(exclude_index, ms.int32)

        self.use_grids = use_grids

        self._pace = get_integer(pace)
        if self._pace < 0:
            raise ValueError('pace cannot be less than 0!')

        if cutoff is None and self.pbc_box is not None:
            cutoff = system.units.length(1, 'nm')

        self.no_mask = False
        if cutoff is None:
            self.cutoff = None
            self.large_dis = Tensor(1e4, ms.float32)
            self._pace = 0
            self.neighbour_list = FullConnectNeighbours(self.num_atoms)
            if self.exclude_index is None:
                self.no_mask = True
        else:
            self.cutoff = get_ms_array(cutoff, ms.float32)
            self.large_dis = self.cutoff * 100
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
                    if self.use_grids is True:
                        print(f'[WARNING] The number of neighbour atoms in `GridNeighbours` '
                              f'({self.neighbour_list.neigh_capacity}) is not less than '
                              f'the number of atoms ({self.num_atoms}). '
                              f'It would be more efficient to use `DistanceNeighbours` '
                              f'(set `use_grids` to `False` or `None`).')
                    else:
                        self.use_grids = False
                else:
                    self.use_grids = True

            if not self.use_grids:
                self.neighbour_list = DistanceNeighbours(
                    cutoff=self.cutoff,
                    num_neighbours=num_neighbours,
                    atom_mask=self.atom_mask,
                    exclude_index=self.exclude_index,
                    use_pbc=use_pbc,
                    cutoff_scale=cutoff_scale,
                    large_dis=self.large_dis,
                    cast_fp16=cast_fp16,
                )

                if num_neighbours is None:
                    self.neighbour_list.set_num_neighbours(self.coordinate, self.pbc_box)

        self.num_neighbours = self.neighbour_list.num_neighbours

        index, mask = self.calculate(self.coordinate, self.pbc_box)

        self.neighbours = None
        self.neighbour_mask = None
        if index is not None:
            self.neighbours = Parameter(index, name='neighbours', requires_grad=False)
            if mask is not None:
                self.neighbour_mask = Parameter(mask, name='neighbour_mask', requires_grad=False)

        self.get_vector = GetVector(use_pbc)
        self.identity = ops.Identity()

        self.norm_last_dim = None
        # MindSpore < 2.0.0-rc1
        if 'ord' not in signature(ops.norm).parameters.keys():
            self.norm_last_dim = nn.Norm(-1, False)

    @property
    def pace(self) -> int:
        r"""Update frequency for neighbour list

        Returns:
            int, update pace

        """
        return self._pace

    def set_exclude_index(self, exclude_index: Tensor):
        r"""set exclude index

        Args:
            exclude_index (Tensor): Tensor of shape :math:`(B, A, Ex)`. Data type is int.

        """
        if exclude_index is None:
            return self
        self.exclude_index = self.neighbour_list.set_exclude_index(exclude_index)
        index, mask = self.update(self.coordinate, self.pbc_box)
        F.assign(self.neighbours, index)
        if self.neighbour_mask is None:
            self.neighbour_mask = Parameter(mask, name='neighbour_mask', requires_grad=False)
        else:
            F.assign(self.neighbour_mask, mask)
        return self

    def print_info(self):
        r"""print information of neighbour list"""
        self.neighbour_list.print_info()
        return self

    def update(self, coordinate: Tensor, pbc_box: Tensor = None) -> Tuple[Tensor, Tensor]:
        r"""
        update neighbour list.

        Args:
            coordinate (Tensor):    Tensor of shape :math:`(B, A, D)`. Data type is float.
                                    Position coordinate.
            pbc_box (Tensor, optional):       Tensor of shape :math:`(B, D)`. Data type is float.
                                    Size of PBC box.

        Returns:
            neigh_idx (Tensor), Tensor of shape :math:`(B, A, N)`. Data type is int.
            Index of neighbouring atoms of each atoms in system.

            neigh_mask (Tensor), Tensor of shape :math:`(B, A, N)`. Data type is bool.
            Mask for neighbour list `neigh_idx`.

        Note:
            - B:  Batchsize, i.e. number of walkers of the simulation.
            - A:  Number of the atoms in the simulation system.
            - N:  Number of the maximum neighbouring atoms.
            - D:  Dimension of position coordinates.

        """

        if self.neighbours is None:
            return None, None

        coordinate = F.stop_gradient(coordinate)
        if pbc_box is not None:
            pbc_box = F.stop_gradient(pbc_box)

        neighbours, neighbour_mask = self.calculate(coordinate, pbc_box)
        neighbours = F.depend(neighbours, self.neighbour_list.check_neighbour_list())

        neighbours = F.depend(neighbours, F.assign(self.neighbours, neighbours))
        if self.neighbour_mask is not None:
            neighbour_mask = F.depend(neighbour_mask, F.assign(self.neighbour_mask, neighbour_mask))

        return neighbours, neighbour_mask

    def calculate(self, coordinate: Tensor, pbc_box: Tensor = None) -> Tuple[Tensor, Tensor]:
        r"""
        calculate neighbour list.

        Args:
            coordinate (Tensor):    Tensor of shape :math:`(B, A, D)`. Data type is float.
                                    Position coordinate.
            pbc_box (Tensor, optional):       Tensor of shape :math:`(B, D)`. Data type is float.
                                    Size of PBC box.

        Returns:
            neigh_idx (Tensor), Tensor of shape :math:`(B, A, N)`. Data type is int.
            Index of neighbouring atoms of each atoms in system.

            neigh_mask (Tensor), Tensor of shape :math:`(B, A, N)`. Data type is bool.
            Mask for neighbour list `neigh_idx`.

        Note:
            - B:  Batchsize, i.e. number of walkers of the simulation.
            - A:  Number of the atoms in the simulation system.
            - N:  Number of the maximum neighbouring atoms.
            - D:  Dimension of position coordinates.

        """

        if self.cutoff is None:
            return self.neighbour_list(self.atom_mask, self.exclude_index)

        if self.use_grids:
            return self.neighbour_list(coordinate, pbc_box)

        _, index, mask = self.neighbour_list(
            coordinate, pbc_box, self.atom_mask, self.exclude_index)

        return index, mask

    def get_neighbour_list(self) -> Tuple[Tensor, Tensor]:
        r"""
        get neighbour list.

        Returns:
            neigh_idx (Tensor), Tensor of shape :math:`(B, A, N)`. Data type is int.
            Index of neighbouring atoms of each atoms in system.

            neigh_mask (Tensor):, Tensor of shape :math:`(B, A, N)`. Data type is bool.
            Mask for neighbour list `neigh_idx`.

        Note:
            - B:  Batchsize, i.e. number of walkers of the simulation.
            - A:  Number of the atoms in the simulation system.
            - N:  Number of the maximum neighbouring atoms.

        """

        if self.neighbours is None:
            return None, None

        index = F.stop_gradient(self.identity(self.neighbours))
        mask = None
        if self.neighbour_mask is not None:
            mask = F.stop_gradient(self.identity(self.neighbour_mask))
        return index, mask

    def construct(self,
                  coordinate: Tensor,
                  pbc_box: Tensor = None
                  ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # pylint: disable=missing-docstring
        # Gather coordinate of neighbours atoms.

        # Args:
        #     coordinate (Tensor):    Tensor of shape :math:`(B, A, D)`. Data type is float.
        #                             Position coordinate.
        #     pbc_box (Tensor):       Tensor of shape :math:`(B, D)`. Data type is float.
        #                             Size of PBC box.

        # Returns:
        #     neigh_idx (Tensor):     Tensor of shape :math:`(B, A, N)`. Data type is int.
        #                             Index of neighbouring atoms of each atoms in system.
        #     neigh_vec (Tensor):     Tensor of shape :math:`(B, A, N, D)`. Data type is float.
        #                             Vectors from central atom to neighbouring atoms.
        #     neigh_dis (Tensor):     Tensor of shape :math:`(B, A, N)`. Data type is float.
        #                             Distance between center atoms and neighbouring atoms.
        #     neigh_mask (Tensor):    Tensor of shape :math:`(B, A, N)`. Data type is bool.
        #                             Mask for neighbour list `neigh_idx`.

        # Note:
        #     - B:  Batchsize, i.e. number of walkers of the simulation.
        #     - A:  Number of the atoms in the simulation system.
        #     - N:  Number of the maximum neighbouring atoms.
        #     - D:  Dimension of position coordinates.

        if self.neighbours is None:
            return None, None, None, None

        neigh_idx, neigh_mask = self.get_neighbour_list()

        # (B, A, 1, D) <- (B, A, D)
        center_pos = F.expand_dims(coordinate, -2)
        # (B, A, N, D) <- (B, A, D)
        neigh_vec = gather_vector(coordinate, neigh_idx)

        neigh_vec = self.get_vector(center_pos, neigh_vec, pbc_box)

        # Add a non-zero value to the neighbour_vector whose mask value is False
        # to prevent them from becoming zero values after Norm operation,
        # which could lead to auto-differentiation errors
        if neigh_mask is not None:
            # (B, A, N)
            large_dis = msnp.broadcast_to(self.large_dis, neigh_mask.shape)
            large_dis = F.select(neigh_mask, F.zeros_like(large_dis), large_dis)
            # (B, A, N, D) = (B, A, N, D) + (B, A, N, 1)
            neigh_vec += F.expand_dims(large_dis, -1)

        # (B, A, N) <- (B, A, N, D)
        if self.norm_last_dim is None:
            neigh_dis = ops.norm(neigh_vec, None, -1)
        else:
            neigh_dis = self.norm_last_dim(neigh_vec)

        return neigh_idx, neigh_vec, neigh_dis, neigh_mask
