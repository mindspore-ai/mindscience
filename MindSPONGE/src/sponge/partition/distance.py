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
Use the distances between atoms to calculate neighbour list
"""

from typing import Tuple
import mindspore as ms
import mindspore.numpy as msnp
from mindspore import Tensor, Parameter
from mindspore.nn import Cell
from mindspore import ops
from mindspore.ops import functional as F

from ..function.functions import get_integer, get_ms_array, reduce_all
from ..function.operations import GetDistance


class DistanceNeighbours(Cell):
    r"""Neighbour list calculated by distance

    Args:
        cutoff (float):         Cutoff distance.

        num_neighbours (int, optional):   Number of neighbours. If ``None`` is given, this value will be calculated by
                                the ratio of the number of neighbouring grids to the total number of grids.
                                Default: ``None``.

        atom_mask (Tensor, optional):     Tensor of shape :math:`(B, A)`. Data type is bool_.
                                Mask of atoms in the system.
                                Default: ``None``.

        exclude_index (Tensor, optional): Tensor of shape :math:`(B, A, Ex)`. Data type is int32.
                                Index of neighbour atoms which could be excluded from the neighbour list.
                                Default: ``None``.

        use_pbc (bool, optional):         Whether to use periodic boundary condition. Default: ``None``.

        cutoff_scale (float, optional):   Factor to scale the cutoff distance. Default: 1.2

        large_dis (float, optional):      A large number to fill in the distances to the masked neighbouring atoms.
                                Default: 1e4

        cast_fp16 (bool, optional):       If this is set to ``True``, the data will be cast to float16 before sort.
                                For use with some devices that only support sorting of float16 data.
                                Default: ``False``.

    Note:

        - B:  Number of simulation walker.

        - A:  Number of atoms in system.

        - N:  Number of neighbour atoms.

        - Ex: Maximum number of excluded neighbour atoms.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import sponge
        >>> from sponge.partition import DistanceNeighbours
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> distance_neighbours = DistanceNeighbours(0.5, use_pbc=False)
        >>> coordinate = Tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])
        >>> distance_neighbours(coordinate)
        (Tensor(shape=[1, 2, 1], dtype=Float32, value=
         [[[ 1.00000000e+00],
         [ 1.00000000e+00]]]),
         Tensor(shape=[1, 2, 1], dtype=Int64, value=
         [[[0],
         [1]]]),
         Tensor(shape=[1, 2, 1], dtype=Bool, value=
         [[[False],
         [False]]]))
        >>> distance = Tensor([[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
        >>> distance_neighbours.calc_max_neighbours(distance, 0.5)
        Tensor(shape=[], dtype=Int32, value= 2)
        >>> distance_neighbours.check_neighbour_list()
        DistanceNeighbours<
          (get_distance): GetDistance<>
          >

    """

    def __init__(self,
                 cutoff: float,
                 num_neighbours: int = None,
                 atom_mask: Tensor = None,
                 exclude_index: Tensor = None,
                 use_pbc: bool = None,
                 cutoff_scale: float = 1.2,
                 large_dis: float = 1e4,
                 cast_fp16: bool = False,
                 ):

        super().__init__()

        self.cutoff = Tensor(cutoff, ms.float32)
        self.cutoff_scale = Tensor(cutoff_scale, ms.float32)
        self.scaled_cutoff = self.cutoff * self.cutoff_scale

        self.num_neighbours = get_integer(num_neighbours)
        if self.num_neighbours is None:
            max_neighbours = Tensor(0, ms.int32)
        else:
            max_neighbours = Tensor(self.num_neighbours, ms.int32)
        self.max_neighbours = Parameter(max_neighbours, name='max_neighbours', requires_grad=False)

        self.large_dis = Tensor(large_dis, ms.float32)
        self.cast_fp16 = cast_fp16

        self.emtpy_atom_shift = 0
        self.atom_mask = None
        self.has_empty_atom = False
        if atom_mask is not None:
            # (B, A)
            self.atom_mask = Tensor(atom_mask, ms.bool_)
            if self.atom_mask.ndim == 1:
                self.atom_mask = F.expand_dims(self.atom_mask, 0)

            self.has_empty_atom = F.logical_not(self.atom_mask.all())
            if self.has_empty_atom:
                emtpy_atom_mask = F.logical_not(self.atom_mask)
                # (B, 1, A)
                self.emtpy_atom_shift = F.expand_dims(emtpy_atom_mask, -2) * self.large_dis

        self.exclude_index = None
        if exclude_index is not None:
            # (B, A, Ex)
            self.exclude_index = Tensor(exclude_index, ms.int32)
            if self.exclude_index.ndim == 2:
                self.exclude_index = F.expand_dims(self.exclude_index, 0)

        self.get_distance = GetDistance(use_pbc)

        self.sort = ops.Sort(-1)

    @staticmethod
    def calc_max_neighbours(distances: Tensor, cutoff: float) -> Tensor:
        r"""Calculate the maximum number of neighbouring atoms.

        Args:
            distances (Tensor):  Tensor of shape :math:`(B, A, N)`. Data type is float.
            cutoff (float):      Cutoff distance.

        """
        mask = distances < cutoff
        max_neighbours = ops.count_nonzero(F.cast(mask, ms.float16), -1, dtype=ms.float16) - 1
        return F.cast(ops.reduce_max(max_neighbours), ms.int32)

    def set_num_neighbours(self,
                           coordinate: Tensor,
                           pbc_box: Tensor = None,
                           scale_factor: float = 1.25
                           ):
        r"""Set maximum number of neighbouring atoms.

        Args:
            coordinate (Tensor):    Tensor of shape :math:`(B, A, D)`. Data type is float.
                                    Position coordinates of atoms
            pbc_box (Tensor, optional):    Tensor of shape :math:`(B, D)`. Data type is bool.
                                           Periodic boundary condition box. Default: ``None``.
            scale_factor (float, optional):   Factor to scale the cutoff distance. Default: ``1.25``.

        """
        distances = self.get_distance(F.expand_dims(coordinate, -2), F.expand_dims(coordinate, -3), pbc_box)
        num_neighbours = self.calc_max_neighbours(distances, self.scaled_cutoff)
        num_neighbours = F.ceil(num_neighbours * scale_factor)
        self.num_neighbours = get_integer(F.minimum(num_neighbours, coordinate.shape[-2] - 1))
        F.assign(self.max_neighbours, self.num_neighbours)
        return self

    def set_exclude_index(self, exclude_index: Tensor) -> Tensor:
        r"""Set the indices of atoms to be excluded from the neighbor list.

        Args:
            exclude_index (Tensor): Tensor of shape :math:`(B, A, Ex)`. Data type is int.
                                    Index of the atoms that should be excluded from the neighbour list.

        """
        # (B, A, Ex)
        self.exclude_index = get_ms_array(exclude_index, ms.int32)
        if self.exclude_index.ndim == 2:
            self.exclude_index = F.expand_dims(self.exclude_index, 0)
        return self.exclude_index

    def print_info(self):
        """Print the information of neighbour list."""
        print(f'[MindSPONGE] Neighbour list: DistanceNeighbours')
        print(f'[MindSPONGE]     Cut-off distance: {self.cutoff}')
        print(f'[MindSPONGE]     Scaled cut-off: {self.scaled_cutoff}')
        print(f'[MindSPONGE]     Max number of neighbour atoms: {self.num_neighbours}')
        return self

    def check_neighbour_list(self):
        """Check the number of neighbouring atoms in neighbour list."""
        if self.num_neighbours is not None and self.max_neighbours > self.num_neighbours:
            raise RuntimeError(f'The max number of neighbour atoms ({self.max_neighbours.asnumpy()}) is larger than '
                               f'the initial neighbouring number of neighbour list ({self.num_neighbours}!')
        return self


    def construct(self,
                  coordinate: Tensor,
                  pbc_box: Tensor = None,
                  atom_mask: Tensor = None,
                  exclude_index: Tensor = None
                  ) -> Tuple[Tensor, Tensor]:
        # pylint: disable=missing-docstring
        # Calculate distances and neighbours.

        # Args:
        #     coordinate (Tensor):    Tensor of shape :math:`(B, A, D)`. Data type is float.
        #                             Position coordinates of atoms
        #     pbc_box (Tensor):       Tensor of shape :math:`(B, D)`. Data type is bool.
        #                             Periodic boundary condition box.
        #                             Default: ``None``.
        #     atom_mask (Tensor):     Tensor of shape :math:`(B, A)`. Data type is bool.
        #                             Atomic mask
        #     exclude_index (Tensor): Tensor of shape :math:`(B, A, Ex)`. Data type is int.
        #                             Index of the atoms that should be excluded from the neighbour list.
        #                             Default: ``None``.

        # Returns:
        #     distances (Tensor):         Tensor of shape :math:`(B, A, N)`. Data type is float.
        #     neighbours (Tensor):        Tensor of shape :math:`(B, A, N)`. Data type is int.
        #     neighbour_mask (Tensor):    Tensor of shape :math:`(B, A, N)`. Data type is bool.

        # Note:
        #     - B:  Batch size.
        #     - A:  Number of atoms in system.
        #     - N:  Number of neighbour atoms.
        #     - D:  Dimension of position coordinates.
        #     - Ex: Maximum number of excluded neighbour atoms.

        # A
        num_atoms = coordinate.shape[-2]
        # (B, A, A) <- (B, A, 1, 3) - (B, 1, A, 3)
        distances = self.get_distance(F.expand_dims(coordinate, -2), F.expand_dims(coordinate, -3), pbc_box)

        if atom_mask is None:
            atom_mask = self.atom_mask
            if self.has_empty_atom:
                # (B, A, A) + (B, 1, A)
                distances += self.emtpy_atom_shift
        else:
            if not atom_mask.all():
                emtpy_atom_mask = F.logical_not(atom_mask)
                # (B, 1, A)
                emtpy_atom_shift = F.expand_dims(emtpy_atom_mask, -2) * self.large_dis
                distances += emtpy_atom_shift

        # (B, A)
        if self.num_neighbours is None:
            num_neighbours = num_atoms - 1
        else:
            num_neighbours = self.num_neighbours
            max_neighbours = self.calc_max_neighbours(distances, self.cutoff)
            distances = F.depend(distances, F.assign(self.max_neighbours, max_neighbours))

        if self.cast_fp16:
            distances = F.cast(distances, ms.float16)
            distances, neighbours = self.sort(distances)
            distances = distances[..., 1:num_neighbours+1]
            distances = F.cast(distances, ms.float32)
            neighbours = neighbours[..., 1:num_neighbours+1]
        else:
            distances, neighbours = F.top_k(-distances, num_neighbours+1)
            distances = -distances[..., 1:]
            neighbours = neighbours[..., 1:]

        neighbour_mask = distances < self.scaled_cutoff

        if exclude_index is None:
            exclude_index = self.exclude_index
        if exclude_index is not None:
            # (B, A, n, Ex) <- (B, A, n, 1) != (B, A, 1, E)
            exc_mask = F.expand_dims(neighbours, -1) != F.expand_dims(exclude_index, -2)
            # (B,A,n)
            exc_mask = reduce_all(exc_mask, -1)
            neighbour_mask = F.logical_and(neighbour_mask, exc_mask)

        if atom_mask is not None:
            # (B, A, n) <- (B, A, n) && (B, A, 1)
            neighbour_mask = F.logical_and(neighbour_mask, F.expand_dims(atom_mask, -1))

        # (1, A, 1)
        no_idx = msnp.arange(num_atoms).reshape(1, -1, 1)
        # (B, A, n)
        no_idx = msnp.broadcast_to(no_idx, neighbours.shape)
        no_idx_tmp = no_idx.astype("int64")
        neighbours_tmp = neighbours.astype("int64")
        neighbours = F.select(neighbour_mask, neighbours_tmp, no_idx_tmp)

        return distances, neighbours, neighbour_mask
