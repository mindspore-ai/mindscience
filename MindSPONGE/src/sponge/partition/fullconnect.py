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
Full connected neighbour list
"""

from typing import Tuple
import mindspore as ms
from mindspore import numpy as msnp
from mindspore import Tensor
from mindspore.nn import Cell
from mindspore.ops import functional as F

from ..function import reduce_all


class FullConnectNeighbours(Cell):
    r"""Full connected neighbour list

    Args:
        num_atoms (int):    Number of atoms

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import sponge
        >>> from sponge.partition import FullConnectNeighbours
        >>> full_connect_neighbours = FullConnectNeighbours(3)
        >>> full_connect_neighbours()
        (Tensor(shape=[1, 3, 2], dtype=Int32, value=
         [[[1, 2],
         [0, 2],
         [0, 1]]]),
         Tensor(shape=[1, 3, 2], dtype=Bool, value=
         [[[ True,  True],
         [ True,  True],
         [ True,  True]]]))

    """

    def __init__(self, num_atoms: int):
        super().__init__()
        self.num_atoms = num_atoms
        self.num_neighbours = num_atoms - 1

        self.fc_idx = None
        self.no_idx = None
        self.idx_mask = None
        self.shape = ()
        self.fc_mask = None

        if self.num_neighbours > 0:
            # neighbours for no connection (A*N)
            # (A, 1)
            no_idx = msnp.arange(self.num_atoms).reshape(-1, 1)

            # (N)
            nrange = msnp.arange(self.num_neighbours)

            # neighbours for full connection (A, N)
            # [[1, 2, 3, ..., N],
            #  [0, 2, 3, ..., N],
            #  [0, 1, 3, ..., N],
            #  ...,
            #  [0, 1, 2, ..., N-1]]
            fc_idx = nrange + F.cast(no_idx <= nrange, ms.int32)
            no_idx = msnp.broadcast_to(
                no_idx, (self.num_atoms, self.num_neighbours))
            idx_mask = fc_idx > no_idx

            # (1, A, N)
            self.fc_idx = F.expand_dims(fc_idx, 0)
            self.no_idx = F.expand_dims(no_idx, 0)
            self.idx_mask = F.expand_dims(idx_mask, 0)

            self.shape = (self.num_atoms, self.num_neighbours)
            self.fc_mask = msnp.broadcast_to(Tensor(True), (1,)+self.shape)

    def set_exclude_index(self, exclude_index: Tensor) -> Tensor:
        # TODO: Dummy
        return exclude_index

    def check_neighbour_list(self):
        # TODO: check the number of neighbouring atoms in neighbour list
        return self

    def print_info(self):
        # TODO: print information
        return self

    def construct(self,
                  atom_mask: Tensor = None,
                  exclude_index: Tensor = None
                  ) -> Tuple[Tensor, Tensor]:
        # pylint: disable=missing-docstring
        # Calculate the full connected neighbour list.

        # Args:
        #     atom_mask (Tensor):     Tensor of :math:`(B, A)`. Data type is bool.
        #     exclude_index (Tensor): Tensor of :math:`(B, A, Ex)`. Data type is int.

        # Returns:
        #     neighbours (Tensor):    Tensor of :math:`(B, A, N)`. Data type is int.
        #     neighbour_mask (Tensor) Tensor of :math:`(B, A, N)`. Data type is bool.

        # Note:
        #     - B:  Batch size.
        #     - A:  Number of atoms in system.
        #     - N:  Number of neighbour atoms.
        #     - D:  Dimension of position coordinates.
        #     - Ex: Maximum number of excluded neighbour atoms.

        if self.fc_idx is None:
            return None, None

        if atom_mask is None:
            neighbours = self.fc_idx
            neighbour_mask = self.fc_mask
            no_idx = self.no_idx
        else:
            # (B, 1, N)
            mask0 = F.expand_dims(atom_mask[:, :-1], -2)
            mask1 = F.expand_dims(atom_mask[:, 1:], -2)

            # (B, A, N)
            neighbour_mask = msnp.where(self.idx_mask, mask1, mask0)
            neighbour_mask = F.logical_and(F.expand_dims(atom_mask, -1), neighbour_mask)
            fc_idx = msnp.broadcast_to(self.fc_idx, neighbour_mask.shape)
            no_idx = msnp.broadcast_to(self.no_idx, neighbour_mask.shape)
            neighbours = F.select(neighbour_mask, fc_idx, no_idx)

        if exclude_index is not None:
            # (B, A, N, Ex) <- (B, A, N, 1) vs (B, A, 1, Ex)
            exc_mask = F.expand_dims(neighbours, -1) != F.expand_dims(exclude_index, -2)
            # (B, A, N)
            exc_mask = reduce_all(exc_mask, -1)
            neighbour_mask = F.logical_and(neighbour_mask, exc_mask)
            if atom_mask is None:
                no_idx = msnp.broadcast_to(no_idx, neighbour_mask.shape)
            neighbours = F.select(neighbour_mask, neighbours, no_idx)

        return neighbours, neighbour_mask
