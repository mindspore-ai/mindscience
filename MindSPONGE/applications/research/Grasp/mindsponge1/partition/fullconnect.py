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
Full connected neighbour list
"""

import mindspore as ms
from mindspore import numpy as msnp
from mindspore import Tensor
from mindspore import ops
from mindspore.nn import Cell
from mindspore.ops import functional as F


class FullConnectNeighbours(Cell):
    r"""
    Full connected neighbour list.

    Args:
        num_atoms (int): Number of atoms.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """

    def __init__(self, num_atoms: int):
        super().__init__()
        self.num_atoms = num_atoms
        self.num_neighbours = num_atoms - 1

        # neighbours for no connection (A*N)
        # (A,1)
        no_idx = msnp.arange(self.num_atoms).reshape(-1, 1)

        # (N)
        nrange = msnp.arange(self.num_neighbours)

        # neighbours for full connection (A,N)
        # [[1,2,3,...,N],
        #  [0,2,3,...,N],
        #  [0,1,3,....N],
        #  .............,
        #  [0,1,2,...,N-1]]
        fc_idx = nrange + F.cast(no_idx <= nrange, ms.int32)
        no_idx = msnp.broadcast_to(
            no_idx, (self.num_atoms, self.num_neighbours))
        idx_mask = fc_idx > no_idx

        # (1,A,N)
        self.fc_idx = F.expand_dims(fc_idx, 0)
        self.no_idx = F.expand_dims(no_idx, 0)
        self.idx_mask = F.expand_dims(idx_mask, 0)

        self.shape = (self.num_atoms, self.num_neighbours)
        self.fc_mask = msnp.broadcast_to(Tensor(True), (1,)+self.shape)

        self.reduce_all = ops.ReduceAll()

    def set_exclude_index(self, _exclude_index: Tensor):
        """
        Dummy.

        Args:
            _exclude_index (Tensor):    Tensor of exclude indexes.
        """
        # pylint: disable=invalid-name
        return self

    def print_info(self):
        """print information"""
        return self

    def construct(self, atom_mask: Tensor = None, exclude_index: Tensor = None):
        r"""
        Calculate the full connected neighbour list.

        Args:
            atom_mask (Tensor):     Tensor of shape (B, A). Data type is bool.
            exclude_index (Tensor): Tensor of shape (B, A, Ex). Data type is int.

        Returns:
            - neighbours (Tensor), Tensor of shape (B, A, N). Data type is int.
            - neighbour_mask (Tensor), Tensor of shape (B, A, N). Data type is bool.

        Symbols:
            B:  Batch size.
            A:  Number of atoms in system.
            N:  Number of neighbour atoms.
            D:  Dimension of position coordinates.
            Ex: Maximum number of excluded neighbour atoms.

        """
        if atom_mask is None:
            neighbours = self.fc_idx
            neighbour_mask = self.fc_mask
        else:

            # (B,1,N)
            mask0 = F.expand_dims(atom_mask[:, :-1], -2)
            mask1 = F.expand_dims(atom_mask[:, 1:], -2)

            # (B,A,N)
            neighbour_mask = msnp.where(self.idx_mask, mask1, mask0)
            neighbour_mask = F.logical_and(F.expand_dims(atom_mask, -1), neighbour_mask)
            neighbours = msnp.where(neighbour_mask, self.fc_idx, self.no_idx)

        if exclude_index is not None:
            # (B,A,N,E) <- (B,A,N,1) vs (B,A,1,E)
            exc_mask = F.expand_dims(
                neighbours, -1) != F.expand_dims(exclude_index, -2)
            # (B,A,N)
            exc_mask = self.reduce_all(exc_mask, -1)
            neighbour_mask = F.logical_and(neighbour_mask, exc_mask)
            neighbours = msnp.where(neighbour_mask, neighbours, self.no_idx)

        return neighbours, neighbour_mask
