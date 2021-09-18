# ============================================================================
# Copyright 2021 The AIMM team at Shenzhen Bay Laboratory & Peking University
#
# People: Yi Isaac Yang, Jun Zhang, Diqing Chen, Yaqiang Zhou, Huiyang Zhang,
#         Yupeng Huang, Yijie Xia, Yao-Kun Lei, Lijiang Yang, Yi Qin Gao
#
# This code is a part of Cybertron-Code package.
#
# The Cybertron-Code is open-source software based on the AI-framework:
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
"""neighbors"""

import mindspore as ms
from mindspore import nn
from mindspore.ops import operations as P
from mindspore.ops import functional as F

from .units import units

__all__ = [
    "GatherNeighbors",
    "Distances",
]


class GatherNeighbors(nn.Cell):
    r"""Gathering the positions of every atom to its neighbors.

    Args:

    """

    def __init__(self, dim, fixed_neigh=False):
        super().__init__()
        self.fixed_neigh = fixed_neigh

        self.broad_ones = P.Ones()((1, 1, dim), ms.int32)

        self.gatherd = P.GatherD()

    def construct(self, inputs, neighbors):
        """construct"""
        # Construct auxiliary index vector
        ns = neighbors.shape

        # Get atomic positions of all neighboring indices

        if self.fixed_neigh:
            return F.gather(inputs, neighbors[0], -2)
        # [B, A, N] -> [B, A*N, 1]
        neigh_idx = F.reshape(neighbors, (ns[0], ns[1] * ns[2], -1))
        neigh_idx = neigh_idx * self.broad_ones
        # [B, A*N, V] gather from [B, A, V]
        outputs = self.gatherd(inputs, 1, neigh_idx)
        # [B, A, N, V]
        return F.reshape(outputs, (ns[0], ns[1], ns[2], -1))


class Distances(nn.Cell):
    r"""Computing distance of every atom to its neighbors.

    Args:


    """

    def __init__(
            self,
            fixed_atoms=False,
            dim=3,
            long_dis=units.length(
                10,
                'nm')):
        super().__init__()
        self.fixed_atoms = fixed_atoms
        self.reducesum = P.ReduceSum()
        self.pow = P.Pow()
        self.gatherd = P.GatherD()
        self.norm = nn.Norm(-1)
        self.long_dis = long_dis

        self.gather_neighbors = GatherNeighbors(dim, fixed_atoms)
        self.maximum = P.Maximum()

        # self.ceil = P.Ceil()
        self.floor = P.Floor()

    def construct(self, positions, neighbors, neighbor_mask=None, pbcbox=None):
        r"""Compute distance of every atom to its neighbors.

        Args:
            positions (ms.Tensor[float], [B, A, 3]): atomic Cartesian coordinates
            neighbors (ms.Tensor[int], [B, A, N] or [1, A, N]): indices of neighboring atoms to consider
            neighbor_mask (ms.Tensor[bool], [B, A, N] or [1, A, N]): boolean mask for neighbor
                positions. Required for the stable computation of forces in
                molecules with different sizes.
            pbcbox (ms.Tensor[float], [B, 3] or [1, 3])

        Returns:
            ms.Tensor[float]: layer output of (N_b x N_at x N_nbh) shape.

        """

        pos_xyz = self.gather_neighbors(positions, neighbors)
        pos_diff = pos_xyz - F.expand_dims(positions, -2)

        if pbcbox is not None:
            # [B, 3] -> [B, 1, 1, 3] or [1, 3] -> [1, 1, 1, 3]
            pbcbox = F.expand_dims(pbcbox, -2)
            pbcbox = F.expand_dims(pbcbox, -2)
            halfbox = F.ones_like(pos_diff) * (pbcbox / 2)
            lmask = pos_diff > halfbox
            smask = pos_diff < -halfbox

            if lmask.any():
                nbox = self.floor(pos_diff / pbcbox - 0.5) + 1
                pos = pos_diff - nbox * pbcbox
                pos_diff = F.select(lmask, pos, pos_diff)

            if smask.any():
                # nbox = self.ceil(-pos_diff/pbcbox - 0.5)
                nbox = self.floor(-pos_diff / pbcbox - 0.5) + 1
                pos = pos_diff + nbox * pbcbox
                pos_diff = F.select(smask, pos, pos_diff)

        if neighbor_mask is not None:
            large_diff = pos_diff + self.long_dis
            smask = (F.ones_like(pos_diff) *
                     F.expand_dims(neighbor_mask, -1)) > 0
            pos_diff = F.select(smask, pos_diff, large_diff)

        return self.norm(pos_diff)
