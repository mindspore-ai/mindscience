# Copyright 2024 Huawei Technologies Co., Ltd
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
"""decoder
"""
import mindspore.nn as nn
import mindspore.mint as mint
from ..gemnet.layers.embedding_block import MAX_ATOMIC_NUM
from ..gemnet.gemnet import GemNetT


class GemNetTDecoder(nn.Cell):
    r"""
    Decoder with GemNetT.

    Args:
        config_path (str): Path to the config file.
        hidden_dim (int): Number of prediction targets. Default: ``128``.
        latent_dim(int): Dimension of the latent parameter 'z'. Default: ``256``.
        max_neighbors(int): Dimension of the latent parameter 'z'. Default: ``20``.
        radius(float): Dimension of the latent parameter 'z'. Default: ``6.``.

    Inputs:
        - **pred_atom_types** (Tensor) - The shape of tensor is :math:`(total\_atoms,)`.
        - **idx_s** (Tensor) - The shape of tensor is :math:`(total\_edges,)`.
        - **idx_t** (Tensor) - The shape of tensor is :math:`(total\_edges,)`.
        - **id3_ca** (Tensor) - The shape of tensor is :math:`(total\_triplets,)`.
        - **id3_ba** (Tensor) - The shape of tensor is :math:`(total\_triplets,)`.
        - **id3_ragged_idx** (Tensor) - The shape of tensor is :math:`(total\_triplets,)`.
        - **id3_ragged_idx_max** (int) - The maximum of id3_ragged_idx.
        - **y_l_m** (Tensor) - The shape of tensor is :math:`(num\_spherical, total\_triplets)`.
        - **d_st** (Tensor) - The shape of tensor is :math:`(total\_edges,)`.
        - **v_st** (Tensor) - The shape of tensor is :math:`(total\_edges, 3)`.
        - **id_swap** (Tensor) - The shape of tensor is :math:`(total\_triplets,)`.
        - **batch** (Tensor) - The shape of tensor is :math:`(total\_atoms,)`.
        - **z_per_atom** (Tensor) - The shape of tensor is :math:`(total\_atoms, latent\_dim)`.
        - **total_atoms** (int) - Total number of atoms.
        - **batch_size** (int) - batch_size.

    Outputs:
        - **atom_frac_coords** (Tensor) - The shape of tensor is :math:`(total\_atoms, 3)`.
        - **atom_types** (Tensor) - The shape of tensor is :math:`(total\_atoms, MAX\_ATOMIC\_NUM)`.
    """

    def __init__(
            self,
            config_path,
            hidden_dim=128,
            latent_dim=256,
            max_neighbors=20,
            radius=6.,
    ):
        super().__init__()
        self.cutoff = radius
        self.max_num_neighbors = max_neighbors

        self.gemnet = GemNetT(
            num_targets=1,
            latent_dim=latent_dim,
            emb_size_atom=hidden_dim,
            emb_size_edge=hidden_dim,
            regress_forces=True,
            cutoff=self.cutoff,
            max_neighbors=self.max_num_neighbors,
            config_path=config_path
        )
        self.fc_atom = mint.nn.Linear(hidden_dim, MAX_ATOMIC_NUM)

    def construct(self, pred_atom_types, idx_s, idx_t, id3_ca, id3_ba,
                  id3_ragged_idx, id3_ragged_idx_max, y_l_m, d_st, v_st, id_swap, batch, z_per_atom,
                  total_atoms, batch_size):
        """construct"""
        h, pred_cart_coord_diff = self.gemnet(
            pred_atom_types, idx_s, idx_t, id3_ca, id3_ba, id3_ragged_idx,
            id3_ragged_idx_max, y_l_m, d_st, v_st, id_swap, batch, z_per_atom, total_atoms, batch_size)
        pred_atom_types = self.fc_atom(h)
        return pred_cart_coord_diff, pred_atom_types
