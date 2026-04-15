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
# ==============================================================================
"""
Embedding
"""

from typing import Union, Tuple

import mindspore as ms
from mindspore import Tensor
from mindspore import ops
from mindspore.common.initializer import Initializer, Normal
from mindspore.nn import Cell

from mindchemistry.cell.embedding import NormBesselBasis, MaskPolynomialCutoff


class AllegroEmbedding(Cell):
    """AllegroEmbedding
    """

    def __init__(
            self,
            num_type: int,
            cutoff: float,
            num_atom_types: int = 64,
            initializer: Union[Initializer, str] = Normal(1.0)
    ):

        super().__init__()
        self.num_type = num_type
        self.num_atom_types = int(num_atom_types)
        self.initializer = initializer
        self.cutoff = cutoff
        self.cutoff_fn = MaskPolynomialCutoff(r_max=cutoff)
        if self.cutoff_fn is not None:
            self.cutoff = self.cutoff_fn.cutoff
        if self.cutoff is not None:
            self.cutoff = ms.Tensor(self.cutoff, ms.float32)

        self.rbf_bessel = NormBesselBasis(r_max=cutoff)

    def construct(self, atom_types: Tensor, pos: Tensor,
                  edge_index: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """construct

        Args:
            atom_types (Tensor): atom_types
            pos (Tensor): pos
            edge_index (Tensor): edge_index

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: outputs
        """
        node_emb = ops.one_hot(atom_types.squeeze(-1), self.num_type, Tensor(1.0, ms.float32), Tensor(0.0, ms.float32))
        node_features = node_emb

        edge_vec = pos[edge_index[0]] - pos[edge_index[1]]
        edge_length = ops.norm(edge_vec, dim=-1)
        rbf_embedding = self.rbf_bessel(edge_length)
        edge_cutoff, edge_mask = self.cutoff_fn(edge_length)
        edge_embedding = rbf_embedding * edge_cutoff.unsqueeze(-1)

        outputs = node_emb, node_features, edge_vec, edge_length, edge_embedding, edge_cutoff, edge_mask
        return outputs
