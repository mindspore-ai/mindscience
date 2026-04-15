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
"""embedding block"""

import mindspore as ms
import mindspore.mint as mint
from .base_layers import DenseWithActivation

MAX_ATOMIC_NUM = 100


class AtomEmbedding(ms.nn.Cell):
    r"""
    Initial atom embeddings based on the atom type

    Args:
        emb_size (int): Atom embeddings size

    Inputs:
        - **z** (Tensor) - The shape of tensor is :math:`(batch\_size, latent\_dim)`.
    Outputs:
        - **h** (Tensor) - The shape of tensor is :math:`(total\_atoms, emb\s_size)`.
    """

    def __init__(self, emb_size):
        super().__init__()
        self.emb_size = emb_size

        # Atom embeddings: We go up to Bi (83).
        self.embeddings = mint.nn.Embedding(MAX_ATOMIC_NUM, emb_size)
        # init by uniform distribution
        self.embeddings.weight = ms.Parameter(ms.common.initializer.initializer(
            ms.common.initializer.Uniform(mint.sqrt(ms.Tensor(3))),
            self.embeddings.weight.shape, ms.float32))
        self.embeddings.weight.init_data()

    def construct(self, z):
        # -1 because Z.min()=1 (==Hydrogen)
        h = self.embeddings(z - 1)
        return h


class EdgeEmbedding(ms.nn.Cell):
    r"""
    Edge embedding based on the concatenation of atom embeddings and subsequent dense layer.

    Args:
        atom_features (int): Number of features in the atom embeddings.
        edge_features (int): Number of features in the edge embeddings.
        out_features (int): Number of output features.
        activation (str): Activation function used in the dense layer.
            Default: None

    Inputs:
        - **h** (Tensor) - The shape of tensor is :math:`(total\_atoms, emb\_size)`.
        - **m_rbf** (Tensor) - The shape of tensor is :math:`(total\_edges, nFeatures)`.
        - **idx_s** (Tensor) - The shape of tensor is :math:`(total\_edges,)`.
        - **idx_t** (Tensor) - The shape of tensor is :math:`(total\_edges,)`.

    Outputs:
        - **m_st** (Tensor) - The shape of tensor is :math:`(total\_edges, emb\_size)`.
    """

    def __init__(self, atom_features, edge_features, out_features, activation=None):
        super().__init__()
        in_features = 2 * atom_features + edge_features
        self.dense = DenseWithActivation(
            in_features, out_features, activation=activation, bias=False
        )

    def construct(self, h, m_rbf, idx_s, idx_t):
        h_s = mint.index_select(h, 0, idx_s)
        h_t = mint.index_select(h, 0, idx_t)

        m_st = mint.cat(
            (h_s, h_t, m_rbf), dim=-1
        )
        m_st = self.dense(m_st)
        return m_st
