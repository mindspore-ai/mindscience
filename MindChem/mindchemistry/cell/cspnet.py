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
"""DiffCSP denoiser file"""
import math

import numpy as np
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from ..graph.graph import (AggregateEdgeToNode, AggregateNodeToGlobal, LiftGlobalToNode)

MAX_ATOMIC_NUM = 100


class SinusoidsEmbedding(nn.Cell):
    """

    Fourier embedding for edge features to address periodic translation invariance
    as described in the paper of DiffCSP.

    Args:
        n_frequencies (int): The number of frequencies for embedding.
        n_space (int): The dimension of edge feature.
    """

    def __init__(self, n_frequencies=10, n_space=3):
        super(SinusoidsEmbedding, self).__init__()
        self.n_frequencies = n_frequencies
        self.n_space = n_space
        self.frequencies = 2 * math.pi * np.arange(self.n_frequencies)
        self.dim = self.n_frequencies * 2 * self.n_space
        self.frequencies_tensor = Tensor(self.frequencies, dtype=mindspore.float32).expand_dims(0).expand_dims(0)

        self.reshape = ops.Reshape()
        self.expand_dims = ops.ExpandDims()
        self.concat = ops.Concat(axis=-1)
        self.sin = ops.Sin()
        self.cos = ops.Cos()

    def construct(self, x):
        """construct

        Args:
            x (Tensor): Distance

        Returns:
            Tensor: Fourier embedding
        """
        emb = self.expand_dims(x, -1) * self.frequencies_tensor
        emb = self.reshape(emb, (-1, self.n_frequencies * self.n_space))
        emb = self.concat((self.sin(emb), self.cos(emb)))
        return emb


def mul_mask(features, mask):
    """Make the padded dim of features to be zeros

    Args:
        features (Tensor): Input tensor
        mask (Tensor): Value 1 specifies the corresponding dimension of input tensor to be valid,
            and value 0 specifies the corresponding dimension of input tensor to be zero.

    Returns:
        Tensor: Output tensor
    """
    return ops.mul(features, ops.reshape(mask, (-1, 1)))


class CSPLayer(nn.Cell):
    r"""One layer of the GNN denoiser. For the input node feature
        :math:`h_i^{(s-1)}` from last layer, the lattice matrix
        :math:`L` and the fractional coordinates :math:`f_i`, the formula is defined as:

    .. math::

        m_{ij}^{(s)} = mlp_m(h_i^{(s-1)}, h_j^{(s-1)}, L^\topL, \text{edge_emb}(f_j - f_i)),

        m_{i}^{(s)} = \sum_{j=1}^N m_{ij}^{(s)},

        h_{i}^{(s)} = h_i^{(s-1)} + mlp_h(h_i^{(s-1)}, m_{i}^{(s)}).

    ...

    Then we can get the new node features `h_i^{(s)}`.

    """

    def __init__(self, hidden_dim=512, act_fn=None, dis_emb=None):
        """Initialization

        Args:
            hidden_dim (int): The dimension of hidden node features. Defaults to 512.
            act_fn (nn): The activation function used in the layer. Defaults to nn.SiLU().
            dis_emb (object): The embbing method used for edge features. Defaults to None.
        """
        super(CSPLayer, self).__init__()
        self.dis_dim = 3
        self.dis_emb = dis_emb
        if dis_emb is not None:
            self.dis_dim = dis_emb.dim
        if act_fn is None:
            act_fn = nn.SiLU()
        self.lattice_matrix_dim = 9
        self.edge_mlp = nn.SequentialCell([
            nn.Dense(hidden_dim * 2 + self.lattice_matrix_dim + self.dis_dim, hidden_dim), act_fn,
            nn.Dense(hidden_dim, hidden_dim), act_fn
        ])
        self.node_mlp = nn.SequentialCell([
            nn.Dense(hidden_dim * 2, hidden_dim), act_fn,
            nn.Dense(hidden_dim, hidden_dim), act_fn
        ])

        self.layer_norm = nn.LayerNorm([hidden_dim], epsilon=1e-5)

        self.edge_scatter = AggregateEdgeToNode(mode='mean', dim=0)

    def edge_model(self, node_features, lattices, edge_index,
                   edge2graph, frac_diff, edge_mask):
        """Edge embbding for edge feature.
        """
        hi, hj = node_features[edge_index[0]], node_features[edge_index[1]]

        frac_diff = self.dis_emb(frac_diff)

        lattice_ips = ops.BatchMatMul()(lattices,
                                        lattices.transpose(0, -1, -2))

        lattice_ips_flatten = ops.Reshape()(lattice_ips, (-1, self.lattice_matrix_dim))
        lattice_ips_flatten_edges = ops.Gather()(lattice_ips_flatten,
                                                 edge2graph, 0)

        edges_input = ops.Concat(axis=1)(
            (hi, hj, lattice_ips_flatten_edges, frac_diff))
        edges_input = mul_mask(edges_input, edge_mask)

        edge_features = self.edge_mlp(edges_input)
        return edge_features

    def node_model(self, node_features, edge_features, edge_index, edge_mask):
        """Aggregate the edge features to be the node features.
        """
        agg = self.edge_scatter(edge_features,
                                edge_index,
                                dim_size=node_features.shape[0],
                                mask=edge_mask)
        agg = ops.Concat(axis=1)((node_features, agg))
        out = self.node_mlp(agg)
        return out

    def construct(self, node_features, lattices, edge_index,
                  edge2graph, frac_diff, node_mask, edge_mask):
        """Apply GNN layer over node features from last layer.

        Args:
            node_features (Tensor): Node features from last layer. Shape: (num_atoms, hidden_dim)
            frac_coords (Tensor): Fractional coordinates for calculating edge features.
                Shape: (num_atoms, 3)
            lattices (Tensor): Lattice mattrix for calculating edge features.
                Shape: (batchsize, 3, 3)
            edge_index (Tensor): Edge index for aggregating the edge features.
                Shape: (2, num_edges)
            edge2graph (Tensor): Graph index to lift the lattice to edge features.
                Shape: (num_edges,)
            frac_diff (Tensor): Distance of fractional coordinates for calculating
                edge features. Shape: (num_edges, 3)
            node_mask (Tensor): Node mask for padded tensor. Shape: (num_atoms,)
            edge_mask (Tensor): Edge mask for padded tensor. Shape: (num_edges,)

        Returns:
            Tensor: The output tensor. Shape: (num_atoms, hidden_dim)
        """
        node_input = node_features
        node_features = mul_mask(node_features, node_mask)
        node_features = self.layer_norm(node_features)
        edge_features = self.edge_model(node_features, lattices,
                                        edge_index, edge2graph, frac_diff,
                                        edge_mask)
        node_output = self.node_model(node_features, edge_features, edge_index,
                                      edge_mask)
        resiual_output = mul_mask(node_input + node_output, node_mask)
        return resiual_output


class CSPNet(nn.Cell):
    """GNN denoiser for DiffCSP.
    """

    def __init__(self,
                 hidden_dim=512,
                 latent_dim=256,
                 num_layers=6,
                 max_atoms=100,
                 num_freqs=128):
        """Initialization

        Args:
            hidden_dim (int): The dimension of hidden node features. Defaults to 512.
            latent_dim (int): The dimension of time embedding. Defaults to 256.
            num_layers (int): The number of layers used in GNN. Defaults to 6.
            max_atoms (int): The number of embedding table lines for atom types. Defaults to 100.
            num_freqs (int): The number of frequencies for Fourier embedding for
                edge features. Defaults to 128.
        """
        super(CSPNet, self).__init__()
        self.node_embedding = nn.Embedding(max_atoms, hidden_dim)
        self.atom_latent_emb = nn.Dense(hidden_dim + latent_dim, hidden_dim)
        self.act_fn = nn.SiLU()
        self.dis_emb = SinusoidsEmbedding(n_frequencies=num_freqs)

        self.csp_layers = nn.CellList([
            CSPLayer(hidden_dim, self.act_fn, self.dis_emb)
            for _ in range(num_layers)
        ])

        self.num_layers = num_layers
        self.coord_out = nn.Dense(hidden_dim, 3, has_bias=False)
        self.lattice_out = nn.Dense(hidden_dim, 9, has_bias=False)
        self.final_layer_norm = nn.LayerNorm([hidden_dim])

        self.node_scatter = AggregateNodeToGlobal(mode='mean')
        self.lift_node = LiftGlobalToNode()

    def construct(self, t, atom_types, frac_coords, lattices, node2graph,
                  edge_index, node_mask, edge_mask):
        """Apply GNN over noised fractional coordinates and lattice matrix.

        Args:
            t (Tensor): Time embeddind features. Shape: (batchsize, latent_dim)
            atom_types (Tensor): Atom types. Shape: (num_atoms,)
            frac_coords (Tensor): Fractional coordinates. Shape: (num_atoms, 3)
            lattices (Tensor): Lattice mattrix. Shape: (batchsize, 3, 3)
            node2graph (Tensor): Graph index for each node. Shape: (num_atoms,)
            edge_index (Tensor): Edge index for aggregating the edge features. Shape: (2, num_edges)
            node_mask (Tensor): Node mask for padded tensor. Shape: (num_atoms,)
            edge_mask (Tensor): Edge mask for padded tensor. Shape: (num_edges,)

        Returns:
            Tuple(Tensor,Tensor): Node features for fractional coordinates denoising terms and
            graph features for lattice matrix denoising terms.
        """
        edge_src = edge_index[0]
        edge_dst = edge_index[1]
        frac_diff = frac_coords[edge_dst] - frac_coords[edge_src]
        edge2graph = ops.Gather()(node2graph, edge_index[0], 0)

        node_features = self.node_embedding(atom_types - 1)
        node_features = mul_mask(node_features, node_mask)

        t_per_atom = self.lift_node(t, node2graph, mask=node_mask)

        node_features = ops.Concat(axis=1)((node_features, t_per_atom))

        node_features = self.atom_latent_emb(node_features)

        for i in range(self.num_layers):
            node_features = self.csp_layers[i](node_features,
                                               lattices, edge_index,
                                               edge2graph, frac_diff,
                                               node_mask, edge_mask)

        node_features = mul_mask(node_features, node_mask)
        node_features = self.final_layer_norm(node_features)
        node_features = mul_mask(node_features, node_mask)

        coord_out = self.coord_out(node_features)

        graph_features = self.node_scatter(node_features,
                                           node2graph,
                                           dim_size=lattices.shape[0],
                                           mask=node_mask)
        lattice_out = self.lattice_out(graph_features)
        lattice_out = ops.Reshape()(lattice_out, (-1, 3, 3))
        lattice_out = ops.BatchMatMul()(lattice_out, lattices)

        return lattice_out, coord_out
