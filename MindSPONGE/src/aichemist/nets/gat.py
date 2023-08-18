# Copyright 2021-2023 @ Shenzhen Bay Laboratory &
#                       Peking University &
#                       Huawei Technologies Co., Ltd
#
# This code is a part of AIchemist package.
#
# The AIchemist is open-source software based on the AI-framework:
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
graph attenion network
"""

from collections.abc import Sequence
from mindspore import ops
from mindspore import nn
import mindspore as ms
from .. import layers
from ..layers import embedding
from ..layers.aggregator import SumAggregation, MeanAggregation


class GraphAttentionNet(nn.Cell):
    """
    Graph Attention Network proposed in `Graph Attention Networks`_.

    .. _Graph Attention Networks:
        https://arxiv.org/pdf/1710.10903.pdf

    Args:
        input_dim (int): input dimension
        hidden_dims (list of int): hidden dimensions
        edge_input_dim (int, optional): dimension of edge features
        n_head (int, optional): number of attention heads
        negative_slope (float, optional): negative slope of leaky relu activation
        short_cut (bool, optional): use short cut or not
        batch_norm (bool, optional): apply batch normalization or not
        activation (str or function, optional): activation function
        concat_hidden (bool, optional): concat hidden representations from all layers as output
        readout (str, optional): readout function. Available functions are ``sum`` and ``mean``.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self, input_dim, hidden_dims, edge_input_dim=None, n_head=1, negative_slope=0.2, short_cut=False,
                 batch_norm=False, activation="ReLU", concat_hidden=False, readout="sum"):
        super().__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.input_dim = input_dim
        self.output_dim = sum(hidden_dims) if concat_hidden else hidden_dims[-1]
        self.dims = [input_dim] + list(hidden_dims)
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden

        self.layers = nn.CellList()
        for i in range(len(self.dims) - 1):
            lay = layers.GraphAttentionConv(self.dims[i], self.dims[i + 1], edge_input_dim, n_head,
                                            negative_slope, batch_norm, activation)
            self.layers.append(lay)
        if readout == "sum":
            self.readout = SumAggregation()
        elif readout == "mean":
            self.readout = MeanAggregation()
        else:
            raise ValueError(f"Unknown readout `{readout}`")

    def construct(self, graph, inputs):
        """
        Compute the node representations and the graph representation(s).

        Args:
            graph (Graph): :math:`n` graph(s)
            inputs (Tensor): input node representations
            all_loss (Tensor, optional): if specified, add loss to this tensor
            metric (dict, optional): if specified, output metrics to this dict

        Returns:
            dict with ``node`` and ``graph`` fields:
                node representations of shape :math:`(|V|, d)`, graph representations of shape :math:`(n, d)`
        """
        hiddens = []
        layer_input = inputs

        for lay in self.layers:
            hidden = lay(graph, layer_input)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden += layer_input
            hiddens.append(hidden)
            layer_input = hidden

        if self.concat_hidden:
            node_feat = ops.concat(hiddens, axis=-1)
        else:
            node_feat = hiddens[-1]
        graph_feat = self.readout(graph.node2graph, node_feat)

        return {
            "graph": graph_feat,
            "node": node_feat
        }


class KnowledgeBaseAttnNet(GraphAttentionNet):
    """
    Knowledge Base Graph Attention Network proposed in
    `Learning Attention-based Embeddings for Relation Prediction in Knowledge Graphs`_.

    .. _Learning Attention-based Embeddings for Relation Prediction in Knowledge Graphs:
        https://arxiv.org/pdf/1906.01195.pdf

    Args:
        n_entity (int): number of entities
        n_relation (int): number of relations
        embedding_dim (int): dimension of embeddings
        hidden_dims (list of int): hidden dimensions
        max_score (float, optional): maximal score for triplets
        **kwargs
    """

    def __init__(self,
                 n_entity,
                 n_relation,
                 embedding_dim=128,
                 hidden_dims=128,
                 max_score=12,
                 **kwargs):
        super().__init__(embedding_dim, hidden_dims, embedding_dim, **kwargs)
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.max_score = max_score

        self.linear = nn.Dense(self.output_dim, embedding_dim, weight_init='xavier_uniform')
        self.output_dim = embedding_dim

        entity = ops.randn(n_entity, embedding_dim)
        relation = ops.randn(n_relation, embedding_dim)
        self.entity = ms.Parameter(entity)
        self.relation = ms.Parameter(relation)

    def construct(self, graph, inputs):
        """
        Compute the score for triplets.

        Args:
            graph (Graph): fact graph
            inputs (List): including three tensors as follows,
                h_index (Tensor): indexes of head entities
                t_index (Tensor): indexes of tail entities
                r_index (Tensor): indexes of relations
        """
        h_index, t_index, r_index = inputs
        graph.edge_feat = self.relation[graph.edge_type]
        output = super().construct(graph, self.entity)
        entity = self.linear(output.get('node'))
        score = embedding.transe_score(entity, self.relation, h_index, t_index, r_index)
        return self.max_score - score
