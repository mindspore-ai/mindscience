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
message passing neural network
"""

from mindspore import ops
from mindspore import nn
from .. import layers


class MessagePassingNet(nn.Cell):
    """
    Message Passing Neural Network proposed in `Neural Message Passing for Quantum Chemistry`_.

    This implements the enn-s2s variant in the original paper.

    .. _Neural Message Passing for Quantum Chemistry:
        https://arxiv.org/pdf/1704.01212.pdf

    Args:
        input_dim (int): input dimension
        hidden_dim (int): hidden dimension
        edge_input_dim (int): dimension of edge features
        n_layer (int, optional): number of hidden layers
        n_gru_layer (int, optional): number of GRU layers in each node update
        n_mlp_layer (int, optional): number of MLP layers in each message function
        n_s2s_step (int, optional): number of processing steps in set2set
        short_cut (bool, optional): use short cut or not
        batch_norm (bool, optional): apply batch normalization or not
        activation (str or function, optional): activation function
        concat_hidden (bool, optional): concat hidden representations from all layers as output

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self, input_dim, hidden_dim, edge_input_dim, n_layer=1, n_gru_layer=1, n_mlp_layer=2,
                 n_s2s_step=3, short_cut=False, batch_norm=False, activation="relu", concat_hidden=False):
        super().__init__()

        self.input_dim = input_dim
        self.edge_input_dim = edge_input_dim
        if concat_hidden:
            feat_dim = hidden_dim * n_layer
        else:
            feat_dim = hidden_dim
        self.output_dim = feat_dim * 2
        self.n_layer = n_layer
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden

        self.linear = nn.Dense(input_dim, hidden_dim, weight_init='xavier_uniform')
        self.layer = layers.GraphConv(hidden_dim, edge_input_dim, [hidden_dim] * (n_mlp_layer - 1),
                                      batch_norm, activation)
        self.gru = nn.GRU(hidden_dim, hidden_dim, n_gru_layer)

        self.readout = layers.Set2SetAggregation(feat_dim, n_s2s_step)

    def construct(self, graph, inputs):
        """
        Compute the node representations and the graph representation(s).

        Args:
            graph (Graph): :math:`n` graph(s)
            inputs (Tensor): input node representations

        Returns:
            dict with ``node`` and ``graph`` fields:
                node representations of shape :math:`(|V|, d)`, graph representations of shape :math:`(n, d)`
        """
        hiddens = []
        layer_input = self.linear(inputs)
        hx = layer_input.repeat(self.gru.n_layers, 1, 1)

        for _ in range(self.n_layer):
            x = self.layer(graph, layer_input)
            hidden, hx = self.gru(x.expand_dims(0), hx)
            hidden = hidden.expand_dims(0)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
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
