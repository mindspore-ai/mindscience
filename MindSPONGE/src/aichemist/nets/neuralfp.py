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
Convolutional Neural fingerprint
"""

from collections.abc import Sequence
from mindspore import nn
from mindspore import ops
from ..layers import NeuralFingerprintConv
from ..layers.aggregator import SumAggregation, MeanAggregation


class NeuralFingerprint(nn.Cell):
    """
    Neural Fingerprints from `Convolutional Networks on Graphs for Learning Molecular Fingerprints`_.

    .. _Convolutional Networks on Graphs for Learning Molecular Fingerprints:
        https://arxiv.org/pdf/1509.09292.pdf

    Args:
        input_dim (int): input dimension
        output_dim (int): fingerprint dimension
        hidden_dims (list of int): hidden dimensions
        edge_input_dim (int, optional): dimension of edge features
        short_cut (bool, optional): use short cut or not
        batch_norm (bool, optional): apply batch normalization or not
        activation (str or function, optional): activation function
        concat_hidden (bool, optional): concat hidden representations from all layers as output
        readout (str, optional): readout function. Available functions are ``sum`` and ``mean``.
    """

    def __init__(self, input_dim, output_dim, hidden_dims, edge_input_dim=None, short_cut=False, batch_norm=False,
                 activation="ReLU", concat_hidden=False, readout="sum"):
        super().__init__()
        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.input_dim = input_dim
        self.output_dim = output_dim * (len(hidden_dims) if concat_hidden else 1)
        self.dims = [input_dim] + list(hidden_dims)
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden

        self.layers = nn.CellList()
        self.linears = nn.CellList()
        for i in range(len(self.dims) - 1):
            self.layers.append(NeuralFingerprintConv(self.dims[i], self.dims[i + 1], edge_input_dim,
                                                     batch_norm, activation))
            self.linears.append(nn.Dense(self.dims[i + 1], output_dim, weight_init='xavier_uniform'))

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
        outputs = []

        layer_input = inputs

        for layer, linear in zip(self.layers, self.linears):
            hidden = layer(graph, layer_input)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            output = linear(hidden)
            output = ops.softmax(output)
            hiddens.append(hidden)
            outputs.append(output)
            layer_input = hidden

        if self.concat_hidden:
            node_feat = ops.concat(hiddens, axis=-1)
            graph_feat = ops.concat(outputs, axis=-1)
        else:
            node_feat = hiddens[-1]
            graph_feat = ops.stack(outputs).sum(axis=0)

        graph_feat = self.readout(graph.node2graph, graph_feat)

        return {
            "graph": graph_feat,
            "node": node_feat
        }
