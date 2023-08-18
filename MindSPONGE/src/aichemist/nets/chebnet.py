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
chebnet
"""

from collections.abc import Sequence
from mindspore import nn
from mindspore import numpy as mnp
from .. import layers
from ..layers.aggregator import SumAggregation, MeanAggregation


class ChebyshevConvNet(nn.Cell):
    """
    Chebyshev convolutional network proposed in
    `Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering`_.

    .. _Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering:
        https://arxiv.org/pdf/1606.09375.pdf

    Args:
        input_dim (int): input dimension
        hidden_dims (list of int): hidden dimensions
        edge_input_dim (int, optional): dimension of edge features
        k (int, optional): number of Chebyshev polynomials
        short_cut (bool, optional): use short cut or not
        batch_norm (bool, optional): apply batch normalization or not
        activation (str or function, optional): activation function
        concat_hidden (bool, optional): concat hidden representations from all layers as output
        readout (str, optional): readout function. Available functions are ``sum`` and ``mean``.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self, input_dim, hidden_dims, edge_input_dim=None, k=1, short_cut=False, batch_norm=False,
                 activation="relu", concat_hidden=False, readout="sum"):
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
            self.layers.append(layers.ChebyshevConv(self.dims[i], self.dims[i + 1], edge_input_dim, k,
                                                    batch_norm, activation))

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
            assert not mnp.isnan(hidden).any()
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            hiddens.append(hidden)
            layer_input = hidden

        if self.concat_hidden:
            node_feat = mnp.concatenate(hiddens, axis=-1)
        else:
            node_feat = hiddens[-1]
        graph_feat = self.readout(graph, node_feat)

        return {
            "graph": graph_feat,
            "node": node_feat
        }
