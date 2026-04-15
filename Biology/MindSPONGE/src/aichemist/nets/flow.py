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
flow
"""
import mindspore as ms
from mindspore import nn
from mindspore import ops

from .. import layers


class GraphAutoregressiveFlow(nn.Cell):
    """
    Graph autoregressive flow proposed in `GraphAF: a Flow-based Autoregressive Model for Molecular Graph Generation`_.

    .. _GraphAF: a Flow-based Autoregressive Model for Molecular Graph Generation:
        https://arxiv.org/pdf/2001.09382.pdf

    Args:
        model (nn.Cell): graph representation model
        prior (nn.Cell): prior distribution
        use_edge (bool, optional): use edge or not
        n_flow_layer (int, optional): number of conditional flow layers
        n_mlp_layer (int, optional): number of MLP layers in each conditional flow
        dequantization_noise (float, optional): scale of dequantization noise
    """

    def __init__(self, model, prior, use_edge=False, n_layer=6, n_mlp_layer=2, dequantization_noise=0.9):
        super().__init__()
        self.model = model
        self.prior = prior
        self.use_edge = use_edge
        self.input_dim = self.output_dim = prior.dim
        self.dequantization_noise = dequantization_noise
        assert dequantization_noise < 1

        self.layers = nn.CellList()
        for _ in range(n_layer):
            condition_dim = model.output_dim * (3 if use_edge else 1)
            lay = layers.ConditionalFlow(self.input_dim, condition_dim,
                                         [model.output_dim] * (n_mlp_layer - 1))
            self.layers.append(lay)

    def construct(self, graph, inputs, edge=None):
        """
        Compute the log-likelihood for the input given the graph(s).

        Args:
            graph (Graph): :math:`n` graph(s)
            inputs (Tensor): discrete data of shape :math:`(n,)`
            edge (Tensor, optional): edge list of shape :math:`(n, 2)`.
                If specified, additionally condition on the edge for each input.
            all_loss (Tensor, optional): if specified, add loss to this tensor
            metric (dict, optional): if specified, output metrics to this dict
        """
        if self.use_edge and edge is None:
            raise ValueError("`use_edge` is true, but no edge is provided")

        edge = self._standarize_edge(graph, edge)

        node_feat = ops.one_hot(graph.node_type, self.model.input_dim, on_value=ms.Tensor(1.), off_value=ms.Tensor(0.))
        feat = self.model(graph, node_feat)
        node_feat = feat.node_feat
        graph_feat = feat.graph_feat
        if self.use_edge:
            condition = ops.concat([node_feat[edge], graph_feat.expand_dims(1)], axis=1).flatten(1)
        else:
            condition = graph_feat

        x = ops.one_hot(inputs, self.input_dim, on_value=ms.Tensor(1.), off_value=ms.Tensor(0.))
        x = x + self.dequantization_noise * ops.rand_like(x)

        log_dets = []
        for lay in self.layers:
            x, log_det = lay(x, condition)
            log_dets.append(log_det)

        log_prior = self.prior(x)
        log_det = ops.stack(log_dets).sum(axis=0)
        log_likelihood = log_prior + log_det
        log_likelihood = log_likelihood.sum(axis=-1)

        return log_likelihood  # (batch_size,)

    def sample(self, graph, edge=None):
        """
        Sample discrete data based on the given graph(s).

        Args:
            graph (Graph): :math:`n` graph(s)
            edge (Tensor, optional): edge list of shape :math:`(n, 2)`.
                If specified, additionally condition on the edge for each input.
            all_loss (Tensor, optional): if specified, add loss to this tensor
            metric (dict, optional): if specified, output metrics to this dict
        """
        if self.use_edge and edge is None:
            raise ValueError("`use_edge` is true, but no edge is provided")

        edge = self._standarize_edge(graph, edge)

        node_feat = ops.one_hot(graph.node_type, self.model.input_dim, on_value=ms.Tensor(1.), off_value=ms.Tensor(0.))
        feat = self.model(graph, node_feat)
        node_feat = feat.node_feat
        graph_feat = feat.graph_feat
        if self.use_edge:
            condition = ops.concat([node_feat[edge], graph_feat.expand_dims(1)], axis=1).flatten(1)
        else:
            condition = graph_feat

        x = self.prior.sample(len(graph))
        for lay in self.layers[::-1]:
            x, _ = lay.reverse(x, condition)

        output = x.argmax(axis=-1)

        return output  # (batch_size,)

    def _standarize_edge(self, graph, edge):
        if edge is not None:
            edge = edge.clone()
            if (edge[:, :2] >= graph.n_nodes.expand_dims(-1)).any():
                raise ValueError("Edge index exceeds the number of nodes in the graph")
            edge[:, :2] += (graph.cum_nodes - graph.n_nodes).expand(-1)
        return edge
