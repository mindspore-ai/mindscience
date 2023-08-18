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
Sampler layers
"""

from mindspore import nn
from mindspore import ops
from ..utils import scatter_add


class NodeSampler(nn.Cell):
    """
    Node sampler from `GraphSAINT: Graph Sampling Based Inductive Learning Method`_.

    .. _GraphSAINT: Graph Sampling Based Inductive Learning Method:
        https://arxiv.org/pdf/1907.04931.pdf

    Args:
        budget (int, optional): number of node to keep
        ratio (int, optional): ratio of node to keep

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self, budget=None, ratio=None):
        super().__init__()
        if budget is None and ratio is None:
            raise ValueError("At least one of `budget` and `ratio` should be provided")
        self.budget = budget
        self.ratio = ratio

    def construct(self, graph):
        """ construct
        Args:
            graph (Graph): input graph

        Returns:
            new_graph (Graph): output graph
        """
        # this is exact for a single graph
        # but approximate for packed graphs
        n_sample = graph.n_node
        if self.budget:
            n_sample = min(n_sample, self.budget)
        if self.ratio:
            n_sample = min(n_sample, int(self.ratio * graph.n_node))

        prob = scatter_add(graph.edge_weight ** 2, graph.edges[1], n_axis=graph.n_node)
        prob /= prob.mean()
        index = ops.multinomial(prob, n_sample)
        new_graph = graph.node_mask(index)
        node_out = new_graph.edges[1]
        new_graph._edge_weight /= n_sample * prob[node_out] / graph.n_node

        return new_graph


class EdgeSampler(nn.Cell):
    """
    Edge sampler from `GraphSAINT: Graph Sampling Based Inductive Learning Method`_.

    .. _GraphSAINT: Graph Sampling Based Inductive Learning Method:
        https://arxiv.org/pdf/1907.04931.pdf

    Args:
        budget (int, optional): number of node to keep
        ratio (int, optional): ratio of node to keep
    """

    def __init__(self, budget=None, ratio=None):
        super().__init__()
        if budget is None and ratio is None:
            raise ValueError("At least one of `budget` and `ratio` should be provided")
        self.budget = budget
        self.ratio = ratio

    def construct(self, graph):
        """
        Sample a subgraph from the graph.

        Args:
            graph (Graph): graph(s)
        """
        # this is exact for a single graph
        # but approximate for packed graphs
        node_in, node_out = graph.edges
        n_sample = graph.n_edge
        if self.budget:
            n_sample = min(n_sample, self.budget)
        if self.ratio:
            n_sample = min(n_sample, int(self.ratio * graph.n_edge))

        prob = 1 / graph.degree_out[node_out] + 1 / graph.degree_in[node_in]
        prob = prob / prob.mean()
        index = ops.multinomial(prob, n_sample)
        new_graph = graph.edge_mask(index)
        new_graph._edge_weight /= n_sample * prob[index] / graph.n_edge

        return new_graph
