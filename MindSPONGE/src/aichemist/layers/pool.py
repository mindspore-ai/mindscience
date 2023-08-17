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
Pool layers
"""

import mindspore as ms
from mindspore import nn
from mindspore import ops
from ..utils import scatter_add, scatter_mean

from ..data import Graph, GraphBatch


class DiffPool(nn.Cell):
    """
    Differentiable pooling operator from `Hierarchical Graph Representation Learning with Differentiable Pooling`_

    .. _Hierarchical Graph Representation Learning with Differentiable Pooling:
        https://papers.nips.cc/paper/7729-hierarchical-graph-representation-learning-with-differentiable-pooling.pdf

    Parameter
        input_dim (int):                    input dimension
        output_node (int):                  number of nodes after pooling
        feature_layer (nn.Cell, optional):   graph convolution layer for embedding
        pool_layer (nn.Cell, optional):      graph convolution layer for pooling assignment
        loss_weight (float, optional):      weight of entropy regularization
        zero_diagonal (bool, optional):     remove self loops in the pooled graph or not
        sparse (bool, optional):            use sparse assignment or not

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    tau = 1
    eps = 1e-10

    def __init__(self, input_dim, output_node, feature_layer=None, pool_layer=None, loss_weight=1, zero_diagonal=False,
                 sparse=False):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = feature_layer.output_dim
        self.output_node = output_node
        self.feature_layer = feature_layer
        self.pool_layer = pool_layer
        self.loss_weight = loss_weight
        self.zero_diagonal = zero_diagonal
        self.sparse = sparse

        if pool_layer is not None:
            self.linear = nn.Dense(pool_layer.output_dim, output_node, weight_init='xavier_uniform')
        else:
            self.linear = nn.Dense(input_dim, output_node, weight_init='xavier_uniform')

    def construct(self, graph, inputs, all_loss=None, metric=None):
        """
        Compute the node cluster assignment and pool the nodes.

        Args:
            graph (Graph):                  graph(s)
            inputs (Tensor):                input node representations
            all_loss (Tensor, optional):    if specified, add loss to this tensor
            metric (dict, optional):        if specified, output metrics to this dict

        Returns:
            (GraphBatch, Tensor, Tensor):
                pooled graph, output node representations, node-to-cluster assignment
        """
        feature = inputs
        if self.feature_layer:
            feature = self.feature_layer(graph, feature)

        x = inputs
        if self.pool_layer:
            x = self.pool_layer(graph, x)
        x = self.linear(x)
        if self.sparse:
            assignment = ops.gumbel_softmax(x, hard=True, tau=self.tau, dim=-1)
            new_graph, output = self.sparse_pool(graph, feature, assignment)
        else:
            assignment = ops.softmax(x, axis=-1)
            new_graph, output = self.dense_pool(graph, feature, assignment)

        if all_loss is not None:
            prob = scatter_mean(assignment, graph.node2graph, axis=0, n_axis=graph.batch_size)
            entropy = -(prob * (prob + self.eps).log()).sum(axis=-1)
            entropy = entropy.mean()
            metric["assignment entropy"] = entropy
            if self.loss_weight > 0:
                all_loss -= entropy * self.loss_weight

        if self.zero_diagonal:
            edge_list = new_graph.edges
            is_diagonal = edge_list[0] == edge_list[1]
            new_graph = Graph.edge_mask(new_graph, ~is_diagonal)

        return new_graph, output, assignment

    def dense_pool(self, graph, inputs, assignment):
        """_summary_

        Args:
            graph (Graph): Graph
            inputs (ms.Tensor): inputs
            assignment (ms.Tensor): assignment

        Returns:
            _type_: _description_
        """
        node_in, node_out = graph.edges
        # S^T A S, O(|V|k^2 + |E|k)
        x = graph.edge_weight.expand_dims(-1) * assignment[node_out]
        x = scatter_add(x, node_in, axis=0, n_axis=graph.n_node)
        x = ops.einsum("np, nq -> npq", assignment, x)
        adjacency = scatter_add(x, graph.node2graph, axis=0, n_axis=graph.batch_size)
        # S^T X
        x = ops.einsum("na, nd -> nad", assignment, inputs)
        output = scatter_add(x, graph.node2graph, axis=0, n_axis=graph.batch_size).flatten(0, 1)

        index = ops.arange(self.output_node).expand(len(graph), self.output_node, -1)
        edge_list = ops.stack([index.transpose(-1, -2), index], axis=-1).flatten(0, -2)
        edge_weight = adjacency.flatten()
        if isinstance(graph, GraphBatch):
            n_nodes = ops.ones(len(graph), dtype=ms.int32) * self.output_node
            n_edges = ops.ones(len(graph), dtype=ms.int32) * self.output_node ** 2
            graph = GraphBatch(edge_list, edge_weight=edge_weight, n_nodes=n_nodes, n_edges=n_edges)
        else:
            graph = Graph(edge_list, edge_weight=edge_weight, n_node=self.output_node)
        return graph, output

    def sparse_pool(self, graph, inputs, assignment):
        """_summary_

        Args:
            graph (Graph): input graph
            inputs (ms.Tensor): input tensor
            assignment (ms.Tensor): assignment tensor

        Returns:
            graph (Graph): output graph
            output (ms.Tensor): output tensor
        """
        assignment = assignment.argmax(axis=-1)
        edge_list = assignment[graph.edges.T]
        pooled_node = graph.node2graph * self.output_node + assignment
        output = scatter_add(inputs, pooled_node, axis=0, n_axis=graph.batch_size * self.output_node)

        edge_weight = graph.edge_weight
        if isinstance(graph, GraphBatch):
            n_nodes = ops.ones(len(graph), dtype=ms.int32) * self.output_node
            n_edges = graph.n_edges
            graph = GraphBatch(edge_list, edge_weight=edge_weight, n_nodes=n_nodes, n_edges=n_edges)
        else:
            graph = Graph(edge_list, edge_weight=edge_weight, n_node=self.output_node)
        return graph, output


class MinCutPool(DiffPool):
    """
    Min cut pooling operator from `Spectral Clustering with Graph Neural Networks for Graph Pooling`_

    .. _Spectral Clustering with Graph Neural Networks for Graph Pooling:
        http://proceedings.mlr.press/v119/bianchi20a/bianchi20a.pdf

    Args:
        input_dim (int): input dimension
        output_node (int): number of nodes after pooling
        feature_layer (nn.Cell, optional): graph convolution layer for embedding
        pool_layer (nn.Cell, optional): graph convolution layer for pooling assignment
        loss_weight (float, optional): weight of entropy regularization
        zero_diagonal (bool, optional): remove self loops in the pooled graph or not
        sparse (bool, optional): use sparse assignment or not

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    eps = 1e-10

    def __init__(self, input_dim, output_node, feature_layer=None, pool_layer=None, loss_weight=1, zero_diagonal=True,
                 sparse=False):
        super().__init__(input_dim, output_node, feature_layer, pool_layer, loss_weight, zero_diagonal, sparse)

    def forward(self, graph, inputs, all_loss=None, metric=None):
        """
        Compute the node cluster assignment and pool the nodes.

        Args:
            graph (Graph): graph(s)
            inputs (Tensor): input node representations
            all_loss (Tensor, optional): if specified, add loss to this tensor
            metric (dict, optional): if specified, output metrics to this dict

        Returns:
            (PackedGraph, Tensor, Tensor):
                pooled graph, output node representations, node-to-cluster assignment
        """
        feature = inputs
        if self.feature_layer:
            feature = self.feature_layer(graph, feature)

        x = inputs
        if self.pool_layer:
            x = self.pool_layer(graph, x)
        x = self.linear(x)
        if self.sparse:
            assignment = ops.gumbel_softmax(x, hard=True, tau=self.tau, dim=-1)
            new_graph, output = self.sparse_pool(graph, feature, assignment)
        else:
            assignment = ops.softmax(x, axis=-1)
            new_graph, output = self.dense_pool(graph, feature, assignment)

        if all_loss is not None:
            edge_list = new_graph.edges.T
            is_diagonal = edge_list[:, 0] == edge_list[:, 1]
            n_intra = scatter_add(new_graph.edge_weight[is_diagonal], new_graph.edge2graph[is_diagonal],
                                  axis=0, n_axis=new_graph.batch_size)
            x = ops.einsum("na, n, nc -> nac", assignment, graph.degree_in, assignment)
            x = scatter_add(x, graph.node2graph, axis=0, n_axis=graph.batch_size)
            n_all = ops.einsum("baa -> b", x)
            cut_loss = (1 - n_intra / (n_all + self.eps)).mean()
            metric["normalized cut loss"] = cut_loss

            x = ops.einsum("na, nc -> nac", assignment, assignment)
            x = scatter_add(x, graph.node2graph, axis=0, n_axis=graph.batch_size)
            x = x / x.flatten(-2).norm(axis=-1, keepdim=True).expand_dims(-1)
            x = x - ops.eye(self.output_node) / (self.output_node ** 0.5)
            regularization = x.flatten(-2).norm(dim=-1).mean()
            metric["orthogonal regularization"] = regularization
            if self.loss_weight > 0:
                all_loss += (cut_loss + regularization) * self.loss_weight

        if self.zero_diagonal:
            edge_list = new_graph.edges
            is_diagonal = edge_list[0] == edge_list[1]
            new_graph = Graph.edge_mask(new_graph, ~is_diagonal)

        return new_graph, output, assignment
