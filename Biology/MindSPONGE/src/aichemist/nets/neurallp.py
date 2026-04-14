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
Neural logic programming
"""

import mindspore as ms
from mindspore import nn, ops
from utils import scatter_add


class NeuralLogicProgramming(nn.Cell):
    """
    Neural Logic Programming proposed in `Differentiable Learning of Logical
        Rules for Knowledge Base Reasoning`_.

    .. _Differentiable Learning of Logical Rules for Knowledge Base Reasoning:
        https://papers.nips.cc/paper/2017/file/0e55666a4ad822e0e34299df3591d979-Paper.pdf

    Args:
        n_relation (int): number of relations
        hidden_dim (int): dimension of hidden units in LSTM
        n_step (int): number of recurrent steps
        n_lstm_layer (int, optional): number of LSTM layers

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    eps = 1e-10

    def __init__(self, n_relation, hidden_dim, n_step, n_lstm_layer=1):
        super().__init__()

        n_relation = int(n_relation)
        self.n_relation = n_relation
        self.n_step = n_step

        self.query = nn.Embedding(n_relation * 2 + 1, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, n_lstm_layer)
        self.weight_linear = nn.Dense(hidden_dim, n_relation * 2)
        self.linear = nn.Dense(1, 1)

    def negative_sample_to_tail(self, h_index, t_index, r_index):
        # convert p(h | t, r) to p(t' | h', r')
        # h' = t, r' = r^{-1}, t' = h
        is_t_neg = (h_index == h_index[:, [0]]).all(axis=-1, keepdim=True)
        new_h_index = ops.where(is_t_neg, h_index, t_index)
        new_t_index = ops.where(is_t_neg, t_index, h_index)
        new_r_index = ops.where(is_t_neg, r_index, r_index + self.n_relation)
        return new_h_index, new_t_index, new_r_index

    def get_t_output(self, graph, h_index, r_index):
        """
        Calculate the t output.

        Args:
            graph (Tensor): fact graph
            h_index (Tensor): indexes of head entities
            t_index (Tensor): indexes of tail entities
            r_index (Tensor): indexes of relations

        Returns:
            output (Tensor): output
        """
        end_index = ops.ones_like(r_index) * graph.n_relation
        q_index = ops.stack(
            [r_index] * (self.n_step - 1) + [end_index], axis=0)
        query = self.query(q_index)

        hidden, _ = self.lstm(query)
        memory = ops.one_hot(h_index, graph.n_node, ms.Tensor(1.), ms.Tensor(0.)).expand_dims(0)

        for i in range(self.n_step):
            key = hidden[i]
            value = hidden[:i + 1]
            x = ops.einsum("bd, tbd -> bt", key, value)
            attention = ops.softmax(x)
            inputs = ops.einsum("bt, tbn -> nb", attention, memory)
            weight = ops.softmax(self.weight_linear(key)).t()

            node_in, node_out = graph
            relation = graph.edge_type
            if graph.n_node * graph.n_relation < graph.n_edge:
                # O(|V|d) memory
                node_out = node_out * graph.n_relation + relation
                shape = (graph.n_node, graph.n_node * graph.n_relation)
                adjacency = ms.COOTensor(ops.stack([node_in, node_out]),
                                         graph.edge_weight, shape)
                output = adjacency.t() @ inputs
                output = output.view(graph.n_node, graph.n_relation, -1)
                output = (output * weight).sum(axis=1)
            else:
                # O(|E|) memory
                message = inputs[node_in]
                message = message * weight[relation]
                edge_weight = graph.edge_weight.expand_dims(-1)
                output = scatter_add(message * edge_weight,
                                     node_out, axis=0, n_axis=graph.n_node)
            output = output / output.sum(axis=0, keepdim=True).clamp(self.eps)

            memory = ops.concat([memory, output.t().expand_dims(0)])

        return output

    def construct(self, graph, h_index, t_index, r_index):
        """
        Compute the score for triplets.

        Args:
            graph (Tensor): fact graph
            h_index (Tensor): indexes of head entities
            t_index (Tensor): indexes of tail entities
            r_index (Tensor): indexes of relations
            all_loss (Tensor, optional): if specified, add loss to this tensor
            metric (dict, optional): if specified, output metrics to this dict
        """
        assert graph.n_relation == self.n_relation
        graph = graph.undirected(add_inverse=True)

        h_index, t_index, r_index = self.negative_sample_to_tail(
            h_index, t_index, r_index)
        hr_index = h_index * graph.n_relation + r_index
        hr_index_set, hr_inverse = hr_index.unique(return_inverse=True)
        h_index_set = hr_index_set // graph.n_relation
        r_index_set = hr_index_set % graph.n_relation

        output = self.get_t_output(graph, h_index_set, r_index_set)

        score = output[t_index, hr_inverse]
        score = self.linear(score.expand_dims(-1)).squeeze(-1)
        return score
