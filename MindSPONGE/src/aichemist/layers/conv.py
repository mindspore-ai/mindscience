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
Common layers
"""

import mindspore as ms
from mindspore import nn
from mindspore import ops
from mindspore.common.initializer import initializer, HeUniform
from .. import data
from .. import utils
from ..utils import scatter_mean, scatter_add, scatter_max


class MessagePassingBase(nn.Cell):
    """
    Base module for message passing.

    Any custom message passing module should be derived from this class.
    """
    gradient_checkpoint = False
    eps = 1e-10

    def message(self, graph, inputs):
        """
        Compute edge messages for the graph.

        Args:
            graph (Graph): graph(s)
            inputs (Tensor): node representations of shape :math:`(|V|, ...)`

        Returns:
            Tensor: edge messages of shape :math:`(|E|, ...)`
        """
        raise NotImplementedError

    def aggregate(self, graph, message):
        """
        Aggregate edge messages to nodes.

        Args:
            graph (Graph): graph(s)
            message (Tensor): edge messages of shape :math:`(|E|, ...)`

        Returns:
            Tensor: node updates of shape :math:`(|V|, ...)`
        """
        raise NotImplementedError

    def message_and_aggregate(self, graph, inputs):
        """
        Fused computation of message and aggregation over the graph.
        This may provide better time or memory complexity than separate calls of
        :meth:`message <MessagePassingBase.message>` and :meth:`aggregate <MessagePassingBase.aggregate>`.

        Args:
            graph (Graph): graph(s)
            inputs (Tensor): node representations of shape :math:`(|V|, ...)`

        Returns:
            Tensor: node updates of shape :math:`(|V|, ...)`
        """
        message = self.message(graph, inputs)
        update = self.aggregate(graph, message)
        return update

    def combine(self, inputs, update):
        """
        Combine node inputs and node update.

        Args:
            inputs (Tensor): node representations of shape :math:`(|V|, ...)`
            update (Tensor): node updates of shape :math:`(|V|, ...)`
        """
        raise NotImplementedError

    def construct(self, graph, inputs):
        """
        Perform message passing over the graph(s).

        Args:
            graph (Graph): graph(s)
            inputs (Tensor): node representations of shape :math:`(|V|, ...)`
        """
        update = self.message_and_aggregate(graph, inputs)
        output = self.combine(inputs, update)
        return output

    def _message_and_aggregate(self, *tensors):
        graph = data.Graph.from_tensors(tensors[:-1])
        inputs = tensors[-1]
        update = self.message_and_aggregate(graph, inputs)
        return update


class GraphConv(MessagePassingBase):
    """
    Graph convolution operator from `Semi-Supervised Classification with Graph Convolutional Networks`_.

    .. _Semi-Supervised Classification with Graph Convolutional Networks:
        https://arxiv.org/pdf/1609.02907.pdf

    Args:
        input_dim (int): input dimension
        output_dim (int): output dimension
        edge_input_dim (int, optional): dimension of edge features
        batch_norm (bool, optional): apply batch normalization on nodes or not
        activation (str or function, optional): activation function

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self, input_dim, output_dim, edge_input_dim=None, batch_norm=False, activation="ReLU"):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.edge_input_dim = edge_input_dim

        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(output_dim)
        else:
            self.batch_norm = None
        if isinstance(activation, str):
            self.activation = getattr(nn, activation)()
        else:
            self.activation = activation

        self.linear = nn.Dense(input_dim, output_dim, weight_init='xavier_uniform')
        if edge_input_dim:
            self.edge_linear = nn.Dense(edge_input_dim, input_dim, weight_init='xavier_uniform')
        else:
            self.edge_linear = None

    def message(self, graph, inputs):
        # add self loop
        node_in = ops.concat([graph.edges[0], ops.arange(int(graph.n_node), dtype=ms.int32)])
        edge_weight = ops.concat([graph.edge_weight, ops.ones(int(graph.n_node))])
        degree_in = ops.expand_dims(graph.degree_in(), -1) + 1
        edge_weight = ops.expand_dims(edge_weight, axis=-1) / (ops.sqrt(degree_in[node_in]) + self.eps)
        message = inputs[node_in]
        if self.edge_linear:
            edge_input = self.edge_linear(graph.edge_feat)
            edge_input = ops.concat([edge_input, ops.zeros(int(graph.n_node), self.input_dim)])
            message += edge_input
        message *= edge_weight
        return message

    def aggregate(self, graph, message):
        # add self loop
        node_out = ops.concat([graph.edges[1], ops.arange(int(graph.n_node), dtype=ms.int32)])
        degree_out = ops.expand_dims(graph.degree_out, axis=-1) + 1
        update = scatter_add(message, node_out, axis=0, n_axis=graph.n_node)
        update = update / (ops.sqrt(degree_out) + self.eps)
        return update

    def combine(self, inputs, update):
        output = self.linear(update)
        if self.batch_norm:
            output = self.batch_norm(output)
        if self.activation:
            output = self.activation(output)
        return output


class GraphAttentionConv(MessagePassingBase):
    """
    Graph attentional convolution operator from `Graph Attention Networks`_.

    .. _Graph Attention Networks:
        https://arxiv.org/pdf/1710.10903.pdf

    Args:
        input_dim (int): input dimension
        output_dim (int): output dimension
        edge_input_dim (int, optional): dimension of edge features
        n_head (int, optional): number of attention heads
        negative_slope (float, optional): negative slope of leaky relu activation
        batch_norm (bool, optional): apply batch normalization on nodes or not
        activation (str or function, optional): activation function

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self, input_dim, output_dim, edge_input_dim=None, n_head=1, negative_slope=0.2,
                 batch_norm=False, activation="ReLU"):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.edge_input_dim = edge_input_dim
        self.n_head = n_head
        self.leaky_relu = nn.LeakyReLU()

        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(output_dim)
        else:
            self.batch_norm = None
        if isinstance(activation, str):
            self.activation = getattr(ops, activation)()
        else:
            self.activation = activation
        assert output_dim % n_head == 0, \
            f"Expect output_dim to be a multiplier of n_head, but found `{output_dim}` and `{n_head}`"

        self.linear = nn.Dense(input_dim, output_dim, weight_init='xavier_uniform')
        if edge_input_dim:
            self.edge_linear = nn.Dense(edge_input_dim, output_dim, weight_init='xavier_uniform')
        else:
            self.edge_linear = None
        query = initializer(HeUniform(negative_slope=negative_slope), (n_head, output_dim * 2 // n_head), ms.float32)
        self.query = ms.Parameter(query)

    def message(self, graph, inputs):
        # add self loop
        node_in = ops.concat([graph.edges[0], ops.arange(int(graph.n_node), dtype=ms.int32)])
        node_out = ops.concat([graph.edges[1], ops.arange(int(graph.n_node), dtype=ms.int32)])
        edge_weight = ops.concat([graph.edge_weight, ops.ones(int(graph.n_node))])
        edge_weight = ops.expand_dims(edge_weight, axis=-1)
        hidden = self.linear(inputs)

        key = ops.stack([hidden[node_in], hidden[node_out]], axis=-1)
        if self.edge_linear:
            edge_input = self.edge_linear(graph.edge_feat)
            edge_input = ops.concat([edge_input, ops.zeros((int(graph.n_node), self.output_dim))])
            key += ops.expand_dims(edge_input, axis=-1)
        key = key.view(-1, *self.query.shape)
        weight = ops.Einsum("hd, nhd -> nh")((self.query, key))
        weight = self.leaky_relu(weight)

        weight = weight - scatter_max(weight, node_out, axis=0, n_axis=graph.n_node)[node_out]
        attention = ops.exp(weight) * edge_weight
        # why mean? because with mean we have normalized message scale across different node degrees
        normalizer = scatter_mean(attention, node_out, axis=0, n_axis=graph.n_node)[node_out]
        attention = attention / (normalizer + self.eps)

        value = hidden[node_in].view(-1, self.n_head, self.query.shape[-1] // 2)
        attention = ops.expand_dims(attention, axis=-1).expand_as(value)
        message = utils.flatten(attention * value, start=1)
        return message

    def aggregate(self, graph, message):
        # add self loop
        node_out = ops.concat([graph.edges[1], ops.arange(int(graph.n_node), dtype=ms.int32)])
        update = scatter_mean(message, node_out, axis=0, n_axis=graph.n_node)
        return update

    def combine(self, inputs, update):
        output = update
        if self.batch_norm:
            output = self.batch_norm(output)
        if self.activation:
            output = self.activation(output)
        return output


class NeuralFingerprintConv(MessagePassingBase):
    """
    Graph neural network operator from `Convolutional Networks on Graphs for Learning Molecular Fingerprints`_.

    Note this operator doesn't include the sparsifying step of the original paper.

    .. _Convolutional Networks on Graphs for Learning Molecular Fingerprints:
        https://arxiv.org/pdf/1509.09292.pdf

    Args:
        input_dim (int): input dimension
        output_dim (int): output dimension
        edge_input_dim (int, optional): dimension of edge features
        batch_norm (bool, optional): apply batch normalization on nodes or not
        activation (str or function, optional): activation function

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self, input_dim, output_dim, edge_input_dim=None, batch_norm=False, activation="ReLU"):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.edge_input_dim = edge_input_dim

        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(output_dim)
        else:
            self.batch_norm = None
        if isinstance(activation, str):
            self.activation = getattr(nn, activation)()
        else:
            self.activation = activation

        self.linear = nn.Dense(input_dim, output_dim, weight_init='xavier_uniform')
        if edge_input_dim:
            self.edge_linear = nn.Dense(edge_input_dim, input_dim, weight_init='xavier_uniform')
        else:
            self.edge_linear = None

    def message(self, graph, inputs):
        node_in = graph.edges[0]
        message = inputs[node_in]
        edge_weight = ops.expand_dims(graph.edge_weight, axis=-1)
        if self.edge_linear:
            message += self.edge_linear(graph.edge_feat)
        message *= edge_weight
        return message

    def aggregate(self, graph, message):
        node_out = graph.edges[1]
        update = scatter_add(message, node_out, axis=0, n_axis=graph.n_node)
        return update

    def combine(self, inputs, update):
        output = self.linear(inputs + update)
        if self.batch_norm:
            output = self.batch_norm(output)
        if self.activation:
            output = self.activation(output)
        return output


class RelationalGraphConv(MessagePassingBase):
    """
    Relational graph convolution operator from `Modeling Relational Data with Graph Convolutional Networks`_.
    .. _Modeling Relational Data with Graph Convolutional Networks: https://arxiv.org/pdf/1703.06103.pdf

    Args:
        input_dim (int): input dimension
        output_dim (int): output dimension
        n_relation (int): number of relations
        edge_input_dim (int, optional): dimension of edge features
        batch_norm (bool, optional): apply batch normalization on nodes or not
        activation (str or function, optional): activation function

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self, input_dim, output_dim, n_relation, edge_input_dim=None, batch_norm=False, activation="relu"):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_relation = n_relation
        self.edge_input_dim = edge_input_dim

        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(output_dim)
        else:
            self.batch_norm = None
        if isinstance(activation, str):
            self.activation = getattr(ops, activation)
        else:
            self.activation = activation

        self.self_loop = nn.Dense(input_dim, output_dim, weight_init='xavier_uniform')
        self.linear = nn.Dense(n_relation * input_dim, output_dim, weight_init='xavier_uniform')
        if edge_input_dim:
            self.edge_linear = nn.Dense(edge_input_dim, input_dim, weight_init='xavier_uniform')
        else:
            self.edge_linear = None

    def message(self, graph, inputs):
        node_in = graph.edges[0]
        message = inputs[node_in]
        if self.edge_linear:
            message += self.edge_linear(graph.edge_feat)
        return message

    def aggregate(self, graph, message):
        assert graph.n_relation == self.n_relation

        node_out = graph.edges[1] * self.n_relation + graph.edge_type
        edge_weight = ops.expand_dims(graph.edge_weight, -1)
        update = scatter_add(message * edge_weight, node_out, axis=0, n_axis=graph.n_node * self.n_relation) / \
            (scatter_add(edge_weight, node_out, axis=0, n_axis=graph.n_node * self.n_relation) + self.eps)
        return update.view(int(graph.n_node), self.n_relation * self.input_dim)

    def message_and_aggregate(self, graph, inputs):
        if graph.edges is None:
            return ops.zeros((int(graph.n_node), self.n_relation * self.input_dim))
        return super().message_and_aggregate(graph, inputs)

    def combine(self, inputs, update):
        output = self.linear(update) + self.self_loop(inputs)
        if self.batch_norm:
            output = self.batch_norm(output)
        if self.activation:
            output = self.activation(output)
        return output


class ChebyshevConv(MessagePassingBase):
    """
    Chebyshev spectral graph convolution operator from
    `Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering`_.

    .. _Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering:
        https://arxiv.org/pdf/1606.09375.pdf

    Args:
        input_dim (int): input dimension
        output_dim (int): output dimension
        edge_input_dim (int, optional): dimension of edge features
        k (int, optional): number of Chebyshev polynomials.
            This also corresponds to the radius of the receptive field.
        hidden_dims (list of int, optional): hidden dims of edge network
        batch_norm (bool, optional): apply batch normalization on nodes or not
        activation (str or function, optional): activation function

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self, input_dim: int, output_dim: int, edge_input_dim: int = None,
                 k: int = 1, batch_norm: bool = False, activation="relu"):
        super(ChebyshevConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.k = k
        self.edge_input_dim = edge_input_dim

        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(output_dim)
        else:
            self.batch_norm = None
        if isinstance(activation, str):
            self.activation = getattr(nn, activation)
        else:
            self.activation = activation

        self.linear = nn.Dense((k + 1) * input_dim, output_dim)
        if edge_input_dim:
            self.edge_linear = nn.Dense(edge_input_dim, input_dim)
        else:
            self.edge_linear = None

    def message(self, graph, inputs):
        node_in = graph.edges[0]
        degree_in = graph.degree_in.expand_dims(-1)
        # because self-loop messages have a different scale, they are processed in combine()
        message = inputs[node_in]
        if self.edge_linear:
            message += self.edge_linear(graph.edge_feature.float())
        message /= (degree_in[node_in].sqrt() + 1e-10)
        return message

    def aggregate(self, graph, message):
        node_out = graph.edges[1]
        edge_weight = graph.edge_weight.expand_dims(-1)
        degree_out = graph.degree_out.expand_dims(-1)
        # because self-loop messages have a different scale, they are processed in combine()
        update = -scatter_add(message * edge_weight, node_out, axis=0, n_axis=graph.n_node)
        update = update / (degree_out.sqrt() + 1e-10)
        return update

    def construct(self, graph, inputs):
        # Chebyshev polynomial bases
        bases = [inputs]
        for i in range(self.k):
            x = super().construct(graph, bases[-1])
            if i > 0:
                x = 2 * x - bases[-2]
            bases.append(x)
        bases = ops.concat(bases, axis=-1)

        output = self.linear(bases)
        if self.batch_norm:
            x = self.batch_norm(output)
        if self.activation:
            output = self.activation(output)
        return output

    def combine(self, inputs, update):
        output = inputs + update
        return output
