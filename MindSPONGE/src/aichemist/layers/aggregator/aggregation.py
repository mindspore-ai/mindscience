# Copyright 2021-2022 @ Shenzhen Bay Laboratory &
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
Extra aggregation function
"""

from mindspore import nn
from mindspore import ops
from ... import utils


class MeanAggregation(nn.Cell):
    """
    Mean readout operator over graphs with variadic sizes.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def construct(self, node2graph, inputs, axis=0):
        """
        Perform readout over the graph(s).

        Args:
            graph (Graph):      graph(s)
            inputs (Tensor):    node representations

        Returns:
            Tensor: graph representations
        """
        batch_size = node2graph.max() + 1
        output = utils.scatter_mean(inputs, node2graph, axis=axis, n_axis=batch_size)
        return output


class SumAggregation(nn.Cell):
    """Sum readout operator over graphs with variadic sizes.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def construct(self, node2graph, inputs, axis=0):
        """
        Perform readout over the graph(s).

        Args:
            graph (Graph):      graph(s)
            inputs (Tensor):    node representations

        Returns:
            Tensor: graph representations
        """
        batch_size = node2graph.max() + 1
        output = utils.scatter_add(inputs, node2graph, axis=axis, n_axis=batch_size)
        return output


class MaxAggregation(nn.Cell):
    """Max readout operator over graphs with variadic sizes.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def construct(self, node2graph, inputs, axis=0):
        """
        Perform readout over the graph(s).

        Args:
            graph (Graph):      graph(s)
            inputs (Tensor):    node representations

        Returns:
            Tensor: graph representations
        """
        batch_size = node2graph.max() + 1
        output = utils.scatter_max(inputs, node2graph, axis=axis, n_axis=batch_size)[0]
        return output


class SoftmaxAggregation(nn.Cell):
    """Softmax operator over graphs with variadic sizes.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    eps = 1e-10

    def construct(self, node2graph, inputs, axis=0):
        """
        Perform softmax over the graph(s).

        Args:
            graph (Graph):      graph(s)
            inputs (Tensor):    node logits

        Returns:
            Tensor: node probabilities
        """
        batch_size = node2graph.max() + 1
        x = inputs - utils.scatter_max(inputs, node2graph, axis=axis, n_axis=batch_size)[0][node2graph]
        x = x.exp()
        normalizer = utils.scatter_add(x, node2graph, axis=axis, n_axis=batch_size)[node2graph]
        return x / (normalizer + self.eps)


class SortAggregation(nn.Cell):
    """
    Sort operator over graphs with variadic sizes.

    Args:
        descending (bool, optional): use descending sort order or not

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self, descending=False):
        super().__init__()
        self.descending = descending

    def construct(self, node2graph, inputs, axis=0):
        """
        Perform sort over graph(s).

        Args:
            graph (Graph):      graph(s)
            inputs (Tensor):    node values

        Returns:
            (Tensor, LongTensor): sorted values, sorted indices
        """
        step = inputs.max(axis=axis) - inputs.min(axis=axis) + 1
        if self.descending:
            step = -step
        x = inputs + node2graph * step
        sort_idx, index = x.sort(axis=axis, descending=self.descending)
        sort_idx = sort_idx - node2graph * step
        return sort_idx, index


class Set2SetAggregation(nn.Cell):
    """
    Set2Set operator from `Order Matters: Sequence to sequence for sets`_.

    .. _Order Matters: Sequence to sequence for sets:
        https://arxiv.org/pdf/1511.06391.pdf

    Args:
        input_dim (int):                input dimension
        num_step (int, optional):       number of process steps
        num_lstm_layer (int, optional): number of LSTM layers

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self, input_dim, num_step=3, num_lstm_layer=1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = self.input_dim * 2
        self.num_step = num_step
        self.lstm = nn.LSTM(input_dim * 2, input_dim, num_lstm_layer)
        self.softmax = SoftmaxAggregation()

    def construct(self, node2graph, inputs, axis=0):
        """
        Perform Set2Set readout over graph(s).

        Args:
            graph (Graph):      graph(s)
            inputs (Tensor):    node representations

        Returns:
            Tensor: graph representations
        """
        batch_size = node2graph.max() + 1
        hx = ops.zeros((self.lstm.num_layers, batch_size, self.lstm.hidden_size)) * 2
        query_star = ops.zeros((batch_size, self.output_dim))

        for _ in range(self.num_step):
            query, hx = self.lstm(query_star.expand_dims(0), hx)
            query = query.squeeze(0)
            product = ops.einsum("bd, bd -> b", query[node2graph], inputs)
            attention = self.softmax(node2graph, product)
            output = utils.scatter_add(attention.expand_dims(-1) * inputs,
                                       node2graph, axis=axis, n_axis=batch_size)
            query_star = ops.concat([query, output], axis=-1)

        return query_star
