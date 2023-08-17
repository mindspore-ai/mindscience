# Copyright 2023 Huawei Technologies Co., Ltd
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
"""layers"""
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import initializer, XavierUniform, Zero


class LstmLayer(nn.Cell):
    """
        Class for LSTM layer.
    """

    def __init__(self, input_dim):
        super(LstmLayer, self).__init__()
        self.lstm_1 = nn.LSTM(input_size=input_dim, hidden_size=512, num_layers=1, has_bias=True, batch_first=True)
        self.lstm_2 = nn.LSTM(input_size=512, hidden_size=512, num_layers=1, has_bias=True, batch_first=True)
        self.concat = ops.Concat(2)

    def construct(self, x):
        """construct"""
        x_l1, _ = self.lstm_1(x)
        x_l2, _ = self.lstm_2(x_l1)
        x = self.concat((x_l1, x_l2))
        return x


class MultiGraphConv(nn.Cell):
    """
        Class for MultiGraphConv layer.
    """

    def __init__(self, in_features,
                 out_features,
                 bias=False,
                 activation=None,
                 device=None,
                 train=False):
        super(MultiGraphConv, self).__init__()
        self.device = device
        self.weight = ms.Parameter(initializer(XavierUniform(), (3 * in_features, out_features), ms.float32))
        if bias:
            self.bias = ms.Parameter(initializer(Zero(), out_features, ms.float32))
        else:
            self.bias = None

        self.activation = nn.get_activation(activation)
        self.sum = ops.ReduceSum()
        self.bmm = ops.BatchMatMul()
        self.cat = ops.Concat(-1)
        self.shape = ops.Shape()
        self.eye = ops.Eye()
        self.sqrt = ops.Sqrt()
        self.train = train
        self.device = ms.get_context("device_target")

    def normalize(self, adj, eps=1e-6):
        """normalize"""
        n = self.shape(adj)[-1]
        batch = self.shape(adj)[0]
        mat = matrix_diagonal_part_diagonal(adj, batch)
        adj -= mat

        adj_hat = adj + self.eye(n, n, adj.dtype).expand_dims(0)
        deg = adj_hat.sum(2)

        degree_symm = matrix_diagonal((1. / (eps + self.sqrt(deg))), batch)
        degree_asymm = matrix_diagonal((1. / (eps + deg)), batch)

        return [adj, self.bmm(degree_asymm, adj_hat), self.bmm(self.bmm(degree_symm, adj_hat), degree_symm)]

    def construct(self, inputs):
        """construct"""
        output = []
        if self.train and self.device == "Ascend":
            inputs = [input_data.astype(ms.float16) for input_data in inputs]

        for i_adj in self.normalize(inputs[1]):
            x = self.bmm(i_adj, inputs[0])
            output.append(x)

        output = self.cat((output[0], output[1], output[2]))
        weight_tmp = self.cast(self.weight, output.dtype)

        output = ops.matmul(output, weight_tmp)

        if self.train and self.device == "Ascend":
            output = output.astype(ms.float32)
        if self.bias is not None:
            output = output + self.bias

        if self.activation is not None:
            output = self.activation(output)

        return output


class SumPooling(nn.Cell):
    """
        Class for SumPooling layer.
    """

    def __init__(self, axis):
        super(SumPooling, self).__init__()
        self.axis = axis

    def construct(self, x):
        """construct"""
        x_pool = x.sum(self.axis)
        return x_pool


class FuncPredictor(nn.Cell):
    """
        Class for FuncPredictor layer.
    """

    def __init__(self, input_dim, output_dim, train=False):
        super(FuncPredictor, self).__init__()

        self.output_dim = output_dim
        self.output_layer = nn.Dense(input_dim, 2 * output_dim, has_bias=True)
        self.reshape = ops.Reshape()
        self.softmax = nn.Softmax(-1)
        self.train = train

    def construct(self, x):
        """construct"""
        x = self.output_layer(x)
        x = self.reshape(x, (-1, self.output_dim, 2))
        if self.train:
            out = x
        else:
            out = self.softmax(x)
        return out


def matrix_diagonal_part_diagonal(adj_matrix, batch):
    """matrix diagonal part diagonal"""
    mat = None
    for i_batch in range(batch):
        concat = ops.Concat(0)
        if mat is None:
            mat = ms.numpy.diag(ms.numpy.diag(adj_matrix[i_batch])).expand_dims(0)
        else:
            diagonal = ms.numpy.diag(ms.numpy.diag(adj_matrix[i_batch])).expand_dims(0)
            mat = concat((mat, diagonal))
    return mat


def matrix_diagonal(deg, batch):
    """matrix diagonal"""
    mat = None
    for i_batch in range(batch):
        concat = ops.Concat(0)
        if mat is None:
            mat = ms.numpy.diag(deg[i_batch]).expand_dims(0)
        else:
            diagonal = ms.numpy.diag(deg[i_batch]).expand_dims(0)
            mat = concat((mat, diagonal))
    return mat
