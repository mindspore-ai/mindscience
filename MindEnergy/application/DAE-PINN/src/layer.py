# Copyright 2025 Huawei Technologies Co., Ltd
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
"""layer modules for DAE-PINN network"""

import numpy as np
import mindspore as ms
from mindspore import ops, nn, Tensor, Parameter
from mindspore.common.initializer import initializer


class Fnn(nn.Cell):
    """
    feed-forward neural network.
    """

    def __init__(self,
                 layer_size,
                 activation,
                 kernel_initializer,
                 dropout_rate=0.0,
                 batch_normalization=None,
                 layer_normalization=None,
                 input_transform=None,
                 output_transform=None,
                 use_bias=True):
        super().__init__()
        self.layer_size = layer_size
        self.activation = activation
        self.initializer = kernel_initializer
        self.dropout_rate = dropout_rate
        self.batch_normalization = batch_normalization
        self.layer_normalization = layer_normalization
        self.input_transform = input_transform
        self.output_transform = output_transform
        self.use_bias = use_bias
        # build neural network
        if self.batch_normalization and self.layer_normalization:
            raise ValueError(
                "cannot apply batch normalization and layer normalization at the same time")
        self.net = nn.CellList()
        if (self.batch_normalization is None) and (self.layer_normalization is None):
            self.build_standard()
        elif (self.batch_normalization == "before") or (self.layer_normalization == "before"):
            self.build_before()
        elif (self.batch_normalization == "after") or (self.layer_normalization == "after"):
            self.build_after()
        else:
            raise ValueError("Neural net was not built")
        # initialize parameters
        self.net.apply(self._init_weights)
        print("NN built...\n")
        print(self.net)

    def construct(self, y):
        r"""
        FNN forward pass.
        Args:
            :input (Tensor): \in [B, d_in]
        Returns:
            :y (Tensor): \in [B, d_out]
        """
        if self.input_transform is not None:
            y = self.input_transform(y)
        for i in range(len(self.net)):
            y = self.net[i](y)
        if self.output_transform is not None:
            y = self.output_transform(y)
        return y

    def _init_weights(self, m):
        """
        initializes layer parameters.
        """
        if isinstance(m, nn.Dense):
            if self.initializer == "Glorot normal":
                m.weight.set_data(
                    initializer('xavier_normal', m.weight.shape).init_data())
            elif self.initializer == "Glorot uniform":
                m.weight.set_data(
                    initializer('xavier_uniform', m.weight.shape).init_data())
            else:
                raise ValueError(
                    "initializer {} not implemented".format(self.initializer))
            m.bias.data.fill(0.0)
        elif isinstance(m, nn.LayerNorm):
            m.beta.data.fill(0.0)
            m.gamma.data.fill(1.0)

    def build_standard(self):
        """build standard module"""
        # FC - activation
        # input layer
        self.net.append(
            nn.Dense(self.layer_size[0], self.layer_size[1], has_bias=self.use_bias))
        for i in range(1, len(self.layer_size)-1):
            self.net.append(nn.Dense(self.layer_size[i], self.layer_size[i+1],
                                     has_bias=self.use_bias, activation=ops.Sin()))
            if self.dropout_rate > 0.0:
                self.net.append(nn.Dropout(keep_prob=1 - self.dropout_rate))

    def build_before(self):
        """build before module"""
        # FC - BN or LN - activation
        self.net.append(
            nn.Dense(self.layer_size[0], self.layer_size[1], has_bias=self.use_bias))
        for i in range(1, len(self.layer_size)-1):
            if self.batch_normalization is not None:
                self.net.append(nn.BatchNorm1d(self.layer_size[i]))
            elif self.layer_normalization is not None:
                self.net.append(nn.LayerNorm((self.layer_size[i],)))
            self.net.append(nn.Dense(self.layer_size[i], self.layer_size[i+1],
                                     has_bias=self.use_bias, activation=ops.Sin()))
            if self.dropout_rate > 0.0:
                self.net.append(nn.Dropout(keep_prob=1 - self.dropout_rate))

    def build_after(self):
        """build after module"""
        # FC - activation - BN or LN
        self.net.append(nn.Dense(self.layer_size[0], self.layer_size[1],
                                 has_bias=self.use_bias, activation=ops.Sin()))
        for i in range(1, len(self.layer_size)-1):
            if self.batch_normalization is not None:
                self.net.append(nn.BatchNorm1d(self.layer_size[i]))
            elif self.layer_normalization is not None:
                self.net.append(nn.LayerNorm((self.layer_size[i],)))
            if self.dropout_rate > 0.0:
                self.net.append(nn.Dropout(keep_prob=1 - self.dropout_rate))
            self.net.append(
                nn.Dense(self.layer_size[i], self.layer_size[i+1], has_bias=self.use_bias))


class Sin(nn.Cell):
    def construct(self, x):
        return ops.sin(x)


class Conv1D(nn.Cell):
    """
    feed-forward neural networks with Conv1D dense layers.
    """

    def __init__(self,
                 layer_size,
                 activation,
                 dropout_rate=0.0,
                 batch_normalization=None,
                 layer_normalization=None,
                 input_transform=None,
                 output_transform=None):
        super().__init__()
        self.layer_size = layer_size
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.batch_normalization = batch_normalization
        self.layer_normalization = layer_normalization
        self.input_transform = input_transform
        self.output_transform = output_transform
        if self.batch_normalization and self.layer_normalization:
            raise ValueError(
                "Can not apply batch_normalization and layer_normalization at the same time.")
        self.net = nn.CellList()
        if (self.batch_normalization is None) and (self.layer_normalization is None):
            self.build_standard()
        elif (self.batch_normalization == "before") or (self.layer_normalization == "before"):
            self.build_before()
        elif (self.batch_normalization == "after") or (self.layer_normalization == "after"):
            self.build_after()
        else:
            raise ValueError("Neural net was not built")
        print("NN built...\n")
        print(self.net)

    def construct(self, y):
        r"""
        FNN forward pass
        Args:
            :y (Tensor): \in [B, d_in]
        Returns:
            :y (Tensor): \in [B, d_out]
        """
        if self.input_transform is not None:
            y = self.input_transform(y)
        for i in range(len(self.net)):
            y = self.net[i](y)
        if self.output_transform is not None:
            y = self.output_transform(y)
        return y

    def build_standard(self):
        """build standard module"""
        # FC - activation
        # input layer
        self.net.append(
            DenseConv1D(self.layer_size[0], self.layer_size[1], activation=ops.Sin()))
        if self.dropout_rate > 0.0:
            self.net.append(nn.Dropout(keep_prob=1 - self.dropout_rate))
        for i in range(1, len(self.layer_size)-2):
            self.net.append(
                DenseConv1D(self.layer_size[i], self.layer_size[i+1], activation=ops.Sin()))
            if self.dropout_rate > 0.0:
                self.net.append(nn.Dropout(keep_prob=1 - self.dropout_rate))
        self.net.append(DenseConv1D(self.layer_size[-2], self.layer_size[-1]))

    def build_before(self):
        """build before module"""
        # FC - BN or LN - activation
        self.net.append(DenseConv1D(self.layer_size[0], self.layer_size[1]))
        for i in range(1, len(self.layer_size)-1):
            if self.batch_normalization is not None:
                self.net.append(nn.BatchNorm1d(self.layer_size[i]))
            elif self.layer_normalization is not None:
                self.net.append(nn.LayerNorm((self.layer_size[i],)))
            self.net.append(
                DenseConv1D(self.layer_size[i], self.layer_size[i+1], activation=ops.Sin()))
            if self.dropout_rate > 0.0:
                self.net.append(nn.Dropout(keep_prob=1 - self.dropout_rate))

    def build_after(self):
        """build after module"""
        # FC - activation - BN or LN
        self.net.append(
            DenseConv1D(self.layer_size[0], self.layer_size[1], activation=ops.Sin()))
        if self.batch_normalization is not None:
            self.net.append(nn.BatchNorm1d(self.layer_size[1]))
        elif self.layer_normalization is not None:
            self.net.append(nn.LayerNorm((self.layer_size[1],)))
        if self.dropout_rate > 0.0:
            self.net.append(nn.Dropout(keep_prob=1 - self.dropout_rate))
        for i in range(1, len(self.layer_size)-2):
            self.net.append(
                DenseConv1D(self.layer_size[i], self.layer_size[i+1], activation=ops.Sin()))
            if self.batch_normalization is not None:
                self.net.append(nn.BatchNorm1d(self.layer_size[i+1]))
            elif self.layer_normalization is not None:
                self.net.append(nn.LayerNorm((self.layer_size[i+1],)))
            if self.dropout_rate > 0.0:
                self.net.append(nn.Dropout(keep_prob=1 - self.dropout_rate))
        self.net.append(DenseConv1D(self.layer_size[-2], self.layer_size[-1]))


class Attention(nn.Cell):
    """
    feed-forward neural network with attention-like architecture.
    """

    def __init__(self,
                 layer_size,
                 activation,
                 kernel_initializer,
                 dropout_rate=0.0,
                 batch_normalization=None,
                 layer_normalization=None,
                 input_transform=None,
                 output_transform=None,
                 use_bias=True):
        super().__init__()
        self.layer_size = layer_size
        self.activation = activation
        self.initializer = kernel_initializer
        self.dropout_rate = dropout_rate
        self.batch_normalization = batch_normalization
        self.layer_normalization = layer_normalization
        self.input_transform = input_transform
        self.output_transform = output_transform
        self.use_bias = use_bias
        # build neural networks
        if self.batch_normalization and self.layer_normalization:
            raise ValueError(
                "cannot apply batch normalization and layer normalization at the same time")
        self.net = nn.CellList()
        if (self.batch_normalization is None) and (self.layer_normalization is None):
            self.build_standard()
        elif self.batch_normalization == "before":
            self.build_before_bn()
        elif self.layer_normalization == "before":
            self.build_before_ln()
        elif self.batch_normalization == "after":
            self.build_after_bn()
        elif self.layer_normalization == "after":
            self.build_after_ln()
        else:
            raise ValueError("Neural net was not built")
        # initialize parameters
        self.net.apply(self._init_weights)
        self.u_net.apply(self._init_weights)
        self.v_net.apply(self._init_weights)

    def _init_weights(self, m):
        """
        initializes layer parameters.
        """
        if isinstance(m, nn.Dense):
            if self.initializer == "Glorot normal":
                m.weight.set_data(
                    initializer('xavier_normal', m.weight.shape).init_data())
            elif self.initializer == "Glorot uniform":
                m.weight.set_data(
                    initializer('xavier_uniform', m.weight.shape).init_data())
            else:
                raise ValueError(
                    "initializer {} not implemented".format(self.initializer))
            m.bias.data.fill(0.0)
        elif isinstance(m, nn.LayerNorm):
            m.beta.data.fill(0.0)
            m.gamma.data.fill(1.0)

    def construct(self, y):
        r"""
        FNN forward pass
        Args:
            :y (Tensor): \in [B, d_in]
        Returns:
            :y (Tensor): \in [B, d_out]
        """
        if self.input_transform is not None:
            y = self.input_transform(y)
        u = self.u_net(y)
        v = self.v_net(y)

        for i in range(len(self.net)-1):
            y = self.net[i](y)
            y = (1 - y) * u + y * v
        y = self.net[-1](y)
        if self.output_transform is not None:
            y = self.output_transform(y)
        return y

    def build_standard(self):
        """build standard U and V nets"""
        self.u_net = nn.SequentialCell([
            nn.Dense(self.layer_size[0],
                     self.layer_size[1], has_bias=self.use_bias),
            Sin(),
            nn.Dropout(
                keep_prob=1 - self.dropout_rate) if self.dropout_rate > 0 else nn.Identity()
        ])
        self.v_net = nn.SequentialCell([
            nn.Dense(self.layer_size[0],
                     self.layer_size[1], has_bias=self.use_bias),
            Sin(),
            nn.Dropout(
                keep_prob=1 - self.dropout_rate) if self.dropout_rate > 0 else nn.Identity()
        ])
        self.net = nn.CellList()
        for k in range(len(self.layer_size)-2):
            self.net.append(nn.Dense(self.layer_size[k], self.layer_size[k+1],
                                     has_bias=self.use_bias, activation=ops.Sin()))
            if self.dropout_rate > 0.0:
                self.net.append(nn.Dropout(keep_prob=1 - self.dropout_rate))
        # output layer
        self.net.append(
            nn.Dense(self.layer_size[-2], self.layer_size[-1], has_bias=self.use_bias))

    def build_before_bn(self):
        """build U and V nets with before batch norm"""
        # FC -BN - activation
        self.u_net = nn.SequentialCell([
            nn.Dense(self.layer_size[0],
                     self.layer_size[1], has_bias=self.use_bias),
            nn.BatchNorm1d(self.layer_size[1]),
            Sin(),
            nn.Dropout(
                keep_prob=1 - self.dropout_rate) if self.dropout_rate > 0 else nn.Identity()
        ])
        self.v_net = nn.SequentialCell([
            nn.Dense(self.layer_size[0],
                     self.layer_size[1], has_bias=self.use_bias),
            nn.BatchNorm1d(self.layer_size[1]),
            Sin(),
            nn.Dropout(
                keep_prob=1 - self.dropout_rate) if self.dropout_rate > 0 else nn.Identity()
        ])
        self.net = nn.CellList()
        for k in range(len(self.layer_size)-2):
            self.net.append(
                nn.Dense(self.layer_size[k], self.layer_size[k+1], has_bias=self.use_bias))
            self.net.append(nn.BatchNorm1d(self.layer_size[k+1]))
            self.net.append(Sin())
            if self.dropout_rate > 0.0:
                self.net.append(nn.Dropout(keep_prob=1 - self.dropout_rate))
        # output layer
        self.net.append(
            nn.Dense(self.layer_size[-2], self.layer_size[-1], has_bias=self.use_bias))

    def build_after_bn(self):
        """build U and V nets with after batch norm"""
        # FC - activation -BN
        self.u_net = nn.SequentialCell([
            nn.Dense(self.layer_size[0],
                     self.layer_size[1], has_bias=self.use_bias),
            Sin(),
            nn.BatchNorm1d(self.layer_size[1]),
            nn.Dropout(
                keep_prob=1 - self.dropout_rate) if self.dropout_rate > 0 else nn.Identity()
        ])
        self.v_net = nn.SequentialCell([
            nn.Dense(self.layer_size[0],
                     self.layer_size[1], has_bias=self.use_bias),
            Sin(),
            nn.BatchNorm1d(self.layer_size[1]),
            nn.Dropout(
                keep_prob=1 - self.dropout_rate) if self.dropout_rate > 0 else nn.Identity()
        ])
        self.net = nn.CellList()
        for k in range(len(self.layer_size)-2):
            self.net.append(nn.Dense(self.layer_size[k], self.layer_size[k+1],
                                     has_bias=self.use_bias, activation=ops.Sin()))
            self.net.append(nn.BatchNorm1d(self.layer_size[k+1]))
            if self.dropout_rate > 0.0:
                self.net.append(nn.Dropout(keep_prob=1 - self.dropout_rate))
        # output layer
        self.net.append(
            nn.Dense(self.layer_size[-2], self.layer_size[-1], has_bias=self.use_bias))

    def build_before_ln(self):
        """build U and V nets with before layer norm"""
        # FC - LN - activation
        self.u_net = nn.SequentialCell([
            nn.Dense(self.layer_size[0],
                     self.layer_size[1], has_bias=self.use_bias),
            nn.LayerNorm((self.layer_size[1],)),
            Sin(),
            nn.Dropout(
                keep_prob=1 - self.dropout_rate) if self.dropout_rate > 0 else nn.Identity()
        ])
        self.v_net = nn.SequentialCell([
            nn.Dense(self.layer_size[0],
                     self.layer_size[1], has_bias=self.use_bias),
            nn.LayerNorm((self.layer_size[1],)),
            Sin(),
            nn.Dropout(
                keep_prob=1 - self.dropout_rate) if self.dropout_rate > 0 else nn.Identity()
        ])
        self.net = nn.CellList()
        for k in range(len(self.layer_size)-2):
            self.net.append(
                nn.Dense(self.layer_size[k], self.layer_size[k+1], has_bias=self.use_bias))
            self.net.append(nn.LayerNorm((self.layer_size[k+1],)))
            self.net.append(Sin())
            if self.dropout_rate > 0.0:
                self.net.append(nn.Dropout(keep_prob=1 - self.dropout_rate))
        # output layer
        self.net.append(
            nn.Dense(self.layer_size[-2], self.layer_size[-1], has_bias=self.use_bias))

    def build_after_ln(self):
        """build U and V nets with after layer norm"""
        # FC - activation - LN
        self.u_net = nn.SequentialCell([
            nn.Dense(self.layer_size[0],
                     self.layer_size[1], has_bias=self.use_bias),
            Sin(),
            nn.LayerNorm((self.layer_size[1],)),
            nn.Dropout(
                keep_prob=1 - self.dropout_rate) if self.dropout_rate > 0 else nn.Identity()
        ])
        self.v_net = nn.SequentialCell([
            nn.Dense(self.layer_size[0],
                     self.layer_size[1], has_bias=self.use_bias),
            Sin(),
            nn.LayerNorm((self.layer_size[1],)),
            nn.Dropout(
                keep_prob=1 - self.dropout_rate) if self.dropout_rate > 0 else nn.Identity()
        ])
        self.net = nn.CellList()
        for k in range(len(self.layer_size)-2):
            self.net.append(nn.Dense(self.layer_size[k], self.layer_size[k+1],
                                     has_bias=self.use_bias, activation=ops.Sin()))
            self.net.append(nn.LayerNorm((self.layer_size[k+1],)))
            if self.dropout_rate > 0.0:
                self.net.append(nn.Dropout(keep_prob=1 - self.dropout_rate))
        # output layer
        self.net.append(
            nn.Dense(self.layer_size[-2], self.layer_size[-1], has_bias=self.use_bias))


class DenseConv1D(nn.Cell):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).
    Basically works like a linear layer but the weights are transposed.
    Args:
        :inputs (int): number of input features.
        :outputs (out): number of output features.
    code adopted from: https://github.com/huggingface/transformers/blob/master/src/transformers/modeling_utils.py
    """

    def __init__(self, inputs, outputs, activation=None):
        super().__init__()
        self.n_out = outputs
        self.weight = Parameter(Tensor(np.random.normal(
            0, 0.02, (inputs, outputs)), dtype=ms.float32))
        self.bias = Parameter(Tensor(np.zeros(outputs), dtype=ms.float32))
        self.activation = activation

    def construct(self, x):
        """construct"""
        size_out = x.shape[:-1] + (self.n_out,)
        y = ops.add(ops.matmul(
            x.view(-1, x.shape[-1]), self.weight), self.bias)
        y = y.view(*size_out)
        if self.activation is not None:
            y = self.activation(y)
        return y
