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
# ==============================================================================
"""models for DeepONet-Grid-UQ"""

from typing import Any

import mindspore as ms
import mindspore.numpy as np
from mindspore import Parameter, mint, nn, ops
from mindspore.common.initializer import XavierNormal, Zero, initializer


class MLP(nn.Cell):
    """MLP module"""

    def __init__(self, layer_size: list, activation: str) -> None:
        super().__init__()
        layers = []
        for k in range(len(layer_size) - 2):
            layers.append(nn.Dense(layer_size[k], layer_size[k + 1], has_bias=True))
            layers.append(get_activation(activation))
        layers.append(nn.Dense(layer_size[-2], layer_size[-1], has_bias=True))
        self.net = nn.SequentialCell(layers)
        self.net.apply(self._init_weights)

    def _init_weights(self, m: Any) -> None:
        if isinstance(m, nn.Dense):
            m.weight.set_data(
                initializer(XavierNormal(), m.weight.shape, m.weight.dtype)
            )
            m.bias.set_data(initializer(Zero(), m.bias.shape, m.bias.dtype))

    def construct(self, x: ms.Tensor):
        return self.net(x)


class ModifiedMLP(nn.Cell):
    """modified MLP module"""

    def __init__(self, layer_size: list, activation: str) -> None:
        super().__init__()
        layers = []
        for k in range(len(layer_size) - 1):
            layers.append(nn.Dense(layer_size[k], layer_size[k + 1], has_bias=True))
        self.net = nn.SequentialCell(layers)

        self.u_net = nn.SequentialCell(
            [nn.Dense(layer_size[0], layer_size[1], has_bias=True)]
        )
        self.v_net = nn.SequentialCell(
            [nn.Dense(layer_size[0], layer_size[1], has_bias=True)]
        )
        self.activation = get_activation(activation)

        self.net.apply(self._init_weights)
        self.u_net.apply(self._init_weights)
        self.v_net.apply(self._init_weights)

        self.len = ms.Tensor(len(layer_size) - 1, ms.int32)
        self.ones = ops.OnesLike()

    def _init_weights(self, m: Any) -> None:
        if isinstance(m, nn.Dense):
            m.weight.set_data(
                initializer(XavierNormal(), m.weight.shape, m.weight.dtype)
            )
            m.bias.set_data(initializer(Zero(), m.bias.shape, m.bias.dtype))

    def construct(self, x: ms.Tensor):
        u = self.activation(self.u_net(x))
        v = self.activation(self.v_net(x))
        for k in range(self.len):
            y = self.net[k](x)
            y = self.activation(y)
            x = y * u + (self.ones(y) - y) * v
        y = self.net[-1](x)
        return y


def get_activation(identifier: str) -> Any:
    """get activation function."""
    return {
        "elu": nn.ELU(),
        "relu": nn.ReLU(),
        "selu": nn.SeLU(),
        "sigmoid": nn.Sigmoid(),
        "leaky": nn.LeakyReLU(),
        "tanh": nn.Tanh(),
        "sin": SinAct(),
        "Rrelu": nn.RReLU(),
        "gelu": nn.GELU(),
        "silu": nn.SiLU(),
        "Mish": nn.Mish(),
    }[identifier]


class SinAct(nn.Cell):
    def construct(self, x: ms.Tensor):
        return mint.sin(x)


class DeepONet(nn.Cell):
    """Base DeepONet class that serves as an interface for different DeepONet implementations."""

    def __init__(self, branch: dict, trunk: dict, use_bias: bool = True) -> None:
        super().__init__()
        if branch["type"] == "MLP":
            self.branch = MLP(branch["layer_size"][:-2], branch["activation"])
        elif branch["type"] == "modified":
            self.branch = ModifiedMLP(branch["layer_size"][:-2], branch["activation"])
        else:
            raise ValueError(
                f"Unsupported branch type: {branch['type']}. Supported: 'MLP', 'modified'."
            )

        if trunk["type"] == "MLP":
            self.trunk = MLP(trunk["layer_size"][:-2], trunk["activation"])
        elif trunk["type"] == "modified":
            self.trunk = ModifiedMLP(trunk["layer_size"][:-2], trunk["activation"])
        else:
            raise ValueError(
                f"Unsupported trunk type: {trunk['type']}. Supported: 'MLP', 'modified'."
            )

        self.use_bias = use_bias

    def _init_weights(self, m):
        """Initialize weights for dense layers."""
        if isinstance(m, nn.Dense):
            m.weight.set_data(
                initializer(XavierNormal(), m.weight.shape, m.weight.dtype)
            )
            m.bias.set_data(initializer(Zero(), m.bias.shape, m.bias.dtype))

    def construct(self, xu, xy):
        """Forward pass interface. To be implemented by subclasses."""
        raise NotImplementedError("construct method must be implemented by subclasses")


class ProbDeepONet(DeepONet):
    """ProbDeepONet inherit from DeepONet"""

    def __init__(self, branch: dict, trunk: dict, use_bias: bool = True) -> None:
        super().__init__(branch, trunk, use_bias)

        # Add probabilistic components
        if use_bias:
            self.bias_mu = Parameter(
                ms.Tensor(np.randn(1), ms.float32), requires_grad=True
            )
            self.bias_std = Parameter(
                ms.Tensor(np.randn(1), ms.float32), requires_grad=True
            )

        self.branch_mu = nn.SequentialCell(
            [
                get_activation(branch["activation"]),
                nn.Dense(
                    branch["layer_size"][-3], branch["layer_size"][-2], has_bias=True
                ),
                get_activation(branch["activation"]),
                nn.Dense(
                    branch["layer_size"][-2], branch["layer_size"][-1], has_bias=True
                ),
            ]
        )

        self.branch_std = nn.SequentialCell(
            [
                get_activation(branch["activation"]),
                nn.Dense(
                    branch["layer_size"][-3], branch["layer_size"][-2], has_bias=True
                ),
                get_activation(branch["activation"]),
                nn.Dense(
                    branch["layer_size"][-2], branch["layer_size"][-1], has_bias=True
                ),
            ]
        )

        self.trunk_mu = nn.SequentialCell(
            [
                get_activation(trunk["activation"]),
                nn.Dense(
                    trunk["layer_size"][-3], trunk["layer_size"][-2], has_bias=True
                ),
                get_activation(trunk["activation"]),
                nn.Dense(
                    trunk["layer_size"][-2], trunk["layer_size"][-1], has_bias=True
                ),
            ]
        )

        self.trunk_std = nn.SequentialCell(
            [
                get_activation(trunk["activation"]),
                nn.Dense(
                    trunk["layer_size"][-3], trunk["layer_size"][-2], has_bias=True
                ),
                get_activation(trunk["activation"]),
                nn.Dense(
                    trunk["layer_size"][-2], trunk["layer_size"][-1], has_bias=True
                ),
            ]
        )

        self.branch_mu.apply(self._init_weights)
        self.branch_std.apply(self._init_weights)
        self.trunk_mu.apply(self._init_weights)
        self.trunk_std.apply(self._init_weights)
        self.reduce_sum = ops.ReduceSum(keep_dims=True)

    def _init_weights(self, m):
        if isinstance(m, nn.Dense):
            m.weight.set_data(
                initializer(XavierNormal(), m.weight.shape, m.weight.dtype)
            )
            m.bias.set_data(initializer(Zero(), m.bias.shape, m.bias.dtype))

    def construct(self, xu, xy):
        u, y = xu, xy
        b = self.branch(u)
        t = self.trunk(y)
        # branch prediction and UQ
        b_mu = self.branch_mu(b)
        b_std = self.branch_std(b)
        # trunk prediction and UQ
        t_mu = self.trunk_mu(t)
        t_std = self.trunk_std(t)

        # dot product using element-wise multiplication and sum
        # Use ReduceSum operation for proper reduction
        mu = self.reduce_sum(b_mu * t_mu, 1)  # Reduce along feature dimension
        # Reduce along feature dimension
        log_std = self.reduce_sum(b_std * t_std, 1)
        if self.use_bias:
            mu += self.bias_mu
            log_std += self.bias_std
        return (mu, log_std)
