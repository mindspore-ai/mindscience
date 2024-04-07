# Copyright 2022 Huawei Technologies Co., Ltd
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
r"""Some basic network blocks."""

import math

from mindspore import dtype as mstype
from mindspore import nn, ops, Tensor
from mindspore.common.initializer import initializer, Uniform


class UniformInitDense(nn.Dense):
    r"""Linear layer (nn.Dense) with Uniform initialization.

    Args:
        dim_in (int): Dimension of the input features.
        dim_out (int): Dimension of the output features.
        has_bias (bool): Whether bias is involved. Default: ``True``.
        scale (float): Scale of initialized weights and biases.
            Default: ``None``, use Kaiming uniform initialization.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(*, dim\_in)`.


    Outputs:
        Tensor of shape :math:`(*, dim\_out)`.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from src.cell.basic_block import UniformInitDense
        >>> dense = UniformInitDense(10, 5, has_bias=True, scale=0.1)
        >>> x = Tensor(np.random.rand(16, 10), mstype.float32)
        >>> y = dense(x)
        >>> print(y.shape)
        (16, 5)
    """

    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 has_bias: bool = True,
                 scale: float = None) -> None:
        super().__init__(dim_in, dim_out, has_bias=has_bias)

        # initialize parameters
        if scale is None:
            if dim_in <= 0:
                raise ValueError(f"'dim_in' should be greater than 0, but got {dim_in}.")
            scale = math.sqrt(1 / dim_in)  # Kaiming uniform initialization
        self.weight.set_data(initializer(
            Uniform(scale), self.weight.shape, self.weight.dtype))
        if has_bias:
            self.bias.set_data(initializer(
                Uniform(scale), self.bias.shape, self.bias.dtype))


class MLP(nn.Cell):
    r"""Multi-layer perceptron (MLP).

    Args:
        dim_in (int): Dimension of the input features.
        dim_out (int): Dimension of the output features.
        dim_hidden (int): Dimension of hidden layer features.
        num_layers (int): Number of Layers. Default: ``3``.
        compute_dtype (mstype.dtype): The computation type of the layer. Default: ``mstype.float16``.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(*, dim\_in)`.

    Outputs:
        Tensor of shape :math:`(*, dim\_out)`.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from mindspore import dtype as mstype
        >>> from src.cell.basic_block import MLP
        >>> mlp = MLP(dim_in=10, dim_out=5, dim_hidden=128, num_layers=3)
        >>> x = Tensor(np.random.rand(16, 10), mstype.float32)
        >>> y = mlp(x)
        >>> print(y.shape)
        (16, 5)
    """

    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 dim_hidden: int,
                 num_layers: int = 3,
                 compute_dtype=mstype.float16) -> None:
        super().__init__()

        if num_layers > 1:
            layers = []
            layers.append(UniformInitDense(dim_in, dim_hidden, has_bias=True).to_float(compute_dtype))
            layers.append(nn.ReLU())
            for _ in range(num_layers - 2):
                layers.append(UniformInitDense(dim_hidden, dim_hidden, has_bias=True).to_float(compute_dtype))
                layers.append(nn.ReLU())
            layers.append(UniformInitDense(dim_hidden, dim_out, has_bias=True).to_float(compute_dtype))
            self.net = nn.SequentialCell(layers)
        elif num_layers == 1:
            self.net = UniformInitDense(dim_in, dim_out, has_bias=True).to_float(compute_dtype)
        else:
            raise ValueError(f"'num_layers' should be greater than 0, but got {num_layers}.")

    def construct(self, x: Tensor) -> Tensor:
        r"""construct"""
        return self.net(x)


class CoordPositionalEncoding(nn.Cell):
    r"""Coordinate positional encoding used in implicit neural representations
    (INRs): x -> [x, sin(x), cos(x), .., sin(2**k * x), cos(2**k * x)] for example.

    Args:
        num_pos_enc (int): Number of frequencies involved in the position
                        encoding. Default: ``0``, no position encoding.
        period (float): Period of the fourier features in the positional
                        encoding. Default: ``2``, for coordinates normalized
                        to [-1, 1].

    Input:
        - **x** (Tensor) - Tensor of shape :math:`(*, dim\_in)`.

    Output:
        Output features of shape (.., dim_out), where
        dim_out = (1 + 2 * num_pos_enc) * dim_in

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from src.cell.basic_block import CoordPositionalEncoding
        >>> pos_enc = CoordPositionalEncoding(num_pos_enc=2, period=2.0)
        >>> x = Tensor(np.random.rand(16, 10), mstype.float32)
        >>> y = pos_enc(x)
        >>> print(y.shape)
        (16, 50)
    """

    def __init__(self, num_pos_enc: int = 0, period: float = 2.0) -> None:
        super().__init__()
        if period == 0:
            raise ValueError("'period' should be non-zero.")
        omega_0 = 2 * math.pi / period
        self.omegas = [2**k * omega_0 for k in range(num_pos_enc)]

    def construct(self, x: Tensor) -> Tensor:
        r"""construct"""
        pos_enc_list = [x]
        for omega in self.omegas:
            pos_enc_list.extend([ops.sin(omega * x), ops.cos(omega * x)])
        x = ops.concat(pos_enc_list, axis=-1)
        return x


class Sine(nn.Cell):
    r"""Sine activation with scaling factor.

    Args:
        w0 (float): scaling factor. Default: ``1.0``.

    Input:
        - **x** (Tensor) - Tensor of shape :math:`(*, dim\_in)`.

    Output:
        Output features of shape :math:`(*, dim\_in)`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from src.cell.basic_block import Sine
        >>> sine = Sine(w0=1.0)
        >>> x = Tensor(np.random.rand(16, 10), mstype.float32)
        >>> y = sine(x)
        >>> print(y.shape)
        (16, 10)
    """

    def __init__(self, w0: float = 1.0) -> None:
        super().__init__()
        self.omega_0 = w0

    def construct(self, x: Tensor) -> Tensor:
        r"""construct"""
        return ops.sin(self.omega_0 * x)
