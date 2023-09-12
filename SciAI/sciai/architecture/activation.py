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
# ==============================================================================
"""activation"""
import copy
from numbers import Number
from types import FunctionType

import mindspore as ms
from mindspore import nn, ops
from mindspore.nn.layer.activation import _activation
import numpy as np

from sciai.utils.check_utils import _batch_check_type


class Swish(nn.Cell):
    """
    Swish(Silu) activation function with backward propagation.

    Inputs:
        - **x** (Tensor) - The input of Swish(Silu).

    Outputs:
        Tensor, activated output with the same type and shape as `x`.

    Supported Platforms:
        ``GPU`` ``CPU`` ``Ascend``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore import ops
        >>> from sciai.architecture import Swish
        >>> swish = Swish()
        >>> x = ops.ones((2, 3), ms.float32)
        >>> y = swish(x)
        >>> print(y)
        [[0.73105854 0.73105854 0.73105854]
         [0.73105854 0.73105854 0.73105854]]
    """

    def __init__(self):  # pylint: disable=W0235
        super().__init__()

    def construct(self, x):
        return x * ops.sigmoid(x)


class SReLU(nn.Cell):
    """
    Sin rectified Linear Unit activation function.
    Applies the sin rectified linear unit function element-wise.

    Inputs:
        - **x** (Tensor) - The input of SReLU.

    Outputs:
        Tensor, activated output with the same type and shape as `x`.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import numpy as np
        >>> from sciai.architecture.activation import SReLU
        >>> from mindspore import Tensor
        >>> input_x = Tensor(np.array([[1.2, 0.1], [0.2, 3.2]], dtype=np.float32))
        >>> srelu = SReLU()
        >>> output = srelu(input_x)
        >>> print(output)
        [[0.         0.05290067]
         [0.15216905 0.        ]]
    """

    def __init__(self):
        super(SReLU, self).__init__()
        self.relu0 = nn.ReLU()
        self.relu1 = nn.ReLU()
        self.sin = ops.Sin()

    def construct(self, x):
        out = 2 * np.pi * x
        out = self.sin(out) * self.relu0(x) * self.relu1(1 - x)
        return out


_all_activation = copy.copy(_activation)
_all_activation.update({
    'swish': Swish,
    'silu': Swish,
    'sin': ops.Sin,
    'srelu': SReLU
})


def get_activation(activation):
    """
    Get the activation function according to its name.

    Args:
        activation (str): The name of the activation function.

    Returns:
        Callable, The corresponding activation function.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from sciai.architecture.activation import get_activation
        >>> sigmoid = get_activation('sigmoid')
        >>> print(sigmoid)
        Sigmoid<>
    """
    if activation is None:
        return None

    if activation not in _all_activation:
        raise KeyError(f"'activation' must be in {list(_activation.keys())}, but got {activation}.")
    return _all_activation[activation]()


class AdaptActivation(nn.Cell):
    """
    Adaptive activation function with trainable Parameter and fixed scale.

    For details of adaptive activation function, please check
    `Adaptive activation functions accelerate convergence in deep and physics-informed neural networks
    <https://www.sciencedirect.com/science/article/pii/S0021999119308411>`_ and
    `Locally adaptive activationfunctions with slope recoveryfor deep and physics-informedneural network
    <https://royalsocietypublishing.org/doi/epdf/10.1098/rspa.2020.0334>`_.

    Args:
         activation (Union[str, Cell, Primitive, function]): Activation function.
         a (Union[Number, Tensor, Parameter]): Trainable parameter `a`.
         scale (Union[Number, Tensor]): Fixed scale parameter.

    Inputs:
        - **x** (Tensor) - The input of AdaptActivation.

    Outputs:
        Tensor, activated output with the same type and shape as `x`.

    Raises:
         TypeError: If types are not correct.

    Supported Platforms:
        ``GPU`` ``CPU`` ``Ascend``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore import ops, nn
        >>> from sciai.architecture import AdaptActivation
        >>> a = ms.Tensor(0.1, ms.float32)
        >>> net = AdaptActivation(nn.Tanh(), a=a, scale=10)
        >>> x = ops.ones((2, 3), ms.float32)
        >>> y = net(x)
        >>> print(y)
        [[0.7615942 0.7615942 0.7615942]
        [0.7615942 0.7615942 0.7615942]]
    """

    def __init__(self, activation, a, scale):
        super().__init__()
        _batch_check_type({"activation": (activation, (str, nn.Cell, ops.Primitive, FunctionType)),
                           "a": (a, (Number, ms.Tensor, ms.Parameter)), "scale": (scale, (Number, ms.Tensor))})
        self.activation = get_activation(activation) if isinstance(activation, str) else activation
        self.a = a if isinstance(a, ms.Parameter) else ms.Parameter(a)
        self.scale = scale

    def construct(self, x):
        return self.activation(self.a * self.scale * x)
