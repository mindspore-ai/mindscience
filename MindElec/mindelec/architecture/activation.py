# Copyright 2021 Huawei Technologies Co., Ltd
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
"""get activation function."""
from __future__ import absolute_import
import numpy as np
import mindspore.ops as ops
import mindspore.nn.layer.activation as activation
import mindspore.nn as nn

__all__ = ['get_activation']


class SReLU(nn.Cell):
    """
    Sin rectified Linear Unit activation function.

    Applies the sin rectified linear unit function element-wise.

    Inputs:
        - **input_data** (Tensor) - The input of SReLU.

    Outputs:
        Tensor, with the same type and shape as the `input_data`.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import numpy as np
        >>> from mindelec.architecture.activation import SReLU
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
        self.relu0 = activation.ReLU()
        self.relu1 = activation.ReLU()
        self.sin = ops.Sin()

    def construct(self, x):
        out = 2 * np.pi * x
        out = self.sin(out) * self.relu0(x) * self.relu1(1 - x)
        return out


_activation = {
    'softmax': activation.Softmax,
    'logsoftmax': activation.LogSoftmax,
    'relu': activation.ReLU,
    'relu6': activation.ReLU6,
    'tanh': activation.Tanh,
    'gelu': activation.GELU,
    'fast_gelu': activation.FastGelu,
    'elu': activation.ELU,
    'sigmoid': activation.Sigmoid,
    'prelu': activation.PReLU,
    'leakyrelu': activation.LeakyReLU,
    'hswish': activation.HSwish,
    'hsigmoid': activation.HSigmoid,
    'logsigmoid': activation.LogSigmoid,
    'sin': ops.Sin,
    'srelu': SReLU,
}


def get_activation(name):
    """
    Gets the activation function.

    Args:
        name (Union[str, None]): The name of the activation function.

    Returns:
        Function, the activation function.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import numpy as np
        >>> from mindelec.architecture import get_activation
        >>> from mindspore import Tensor
        >>> input_x = Tensor(np.array([[1.2, 0.1], [0.2, 3.2]], dtype=np.float32))
        >>> sigmoid = get_activation('sigmoid')
        >>> output = sigmoid(input_x)
        >>> print(output)
        [[0.7685248  0.5249792 ]
         [0.54983395 0.96083426]]
    """
    if name is None:
        return None
    if not isinstance(name, str):
        raise TypeError("the type of name should be str, but got {}".format(type(name)))
    name = name.lower()
    if name not in _activation:
        return activation.get_activation(name)
    return _activation[name]()
