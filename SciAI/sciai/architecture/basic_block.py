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
"""basic block"""
from numbers import Number
from types import FunctionType

import mindspore as ms
from mindspore import nn, ops, mutable
from mindspore.common.initializer import Initializer
from mindspore.ops import Primitive

from sciai.architecture.activation import get_activation, AdaptActivation
from sciai.utils.check_utils import _batch_check_type, _check_value_in, _recursive_type_check
from sciai.utils.ms_utils import to_tensor


class MLP(nn.Cell):
    """
    Multi-layer perceptron. The last layer is without activation function.

    The first value in `layers` in Args should be equal to the size of last axis `in_channels` in input Tensor.

    Args:
        layers (Union(tuple[int], list[int])): List of numbers of neurons in each layer, e.g., [2, 10, 10, 1].
        weight_init (Union[str, Initializer]): The `weight_init` parameter for Dense.
            The dtype is the same as `x`. The values of str refer to the function `initializer`.
            Default: 'xavier_trunc_normal'.
        bias_init (Union[str, Initializer]): The `bias_init` parameter for Dense. The
            dtype is same as `x`. The values of str refer to the function `initializer`. Default: 'zeros'.
        activation (Union[str, Cell, Primitive, FunctionType, None]): Activation function applied to the output of each
            fully connected layer excluding the last layer. Both activation name, e.g. 'relu', and mindspore activation
            function, e.g. nn.ReLU(), are supported. Default: 'tanh'.
        last_activation (Union[str, Cell, Primitive, FunctionType, None]): Activation function applied to the output of
            the last dense layer. The type rule is the same as `activation`.

    Inputs:
        - **x** (Tensor) - Tensor of shape (∗, in_channels).

    Outputs:
        Union(Tensor, tuple[Tensor]), Output Tensor of the network.

    Raises:
        TypeError: If `layers` is not one of list, tuple, or elements in `layers` are not ints.
        TypeError: If `activation` is not one of str, Cell, Primitive, None.
        TypeError: If `last_activation` is not one of str, Cell, Primitive, None.
        ValueError: If `weight_init` is not one of str, Initializer.
        ValueError: If `bias_init` is not one of str, Initializer.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> import numpy as np
        >>> from sciai.architecture import MLP
        >>> x = ms.Tensor(np.array([[180, 234, 154], [244, 48, 247]]), ms.float32)
        >>> net = MLP((3, 10, 4))
        >>> output = net(x)
        >>> print(output.shape)
        (2, 4)
    """

    def __init__(self, layers, weight_init='xavier_trunc_normal', bias_init='zeros', activation='tanh',
                 last_activation=None):
        super(MLP, self).__init__()
        _batch_check_type({"layers": (layers, (list, tuple)),
                           "weight_init": (weight_init, (str, Initializer)),
                           "bias_init": (bias_init, (str, Initializer)),
                           "activation": (activation, (str, nn.Cell, Primitive, FunctionType, None)),
                           "last_activation": (last_activation, (str, nn.Cell, Primitive, FunctionType, None))})
        _recursive_type_check(layers, int)
        self.activation = get_activation(activation) if isinstance(activation, str) else activation
        self.last_activation = get_activation(last_activation) if isinstance(last_activation, str) else last_activation
        self.cell_list = nn.SequentialCell()
        for in_channels, out_channels in zip(layers[:-2], layers[1:-1]):
            dense = nn.Dense(in_channels=in_channels, out_channels=out_channels, weight_init=weight_init,
                             bias_init=bias_init, activation=self.activation)
            self.cell_list.append(dense)
        dense = nn.Dense(in_channels=layers[-2], out_channels=layers[-1], weight_init=weight_init, bias_init=bias_init,
                         activation=self.last_activation)
        self.cell_list.append(dense)

    def construct(self, x):
        return self.cell_list(x)

    def weights(self):
        """
        Weight parameter list for all Dense layers.

        Returns:
            list[Parameter], All weight Parameters.
        """
        return [_.weight for _ in self.cell_list.cells()]

    def biases(self):
        """
        Bias parameter list for all Dense layers.

        Returns:
            list[Parameter], All bias Parameters.
        """
        return [_.bias for _ in self.cell_list.cells()]


class MLPAAF(nn.Cell):
    """
    Multi-layer perceptron with adaptive activation function. The last layer is without activation function.

    The first value in `layers` in Args should be equal to the size of last axis `in_channels` in input Tensor.

    More information about the improvement for MLP, please refer to
    `Locally adaptive activation functions with slope recovery for deep and physics-informed neural networks
    <https://royalsocietypublishing.org/doi/10.1098/rspa.2020.0334>`_.

    Args:
        layers (Union(tuple[int], list[int])): List of numbers of neurons in each layer, e.g., [2, 10, 10, 1].
        weight_init (Union[str, Initializer]): The `weight_init` parameter for Dense.
            The dtype is the same as `x`. The values of str refer to the function `initializer`.
            Default: 'xavier_trunc_normal'.
        bias_init (Union[str, Initializer]): The `bias_init` parameter for Dense. The
            dtype is same as `x`. The values of str refer to the function `initializer`. Default: 'zeros'.
        activation (Union[str, Cell, Primitive, FunctionType, None]): Activation function applied to the output of each
            fully connected layer excluding the last layer. Both activation name, e.g. 'relu', and mindspore activation
            function, e.g. nn.ReLU(), are supported. Default: 'tanh'.
        last_activation (Union[str, Cell, Primitive, FunctionType, None]): Activation function applied to the output of
            the last dense layer. The type rule is the same as `activation`.
        a_value (Union[Number, Tensor, Parameter]): Adaptive trainable parameter `a`. Default: 1.0.
        scale (Union[Number, Tensor]): Fixed scale parameter `scale`. Default: 1.0.
        share_type (str): The sharing level of trainable parameter of adaptive function, can be `layer_wise`, `global`.
            default: `layer_wise`.

    Inputs:
        - **x** (Tensor) - Tensor of shape (∗,in_channels).

    Outputs:
        Union(Tensor, tuple[Tensor]), Output of the network.

    Raises:
        TypeError: If `layers` is not one of list, tuple, or elements in `layers` are not ints.
        TypeError: If `activation` is not one of str, Cell, Primitive, None.
        TypeError: If `last_activation` is not one of str, Cell, Primitive, None.
        TypeError: If `weight_init` is not one of str, Initializer.
        TypeError: If `bias_init` is not one of str, Initializer.
        TypeError: If `a_value` is not one of Number, Tensor, Parameter.
        TypeError: If `scale` is not one of Number, Tensor.
        TypeError: If `share_type` is not str.
        ValueError: If `share_type` is not supported.

    Supported Platforms:
        ``GPU`` ``CPU`` ``Ascend``

    Examples:
        >>> import mindspore as ms
        >>> import numpy as np
        >>> from sciai.architecture import MLPAAF
        >>> x = ms.Tensor(np.array([[180, 234, 154], [244, 48, 247]]), ms.float32)
        >>> net = MLPAAF((3, 10, 4))
        >>> output = net(x)
        >>> print(output.shape)
        (2, 4)
    """

    def __init__(self, layers, weight_init='xavier_trunc_normal', bias_init='zeros', activation='tanh',
                 last_activation=None, a_value=1.0, scale=1.0, share_type='layer_wise'):
        super(MLPAAF, self).__init__()
        _batch_check_type({"layers": (layers, (list, tuple)),
                           "weight_init": (weight_init, (str, Initializer)),
                           "bias_init": (bias_init, (str, Initializer)),
                           "activation": (activation, (str, nn.Cell, Primitive, FunctionType, None)),
                           "last_activation": (last_activation, (str, nn.Cell, Primitive, FunctionType, None)),
                           "a_value": (a_value, (Number, ms.Tensor, ms.Parameter)),
                           "scale": (scale, (Number, ms.Tensor)),
                           "share_type": (share_type, str)})
        _recursive_type_check(layers, int)
        self.share_type = share_type.lower()
        _check_value_in(self.share_type, "share_type", ("layer_wise", "global"))
        self.activation = get_activation(activation) if isinstance(activation, str) else activation
        self.last_activation = get_activation(last_activation) if isinstance(last_activation, str) else last_activation
        self.cell_list = nn.SequentialCell()
        if self.share_type == "global":
            self.a_list = [ms.Parameter(a_value)] * (len(layers) - 2)
        elif self.share_type == "layer_wise":
            self.a_list = [ms.Parameter(a_value) for _ in range(len(layers) - 2)]
        for i, (in_channels, out_channels) in enumerate(zip(layers[:-2], layers[1:-1])):
            act_layer = AdaptActivation(activation, self.a_list[i], scale)
            dense = nn.Dense(in_channels=in_channels, out_channels=out_channels, weight_init=weight_init,
                             bias_init=bias_init)
            self.cell_list.append(dense)
            self.cell_list.append(act_layer)
        dense = nn.Dense(in_channels=layers[-2], out_channels=layers[-1], weight_init=weight_init, bias_init=bias_init,
                         activation=self.last_activation)
        self.cell_list.append(dense)

    def construct(self, x):
        return self.cell_list(x)

    def a_value(self):
        """
        Get trainable local adaptive parameter value(s) of the MLP.

        Returns:
            Union(Parameter, tuple[Parameter]), The common trainable Parameter `a` if share_type is "global";
                a list of Parameters `a` in all layers if "layer_wise".
        """
        return self.a_list[0] if self.share_type == "global" else self.a_list


class MLPShortcut(nn.Cell):
    """
    Multi-layer perceptron with shortcuts. The last layer is without activation function.
    For details of this ameliorated MLP architecture, please check:
    `Understanding and mitigating gradient pathologies in physics-informed neural networks
    <https://arxiv.org/abs/2001.04536>`_.

    Args:
        layers (Union(tuple[int], list[int])): List of numbers of neurons in each layer, e.g., [2, 10, 10, 1].
        weight_init (Union(str, Initializer)): The `weight_init` parameter for Dense.
            The dtype is the same as `x`. The values of str refer to the function `initializer`.
            Default: 'xavier_trunc_normal'.
        bias_init (Union(str, Initializer)): The `bias_init` parameter for Dense. The
            dtype is same as `x`. The values of str refer to the function `initializer`. Default: 'zeros'.
        activation (Union(str, Cell, Primitive, FunctionType, None)): Activation function applied to the output of each
            fully connected layer excluding the last layer. Both activation name, e.g. 'relu', and mindspore activation
            function, e.g. nn.ReLU(), are supported. Default: 'tanh'.
        last_activation (Union(str, Cell, Primitive, FunctionType, None)): Activation function applied to the output of
            the last dense layer. The type rule is the same as `activation`.

    Inputs:
        - **x** (Tensor) - Input Tensor of the network.

    Outputs:
        Union(Tensor, tuple[Tensor]), Output Tensor of the network.

    Raises:
        TypeError: If `layers` is not list, tuple or any element is not an int.
        TypeError: If `activation` is not one of str, Cell, Primitive, FunctionType or None.
        TypeError: If `last_activation` is not one of str, Cell, Primitive, FunctionType or None.
        TypeError: If `weight_init` is not str or Initializer.
        TypeError: If `bias_init` is not str or Initializer.

    Supported Platforms:
        ``GPU`` ``CPU`` ``Ascend``

    Examples:
        >>> import mindspore as ms
        >>> import numpy as np
        >>> from sciai.architecture import MLPShortcut
        >>> x = ms.Tensor(np.array([[180, 234, 154], [244, 48, 247]]), ms.float32)
        >>> net = MLPShortcut((3, 10, 4))
        >>> output = net(x)
        >>> print(output.shape)
        (2, 4)
    """

    def __init__(self, layers, weight_init='xavier_trunc_normal', bias_init='zeros', activation='tanh',
                 last_activation=None):
        super(MLPShortcut, self).__init__()
        _batch_check_type({"layers": (layers, (list, tuple)),
                           "weight_init": (weight_init, (str, Initializer)),
                           "bias_init": (bias_init, (str, Initializer)),
                           "activation": (activation, (str, nn.Cell, Primitive, FunctionType, None)),
                           "last_activation": (last_activation, (str, nn.Cell, Primitive, FunctionType, None))})
        _recursive_type_check(layers, int)
        self.activation = get_activation(activation) if isinstance(activation, str) else activation
        self.last_activation = get_activation(last_activation) if isinstance(last_activation, str) else last_activation
        self.dense_u = nn.Dense(in_channels=layers[0], out_channels=layers[1], weight_init=weight_init,
                                bias_init=bias_init, activation=self.activation)
        self.dense_v = nn.Dense(in_channels=layers[0], out_channels=layers[1], weight_init=weight_init,
                                bias_init=bias_init, activation=self.activation)
        self.cell_list = nn.CellList()
        for in_channel, out_channel in zip(layers[:-2], layers[1:-1]):
            self.cell_list.append(nn.Dense(in_channels=in_channel, out_channels=out_channel, weight_init=weight_init,
                                           bias_init=bias_init, activation=self.activation))
        self.dense_final = nn.Dense(in_channels=layers[-2], out_channels=layers[-1], weight_init=weight_init,
                                    bias_init=bias_init, activation=self.last_activation)

    def construct(self, x):
        u = self.dense_u(x)
        v = self.dense_v(x)
        for dense in self.cell_list:
            y = dense(x)
            x = y * u + (1 - y) * v
        y = self.dense_final(x)
        return y

    def main_weights(self):
        res = []
        for cell in self.cell_list:
            res.append(cell.weight)
        res.append(self.dense_final.weight)
        return res


class MSE(nn.Cell):
    """
    Mean square error with 0.

    Inputs:
        - **x** (Tensor) - Input Tensor for MSE calculation.

    Outputs:
        Tensor, Mean square error between `x` and 0.

    Supported Platforms:
        ``GPU`` ``CPU`` ``Ascend``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore import ops
        >>> from sciai.architecture import MSE
        >>> mse = MSE()
        >>> x = ops.ones((2, 3), ms.float32)
        >>> res = mse(x)
        >>> print(res)
        1.0
    """

    def __init__(self):  # pylint: disable=W0235
        super().__init__()

    def construct(self, x):
        return ops.reduce_mean(ops.square(x))


class SSE(nn.Cell):
    """
    Sum square error with 0.

    Inputs:
        - **x** (Tensor) - Input Tensor for SSE calculation.

    Outputs:
        Tensor, Sum square error between `x` and 0.

    Supported Platforms:
        ``GPU`` ``CPU`` ``Ascend``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore import ops
        >>> from sciai.architecture import SSE
        >>> mse = SSE()
        >>> x = ops.ones((2, 3), ms.float32)
        >>> res = mse(x)
        >>> print(res)
        6.0
    """

    def __init__(self):  # pylint: disable=W0235
        super().__init__()

    def construct(self, x):
        return ops.reduce_sum(ops.square(x))


class FirstOutputCell(nn.Cell):
    r"""
    Network that return the first output of given network.

    Args:
        backbone (Callable): Original network.

    Inputs:
        - **\*inputs** (Tensor) - Original network inputs.

    Outputs:
        Union(Tensor, tuple[Tensor]), The first output of the original network.

    Supported Platforms:
        ``GPU`` ``CPU`` ``Ascend``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore import ops, nn
        >>> from sciai.architecture.basic_block import FirstOutputCell
        >>> class Net2In3Out(nn.Cell):
        >>>     def __init__(self):
        >>>         super().__init__()
        >>>     def construct(self, x, y):
        >>>         out1 = x + y
        >>>         out2 = 2 * x + y
        >>>         out3 = x * x + 4 * y * y + 3 * y
        >>>         return out1.sum(), out2.sum(), out3.sum()
        >>> net = Net2In3Out()
        >>> first_output_cell = FirstOutputCell(net)
        >>> x, y = ops.ones((2, 3), ms.float32), ops.ones((2, 3), ms.float32)
        >>> res = first_output_cell(x, y)
        >>> print(res)
        12.0
    """

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def construct(self, *inputs):
        outputs = self.backbone(*inputs)
        return outputs[0] if len(outputs) > 1 else outputs


class NoArgNet(nn.Cell):
    """
    Convert a net with inputs into a net without inputs in construct.

    Args:
        backbone (Cell): Original network.
        *inputs (tuple): Inputs of the original network.

    Outputs:
        Union(Tensor, tuple[Tensor]), The result of original network with the given inputs.

    Supported Platforms:
        ``GPU`` ``CPU`` ``Ascend``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore import ops, nn
        >>> from sciai.architecture.basic_block import NoArgNet
        >>> class Net(nn.Cell):
        >>>     def __init__(self):
        >>>         super().__init__()
        >>>     def construct(self, x, y):
        >>>         out1 = x + y
        >>>         return out1.sum()
        >>> net = Net()
        >>> x, y = ops.ones((2, 3), ms.float32), ops.ones((2, 3), ms.float32)
        >>> no_arg_cell = NoArgNet(net, x, y)
        >>> res = no_arg_cell()
        >>> print(res)
        12.0
    """

    def __init__(self, backbone, *inputs):
        super().__init__()
        self.backbone = backbone
        self.inputs = mutable(inputs)

    def construct(self):
        res = self.backbone(*self.inputs)
        return res


class Normalize(nn.Cell):
    """
    Normalize inputs with given lower bound and upper bound.

    Args:
        lb (Tensor): Lower bound.
        ub (Tensor): Upper bound.

    Inputs:
        - **inputs** (Tensor) - Input tensor to be normalized.

    Outputs:
        Tensor, The normalized tensor projected into [-1, 1].

    Supported Platforms:
        ``GPU`` ``CPU`` ``Ascend``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore import ops, nn
        >>> from sciai.architecture.basic_block import Normalize
        >>> x = ops.ones((3, 2), ms.float32)
        >>> lb, ub = ops.Tensor([0, -0.5], ms.float32), ops.Tensor([2, 3.5], ms.float32)
        >>> normalize = Normalize(lb, ub)
        >>> res = normalize(x)
        >>> print(res)
        [[ 0.   -0.25]
         [ 0.   -0.25]
         [ 0.   -0.25]]
    """

    def __init__(self, lb, ub):
        super().__init__()
        _batch_check_type({'lb': (lb, (Number, ms.Tensor)), 'ub': (ub, (Number, ms.Tensor))})
        lb, ub = to_tensor((lb, ub))
        self.lb = lb
        self.ub = ub

    def construct(self, inputs):
        return 2.0 * (inputs - self.lb) / (self.ub - self.lb) - 1.0
