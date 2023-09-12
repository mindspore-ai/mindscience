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
"""jacobian weights"""
from types import FunctionType, MethodType

import mindspore as ms
from mindspore import Tensor, nn, ops
from mindspore.ops import composite as C
from mindspore.ops import constexpr
from mindspore.ops import functional as F

cast_grad = C.MultitypeFuncGraph("cast_grad")


class JacobianWeights(nn.Cell):
    """
    Jacobian matrix with respect to weight(s).
    The last tensor in the input Tensor tuple is the weight Parameter, and the remainders are the inputs of network.

    Args:
        model (Cell): Network for jacobian result with respect to weights.
        out_shape (tuple): Output shape of the netword.
        out_type (type): Mindspore data type. Default: ms.float32.

    Inputs:
        - **x** (tuple[Tensor]) - Tensors of the network input and the weight to find jacobian matrix.

    Outputs:
        Tensor, Jacobian matrix with respect to the given weights.

    Examples:
        >>> import mindspore as ms
        >>> from mindspore import nn, ops
        >>> from sciai.operators import JacobianWeights
        >>> class Net1In1OutTensor(nn.Cell):
        >>>     def __init__(self):
        >>>         super().__init__()
        >>>         self.dense1 = nn.Dense(2, 1)
        >>>     def construct(self, x):
        >>>         return self.dense1(x)
        >>> net = Net1In1OutTensor()
        >>> x = ops.ones((100, 2), ms.float32)
        >>> params = net.trainable_params()
        >>> out = net(x)
        >>> jw = JacobianWeights(net, out.shape)
        >>> jacobian_weights = jw(x, params[0])
        >>> print(jacobian_weights.shape)
        (100, 1, 1, 2)
    """

    def __init__(self, model, out_shape, out_type=ms.float32):
        super(JacobianWeights, self).__init__()
        if not isinstance(model, (nn.Cell, FunctionType, MethodType)):
            raise TypeError("The type of model should be a Cell, Function or Method, but got {}".format(type(model)))
        self.model = model
        self.hyper_map = C.HyperMap()
        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
        self.gather = ops.Gather()
        self.cast = ops.Cast()
        self.dtype = ops.DType()
        self.stack_op = ops.Stack(axis=0)
        self.out_shape = out_shape
        self.sens = get_vmap_sens_list(*out_shape, out_type)
        self.sens = self.stack_op(self.sens)
        self.sens = self.cast(self.sens, out_type)

    def construct(self, *x):
        inputs, weight = x[:-1], x[-1]
        gradient_function = self.grad(self.model, weight)
        gradient_function_vmap = F.vmap(gradient_function, in_axes=(None, 0), out_axes=0)
        gradient = gradient_function_vmap(*inputs, self.sens)
        if isinstance(gradient, (tuple, list)):
            gradient = [ops.reshape(g, self.out_shape + w.shape) for w, g in zip(weight, gradient)]
        else:
            gradient = ops.reshape(gradient, self.out_shape + weight.shape)
        return gradient


@constexpr
def _generate_sens(batch_size, out_channel, row_ind, col_ind, dtype):
    r"""
    Generate sens tensors.

    Args:
        batch_size (int): Batch size.
        out_channel (int): Output channel.
        row_ind (int): Row index of retained part.
        col_ind (int): Column index of retained part.
        dtype (type): Mindspore data type.

    Returns:
        Tensor, Sense tensor of shape :math:`(batch\_size, out\_channel)`.
    """
    sens = ops.zeros((batch_size, out_channel), dtype)
    sens[row_ind, col_ind] = 1
    return Tensor(sens)


@constexpr
def get_vmap_sens_list(batch_size, out_channel, dtype):
    r"""
    Generate a list of sens tensors.

    Args:
        batch_size (int): Batch size.
        out_channel (int): Out channel.
        dtype (type): Mindspore data type.

    Returns:
        list, Tensors in shape :math:`(batch\_size, out\_channel)`.
    """
    sens = []
    for row_id in range(batch_size):
        for cow_id in range(out_channel):
            sen = _generate_sens(batch_size, out_channel, row_id, cow_id, dtype)
            sens.append(sen)
    return sens
