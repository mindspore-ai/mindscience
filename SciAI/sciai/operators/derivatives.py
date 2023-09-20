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
# -*- coding: utf-8 -*-
"""derivatives"""
from mindspore import nn
from mindspore import ops

from sciai.common.train_cell import to_tuple
from sciai.utils.check_utils import _recursive_type_check


class _Grad(nn.Cell):
    r"""
    The derivative net of given net according to given output index and input index(es). All output indices will be
        used for differentiation and summed, and all input index(es) will be differentiated separately.

    Args:
        net (Cell): Net to be auto-differentiated.
        output_index (int): Output index starting from 0. Default: 0.
        input_index (Union[int, tuple[int]]): Input index(es) starting from 0, and only forward indexes are allowed.
            If -1, all specified inputs would be differentiated respectively. Default: -1.

    Inputs:
        - **\*inputs** (tuple[Tensor]) - The inputs of the original network.

    Returns:
        Union(Tensor, tuple[Tensor]), The outputs of the fist order derivative net.

    Raises:
        TypeError: If out_index is not int.
        TypeError: If input_index is neither int nor tuple/list of int.
        TypeError: If output of the nerwork are neither Tensor, not tuple of Tensors.
        TypeError: If input_index type is neither int nor tuple of int.
        IndexError: If input_index or output_index is out of range.

    Example:
    >>> import mindspore as ms
    >>> class Net(nn.Cell):
    >>>     def __init__(self):
    >>>         super().__init__()
    >>>     def construct(self, x, y):
    >>>         out1 = x + y
    >>>         out2 = 2 * x + y
    >>>         out3 = x * x + 4 * y * y + 3 * y
    >>>         f, g, h = out1.sum(), out2.sum(), out3.sum()
    >>>         return f, g, h
    >>> net = Net() # net: f, g, h = net(x, y)
    >>> x = ops.ones((2, 3), ms.float32)
    >>> y = ops.ones((2, 3), ms.float32)
    >>> first_grad_net = grad(net, 2, 1) # ∂h/∂y, since (f, g, h)[2] == h, (x, y)[1] == y
    >>> second_grad_net = grad(first_grad_net, 0, 1) # ∂2h/∂y2, since (∂h)[0] == ∂h, (x, y)[1] == y
    >>> print(first_grad_net(x, y))
    [[11. 11. 11.]
     [11. 11. 11.]]
    >>> print(second_grad_net(x, y))
    [[8. 8. 8.]
     [8. 8. 8.]]
    >>> class Net2(nn.Cell):
    >>>    def __init__(self):
    >>>        super().__init__()
    >>>    def construct(self, x, y):
    >>>        out1 = 2 * x + y
    >>>        out2 = x * x + 4 * x * y + 3 * y
    >>>        f, g = out1.sum(), out2.sum()
    >>>        return f, g
    >>> net = Net2()  # output: (f, g), input:(x, y)
    >>> x = ops.ones((2, 3), ms.float32)
    >>> y = ops.ones((2, 3), ms.float32)
    >>> first_grad_net = grad(net, 1, (0, 1))  # (∂g/∂x, ∂g/∂y), since (f, g)[1] == g
    >>> second_grad_net = grad(first_grad_net, 0, (0, 1))  # (∂2g/∂x2, ∂2g/∂x∂y), since (∂g/∂x, ∂g/∂y)[0] == ∂g/∂x
    >>> print(first_grad_net(x, y))
    (Tensor(shape=[2, 3], dtype=Float32, value=
    [[ 6.00000000e+00,  6.00000000e+00,  6.00000000e+00],
     [ 6.00000000e+00,  6.00000000e+00,  6.00000000e+00]]), Tensor(shape=[2, 3], dtype=Float32, value=
    [[ 7.00000000e+00,  7.00000000e+00,  7.00000000e+00],
     [ 7.00000000e+00,  7.00000000e+00,  7.00000000e+00]]))
    >>> print(second_grad_net(x, y))
    (Tensor(shape=[2, 3], dtype=Float32, value=
    [[ 2.00000000e+00,  2.00000000e+00,  2.00000000e+00],
     [ 2.00000000e+00,  2.00000000e+00,  2.00000000e+00]]), Tensor(shape=[2, 3], dtype=Float32, value=
    [[ 4.00000000e+00,  4.00000000e+00,  4.00000000e+00],
     [ 4.00000000e+00,  4.00000000e+00,  4.00000000e+00]]))
    """

    def __init__(self, net, output_index=0, input_index=-1):
        super().__init__()
        if not isinstance(output_index, int):
            raise TypeError(f"output_index type is {type(output_index)}, which can only be int.")
        if not _recursive_type_check(input_index, int):
            raise TypeError(f"input_index is {output_index}, which can only be None, int or tuple/list of int.")
        self.net = net
        self.grad = ops.GradOperation(get_all=True, sens_param=True)
        self.grad_net = self.grad(self.net)
        self.output_index = output_index
        self.input_index = input_index
        if isinstance(self.input_index, int):
            self.input_index = to_tuple(self.input_index)
        self.cast = ops.Cast()

    def construct(self, *inputs):
        """construct"""
        outputs = self.net(*inputs)
        out_tup = to_tuple(outputs)
        data_type = out_tup[0].dtype
        sens = [ops.zeros_like(output) for output in out_tup]
        sens[self.output_index] = ops.ones_like(out_tup[self.output_index])
        sens_tuple = tuple([self.cast(_, data_type) for _ in sens])
        first_grad = self.grad_net(*inputs, sens_tuple if len(sens_tuple) > 1 else sens_tuple[0])
        if len(self.input_index) == 1:
            if self.input_index == (-1,):
                return first_grad
            return first_grad[self.input_index[0]]
        return tuple(first_grad[ind] for ind in self.input_index)

    def grad_func(self, *inputs):
        def currying(sens):
            return self.grad_net(*inputs, sens if len(sens) > 1 else sens[0])

        return currying


def grad(net, output_index=0, input_index=-1):
    r"""
    Gradient function. Refer to _Grad.

    Args:
        net (Cell): Net to be auto-differentiated.
        output_index (int): Output index starting from 0. Default: 0.
        input_index (Union(int, tuple[int])): Input index(es) starting from 0, and only forward indexes are allowed.
            If -1, all specified inputs would be differentiated respectively. Default: -1.

    Inputs:
        - **\*inputs** (tuple[Tensor]) - The inputs of the original network.

    Outputs:
        Union(Tensor, tuple[Tensor]), The outputs of the fist order derivative net.

    Raises:
        TypeError: If out_index is not int.
        TypeError: If input_index is neither int nor tuple/list of int.
        TypeError: If output of the network are neither Tensor, not tuple of Tensors.
        TypeError: If input_index type is neither int nor tuple of int.
        IndexError: If input_index or output_index is out of range.
    """
    return _Grad(net, output_index=output_index, input_index=input_index)
