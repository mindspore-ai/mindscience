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
# ============================================================================
"""
derivative
"""
from inspect import isfunction
import numpy as np

import mindspore
import mindspore.numpy as mnp
from mindspore import Tensor, nn, ops
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import constexpr
from mindspore import dtype as mstype

cast_grad = C.MultitypeFuncGraph("cast_grad")


@cast_grad.register("Tensor")
def _cast_grad(grad):
    return F.cast(grad, mstype.float32)


def _transfer_tensor_to_tuple(inputs):
    """
    If the input is a tensor, convert it to a tuple. If not, the output is unchanged.
    """
    if isinstance(inputs, Tensor):
        return (inputs,)
    return inputs


class _GenerateMultiSens(nn.Cell):
    """generate sens for multi-outputs"""
    def construct(self, o, net_out, sens):
        if len(net_out) == 1:
            return sens
        all_sens = ()
        for i, _ in enumerate(net_out):
            if i != o:
                all_sens += (mnp.zeros(net_out[i].shape, mnp.float32),)
            else:
                all_sens += (sens,)
        return all_sens


class _MergeOutput(nn.Cell):
    """merge output"""
    def construct(self, out_tmp, gout, iters):
        for i in range(iters):
            out_tmp[i] = out_tmp[i] + (gout[i],)
        return out_tmp


@constexpr
def _generate_sens(batch_size, out_channels, i):
    sens = np.zeros((batch_size, out_channels), np.float32)
    sens[:, i] = 1
    return Tensor(sens)


@constexpr
def _generate_indices(j):
    return Tensor([j], mindspore.int32)


@constexpr
def _check_type(net_in, net_out, input_idx=None, output_idx=None):
    """check type of input"""
    if net_in is not None:
        raise TypeError("The Type of network input should be Tensor but got {}".format(type(net_in)))
    if input_idx is not None and (not isinstance(input_idx, int) or isinstance(input_idx, bool)):
        raise TypeError("The Type of column index of input should be int but got {}".format(type(input_idx)))
    if output_idx is not None and (not isinstance(output_idx, int) or isinstance(output_idx, bool)):
        raise TypeError("The Type of column index of output should be int but got {}".format(type(output_idx)))
    if net_out is not None:
        raise TypeError("The Type of network output should be Tensor but got {}".format(type(net_out)))


@constexpr
def _check_dimension(in_shape, out_shape, in_idx, out_idx):
    """check dimension of input"""
    if len(in_shape) != 2:
        raise ValueError("The dimension of network input should be 2, but got {}".format(len(in_shape)))
    if len(out_shape) != 2:
        raise ValueError("The dimension of network output should be 2, but got {}".format(len(out_shape)))
    if in_idx is not None and out_idx is not None:
        if in_idx >= in_shape[1]:
            raise ValueError("input index should be in range (0, {}), but got {}".format(in_shape[1], in_idx))
        if out_idx >= out_shape[1]:
            raise ValueError("output index should be in range (0, {}), but got {}".format(out_shape[1], out_idx))


class Grad(nn.Cell):
    """
    Computes and returns the gradients of the specified column of outputs with respect to the specified column of
    inputs.

    Args:
        model (Cell): a function or network that takes Tensor inputs.
        argnum (int): specifies which input the output takes the first derivative of. Default: 0.

    Inputs:
        - **x** (list) - The input is variable-length argument. The first input is a 2D network inputs (Tensor),
          the last three inputs are column index of input (int), column index of output (int) and output of
          network (Tensor).

    Outputs:
        Tensor.

    Raises:
        TypeError: If the type of `argnum` is not int.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import nn, Tensor
        >>> from mindflow.operators import Grad
        ...
        >>> class Net(nn.Cell):
        ...    def __init__(self):
        ...        super(Net, self).__init__()
        ...    def construct(self, x):
        ...        return x * x
        ...
        >>> x = Tensor(np.array([[1.0, -2.0], [-3.0, 4.0]]).astype(np.float32))
        >>> net = Net()
        >>> out = net(x)
        >>> grad = Grad(net)
        >>> print(grad(x, 0, 0, out).asnumpy())
        [[ 2.]
         [-6.]]
    """
    def __init__(self, model, argnum=0):
        super(Grad, self).__init__()
        if not isinstance(model, nn.Cell) and not isfunction(model):
            raise TypeError("The type of model should be a function or network, but got {}".format(type(model)))
        self.model = model
        if isinstance(argnum, bool) or not isinstance(argnum, int):
            raise TypeError("The type of argnum should be int, but get {}".format(type(argnum)))
        self.hyper_map = C.HyperMap()
        self.argnum = argnum
        self.grad = ops.GradOperation(get_all=True, sens_param=True)
        self.gather = ops.Gather()
        self.cast = ops.Cast()
        self.dtype = ops.DType()

    def construct(self, *x):
        x = _transfer_tensor_to_tuple(x)
        input_idx, output_idx, net_out = x[-3], x[-2], x[-1]
        net_in = x[:-3]
        _check_type(net_in[0], net_out, input_idx, output_idx)
        if net_out is None:
            net_out = self.model(*net_in)
        net_out = _transfer_tensor_to_tuple(net_out)[0]
        _check_dimension(net_in[self.argnum].shape, net_out.shape, input_idx, output_idx)
        batch_size, out_channels = net_out.shape
        sens = _generate_sens(batch_size, out_channels, output_idx)
        gradient_function = self.grad(self.model)
        sens = self.cast(sens, self.dtype(net_out))
        gradient = gradient_function(*net_in, sens)
        gradient = self.hyper_map(cast_grad, gradient)
        if input_idx is None:
            output = gradient[self.argnum]
        else:
            out_indices = _generate_indices(input_idx)
            output = self.gather(gradient[self.argnum], out_indices, 1)
        return output


class SecondOrderGrad(nn.Cell):
    """
    Computes and returns the second order gradients of the specified column of outputs with respect to the specified
    column of inputs.

    Args:
        model (Cell): a function or network that takes a single Tensor input and returns a single Tensor.
        input_idx1 (int): specifies the column index of input to take the first derivative,
            takes values in [0, model input size - 1].
        input_idx2 (int): specifies the column index of input to take the second derivative,
            takes values in [0, model input size - 1].
        output_idx (int): specifies the column index of output, takes values in [0, model output size - 1].

    Inputs:
        - **input** - The input of given function or network `model`.

    Outputs:
        Tensor.

    Raises:
        TypeError: If the type of `input_idx1`,  `input_idx2` or `output_idx` is not int.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import nn, Tensor
        >>> from mindflow.operators import SecondOrderGrad
        >>> class Net(nn.Cell):
        ...    def __init__(self):
        ...        super(Net, self).__init__()
        ...
        ...    def construct(self, x):
        ...        return x * x * x
        >>> x = Tensor(np.array([[1.0, -2.0], [-3.0, 4.0]]).astype(np.float32))
        >>> net = Net()
        >>> out = net(x)
        >>> grad = SecondOrderGrad(net, 0, 0, 0)
        >>> print(grad(x).asnumpy())
        [[  6.]
         [-18.]]
    """
    def __init__(self, model, input_idx1, input_idx2, output_idx):
        super(SecondOrderGrad, self).__init__()
        if not isinstance(model, nn.Cell) and not isfunction(model):
            raise TypeError("The type of model should be a function or network, but got {}".format(type(model)))
        if isinstance(input_idx1, bool) or not isinstance(input_idx1, int):
            raise TypeError("The type of input_idx1 should be int, but got {}".format(type(input_idx1)))
        if isinstance(input_idx2, bool) or not isinstance(input_idx2, int):
            raise TypeError("The type of input_idx1 should be int, but got {}".format(type(input_idx2)))
        if isinstance(output_idx, bool) or not isinstance(output_idx, int):
            raise TypeError("The type of input_idx1 should be int, but got {}".format(type(output_idx)))
        self.jac1 = _FirstOrderGrad(model, input_idx=input_idx1, output_idx=output_idx)
        self.jac2 = _FirstOrderGrad(self.jac1, input_idx=input_idx2, output_idx=0)

    def construct(self, x):
        hes = self.jac2(x)
        return hes


@constexpr
def get_vmap_sens_list(batch_size, out_channel, output_idxs):
    sens = []
    for output_idx in output_idxs:
        sen = _generate_sens(batch_size, out_channel, output_idx)
        sens.append(sen)
    return sens


class GradVmap(nn.Cell):
    """
    Computes and returns the gradients of the specified column of outputs with respect to the specified column of
    inputs using Vmap.

    Args:
        model (Cell): a function or network that takes Tensor inputs.
        argnum (int): specifies which input the output takes the first derivative of. Default: 0.

    Inputs:
        - **x** (list) - The input is variable-length argument. The first input is a 2D network inputs (Tensor),
          the last three inputs are column index of input (int), column index of output (int) and output of
          network (Tensor).

    Outputs:
        Tensor.

    Raises:
        TypeError: If the type of `argnum` is not int.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import nn, Tensor
        >>> from mindflow.operators import Grad
        ...
        >>> class Net(nn.Cell):
        ...    def __init__(self):
        ...        super(Net, self).__init__()
        ...    def construct(self, x):
        ...        return x * x
        ...
        >>> x = Tensor(np.array([[1.0, -2.0], [-3.0, 4.0]]).astype(np.float32))
        >>> net = Net()
        >>> out = net(x)
        >>> grad = Grad(net)
        >>> print(grad(x, 0, 0, out).asnumpy())
        [[ 2.]
         [-6.]]
    """
    def __init__(self, model, argnum=0):
        super(GradVmap, self).__init__()
        if not isinstance(model, nn.Cell) and not isfunction(model):
            raise TypeError("The type of model should be a function or network, but got {}".format(type(model)))
        self.model = model
        if isinstance(argnum, bool) or not isinstance(argnum, int):
            raise TypeError("The type of argnum should be int, but get {}".format(type(argnum)))
        self.hyper_map = C.HyperMap()
        self.argnum = argnum
        self.grad = ops.GradOperation(get_all=True, sens_param=True)
        self.gather = ops.Gather()
        self.cast = ops.Cast()
        self.dtype = ops.DType()
        self.stack_op = ops.Stack(axis=0)

    def construct(self, *x):
        x = _transfer_tensor_to_tuple(x)
        input_idx, output_idxs, net_out = x[-3], x[-2], x[-1]
        net_in = x[:-3]

        if net_out is None:
            net_out = self.model(*net_in)
        net_out = _transfer_tensor_to_tuple(net_out)[0]

        batch_size, out_channels = net_out.shape
        sens = get_vmap_sens_list(batch_size, out_channels, output_idxs)
        sens = self.stack_op(sens)
        sens = self.cast(sens, self.dtype(net_out))
        gradient_function = self.grad(self.model)
        gradient_function_vmap = F.vmap(gradient_function, in_axes=(None, 0), out_axes=0)
        gradient = gradient_function_vmap(*net_in, sens)
        gradient = self.hyper_map(cast_grad, gradient)

        if input_idx is None:
            output = gradient[self.argnum]
        else:
            out_indices = _generate_indices(input_idx)
            output = self.gather(gradient[self.argnum], out_indices, 2)

        return output


class _FirstOrderGrad(nn.Cell):
    """compute first-order derivative"""
    def __init__(self, model, argnums=0, input_idx=None, output_idx=1):
        super(_FirstOrderGrad, self).__init__()
        self.model = model
        self.argnums = argnums
        self.input_idx = input_idx
        self.output_idx = output_idx
        self.grad = ops.GradOperation(get_all=True, sens_param=True)
        self.gather = ops.Gather()
        self.cast = ops.Cast()
        self.dtype = ops.DType()
        self.hyper_map = C.HyperMap()

    def construct(self, *x):
        """Defines the computation to be performed"""
        x = _transfer_tensor_to_tuple(x)
        _check_type(x[self.argnums], None)
        net_out = self.model(*x)
        net_out = _transfer_tensor_to_tuple(net_out)[0]
        _check_dimension(x[self.argnums].shape, net_out.shape, self.input_idx, self.output_idx)
        batch_size, out_channels = net_out.shape
        sens = _generate_sens(batch_size, out_channels, self.output_idx)
        gradient_function = self.grad(self.model)
        sens = self.cast(sens, self.dtype(net_out))
        gradient = gradient_function(*x, sens)
        gradient = self.hyper_map(cast_grad, gradient)
        outout_indices = _generate_indices(self.input_idx)
        output = self.gather(gradient[self.argnums], outout_indices, 1)
        return output
        
