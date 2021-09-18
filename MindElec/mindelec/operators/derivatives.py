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
from mindspore.ops import constexpr
from mindspore import dtype as mstype
from ..architecture.util import check_mode


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
        for i in range(len(net_out)):
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
def _generate_sens(batch_size, out_chanel, i):
    sens = np.zeros((batch_size, out_chanel), np.float32)
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
        - **x** - The input is variable-length argument. Notes that the last three inputs are column index of
          input (int), column index of output (int) and output of network (Tensor). Besides these inputs, the
          first is the network inputs (Tensor), which should be two dimensions.

    Outputs:
        Tensor.

    Raises:
        TypeError: If the type of `argnum` is not int.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import numpy as np
        >>> from mindspore import nn, Tensor
        >>> from mindelec.operators import Grad
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
        check_mode("Grad")
        if not isinstance(model, nn.Cell) and not isfunction(model):
            raise TypeError("The type of model should be a function or network, but got {}".format(type(model)))
        self.model = model
        if isinstance(argnum, bool) or not isinstance(argnum, int):
            raise TypeError("The type of argnum should be int, but get {}".format(type(argnum)))
        self.argnum = argnum
        self.grad = ops.GradOperation(get_all=True, sens_param=True)
        self.gather = ops.Gather()
        self.cast = ops.Cast()
        self.dtype = ops.DType()

    def construct(self, *x):
        """define computation to be performed"""
        x = _transfer_tensor_to_tuple(x)
        input_idx, output_idx, net_out = x[-3], x[-2], x[-1]
        net_in = x[:-3]
        _check_type(net_in[0], net_out, input_idx, output_idx)
        if net_out is None:
            net_out = self.model(*net_in)
        net_out = _transfer_tensor_to_tuple(net_out)[0]
        _check_dimension(net_in[self.argnum].shape, net_out.shape, input_idx, output_idx)
        batch_size, out_chanel = net_out.shape
        sens = _generate_sens(batch_size, out_chanel, output_idx)
        gradient_function = self.grad(self.model)
        sens = self.cast(sens, self.dtype(net_out))
        gradient = gradient_function(*net_in, sens)
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
        input_idx1 (int): specifies the column index of input to take the first derivative of.
        input_idx2 (int): specifies the column index of input to take the second derivative of.
        output_idx (int): specifies the column index of output.

    Inputs:
        - **input** - The input of given function or network `model`.

    Outputs:
        Tensor.

    Raises:
        TypeError: If the type of `input_idx1`,  `input_idx2` or `output_idx` is not int.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import numpy as np
        >>> from mindspore import nn, Tensor
        >>> from mindelec.operators import SecondOrderGrad
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
        check_mode("SecondOrderGrad")
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


class Jacobian(nn.Cell):
    r"""
        Computes the Jacobian of a given function or network.

        Note:
            The output of the given function or network should be a single Tensor.

        Args:
            net (Union[function, Cell]): a function or network that takes Tensor inputs.
            arg_nums (int): specifies which input the output takes the first derivative of.
            out_idx (int): specifies which output to take the first derivative of.

        Inputs:
            - **x** (Tensor) - The inputs of the function or network `net`.

        Outputs:
            Tensor or tuple of Tensors. If `arg_nums` is int, output will be a Tensor whose shape is the shape of
            specified output * the shape of specified input. If `arg_nums` is None, output will be a tuple of Tensors
            where output[i] will contain the Jacobian of the specified output and ith input and will have as size the
            concatenation of the sizes of the corresponding output and the corresponding input

        Raises:
            TypeError: if the type of `arg_nums` or `out_idx` is not int.

        Supported Platforms:
            ``Ascend``

        Examples:
            >>> import numpy as np
            >>> from mindelec.operators import Jacobian
            >>> from mindspore import Tensor
            >>> def func(x, y):
            >>>     return (x * x * x * y + 3 * y * y * x).sum()
            ...
            >>> a = Tensor(np.array([[1, 3], [5, 9], [8, 2]], np.float32))
            >>> b = Tensor(np.array([[4, 6], [7, 2], [2, 1]], np.float32))
            >>> jac = Jacobian(func, 0, 0)
            >>> output = jac(a, b)
            >>> print(output.shape)
            (3, 2)
        """
    def __init__(self, net, argnums=0, out_idx=0):
        super(Jacobian, self).__init__()
        if not (isinstance(argnums, int) or argnums is None) or not isinstance(out_idx, int):
            raise TypeError("The type of argnums should be int or None and out_idx should be int.")
        self.net = net
        self.argnums = argnums
        self.out_idx = out_idx
        self.grad_op = ops.GradOperation(get_all=True, sens_param=True)
        self.eye = ops.Eye()
        self.concat = ops.Concat()
        self.reshape = ops.Reshape()
        self.tuple_len = ops.Primitive("tuple_len")
        self.make_list = ops.Primitive("make_list")
        self._merge_output = _MergeOutput()
        self._generate_multi_sens = _GenerateMultiSens()

    def construct(self, *x):
        """
        forward

        Args:
            inputs (tuple): input tensor.
        """
        net_out = _transfer_tensor_to_tuple(self.net(*x))
        net_out_target = net_out[self.out_idx]
        grad_fn = self.grad_op(self.net)
        input_len = self.tuple_len(x)

        identity_matrix = self.eye(net_out_target.size, net_out_target.size, mstype.float32)
        identity_matrix = ops.Split(0, net_out_target.size)(identity_matrix)
        if self.argnums is None:
            out_tmp = [()] * input_len
            for line in identity_matrix:
                sens = self.reshape(line, net_out_target.shape)
                grad_wrt_output = self._generate_multi_sens(self.out_idx, net_out, sens)
                grad = grad_fn(*x, grad_wrt_output)
                out_tmp = self._merge_output(out_tmp, grad, input_len)
            output = ()
            for i in range(input_len):
                out_tmp[i] = self.concat(out_tmp[i])
                output = output + (self.reshape(out_tmp[i], net_out_target.shape + x[i].shape),)
            return output
        output = ()
        for line in identity_matrix:
            sens = self.reshape(line, net_out_target.shape)
            grad_wrt_output = self._generate_multi_sens(self.out_idx, net_out, sens)
            grad = grad_fn(*x, grad_wrt_output)
            output = output + (grad[self.argnums],)
        output = self.concat(output)
        return self.reshape(output, net_out_target.shape + x[self.argnums].shape)


class Hessian(nn.Cell):
    r"""
    Computes the Hessian of a given function or network.

    Note:
        The output of the given function or network should be a single Tensor.

    Args:
        net (Union[function, Cell]): a function or network that takes Tensor inputs and returns a single Tensor.
        diff1_idx (int): specifies which input the output takes the first derivative of.
        diff2_idx (int): specifies which input the output takes the second derivative of.

    Inputs:
        - **x** (Tensor) - The inputs of the function or network `net`.

    Outputs:
        Tensor, the shape is the shape output * shape of specified input * the shape of specified input.

    Raises:
        TypeError: if the type of `diff1_idx` or `diff2_idx` is not int.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import numpy as np
        >>> from mindelec.operators import Hessian
        >>> from mindspore import Tensor
        >>> def func(x, y):
        >>>     return (x * x * x * y + 3 * y * y * x).sum()
        >>> a = Tensor(np.array([[1, 3], [5, 9], [8, 2]], np.float32))
        >>> b = Tensor(np.array([[4, 6], [7, 2], [2, 1]], np.float32))
        >>> hes = Hessian(func, 0, 0)
        >>> output = hes(a, b)
        >>> print(output.shape)
        (3, 2, 3, 2)
    """
    def __init__(self, net, diff1_idx, diff2_idx, out_idx=0):
        super(Hessian, self).__init__()
        if not isinstance(diff1_idx, int) or not (isinstance(diff2_idx, int) or diff2_idx is None):
            raise TypeError("The type of diff1 should be int and diff2 should be int or None.")
        self.jac1 = Jacobian(net, argnums=None, out_idx=out_idx)
        self.jac2 = Jacobian(self.jac1, argnums=diff2_idx, out_idx=diff1_idx)

    def construct(self, *x):
        return self.jac2(*x)


def jacobian(func, inputs):
    r"""
    Function that computes the Jacobian of a given function or network.

    Parameters:
        func: a function or network that takes Tensor inputs.
        inputs: The inputs of the function or network `net`.
    """
    inputs = _transfer_tensor_to_tuple(inputs)
    func_out = _transfer_tensor_to_tuple(func(*inputs))
    output = ()
    for i in range(len(func_out)):
        jac = Jacobian(func, argnums=None, out_idx=i)
        output = output + _transfer_tensor_to_tuple(jac(*inputs))
    return output


def hessian(func, inputs):
    r"""
    Function that computes the Hessian of a given function or network.

    Parameters:
        func: a function or network that takes Tensor inputs.
        inputs: The inputs of the function or network `net`.
    """
    inputs = _transfer_tensor_to_tuple(inputs)
    func_out = _transfer_tensor_to_tuple(func(*inputs))
    output = ()
    for i in range(len(func_out)):
        out_tmp = ()
        for j in range(len(inputs)):
            hes = Hessian(func, diff1_idx=j, diff2_idx=None, out_idx=i)
            out_tmp = out_tmp + _transfer_tensor_to_tuple(hes(*inputs))
        output = output + out_tmp
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

    def construct(self, *x):
        """Defines the computation to be performed"""
        x = _transfer_tensor_to_tuple(x)
        _check_type(x[self.argnums], None)
        net_out = self.model(*x)
        net_out = _transfer_tensor_to_tuple(net_out)[0]
        _check_dimension(x[self.argnums].shape, net_out.shape, self.input_idx, self.output_idx)
        batch_size, out_chanel = net_out.shape
        sens = _generate_sens(batch_size, out_chanel, self.output_idx)
        gradient_function = self.grad(self.model)
        sens = self.cast(sens, self.dtype(net_out))
        gradient = gradient_function(*x, sens)
        outout_indices = _generate_indices(self.input_idx)
        output = self.gather(gradient[self.argnums], outout_indices, 1)
        return output
