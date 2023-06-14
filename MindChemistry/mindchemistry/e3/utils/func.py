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
import numpy as np
from scipy.linalg import expm

from mindspore import Tensor, ops
from mindspore.ops import operations as P


def norm_keep(input_x, axis):
    r"""
    Compute the matrix norm or vector norm of a given tensor, and the output tensors have dimension retained.

    Args:
        input_x (Tensor): Input tensor. The dtype must be float32 or float16.
        axis (Union[int, list, tuple]): Specifies which dimension or dimensions of input to calculate the norm across.

    Returns:
        Tensor, has the same dtype and shape as `input`.
    """
    return ops.expand_dims(input_x.norm(None, axis, False), axis=axis)


def _to_tensor(arg):
    if isinstance(arg, (int, float)):
        return Tensor(arg)
    elif isinstance(arg, (np.ndarray, list, tuple)):
        return Tensor(arg)
    elif isinstance(arg, Tensor):
        return arg
    else:
        raise TypeError


def broadcast_shapes(*shapes):
    r"""
    Return the broadcast shape of the shapes of input tensors.

    Args:
        shapes (tuple): Any number of shapes of tensors to be broadcasted.

    Returns:
        Tuple, a shape compatible with all input shapes.
    """
    max_len = 0
    for shape in shapes:
        if isinstance(shape, int):
            if max_len < 1:
                max_len = 1
        elif isinstance(shape, tuple) or isinstance(shape, list):
            s = len(shape)
            if max_len < s:
                max_len = s
    result = [1] * max_len
    for shape in shapes:
        if isinstance(shape, int):
            shape = (shape,)
        if isinstance(shape, tuple) or isinstance(shape, list):
            for i in range(-1, -1 - len(shape), -1):
                if shape[i] < 0:
                    raise RuntimeError("Trying to create tensor with negative dimension ({}): ({})"
                                       .format(shape[i], shape[i]))
                if shape[i] == 1 or shape[i] == result[i]:
                    continue
                if result[i] != 1:
                    raise RuntimeError(
                        "Shape mismatch: objects cannot be broadcast to a single shape")
                result[i] = shape[i]
        else:
            raise RuntimeError(
                "Input shapes should be of type ints, a tuple of ints, or a list of ints, got ", shape)
    return tuple(result)


def broadcast_tensors(*tensors):
    r"""
    Broadcasts the given tensors.

    Args:
        tensors (Tensor): Any number of tensors of the same type.

    Returns:
        A list of tensors, tensors after broadcast.
    """
    shapes = []
    for tensor in tensors:
        shapes.append(tensor.shape)
    shape = broadcast_shapes(*shapes)
    res = []
    for tensor in tensors:
        if len(shape):
            res.append(ops.broadcast_to(tensor, shape))
        else:
            res.append(tensor)
    return res


def broadcast_args(*args):
    r"""
    Broadcasts the given data with multiple types.

    Args:
        *arg (Union[Tensor[float32], list[float], tuple[float], ndarray[np.float32], float]): Any number of data to be broadcasted.

    Returns:
        A list of tensors, tensors after broadcast.
    """
    tensors = []
    for arg in args:
        tensors.append(_to_tensor(arg))
    res = broadcast_tensors(*tensors)
    return res


def _ndexpm(mat):
    """Compute matrix-product exponential of matrices."""
    if isinstance(mat, Tensor):
        mat = mat.asnumpy()
    mat_shape = mat.shape
    if len(mat_shape) < 2:
        raise ValueError
    elif len(mat_shape) == 2:
        return Tensor(expm(mat))
    else:
        mat = np.reshape(mat, (-1, mat_shape[-1], mat_shape[-1]))
        n = mat.shape[0]
        for i in range(n):
            mat[i] = expm(mat[i])
        mat = np.reshape(mat, mat_shape)
        return Tensor(mat)


def _expand_last_dims(x):
    if isinstance(x, Tensor):
        x = ops.expand_dims(x, -1)
        x = ops.expand_dims(x, -1)
    else:
        x = x[..., None, None]
    return x


def narrow(inputs, axis, start, length):
    """tmp narrow API"""
    begins = [0] * inputs.ndim
    begins[axis] = start
    sizes = [i for i in inputs.shape]
    sizes[axis] = length
    return P.Slice()(inputs, begins, sizes)
