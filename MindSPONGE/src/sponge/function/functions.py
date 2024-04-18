# Copyright 2021-2023 @ Shenzhen Bay Laboratory &
#                       Peking University &
#                       Huawei Technologies Co., Ltd
#
# This code is a part of MindSPONGE:
# MindSpore Simulation Package tOwards Next Generation molecular modelling.
#
# MindSPONGE is open-source software based on the AI-framework:
# MindSpore (https://www.mindspore.cn/)
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
Common functions
"""

from typing import Union, List, Tuple, Iterable
from datetime import time, timedelta, date
import numpy as np
from numpy import ndarray
import mindspore as ms
try:
    # MindSpore 2.X
    from mindspore import jit
except ImportError:
    # MindSpore 1.X
    from mindspore import ms_function as jit
import mindspore.numpy as msnp
from mindspore.numpy.utils import _to_tensor
from mindspore import ops, Tensor, Parameter
from mindspore.ops import functional as F
from mindspore.common.initializer import Initializer, _INITIALIZER_ALIAS


__all__ = [
    'PI',
    'keepdims_sum',
    'keepdims_mean',
    'keepdims_prod',
    'reduce_any',
    'reduce_all',
    'reduce_prod',
    'concat_first_dim',
    'concat_last_dim',
    'concat_penulti',
    'stack_first_dim',
    'stack_last_dim',
    'stack_penulti',
    'squeeze_first_dim',
    'squeeze_last_dim',
    'squeeze_penulti',
    'identity',
    'periodic_variable',
    'periodic_difference',
    'gather_vector',
    'gather_value',
    'pbc_box_reshape',
    'pbc_image',
    'coordinate_in_pbc',
    'vector_in_pbc',
    'calc_vector_nopbc',
    'calc_vector_pbc',
    'calc_vector',
    'calc_distance_nopbc',
    'calc_distance_pbc',
    'calc_distance',
    'calc_angle_by_vectors',
    'calc_angle_nopbc',
    'calc_angle_pbc',
    'calc_angle',
    'calc_torsion_by_vectors',
    'calc_torsion_nopbc',
    'calc_torsion_pbc',
    'calc_torsion',
    'coulomb_interaction',
    'lennard_jones_potential',
    'lennard_jones_potential2',
    'get_integer',
    'get_ndarray',
    'get_tensor',
    'get_ms_array',
    'check_broadcast',
    'any_none',
    'all_none',
    'any_not_none',
    'all_not_none',
    'get_arguments',
    'get_initializer',
    'bonds_in'
]

PI = 3.141592653589793238462643383279502884197169399375105820974944592307
r""":math:`\pi`"""

keepdims_sum_ = ops.ReduceSum(True)
keepdims_mean_ = ops.ReduceMean(True)
keepdims_prod_ = ops.ReduceProd(True)
reduce_any_ = ops.ReduceAny()
reduce_all_ = ops.ReduceAll()
reduce_prod_ = ops.ReduceProd()
concat_first_dim_ = ops.Concat(0)
concat_last_dim_ = ops.Concat(-1)
concat_penulti_ = ops.Concat(-2)
stack_first_dim_ = ops.Stack(0)
stack_last_dim_ = ops.Stack(-1)
stack_penulti_ = ops.Stack(-2)
squeeze_first_dim_ = ops.Squeeze(0)
squeeze_last_dim_ = ops.Squeeze(-1)
squeeze_penulti_ = ops.Squeeze(-2)
identity_ = ops.Identity()


def keepdims_sum(x: Tensor, axis: Union[int, Tuple[int], List[int]] = ()) -> Tensor:
    """
    Reduces a dimension to 1 by summing the elements in the dimension of `x` along the axis,
    and the dimensions of the output and input are the same.

    Args:
        x (Tensor[Number]): The input tensor. The dtype of the tensor to be reduced is number.
            :math:`(N,*)` where :math:`*` means, any number of additional dimensions, its rank should be less than 8.
        axis (Union[int, tuple(int), list(int)]): The dimensions to reduce. Default: (), reduce all dimensions.
            Only constant value is allowed. Must be in the range [-rank(`x`), rank(`x`)).

    Outputs:
        Tensor, has the same dtype as the `x`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    """
    return keepdims_sum_(x, axis)


def keepdims_mean(x: Tensor, axis: Union[int, Tuple[int], List[int]] = ()) -> Tensor:
    """
    Reduces a dimension to 1 by averaging the elements in the dimension of `x` along the axis,
    and the dimensions of the output and input are the same.

    Args:
        x (Tensor[Number]): The input tensor. The dtype of the tensor to be reduced is number.
            :math:`(N,*)` where :math:`*` means, any number of additional dimensions, its rank should be less than 8.
        axis (Union[int, tuple(int), list(int)]): The dimensions to reduce. Default: (), reduce all dimensions.
            Only constant value is allowed. Must be in the range [-rank(`x`), rank(`x`)).

    Outputs:
        Tensor, has the same dtype as the `x`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    """
    return keepdims_mean_(x, axis)


def keepdims_prod(x: Tensor, axis: Union[int, Tuple[int], List[int]] = ()) -> Tensor:
    """
    Reduces a dimension to 1 by multiplying the elements in the dimension of `x` along the axis,
    and the dimensions of the output and input are the same.

    Args:
        x (Tensor[Number]): The input tensor. The dtype of the tensor to be reduced is number.
            :math:`(N,*)` where :math:`*` means, any number of additional dimensions, its rank should be less than 8.
        axis (Union[int, tuple(int), list(int)]): The dimensions to reduce. Default: (), reduce all dimensions.
            Only constant value is allowed. Must be in the range [-rank(`x`), rank(`x`)).

    Outputs:
        Tensor, has the same dtype as the `x`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    """
    return keepdims_prod_(x, axis)


def reduce_any(x: Tensor, axis: Union[int, Tuple[int], List[int]] = ()) -> Tensor:
    r"""
    Reduces a dimension of a tensor by the "logical OR" of all elements in the dimension, by default. And also can
    reduce a dimension of `x` along the axis. See `mindspore.ops.ReduceAny` for detailed information.

    Args:
        x (Tensor[bool]): The input tensor. The dtype of the tensor to be reduced is bool.
            :math:`(N,*)` where :math:`*` means, any number of additional dimensions, its rank should be less than 8.
        axis (Union[int, tuple(int), list(int)]): The dimensions to reduce. Default: (), reduce all dimensions.
            Only constant value is allowed. Must be in the range [-rank(x), rank(x)).

    Outputs:
        Tensor, the dtype is bool.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    """
    return reduce_any_(x, axis)


def reduce_all(x: Tensor, axis: Union[int, Tuple[int], List[int]] = ()) -> Tensor:
    r"""
    Reduces a dimension of a tensor by the "logicalAND" of all elements in the dimension, by default. And also can
    reduce a dimension of `x` along the axis. See `mindspore.ops.ReduceAll` for detailed information.

    Args:
        x (Tensor[bool]): The input tensor. The dtype of the tensor to be reduced is bool.
            :math:`(N,*)` where :math:`*` means, any number of additional dimensions, its rank should be less than 8.
        axis (Union[int, tuple(int), list(int)]): The dimensions to reduce. Default: (), reduce all dimensions.
            Only constant value is allowed. Must be in the range [-rank(x), rank(x)).

    Outputs:
        Tensor, the dtype is bool.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    """
    return reduce_all_(x, axis)


def reduce_prod(x: Tensor, axis: Union[int, Tuple[int], List[int]] = ()) -> Tensor:
    r"""
    Reduces a dimension of a tensor by multiplying all elements in the dimension, by default. And also can
    reduce a dimension of `x` along the axis. See `mindspore.ops.ReduceProd` for detailed information.

    Args:
        x (Tensor[Number]): The input tensor. The dtype of the tensor to be reduced is number.
            :math:`(N,*)` where :math:`*` means, any number of additional dimensions, its rank should be less than 8.
        axis (Union[int, tuple(int), list(int)]): The dimensions to reduce. Default: (), reduce all dimensions.
            Only constant value is allowed. Must be in the range [-r, r).

    Outputs:
        Tensor, has the same dtype as the `x`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    """
    return reduce_prod_(x, axis)


def concat_first_dim(input_x: Tensor) -> Tensor:
    r"""
    Connect tensor in the first axis (axis=0).

    Connect input tensors along with the first axis.

    Args:
        input_x (Union[tuple, list]): A tuple or a list of input tensors.

    Returns:
        Tensor. A concatenated Tensor with the same type as `input_x`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    """
    return concat_first_dim_(input_x)


def concat_last_dim(input_x: Tensor) -> Tensor:
    r"""
    Connect tensor in the last axis (axis=-1).

    Connect input tensors along with the last axis.

    Args:
        input_x (Union[tuple, list]): A tuple or a list of input tensors.

    Returns:
        Tensor. A concatenated Tensor with the same type as `input_x`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    """
    return concat_last_dim_(input_x)


def concat_penulti(input_x: Tensor) -> Tensor:
    r"""
    Connect tensor in the penultimate axis (axis=-2).

    Connect input tensors along with the penultimate axis.

    Args:
        input_x (Union[tuple, list]): A tuple or a list of input tensors.

    Returns:
        Tensor. A concatenated Tensor with the same type as `input_x`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    """
    return concat_penulti_(input_x)


def stack_first_dim(input_x: Tensor) -> Tensor:
    r"""
    Stacks a list of tensors in the first axis (axis=0).

    Args:
        input_x (Union[tuple, list]): A Tuple or list of Tensor objects with the same shape and type.

    Returns:
        Tensor. A stacked Tensor with the same type as `input_x`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    """
    return stack_first_dim_(input_x)


def stack_last_dim(input_x: Tensor) -> Tensor:
    r"""
    Stacks a list of tensors in the last axis (axis=-1).

    Args:
        input_x (Union[tuple, list]): A Tuple or list of Tensor objects with the same shape and type.

    Returns:
        Tensor. A stacked Tensor with the same type as `input_x`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    """
    return stack_last_dim_(input_x)


def stack_penulti(input_x: Tensor) -> Tensor:
    r"""
    Stacks a list of tensors in the penultimate axis (axis=-2).

    Args:
        input_x (Union[tuple, list]): A Tuple or list of Tensor objects with the same shape and type.

    Returns:
        Tensor. A stacked Tensor with the same type as `input_x`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    """
    return stack_penulti_(input_x)


def squeeze_first_dim(input_x: Tensor) -> Tensor:
    r"""
    Return the Tensor after deleting the dimension of size 1 from the first axis (axis=0).

    Args:
        input_x (Tensor): The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.

    Returns:
        Tensor, the shape of tensor is :math:`(x_2, ..., x_R)`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    """
    return squeeze_first_dim_(input_x)


def squeeze_last_dim(input_x: Tensor) -> Tensor:
    r"""
    Return the Tensor after deleting the dimension of size 1 from the last axis (axis=-1).

    Args:
        input_x (Tensor): The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.

    Returns:
        Tensor, the shape of tensor is :math:`(x_1, x_2, ..., x_{R-1})`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    """
    return squeeze_last_dim_(input_x)


def squeeze_penulti(input_x: Tensor) -> Tensor:
    r"""
    Return the Tensor after deleting the dimension of size 1 from the penultimate axis (axis=-2).

    Args:
        input_x (Tensor): The shape of tensor is :math:`(x_1, x_2, ..., x_{R-1}, x_R)`.

    Returns:
        Tensor, the shape of tensor is :math:`(x_1, x_2, ..., x_R)`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    """
    return squeeze_penulti_(input_x)


def identity(x: Tensor) -> Tensor:
    r"""
    Returns a Tensor with the same shape and contents as input.

    Args:
        x (Tensor): The shape of tensor is :math:`(x_1, x_2, ..., x_R)`. The data type is Number.

    Returns:
        Tensor, the shape of tensor and the data type are the same as `x`, :math:`(x_1, x_2, ..., x_R)`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    """
    return identity_(x)


@jit
def periodic_variable(variable: Tensor,
                      upper: Tensor,
                      lower: Tensor = 0,
                      mask: Tensor = None,
                      ) -> Tensor:
    r"""get the value in the periodic range.

    Args:
        variable (Tensor):  Tensor of shape `(...)`. Data type is float.
                            Periodic variable
        upper (Tensor):     Tensor of shape `(...)`. Data type is float.
                            Upper boundary of perodicity.
        lower (Tensor):     Tensor of shape `(...)`. Data type is float.
                            Lower boundary of perodicity. Default: 0
        mask (Tensor):      Tensor of shape `(...)`. Data type is bool_.
                            Mask for the periodic variable.

    Returns:
        period_value (Tensor), Tensor of shape `(...)`. Data type is float.
        Variable with value in the periodic range.

    Supported Platforms:
        ``Ascend`` ``GPU``

    """

    period = upper - lower
    period_value = variable - period * F.floor((variable - lower) / period)
    if mask is None:
        return period_value
    if mask.shape != variable.shape:
        mask = msnp.broadcast_to(mask, variable.shape)
    return F.select(mask, period_value, variable)


@jit
def periodic_difference(difference: Tensor,
                        period: Tensor,
                        mask: Tensor = None,
                        offset: float = -0.5,
                        ) -> Tensor:
    r"""get value of difference between periodic variables.

    Args:
        variable (Tensor):  Tensor of shape `(...)`. Data type is float.
                            Periodic variable
        period (Tensor):    Tensor of shape `(...)`. Data type is float.
                            Upper boundary of perodicity.
        mask (Tensor):      Tensor of shape `(...)`. Data type is bool_.
                            Mask for the periodic variable.
        offset (float):     Offset ratio :math:`c` with relative to the period :math:`\theta`.
                            Default: -0.5

    Returns:
        period_diff (Tensor), Tensor of shape `(...)`. Data type is float.
        Variable with value in the periodic range.

    Supported Platforms:
        ``Ascend`` ``GPU``

    """

    period_diff = difference - period * F.floor(difference / period - offset)
    if mask is None:
        return period_diff
    if mask.shape != difference.shape:
        mask = msnp.broadcast_to(mask, difference.shape)
    return F.select(mask, period_diff, difference)


@jit
def gather_vector(tensor: Tensor, index: Tensor) -> Tensor:
    r"""Gather vector from the penultimate axis (`axis=-2`) of the tensor according to index.

    Args:
        tensor (Tensor):    Tensor of shape :math:`(B, X, D)`, where :math:`B` is batch size, :math:`X` is an
                            arbitrary value., :math:`D` is spatial dimension of the simulation system, usually is 3.
        index (Tensor):     Tensor of shape :math:`(B, ...,)`. Data type is int.

    Returns:
        vector (Tensor), a tensor of shape :math:`(B, ..., D)`

    Supported Platforms:
        ``Ascend`` ``GPU``
    """

    if index.shape[0] == 1:
        return F.gather(tensor, index[0], -2)
    if tensor.shape[0] == 1:
        return F.gather(tensor[0], index, -2)

    # (B, N, M)
    shape0 = index.shape
    # (B, N * M, 1) <- (B, N, M)
    index = F.reshape(index, (shape0[0], -1, 1))
    # (B, N * M, D) <- (B, N, D)
    vectors = msnp.take_along_axis(tensor, index, axis=-2)
    # (B, N, M, D) <- (B, N, M) + (D,)
    output_shape = shape0 + tensor.shape[-1:]
    # (B, N, M, D)
    return F.reshape(vectors, output_shape)


@jit
def gather_value(tensor: Tensor, index: Tensor) -> Tensor:
    r"""Gather value from the last axis (`axis=-1`) of the tensor according to index.

    Args:
        tensor (Tensor):    Tensor of shape `(B, X)`, where :math:`B` is batch_size,
                            and :math:`X` is an arbitrary value.
        index (Tensor):     Tensor of shape `(B, ...,)`. Data type is int.

    Returns:
        value (Tensor), a tensor of shape `(B, ...,)` .

    Supported Platforms:
        ``Ascend`` ``GPU``
    """

    if index.shape[0] == 1:
        return F.gather(tensor, index[0], -1)
    if tensor.shape[0] == 1:
        return F.gather(tensor[0], index, -1)

    # (B, N, M)
    origin_shape = index.shape
    # (B, N * M) <- (B, N, M)
    index = F.reshape(index, (origin_shape[0], -1))
    # (B, N * M)
    values = F.gather_d(tensor, -1, index)
    # (B, N, M)
    return F.reshape(values, origin_shape)


@jit
def pbc_box_reshape(pbc_box: Tensor, ndim: int) -> Tensor:
    r"""Reshape the pbc_box as the same ndim.

    Args:
        pbc_box (Tensor):   Tensor of shape :math:`(B, D)`. Data type is float.
                            B is batchsize, i.e. number of walkers in simulation.
                            D is spatial dimension of the simulation system. Usually is 3.
        ndim (int):         The rank (number of dimension) of the pbc_box

    Returns:
        pbc_box (Tensor), a tensor of shape :math:`(B, 1, .., 1, D)`. Data type is float.

    Supported Platforms:
        ``Ascend`` ``GPU``

    """

    if ndim <= 2:
        return pbc_box
    shape = pbc_box.shape[:1] + (1,) * (ndim - 2) + pbc_box.shape[-1:]
    return F.reshape(pbc_box, shape)


@jit
def pbc_image(position: Tensor, pbc_box: Tensor, offset: float = 0) -> Tensor:
    r"""calculate the periodic image of the PBC box

    Args:
        position (Tensor):  Tensor of shape :math:`(B, ..., D)`. Data type is float.
                            B is batchsize, i.e. number of walkers in simulation
                            D is spatial dimension of the simulation system. Usually is 3.
        pbc_box (Tensor):   Tensor of shape :math:`(B, D)`. Data type is float.
        offset (float):     Offset ratio :math:`c` relative to box size :math:`\vec{L}`.
                            Default: ``0``

    Returns:
        image (Tensor), a tensor of shape :math:`(B, ..., D)`. Data type is int32.

    Supported Platforms:
        ``Ascend`` ``GPU``

    """

    pbc_box = pbc_box_reshape(F.stop_gradient(pbc_box), position.ndim)
    image = -F.floor(position / pbc_box - offset)
    return F.cast(image, ms.int32)


@jit
def coordinate_in_pbc(position: Tensor, pbc_box: Tensor, offset: float = 0) -> Tensor:
    r"""get coordinate in main PBC box

    Args:
        position (Tensor):  Tensor of shape `(B, ..., D)`. Data type is float.
                            Position coordinate :math:`R`
                            B means batchsize, i.e. number of walkers in simulation.
                            D means spatial dimension of the simulation system. Usually is 3.
        pbc_box (Tensor):   Tensor of shape `(B, D)`. Data type is float.
                            Size of PBC box :math:`\vec{L}`
        offset (float):     Offset ratio :math:`c` relative to box size :math:`\vec{L}`.
                            Default: 0

    Returns:
        coordinate (Tensor), Tensor of shape `(B, ..., D)`. Data type is float.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import mindspore as ms
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from sponge.function import coordinate_in_pbc
        >>> crd = Tensor(np.random.random((4, 3)), ms.float32)
        >>> pbc_box = Tensor([[0.5, 0.5, 0.5]], ms.float32)
        >>> crd
        Tensor(shape=[4, 3], dtype=Float32, value=
        [[ 4.94492769e-01,  4.85243529e-01,  1.63403198e-01],
         [ 5.60526669e-01,  6.17091954e-01,  3.65307808e-01],
         [ 7.81092644e-01,  3.17117482e-01,  1.41929969e-01],
         [ 9.59174633e-01,  3.53236049e-02,  4.85624045e-01]])
        >>> coordinate_in_pbc(crd, pbc_box)
        Tensor(shape=[4, 3], dtype=Float32, value=
        [[ 4.94492769e-01,  4.85243529e-01,  1.63403198e-01],
         [ 6.05266690e-02,  1.17091954e-01,  3.65307808e-01],
         [ 2.81092644e-01,  3.17117482e-01,  1.41929969e-01],
         [ 4.59174633e-01,  3.53236049e-02,  4.85624045e-01]])
    """

    pbc_box = pbc_box_reshape(F.stop_gradient(pbc_box), position.ndim)
    return position - pbc_box * F.floor(position / pbc_box - offset)


@jit
def vector_in_pbc(vector: Tensor, pbc_box: Tensor, offset: float = -0.5) -> Tensor:
    r"""
    Make the value of vector :math:`\vec{v}` at a single PBC box :math:`\vec{L}`.

    Args:
        vector (Tensor):    Tensor of shape `(B, ..., D)`. Data type is float.
                            Vector :math:`\vec{v}`.
                            B means batchsize, i.e. number of walkers in simulation.
                            D means spatial dimension of the simulation system. Usually is 3.
        pbc_box (Tensor):   Tensor of shape `(B, D)`. Data type is float.
                            Size of PBC box :math:`\vec{L}`.
        offset (float):     Offset ratio :math:`c` of the vector relative to box size :math:`\vec{L}`.
                            The value of vector will be between :math:`c \vec{L}` and
                            :math:`(c+1) \vec{L}`. Default: ``-0.5``.

    Returns:
        pbc_vector (Tensor), a tensor of shape `(B, ..., D)`. Data type is float.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """

    pbc_box = pbc_box_reshape(pbc_box, vector.ndim)
    box_nograd = F.stop_gradient(pbc_box)
    inv_box = msnp.reciprocal(box_nograd)
    vector -= box_nograd * F.floor(vector * inv_box - offset)
    return vector * inv_box * pbc_box


@jit
def calc_vector_nopbc(initial: Tensor, terminal: Tensor) -> Tensor:
    r"""Compute vector from initial point to terminal point without perodic bundary condition.

    Args:
        initial (Tensor):   Tensor of shape :math:`(..., D)`, where :math:`D` is the spatial
                            dimension of the simulation system (usually 3). Data type is float.
                            Position coordinate of initial point
        terminal (Tensor):  Tensor of shape :math:`(..., D)`. Data type is float.
                            Position coordinate of terminal point

    Returns:
        vector (Tensor), a tensor of shape :math:`(..., D)`. Data type is float.

    Supported Platforms:
        ``Ascend`` ``GPU``

    """

    return terminal - initial


@jit
def calc_vector_pbc(initial: Tensor, terminal: Tensor, pbc_box: Tensor) -> Tensor:
    r"""Compute vector from initial point to terminal point at perodic bundary condition.

    Args:
        initial (Tensor):   Tensor of shape :math:`(..., D)`, where :math:`D` is the spatial
                            dimension of the simulation system (usually 3). Data type is float.
                            Position coordinate of initial point
        terminal (Tensor):  Tensor of shape :math:`(..., D)`. Data type is float.
                            Position coordinate of terminal point
        pbc_box (Tensor):   Tensor of shape :math:`(D)` or :math:`(B, D)`, where :math:`B` is
                            the batchsize (i.e., number of walkers in simulation). Data type is float.
                            Size of PBC box.

    Returns:
        vector (Tensor), a tensor of shape :math:`(..., D)`. Data type is float.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import mindspore as ms
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from sponge.function import calc_vector_pbc
        >>> crd = Tensor(np.random.random((4, 3)), ms.float32)
        >>> pbc_box = Tensor([[3, 3, 3]], ms.float32)
        >>> calc_vector_pbc(crd[0], crd[1], pbc_box)
        Tensor(shape=[1, 3], dtype=Float32, value=
        [[ 6.60338998e-02,  1.31848425e-01,  2.01904625e-01]])
    """

    return vector_in_pbc(terminal-initial, pbc_box)

@jit
def calc_vector(initial: Tensor, terminal: Tensor, pbc_box: Tensor = None) -> Tensor:
    r"""Compute vector from initial point to terminal point.

    Args:
        initial (Tensor):   Tensor of shape :math:`(..., D)`, where :math:`D` is the spatial
                            dimension of the simulation system (usually 3). Data type is float.
                            Position coordinate of initial point
        terminal (Tensor):  Tensor of shape :math:`(..., D)`. Data type is float.
                            Position coordinate of terminal point.
        pbc_box (Tensor):   Tensor of shape :math:`(D)` or :math:`(B, D)`, where :math:`B` is
                            the batchsize (i.e., number of walkers in simulation). Data type is float.
                            Default: ``None``

    Returns:
        vector (Tensor, a tensor of shape `(..., D)`). Data type is float.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import mindspore as ms
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from sponge.function import calc_vector
        >>> crd = Tensor(np.random.random((4, 3)), ms.float32)
        >>> pbc_box = Tensor([[3, 3, 3]], ms.float32)
        >>> calc_vector(crd[0], crd[1])
        Tensor(shape=[3], dtype=Float32, value= [ 6.60338998e-02,  1.31848425e-01,  2.01904610e-01])
        >>> calc_vector(crd[0], crd[1], pbc_box)
        Tensor(shape=[1, 3], dtype=Float32, value=
        [[ 6.60338998e-02,  1.31848425e-01,  2.01904625e-01]])
    """

    vector = terminal - initial
    if pbc_box is None:
        return vector
    return vector_in_pbc(vector, pbc_box)


@jit
def calc_distance_nopbc(position_a: Tensor,
                        position_b: Tensor,
                        keepdims: bool = False,
                        ) -> Tensor:
    r"""Compute distance between position A and B without perodic bundary condition.

    Args:
        position_a (Tensor):    Tensor of shape :math:`(..., D)`. Data type is float.
                                D is spatial dimension of the simulation system. Usually is 3.
                                Position coordinate of point :math:`A`.
        position_b (Tensor):    Tensor of shape :math:`(..., D)`. Data type is float.
                                Position coordinate of point :math:`B`.
        keepdims (bool):        If this is set to ``True``, the last axis will be left
                                in the result as dimensions with size one.
                                Default: ``False``.

    Returns:
        distance (Tensor), a tensor of shape :math:`(...)` or :math:`(..., 1)`. Data type is float.

    Supported Platforms:
        ``Ascend`` ``GPU``

    """

    vec = calc_vector_nopbc(position_a, position_b)
    return msnp.norm(vec, axis=-1, keepdims=keepdims)


@jit
def calc_distance_pbc(position_a: Tensor,
                      position_b: Tensor,
                      pbc_box: Tensor = None,
                      keepdims: bool = False
                      ) -> Tensor:
    r"""Compute distance between position :math:`A` and :math:`B` at perodic bundary condition.

    Args:
        position_a (Tensor):    Tensor of shape :math:`(..., D)`. Data type is float.
                                D means spatial dimension of the simulation system. Usually is 3.
                                Position coordinate of point :math:`A`.
        position_b (Tensor):    Tensor of shape :math:`(..., D)`. Data type is float.
                                Position coordinate of point :math:`B`.
        pbc_box (Tensor):       Tensor of shape :math:`(D)` or :math:`(B, D)`. Data type is float.
                                B means batchsize, i.e. number of walkers in simulation
                                Size of PBC box :math:`\vec{L}`
        keepdims (bool):        If this is set to ``True``, the last axis will be left
                                in the result as dimensions with size one.
                                Default: ``False``.

    Returns:
        distance (Tensor), a tensor of shape :math:`(...)` or :math:`(..., 1)`. Data type is float.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore import Tensor
        >>> from sponge.function import calc_distance_pbc
        >>> tensor_a = Tensor([[1, 2, 3], [4, 5, 6]], ms.float32)
        >>> tensor_b = Tensor([[1, 2, 4], [6, 8, 10]], ms.float32)
        >>> pbc_box = Tensor([[3, 3, 3]], ms.float32)
        >>> calc_distance_pbc(tensor_a, tensor_b, pbc_box)
        Tensor(shape=[2], dtype=Float32, value= [ 1.00000000e+00,  1.41421354e+00])
    """

    vec = calc_vector_pbc(position_a, position_b, pbc_box)
    return msnp.norm(vec, axis=-1, keepdims=keepdims)


@jit
def calc_distance(position_a: Tensor,
                  position_b: Tensor,
                  pbc_box: Tensor = None,
                  keepdims: bool = False,
                  ) -> Tensor:
    r"""Compute distance between position :math:`A` and :math:`B`.

    Args:
        position_a (Tensor):    Tensor of shape :math:`(..., D)`. Data type is float.
                                Position coordinate of point :math:`A`.
        position_b (Tensor):    Tensor of shape :math:`(..., D)`. Data type is float.
                                Position coordinate of point :math:`B`.
        pbc_box (Tensor):       Tensor of shape :math:`(D)` or :math:`(B, D)`. Data type is float.
                                Size of PBC box :math:`\vec{L}`
        keepdims (bool):        If this is set to ``True`` , the last axis will be left
                                in the result as dimensions with size one.
                                Default: ``False`` .

    Returns:
        distance (Tensor), a tensor of shape :math:`(...)` or :math:`(..., 1)` . Data type is float.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore import Tensor
        >>> from sponge.function import calc_distance
        >>> tensor_a = Tensor([[1, 2, 3], [4, 5, 6]], ms.float32)
        >>> tensor_b = Tensor([[1, 2, 4], [6, 8, 10]], ms.float32)
        >>> calc_distance(tensor_a, tensor_b)
        Tensor(shape=[2], dtype=Float32, value= [ 1.00000000e+00,  5.38516474e+00])
        >>> pbc_box = Tensor([[3, 3, 3]], ms.float32)
        >>> calc_distance(tensor_a, tensor_b, pbc_box)
        >>> Tensor(shape=[2], dtype=Float32, value= [ 1.00000000e+00,  1.41421354e+00])
    """

    vec = calc_vector_nopbc(position_a, position_b)
    if pbc_box is not None:
        vec = vector_in_pbc(vec, pbc_box)
    return msnp.norm(vec, axis=-1, keepdims=keepdims)


@jit
def calc_angle_by_vectors(vector1: Tensor,
                          vector2: Tensor,
                          keepdims: bool = False
                          ) -> Tensor:
    r"""
    Compute angle between two vectors.
    For vector :math:`\vec {v_1} = (x_1, x_2, x_3, ..., x_n)` and
    :math:`\vec {v_2} = (y_1, y_2, y_3, ..., y_n)` , the formula is

    .. math::

        \theta = \arccos {\frac{|x_1y_1 + x_2y_2 + \cdots + x_ny_n|}{\sqrt{x_1^2 + x_2^2 +
                 \cdots + x_n^2}\sqrt{y_1^2 + y_2^2 + \cdots + y_n^2}}}

    Args:
        vector1 (Tensor):    Tensor of shape :math:`(..., D)`. Data type is float.
                             D means spatial dimension of the simulation system. Usually is 3.
                             Vector of :math:`\vec{v_1}`.
        vector2 (Tensor):    Tensor of shape :math:`(..., D)`. Data type is float.
                             Vector of :math:`\vec{v_2}`.
        keepdims (bool):     If this is set to True, the last axis will be left
                             in the result as dimensions with size one.
                             Default: ``False``.

    Returns:
        angle (Tensor), a tensor of shape :math:`(...)` or :math:`(..., 1)`. Data type is float.

    Supported Platforms:
        ``Ascend`` ``GPU``

    """

    # (...) or (..., 1) <- (..., D)
    dis1 = msnp.norm(vector1, axis=-1, keepdims=keepdims)
    dis2 = msnp.norm(vector2, axis=-1, keepdims=keepdims)
    dot12 = msnp.sum(vector1*vector2, axis=-1, keepdims=keepdims)
    # (...) or (..., 1)
    cos_theta = dot12 / dis1 / dis2
    return F.acos(cos_theta)


@jit
def calc_angle_nopbc(position_a: Tensor,
                     position_b: Tensor,
                     position_c: Tensor,
                     keepdims: bool = False,
                     ) -> Tensor:
    r"""
    Compute angle :math:`\angle{ABC}` formed by the position coordinates of three positions
    :math:`A`, :math:`B` and :math:`C` without periodic boundary condition.

    Args:
        position_a (Tensor):    Tensor of shape `(..., D)`. Data type is float.
                                D means spatial dimension of the simulation system. Usually is 3.
                                Position coordinate of point :math:`A`.
        position_b (Tensor):    Tensor of shape `(..., D)`. Data type is float.
                                Position coordinate of point :math:`B`.
        position_c (Tensor):    Tensor of shape `(..., D)`. Data type is float.
                                Position coordinate of point :math:`C`.
        keepdims (bool):        If this is set to True, the last axis will be left
                                in the result as dimensions with size one.
                                Default: ``False``.

    Returns:
        angle (Tensor), a tensor of shape `(...)` or `(..., 1)`. Data type is float.
        Value of angle :math:`\angle{ABC}`.


    Supported Platforms:
        ``Ascend`` ``GPU``

    """

    # (...,D)
    vec_ba = calc_vector_nopbc(position_b, position_a)
    vec_bc = calc_vector_nopbc(position_b, position_c)
    return calc_angle_by_vectors(vec_ba, vec_bc, keepdims=keepdims)


@jit
def calc_angle_pbc(position_a: Tensor,
                   position_b: Tensor,
                   position_c: Tensor,
                   pbc_box: Tensor,
                   keepdims: bool = False,
                   ) -> Tensor:
    r"""
    Compute angle :math:`\angle{ABC}` formed by the position coordinates of three positions
    :math:`A`, :math:`B` and :math:`C` at periodic boundary condition.

    Args:
        position_a (Tensor):    Tensor of shape :math:`(..., D)` . Data type is float.
                                D means spatial dimension of the simulation system. Usually is 3.
                                Position coordinate of point :math:`A`.
        position_b (Tensor):    Tensor of shape :math:`(..., D)` . Data type is float.
                                Position coordinate of point :math:`B`.
        position_c (Tensor):    Tensor of shape :math:`(..., D)` . Data type is float.
                                Position coordinate of point :math:`C`.
        pbc_box (Tensor):       Tensor of shape :math:`(D)` or :math:`(B, D)` . Data type is float.
                                B means batchsize, i.e. number of walkers in simulation
                                Size of PBC box :math:`\vec{L}`
        keepdims (bool):        If this is set to True, the last axis will be left
                                in the result as dimensions with size one.
                                Default: ``False`` .

    Returns:
        angle (Tensor), a tensor of shape :math:`(...)` or :math:`(..., 1)` . Data type is float.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore import Tensor
        >>> import numpy as np
        >>> from sponge.function import calc_angle_pbc
        >>> crd = Tensor(np.random.random((3, 3)), ms.float32)
        >>> pbc_box = Tensor([[3, 3, 3]], ms.float32)
        >>> calc_angle_pbc(crd[0], crd[1], crd[2], pbc_box)
        >>> Tensor(shape=[1], dtype=Float32, value= [ 6.17621064e-01])
    """

    # (B, ..., D)
    vec_ba = calc_vector_pbc(position_b, position_a, pbc_box)
    vec_bc = calc_vector_pbc(position_b, position_c, pbc_box)
    return calc_angle_by_vectors(vec_ba, vec_bc, keepdims=keepdims)


@jit
def calc_angle(position_a: Tensor,
               position_b: Tensor,
               position_c: Tensor,
               pbc_box: Tensor = None,
               keepdims: bool = False,
               ) -> Tensor:
    r"""
    Compute angle formed by three positions :math:`A`, :math:`B` and :math:`C`
    with or without periodic boundary condition.

    Args:
        position_a (Tensor):    Tensor of shape :math:`(..., D)` . Data type is float.
                                D means spatial dimension of the simulation system. Usually is 3.
                                Position coordinate of point :math:`A`.
        position_b (Tensor):    Tensor of shape :math:`(..., D)` . Data type is float.
                                Position coordinate of point :math:`B`.
        position_c (Tensor):    Tensor of shape :math:`(..., D)` . Data type is float.
                                Position coordinate of point :math:`C`.
        pbc_box (Tensor):       Tensor of shape :math:`(D)` or :math:`(B, D)` . Data type is float.
                                B means batchsize, i.e. number of walkers in simulation
                                Size of PBC box :math:`\vec{L}`. Default: ``None``.
        keepdims (bool):        If this is set to True, the last axis will be left
                                in the result as dimensions with size one.
                                Default: ``False``.

    Returns:
        angle (Tensor), a tensor of shape :math:`(...)` or :math:`(..., 1)`. Data type is float.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore import Tensor
        >>> import numpy as np
        >>> from sponge.function import calc_angle
        >>> crd = Tensor(np.random.random((3, 3)), ms.float32)
        >>> pbc_box = Tensor([[3, 3, 3]], ms.float32)
        >>> calc_angle(crd[0], crd[1], crd[2], pbc_box)
        >>> Tensor(shape=[1], dtype=Float32, value= [ 6.17621064e-01])
    """

    if pbc_box is None:
        return calc_angle_nopbc(position_a, position_b, position_c, keepdims=keepdims)
    return calc_angle_pbc(position_a, position_b, position_c, pbc_box=pbc_box, keepdims=keepdims)


@jit
def calc_torsion_by_vectors(vector1: Tensor,
                            vector2: Tensor,
                            axis_vector: Tensor = None,
                            keepdims: bool = False,
                            ) -> Tensor:
    r"""
    Compute torsion angle formed by two direction vectors :math:`\vec{v_1}` and :math:`\vec{v_2}`
    and an axis vector :math:`\vec{v_{axis}}`.

    Args:
        vector1 (Tensor):       Tensor of shape :math:`(..., D)`. Data type is float.
                                D is spatial dimension of the simulation system. Usually is 3.
                                Direction vector :math:`\vec{v_1}`
        vector2 (Tensor):       Tensor of shape :math:`(..., D)`. Data type is float.
                                Direction vector :math:`\vec{v_2}`
        axis_vector (Tensor):   Tensor of shape :math:`(..., D)`. Data type is float.
                                Axis vector :math:`\vec{v_{axis}}`.
                                Default: ``None``.
        keepdims (bool):        If this is set to True, the last axis will be left
                                in the result as dimensions with size one.
                                Default: ``False``.

    Returns:
        torsion (Tensor), a tensor of shape :math:`(...)` or :math:`(..., 1)`. Data type is float.

    Supported Platforms:
        ``Ascend`` ``GPU``

    """

    if axis_vector is None:
        return calc_angle_by_vectors(vector1, vector2, keepdims=keepdims)

    # (..., D)
    vec_a = msnp.cross(vector1, axis_vector)
    vec_b = msnp.cross(vector2, axis_vector)
    cross_ab = msnp.cross(vec_a, vec_b)

    # (..., 1) <- (..., D)
    axis_norm = msnp.norm(axis_vector, axis=-1, keepdims=True)
    # (..., D) = (..., D) / (..., 1)
    axis_vector *= msnp.reciprocal(axis_norm)

    # (...) or (..., 1)
    sin_phi = msnp.sum(axis_vector*cross_ab, axis=-1, keepdims=keepdims)
    cos_phi = msnp.sum(vec_a*vec_b, axis=-1, keepdims=keepdims)

    return F.atan2(sin_phi, cos_phi)


@jit
def calc_torsion_nopbc(position_a: Tensor,
                       position_b: Tensor,
                       position_c: Tensor,
                       position_d: Tensor,
                       keepdims: bool = False,
                       ) -> Tensor:
    r"""
    Compute torsion angle `A-B-C-D` formed by four positions :math:`A`, :math:`B`,
    :math:`C` and :math:`D` without periodic boundary condition.

    Args:
        position_a (Tensor):    Tensor of shape :math:`(..., D)`. Data type is float.
                                D is spatial dimension of the simulation system. Usually is 3.
                                Position coordinate of point :math:`A`.
        position_b (Tensor):    Tensor of shape :math:`(..., D)`. Data type is float.
                                Position coordinate of point :math:`B`.
        position_c (Tensor):    Tensor of shape :math:`(..., D)`. Data type is float.
                                Position coordinate of point :math:`C`.
        position_d (Tensor):    Tensor of shape :math:`(..., D)`. Data type is float.
                                Position coordinate of point :math:`D`.
        keepdims (bool):        If this is set to ``True``, the last axis will be left
                                in the result as dimensions with size one.
                                Default: ``False``.

    Returns:
        torsion (Tensor), a tensor of shape :math:`(...)` or :math:`(..., 1)`. Data type is float.

    Supported Platforms:
        ``Ascend`` ``GPU``

    """

    vec_ba = calc_vector_nopbc(position_b, position_a)
    vec_cd = calc_vector_nopbc(position_c, position_d)
    vec_bc = calc_vector_nopbc(position_b, position_c)
    return calc_torsion_by_vectors(vec_ba, vec_cd, axis_vector=vec_bc, keepdims=keepdims)


@jit
def calc_torsion_pbc(position_a: Tensor,
                     position_b: Tensor,
                     position_c: Tensor,
                     position_d: Tensor,
                     pbc_box: Tensor,
                     keepdims: bool = False,
                     ) -> Tensor:
    r"""
    Compute torsion angle `A-B-C-D` formed by four positions :math:`A`, :math:`B`,
    :math:`C` and :math:`D` at periodic boundary condition.

    Args:
        position_a (Tensor):    Tensor of shape :math:`(..., D)`. Data type is float.
                                D is spatial dimension of the simulation system. Usually is 3.
                                Position coordinate of point :math:`A`.
        position_b (Tensor):    Tensor of shape :math:`(..., D)`. Data type is float.
                                Position coordinate of point :math:`B`.
        position_c (Tensor):    Tensor of shape :math:`(..., D)`. Data type is float.
                                Position coordinate of point :math:`C`.
        position_d (Tensor):    Tensor of shape :math:`(..., D)`. Data type is float.
                                Position coordinate of point :math:`D`.
        pbc_box (Tensor):       Tensor of shape :math:`(D)` or :math:`(B, D)`. Data type is float.
                                B is batchsize, i.e. number of walkers in simulation
                                Size of PBC box :math:`\vec{L}`.
        keepdims (bool):        If this is set to ``True``, the last axis will be left
                                in the result as dimensions with size one.
                                Default: ``False``.

    Returns:
        torsion (Tensor), a tensor of shape :math:`(...)` or :math:`(..., 1)`. Data type is float.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import mindspore as ms
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from sponge.function import calc_torsion_pbc
        >>> crd = Tensor(np.random.random((4, 3)), ms.float32)
        >>> pbc_box = Tensor([[3, 3, 3]], ms.float32)
        >>> calc_torsion_pbc(crd[0], crd[1], crd[2], crd[3], pbc_box)
        Tensor(shape=[1], dtype=Float32, value= [-2.33294296e+00])
    """

    vec_ba = calc_vector_pbc(position_b, position_a, pbc_box)
    vec_cd = calc_vector_pbc(position_c, position_d, pbc_box)
    vec_bc = calc_vector_pbc(position_b, position_c, pbc_box)
    return calc_torsion_by_vectors(vec_ba, vec_cd, axis_vector=vec_bc, keepdims=keepdims)


@jit
def calc_torsion(position_a: Tensor,
                 position_b: Tensor,
                 position_c: Tensor,
                 position_d: Tensor,
                 pbc_box: Tensor = None,
                 keepdims: bool = False,
                 ) -> Tensor:

    r"""
    Compute torsion angle :math:`A-B-C-D` formed by four positions :math:`A`, :math:`B`, :math:`C` and :math:`D`
    with or without periodic boundary condition.

    Args:
        position_a (Tensor):    Tensor of shape :math:`(..., D)`. Data type is float.
                                D is spatial dimension of the simulation system. Usually is 3.
                                Position coordinate of point :math:`A`.
        position_b (Tensor):    Tensor of shape :math:`(..., D)`. Data type is float.
                                Position coordinate of point :math:`B`.
        position_c (Tensor):    Tensor of shape :math:`(..., D)`. Data type is float.
                                Position coordinate of point :math:`C`.
        position_d (Tensor):    Tensor of shape :math:`(..., D)`. Data type is float.
                                Position coordinate of point :math:`D`.
        pbc_box (Tensor):       Tensor of shape :math:`(D)` or :math:`(B, D)`. Data type is float.
                                B is batchsize, i.e. number of walkers in simulation.
                                Size of PBC box :math:`\vec{L}`. Default: ``None``.
        keepdims (bool):        If this is set to ``True`` , the last axis will be left
                                in the result as dimensions with size one.
                                Default: ``False`` .

    Returns:
        torsion (Tensor), a tensor of shape :math:`(...)` or :math:`(..., 1)`. Data type is float.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import mindspore as ms
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from sponge.function import calc_torsion
        >>> crd = Tensor(np.random.random((4, 3)), ms.float32)
        >>> pbc_box = Tensor([[3, 3, 3]], ms.float32)
        >>> calc_torsion(crd[0], crd[1], crd[2], crd[3], pbc_box)
        Tensor(shape=[1], dtype=Float32, value= [-2.33294296e+00])
    """

    if pbc_box is None:
        return calc_torsion_nopbc(
            position_a, position_b, position_c, position_d, keepdims=keepdims)

    return calc_torsion_pbc(
        position_a, position_b, position_c, position_d, pbc_box=pbc_box, keepdims=keepdims)


@jit
def coulomb_interaction(q_i: Tensor,
                        q_j: Tensor,
                        r_ij: Tensor,
                        mask: Tensor = None,
                        coulomb_const: float = 1,
                        ):
    r"""Calculate Coulomb interaction.

    Math:

    .. math::

        E_{coulomb}(r_{ij}) = k \frac{q_i q_j}{r_{ij}}

    Args:
        q_i (Tensor):           Tensor of shape `(...)`. Data type is float.
                                Charge of the :math:`i`-th atom :math:`q_i`.
        q_j (Tensor):           Tensor of shape `(...)`. Data type is float.
                                Charge of the :math:`j`-th atom :math:`q_j`.
        r_ij (Tensor):          Tensor of shape `(...)`. Data type is float.
                                Distance :math:`r_{ij}` between atoms :math:`i` and :math:`i`.
        mask (Tensor):          Tensor of shape `(...)`. Data type is bool.
                                Mask for distance :math:`r_{ij}`. Default: ``None``.
        coulomb_const (float):  Coulomb constant :math:`k`. Default: 1

    Returns:
        E_coulomb (Tensor), Tensor of shape `(...)`. Data type is float.

    Supported Platforms:
        ``Ascend`` ``GPU``

    """

    energy = coulomb_const * q_i * q_j * msnp.reciprocal(r_ij)

    if mask is None:
        return energy

    return energy * mask


@jit
def lennard_jones_potential(epsilon: Tensor, sigma: Tensor, r_ij: Tensor, mask: Tensor = None) -> Tensor:
    r"""Calculate Lennard-Jones (LJ) potential with :math:`\epsilon` and :math:`\sigma`.

    Math:

    .. math::

        E_{lj}(r_{ij}) = 4 \epsilon \left [\left ( \frac{\sigma}{r_{ij}} \right ) ^{12} -
                                           \left ( \frac{\sigma}{r_{ij}} \right ) ^{6} \right]

    Args:
        epsilon (Tensor):   Tensor of shape `(...)`. Data type is float.
                            Well depth :math:`\epsilon`.
        sigma (Tensor):     Tensor of shape `(...)`. Data type is float.
                            Characteristic distance :math:`\sigma`.
        r_ij (Tensor):      Tensor of shape `(...)`. Data type is float.
                            Distance :math:`r_{ij}` between atoms :math:`i` and :math:`i`.
        mask (Tensor):      Tensor of shape `(...)`. Data type is bool.
                            Mask for distances :math:`r_{ij}`. Default: ``None``.

    Returns:
        E_coulomb (Tensor), Tensor of shape (...). Data type is float.

    Supported Platforms:
        ``Ascend`` ``GPU``

    """
    # (\sigma / r_{ij}) ^ 6
    r0_6 = F.pows(sigma * msnp.reciprocal(r_ij), 6)
    # 4 * \epsilon * (\sigma / r_{ij}) ^ 6
    ene_bcoeff = 4 * epsilon * r0_6
    # 4 * \epsilon * (\sigma / r_{ij}) ^ 12
    ene_acoeff = ene_bcoeff * r0_6

    energy = ene_acoeff - ene_bcoeff

    if mask is None:
        return energy
    return energy * mask


@jit
def lennard_jones_potential2(epsilon: Tensor, r_0: Tensor, r_ij: Tensor, mask: Tensor = None) -> Tensor:
    r"""Calculate Lennard-Jones (LJ) potential with :math:`\epsilon` and :math:`r_0`.

    Math:

    .. math::

        E_{lj}(r_{ij}) = 4 \epsilon \left [\frac{1}{4} \left ( \frac{r_0}{r_{ij}} \right ) ^{12} -
                                           \frac{1}{2} \left ( \frac{r_0}{r_{ij}} \right ) ^{6} \right]

    Args:
        epsilon (Tensor):   Tensor of shape `(...)`. Data type is float.
                            Well depth :math:`\epsilon`.
        r_0 (Tensor):       Tensor of shape `(...)`. Data type is float.
                            Atomic radius :math:`r_0`.
        r_ij (Tensor):      Tensor of shape `(...)`. Data type is float.
                            Distance :math:`r_{ij}` between atoms :math:`i` and :math:`i`.
        mask (Tensor):      Tensor of shape `(...)`. Data type is bool.
                            Mask for distances :math:`r_{ij}`. Default: ``None``.
    Returns:
        E_coulomb (Tensor), Tensor of shape `(...)`. Data type is float.

    Supported Platforms:
        ``Ascend`` ``GPU``

    """
    # (\r_0 / r_{ij}) ^ 6
    r0_6 = F.pows(r_0 * msnp.reciprocal(r_ij), 6)
    # 2 * \epsilon * (r_0 / r_{ij}) ^ 6
    ene_bcoeff = 2 * epsilon * r0_6
    # \epsilon * (r_0 / r_{ij}) ^ 12
    ene_acoeff = epsilon * r0_6 * r0_6

    energy = ene_acoeff - ene_bcoeff

    if mask is None:
        return energy
    return energy * mask


def get_integer(value: Union[int, Tensor, Parameter, ndarray]) -> int:
    r"""get integer type of the input value

    Args:
        value (Union[int, Tensor, Parameter, ndarray]): Input value

    Returns:
        integer (int)

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    """
    if value is None:
        return None
    if isinstance(value, Tensor):
        value = value.asnumpy()
    return int(value)


def get_ndarray(value: Union[Tensor, Parameter, ndarray, List[float], Tuple[float]],
                dtype: type = None) -> ndarray:
    r"""get ndarray type of the input value

    Args:
        value (Union[Tensor, Parameter, ndarray, List[float], Tuple[float]]):  Input value
        dtype (type):                               Data type. Default: ``None``.

    Returns:
        array (ndarray)

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    """
    if value is None:
        return None
    if isinstance(value, (Tensor, Parameter)):
        value = value.asnumpy()
        if dtype is not None:
            value = value.astype(dtype)
    else:
        value = np.array(value, dtype)
    return value


def get_tensor(value: Union[float, int, Tensor, Parameter, ndarray, List[float], Tuple[float]],
               dtype: type = None) -> Tensor:
    r"""get mindspore.Tensor type of the input value

    Args:
        value (Union[float, int, Tensor, Parameter, ndarray, list, tuple]):
                        Input value
        dtype (type):   Data type. Default: ``None``.

    Returns:
        tensor (Tensor)

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    """
    if value is None:
        return None

    if isinstance(value, ndarray):
        value = Tensor(value, dtype)
    elif isinstance(value, Parameter):
        value = identity(value)
    elif not isinstance(value, Tensor):
        (value,) = _to_tensor((value,))

    if dtype is not None and dtype != value.dtype:
        value = F.cast(value, dtype)

    return value


def get_ms_array(value: Union[float, int, Tensor, Parameter, ndarray, list, tuple],
                 dtype: type = None
                 ) -> Union[Tensor, Parameter]:
    r"""get mindspore.Tensor type of the input value

    Args:
        value (Union[float, int, Tensor, Parameter, ndarray, list, tuple]):
                        Input value
        dtype (type):   Data type. Default: ``None``.

    Returns:
        array (Tensor or Parameter)

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    """

    if value is None:
        return None

    if isinstance(value, ndarray):
        value = Tensor(value, dtype)
    elif not isinstance(value, (Tensor, Parameter)):
        (value,) = _to_tensor((value,))

    if dtype is not None and value.dtype != dtype:
        value = F.cast(value, dtype)

    return value


def check_broadcast(shape0: tuple, shape1: tuple) -> tuple:
    r"""Check whether the two shapes match the rule of broadcast.

    Args:
        shape0 (tuple): First shape
        shape1 (tuple): Second shape

    Returns:
        shape (tuple), Shape after broadcast

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    """
    if shape0 is None:
        return shape1
    if shape1 is None:
        return shape0

    if len(shape0) < len(shape1):
        shape0 = (1,) * (len(shape1) - len(shape0)) + shape0
    if len(shape0) > len(shape1):
        shape1 = (1,) * (len(shape0) - len(shape1)) + shape1

    shape = ()
    for s0, s1 in zip(shape0, shape1):
        if s0 == s1:
            s = s0
        else:
            if s0 == 1:
                s = s1
            elif s1 == 1:
                s = s0
            else:
                raise ValueError(f'{shape0} and {shape1} cannot be broadcast to each other!')
        shape += (s,)
    return shape


def any_none(iterable: Iterable) -> bool:
    r"""Return True if ANY values x in the iterable is None.

    Args:
        iterable (Iterable): Iterable variable

    Returns:
        any (bool), If any values x in the iterable is None

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    """
    return any([i is None for i in iterable])


def all_none(iterable: Iterable) -> bool:
    r"""Return True if ALL values `x` in the `iterable` is None..

    Args:
        iterable (Iterable): Iterable variable

    Returns:
        all (bool), If all values `x` in the `iterable` is None

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    """
    return all([i is None for i in iterable])


def any_not_none(iterable: Iterable) -> bool:
    r"""Return True if ANY values `x` in the `iterable` is NOT None.

    Args:
        iterable (Iterable): Iterable variable

    Returns:
        any (bool), If any values `x` in the `iterable` is not None

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    """
    return any([i is not None for i in iterable])


def all_not_none(iterable: Iterable) -> bool:
    r"""Return True if ALL values `x` in the `iterable` is Not None.

    Args:
        iterable (Iterable): Iterable variable

    Returns:
        all (bool), If all values `x` in the `iterable` is Not None

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    """
    return all([i is not None for i in iterable])


def get_arguments(locals_: dict, kwargs: dict = None) -> dict:
    r"""get arguments of a class.

    Args:
        locals\_ (dict): Dictionary of the arguments from `locals()`.
        kwargs (dict): Dictionary of keyword arguments (kwargs) of the class.

    Returns:
        args (dict), Dictionary of arguments.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    """

    if '__class__' in locals_.keys():
        locals_.pop('__class__')

    arguments = {}
    if 'self' in locals_.keys():
        cls = locals_.pop('self')
        arguments['cls_name'] = cls.__class__.__name__

    def _set_arguments(args_: dict):
        def _convert(value):
            if value is None or isinstance(value, (int, float, bool, str,
                                                   time, timedelta, date)):
                return value
            if isinstance(value, ndarray):
                return value.tolist()
            if isinstance(value, (Tensor, Parameter)):
                return value.asnumpy().tolist()
            if isinstance(value, (list, tuple)):
                return [_convert(v) for v in value]
            if isinstance(value, dict):
                if 'cls_name' in value.keys():
                    return value
                dict_ = value.copy()
                for k, v in value.items():
                    dict_[k] = _convert(v)
                return dict_

            cls_name = value.__class__.__name__
            if hasattr(value, '_kwargs'):
                value = value.__dict__['_kwargs']
            elif hasattr(value, 'init_args'):
                value = value.__dict__['init_args']
            else:
                value = value.__class__.__name__

            if isinstance(value, dict) and 'cls_name' not in value.keys():
                dict_ = {'cls_name': cls_name}
                dict_.update(_set_arguments(value))
                value = dict_

            return value

        for k, v in args_.items():
            args_[k] = _convert(v)
        return args_

    kwargs_ = {}
    if 'kwargs' in locals_.keys():
        kwargs_: dict = locals_.pop('kwargs')

    if kwargs is None:
        kwargs = kwargs_

    if 'cls_name' in kwargs.keys():
        kwargs.pop('cls_name')

    arguments.update(_set_arguments(locals_))
    arguments.update(_set_arguments(kwargs))

    return arguments


def get_initializer(cls_name: Union[Initializer, str, dict, Tensor], **kwargs) -> Initializer:
    r"""get initializer by name.

    Args:
        cls_name (Union[Initializer, str, dict, Tensor]): Class name of Initializer.
        kwargs (dict): Dictionary of keyword arguments (kwargs) of the class.

    Returns:
        Initializer.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    """
    if isinstance(cls_name, Initializer):
        return cls_name

    if isinstance(cls_name, (Tensor, Parameter, ndarray)):
        return get_tensor(cls_name, ms.float32)

    if isinstance(cls_name, dict):
        return get_initializer(**cls_name)

    if isinstance(cls_name, str):
        init = _INITIALIZER_ALIAS.get(cls_name.lower())
        if init is None:
            raise ValueError(f"For 'initializer', the class corresponding to '{cls_name}' was not found.")
        return init(**kwargs)

    raise TypeError(f'The cls_name must be Initializer, str, dict or Tensor but got: {init}')


def _bonds_in(bonds, bond):
    """ Check bonds exists in both sets. """
    return (bonds == bond).all(-1)


def bonds_in(bonds, batch_bond):
    """ Return if batch_bond exists in bonds.

    Args:
        bonds (Tensor): The total bonds set.
        batch_bond (Tensor): The input bond set.

    Returns:
        If batch_bond exists in bonds, the mask will be 1, else 0.
    """
    return ops.vmap(_bonds_in, in_axes=(None, -2), out_axes=0)(bonds, batch_bond).sum(axis=-3)
