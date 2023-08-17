# Copyright 2021-2023 @ Shenzhen Bay Laboratory &
#                       Peking University &
#                       Huawei Technologies Co., Ltd
#
# This code is a part of AIchemist package.
#
# The AIchemist is open-source software based on the AI-framework:
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
scatter
"""

from typing import Optional

import numpy as np
import mindspore as ms
from mindspore import ops
from .operator import flatten


def broadcast(index: ms.Tensor, src: ms.Tensor, axis: int) -> ms.Tensor:
    """
    Broadcast the index tensor to obtain the detailed information for the scatter operators.

    Args:
        src (ms.Tensor): The source tensor.
        index (ms.Tensor): The indices of elements to scatter.
        axis (ms.Tensor): The axis along which to index.

    Returns:
        ix (ms.Tensor): the indices with detailed information.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    if axis < 0:
        axis += src.dim()
    if index.dim() == 1:
        for _ in range(axis):
            index = index.expand_dims(0)
    for _ in range(index.dim(), axis+1):
        index = index.expand_dims(-1)
    shape = src.shape[:axis+1]
    index = index.broadcast_to(shape)

    ix = ms.Tensor(np.transpose(np.nonzero(index.asnumpy() > -1)))
    index = index.reshape(-1)
    ix[:, axis] = index
    return ix


def scatter(src: ms.Tensor, index: ms.Tensor, axis: int = -1,
            out: Optional[ms.Tensor] = None,
            n_axis: Optional[int] = None, reduce: str = 'update') -> ms.Tensor:
    r"""
    Reduces all values from the :attr:`src` tensor into :attr:`out` at the
    indices specified in the :attr:`index` tensor along a given axis
    :attr:`axis`.
    For each value in :attr:`src`, its output index is specified by its index
    in :attr:`src` for dimensions outside of :attr:`axis` and by the
    corresponding value in :attr:`index` for dimension :attr:`axis`.
    The applied reduction is defined via the :attr:`reduce` argument.
    Formally, if :attr:`src` and :attr:`index` are :math:`n`-dimensional
    tensors with size :math:`(x_0, ..., x_{i-1}, x_i, x_{i+1}, ..., x_{n-1})`
    and :attr:`axis` = `i`, then :attr:`out` must be an :math:`n`-dimensional
    tensor with size :math:`(x_0, ..., x_{i-1}, y, x_{i+1}, ..., x_{n-1})`.
    Moreover, the values of :attr:`index` must be between :math:`0` and
    :math:`y - 1`, although no specific ordering of indices is required.
    The :attr:`index` tensor supports broadcasting in case its dimensions do
    not match with :attr:`src`.
    For one-dimensional tensors with :obj:`reduce="sum"`, the operation
    computes
    .. math::
        \mathrm{out}_i = \mathrm{out}_i + \sum_j~\mathrm{src}_j
    where :math:`\sum_j` is over :math:`j` such that
    :math:`\mathrm{index}_j = i`.

    Args:
        src (ms.Tensor): The source tensor.
        index (ms.Tensor): The indices of elements to scatter.
        dim (ms.Tensor): The axis along which to index. (default: :obj:`-1`)
        out (ms.Tensor): The destination tensor.
        n_axis (int): If :attr:`out` is not given, automatically create output
            with size :attr:`n_axis` at axisension :attr:`axis`.
            If :attr:`n_axis` is not given, a minimal sized output tensor
            according to :obj:`index.max() + 1` is returned.
        reduce (str): The reduce operation (:obj:`"add"`, :obj:`"mul"`,
            :obj:`"mean"`, :obj:`"min"` or :obj:`"max"`, :obj:`"max"`). (default: :obj:`"update"`)

    Outputs:
        out (ms.Tensor): The output tensor.

    Examples:
        >>> from ever import scatter
        >>> from mindspore import numpy as np
        >>> src = np.randn(10, 6, 64)
        >>> index = np.array([0, 1, 0, 1, 2, 1])
        # Broadcasting in the first and last axis.
        >>> out = scatter(src, index, axis=1, reduce="add")
        >>> print(out.shape)
            Tensor([10, 3, 64])

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    index = broadcast(index, src, axis=axis)
    assert reduce in ['update', 'min', 'max', 'add', 'mul', 'mean', 'div']
    if reduce == 'mean':
        fn = getattr(ops, 'tensor_scatter_add')
    else:
        fn = getattr(ops, 'TensorScatter' + reduce.capitalize())()
    if out is None:
        shape = list(src.shape)
        if n_axis is not None:
            shape[axis] = int(n_axis)
        else:
            shape[axis] = int(index.max()) + 1
        out = ops.zeros(shape, dtype=src.dtype)
    src = flatten(src, end=axis)

    if reduce in ['mul', 'div']:
        aux = ops.ones_like(src)
        out = ops.tensor_scatter_update(out, index, aux)
    if reduce in ['max', 'min']:
        out = out.fill(np.inf if reduce == 'min' else -np.inf)
    out = fn(out, index, src)
    if reduce in ['max', 'min']:
        out[out.isinf()] = 0
    if reduce == 'mean':
        aux = ops.ones_like(src)
        aux = fn(ops.ones_like(out), index, aux)
        out = out / aux
    return out


def scatter_(src: np.ndarray, index: np.ndarray, axis: int = -1,
             out: Optional[np.ndarray] = None,
             n_axis: Optional[int] = None, reduce: str = 'update') -> np.ndarray:
    src = ms.Tensor(src)
    index = ms.Tensor(index)
    out = scatter(src, index, axis=axis, out=out, n_axis=n_axis, reduce=reduce)
    return out.asnumpy()


def scatter_add(src: ms.Tensor, index: ms.Tensor, axis: int = -1,
                out: Optional[ms.Tensor] = None,
                n_axis: Optional[int] = None) -> ms.Tensor:
    return scatter(src, index, axis=axis, out=out, n_axis=n_axis, reduce='add')


def scatter_max(src: ms.Tensor, index: ms.Tensor, axis: int = -1,
                out: Optional[ms.Tensor] = None,
                n_axis: Optional[int] = None) -> ms.Tensor:
    return scatter(src, index, axis=axis, out=out, n_axis=n_axis, reduce='max')


def scatter_mean(src: ms.Tensor, index: ms.Tensor, axis: int = -1,
                 out: Optional[ms.Tensor] = None,
                 n_axis: Optional[int] = None) -> ms.Tensor:
    return scatter(src, index, axis=axis, out=out, n_axis=n_axis, reduce='mean')


def scatter_min(src: ms.Tensor, index: ms.Tensor, axis: int = -1,
                out: Optional[ms.Tensor] = None,
                n_axis: Optional[int] = None) -> ms.Tensor:
    return scatter(src, index, axis=axis, out=out, n_axis=n_axis, reduce='min')


def scatter_mul(src: ms.Tensor, index: ms.Tensor, axis: int = -1,
                out: Optional[ms.Tensor] = None,
                n_axis: Optional[int] = None) -> ms.Tensor:
    return scatter(src, index, axis=axis, out=out, n_axis=n_axis, reduce='mul')


def scatter_update(src: ms.Tensor, index: ms.Tensor, axis: int = -1,
                   out: Optional[ms.Tensor] = None,
                   n_axis: Optional[int] = None) -> ms.Tensor:
    return scatter(src, index, axis=axis, out=out, n_axis=n_axis, reduce='update')


def scatter_softmax(src: ms.Tensor, index: ms.Tensor,
                    axis: int = -1,
                    n_axis: Optional[int] = None) -> ms.Tensor:
    """scatter softmax

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    max_value_per_index = scatter_max(src, index, axis=axis, n_axis=n_axis)
    ix = broadcast(index, src, axis)
    max_per_src_element = ops.gather_nd(max_value_per_index, ix).reshape(src.shape)

    recentered_scores = src - max_per_src_element
    recentered_scores_exp = ops.exp(recentered_scores)

    sum_per_index = scatter_add(recentered_scores_exp, index, axis, n_axis=n_axis)
    normalizing_constants = ops.gather_nd(sum_per_index, ix).reshape(src.shape)
    return ops.div(recentered_scores_exp, normalizing_constants)


def scatter_log_softmax(src: ms.Tensor, index: ms.Tensor, axis: int = -1,
                        eps: float = 1e-12,
                        n_axis: Optional[int] = None) -> ms.Tensor:
    """scatter softmax

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    max_value_per_index = scatter_max(src, index, axis=axis, n_axis=n_axis)
    ix = broadcast(index, src, axis)
    max_per_src_element = ops.gather_nd(max_value_per_index, ix).reshape(src.shape)

    recentered_scores = src - max_per_src_element
    recentered_scores_exp = ops.exp(recentered_scores)

    sum_per_index = scatter_add(recentered_scores_exp, index, axis, n_axis=n_axis)
    normalizing_constants = ops.gather_nd(ops.log(sum_per_index + eps), ix).reshape(src.shape)
    return recentered_scores - normalizing_constants
