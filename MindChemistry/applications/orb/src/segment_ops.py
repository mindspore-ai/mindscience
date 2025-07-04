# ============================================================================
# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Segment operations."""

from typing import Optional
import numpy as np
import mindspore as ms
from mindspore import ops, Tensor, mint

MSINT = [ms.int64, ms.int32, ms.int16, ms.int8, ms.uint8]


def aggregate_nodes(tensor: Tensor, n_node: Tensor, reduction: str = "mean", deterministic: bool = False) -> Tensor:
    """Aggregates over a tensor based on graph sizes."""
    count = len(n_node)
    if deterministic:
        ms.set_seed(1)
    segments = ops.arange(count).repeat_interleave(n_node).astype(ms.int32)
    if reduction == "sum":
        return scatter_sum(tensor, segments, dim=0)
    if reduction == "mean":
        return scatter_mean(tensor, segments, dim=0)
    if reduction == "max":
        return scatter_max(tensor, segments, dim=0)
    raise ValueError("Invalid reduction argument. Use sum, mean or max.")


def segment_sum(data: Tensor, segment_ids: Tensor, num_segments: int):
    """Computes index based sum over segments of a tensor."""
    return scatter_sum(data, segment_ids, dim=0, dim_size=num_segments)


def segment_max(data: Tensor, segment_ids: Tensor, num_segments: int):
    """Computes index based max over segments of a tensor."""
    assert segment_ids is not None, "segment_ids must not be None"
    assert num_segments > 0, "num_segments must be greater than 0"
    max_op = ops.ArgMaxWithValue(axis=0)
    _, max_values = max_op(data)
    return max_values


def segment_mean(data: Tensor, segment_ids: Tensor, num_segments: int):
    """Computes index based mean over segments of a tensor."""
    sum_v = segment_sum(data, segment_ids, num_segments)
    count = ops.scatter_add(ops.zeros(
        (num_segments,), dtype=ms.int32), segment_ids, ops.ones_like(segment_ids))
    return sum_v / count.astype(sum_v.dtype)


def segment_softmax(data: Tensor, segment_ids: Tensor, num_segments: int, weights: Optional[Tensor] = None):
    """Computes a softmax over segments of the tensor."""
    data_max = segment_max(data, segment_ids, num_segments)
    data = data - data_max[segment_ids]

    unnormalised_probs = ops.exp(data)
    if weights is not None:
        unnormalised_probs = unnormalised_probs * weights
    denominator = segment_sum(unnormalised_probs, segment_ids, num_segments)

    return safe_division(unnormalised_probs, denominator, segment_ids)


def safe_division(numerator: Tensor, denominator: Tensor, segment_ids: Tensor):
    """Divides logits by denominator, setting 0 where the denominator is zero."""
    result = ops.where(denominator[segment_ids] ==
                       0, 0, numerator / denominator[segment_ids])
    return result


def _broadcast(src: Tensor, other: Tensor, dim: int):
    """Broadcasts the source tensor to match the shape of the other tensor along the specified dimension."""
    if dim < 0:
        dim = other.ndim + dim
    if src.ndim == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.ndim, other.ndim):
        src = src.unsqueeze(-1)
    src = src.expand_as(other)
    return src


def scatter_sum(
        src: Tensor, index: Tensor, dim: int = -1, out: Optional[Tensor] = None,
        dim_size: Optional[int] = None, reduce: str = "sum"
) -> Tensor:
    """Applies a sum reduction of the orb_models tensor along the specified dimension."""
    assert reduce == "sum"
    index = _broadcast(index, src, dim)
    if out is None:
        size = list(src.shape)
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = ops.zeros(size, dtype=src.dtype)
        return mint.scatter_add(out, dim, index, src)
    return mint.scatter_add(out, dim, index, src)


def scatter_std(
        src: Tensor, index: Tensor, dim: int = -1, out: Optional[Tensor] = None,
        dim_size: Optional[int] = None, unbiased: bool = True
) -> Tensor:
    """Computes the standard deviation of the orb_models tensor along the specified dimension."""
    if out is not None:
        dim_size = out.shape[dim]

    if dim < 0:
        dim = src.ndim + dim

    count_dim = dim
    if index.ndim <= dim:
        count_dim = index.ndim - 1

    ones = ops.ones(index.shape, dtype=src.dtype)
    count = scatter_sum(ones, index, count_dim, dim_size=dim_size)

    index = _broadcast(index, src, dim)
    tmp = scatter_sum(src, index, dim, dim_size=dim_size)
    count = _broadcast(count, tmp, dim).clip(1)
    mean = tmp / count

    var = src - mean.gather(dim, index)
    var = var * var
    out = scatter_sum(var, index, dim, out=out, dim_size=dim_size)

    if unbiased:
        count = count - 1
        count = count.clip(1)
    out = out / (count + 1e-6)
    out = ops.sqrt(out)
    return out


def scatter_mean(
        src: Tensor, index: Tensor, dim: int = -1, out: Optional[Tensor] = None,
        dim_size: Optional[int] = None
) -> Tensor:
    """Computes the mean of the orb_models tensor along the specified dimension."""
    out = scatter_sum(src, index, dim, out=out, dim_size=dim_size)
    dim_size = out.shape[dim]

    index_dim = dim
    if index_dim < 0:
        index_dim = index_dim + src.ndim
    if index.ndim <= index_dim:
        index_dim = index.ndim - 1

    ones = ops.ones(index.shape, dtype=src.dtype)
    count = scatter_sum(ones, index, index_dim, dim_size=dim_size)
    count = count.clip(1)
    count = _broadcast(count, out, dim)
    out = out / count
    return out


def scatter_max(
        src: Tensor, index: Tensor, dim: int = -1, out: Optional[Tensor] = None,
        dim_size: Optional[int] = None
) -> Tensor:
    """Computes the maximum of the orb_models tensor for each group defined by index along the specified dimension."""
    if out is not None:
        raise NotImplementedError(
            "The 'out' argument is not supported for scatter_max")

    if src.dtype in MSINT:
        init_value = np.iinfo(src.dtype).min
    else:
        init_value = np.finfo(src.dtype).min

    if dim < 0:
        dim = src.ndim + dim

    if dim_size is None:
        dim_size = int(index.max()) + 1

    result = ops.ones(
        (dim_size, *src.shape[:dim], *src.shape[dim + 1:]), dtype=src.dtype)
    result = init_value * result
    broadcasted_index = _broadcast(index, src, dim)

    scatter_result = ops.ZerosLike()(result)
    index = ops.expand_dims(broadcasted_index, dim)
    scatter_result = scatter_result.scatter_update(index, src)
    result = ops.Maximum()(result, scatter_result)
    return result
