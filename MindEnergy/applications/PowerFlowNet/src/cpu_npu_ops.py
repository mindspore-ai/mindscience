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
"""
NPU/Ascend optimized operations for MindSpore.
This module provides operations optimized for Ascend devices.
"""

import mindspore as ms
from mindspore import Tensor, ops, mint


def gather_cpu_npu_compatible(tensor: Tensor, indices: Tensor, axis: int = 0) -> Tensor:
    """
    Gather operation optimized for Ascend.

    Args:
        tensor: Source tensor to gather from
        indices: Indices to gather
        axis: Axis along which to gather

    Returns:
        Gathered tensor
    """
    if axis == 0:
        if len(indices.shape) == 1:
            return ops.gather_nd(tensor, indices.expand_dims(-1))
        return ops.gather_nd(tensor, indices)
    return ops.gather(tensor, indices, axis=axis)


def scatter_sum_cpu_npu_compatible(
        updates: Tensor, indices: Tensor, shape: tuple, dtype=None) -> Tensor:  # pylint: disable=unused-argument
    """
    Scatter sum operation optimized for Ascend.

    Args:
        updates: Values to scatter
        indices: Target indices
        shape: Output tensor shape
        dtype: Output data type

    Returns:
        Scattered tensor
    """
    if len(shape) == 1:
        return ops.unsorted_segment_sum(updates, indices, num_segments=shape[0])
    return ops.unsorted_segment_sum(updates, indices, num_segments=shape[0])


def where_cpu_npu_compatible(condition: Tensor, x: Tensor, y: Tensor) -> Tensor:
    """
    Where operation (conditional select) for Ascend.
    """
    return ops.where(condition, x, y)


def pow_cpu_npu_compatible(tensor: Tensor, exponent: float) -> Tensor:
    """
    Power operation optimized for Ascend.
    """
    return ops.pow(tensor, exponent)


def randint_like_cpu_npu_compatible(tensor: Tensor, low: int = 0, high: int = 2) -> Tensor:
    """
    Generate random integers with same shape as input tensor.
    Optimized for Ascend.
    """
    return mint.randint(low, high, shape=tensor.shape, dtype=tensor.dtype)


def degree_cpu_npu_compatible(index: Tensor, num_nodes: int, dtype=ms.float32) -> Tensor:
    """
    Compute node degrees from edge indices.
    Optimized for Ascend devices.

    Args:
        index: Node indices, shape (num_edges,)
        num_nodes: Total number of nodes
        dtype: Output data type

    Returns:
        Degree tensor, shape (num_nodes,)
    """
    one = mint.ones((index.shape[0],), dtype=dtype)
    out = ops.unsorted_segment_sum(one, index, num_nodes)
    return out
