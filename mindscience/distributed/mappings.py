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
Distributed tensor mapping utilities.

This module provides helper functions that implement common distributed tensor
mappings (reduce, gather, scatter, reduce-scatter and all-to-all) using the
MindSpore communication backend. These are intended to be used by model-parallel
layers to move and combine tensor partitions across parallel groups.
"""

from mindspore import mint, _Function
from mindspore.communication import comm_func

def _reduce(x, group):
    if group.size == 1:
        return x
    out, _ = comm_func.all_reduce(x, group=group.group_name)
    return out

def _gather_along_last_dim(x, group):
    if group.size == 1:
        return x
    gather_x, _ = comm_func.all_gather_into_tensor(x, group=group.group_name)
    tensor_list = mint.split(gather_x, x.shape[0], dim=0)
    out = mint.cat(tensor_list, dim=-1)
    return out

def _scatter_along_last_dim(x, group):
    group_size = group.size
    if group_size == 1:
        return x
    per_partition_size = x.shape[-1] // group_size
    tensor_list = mint.split(x, per_partition_size, dim=-1)
    return tensor_list[group.rank].contiguous()

def _gather_along_first_dim(x, group):
    if group.size == 1:
        return x
    out, _ = comm_func.all_gather_into_tensor(x, group=group.group_name)
    return out

def _scatter_along_first_dim(x, group):
    group_size = group.size
    if group_size == 1:
        return x
    per_partition_size = x.shape[0] // group_size
    tensor_list = mint.split(x, per_partition_size, dim=0)
    return tensor_list[group.rank].contiguous()

def _reduce_scatter_along_first_dim(x, group):
    if group.size == 1:
        return x
    out, _ = comm_func.reduce_scatter_tensor(x, group=group.group_name)
    return out

def _alltoall_s2h(x, group):
    group_size = group.size
    if group_size == 1:
        return x
    per_partition_size = x.shape[1] // group_size
    tensor_list = mint.split(x, per_partition_size, dim=1)
    shape_list = [x.shape[:1] + (per_partition_size,) + x.shape[2:]] * group_size
    recv_list, _ = comm_func.all_to_all_with_output_shape(shape_list, tensor_list, group=group.group_name)
    output = mint.concat(recv_list, dim=0)
    return output

def _alltoall_h2s(x, group):
    group_size = group.size
    if group_size == 1:
        return x
    per_partition_size = x.shape[0] // group_size
    tensor_list = mint.split(x, per_partition_size, dim=0)
    shape_list = [(per_partition_size,) + x.shape[1:]] * group_size
    recv_list, _ = comm_func.all_to_all_with_output_shape(shape_list, tensor_list, group=group.group_name)
    output = mint.concat(recv_list, dim=1)
    return output


class CopyToAll(_Function):
    """Forwards the input to all ranks and
    reduces gradients across the group in backward."""
    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return x

    @staticmethod
    def backward(ctx, grad):
        return _reduce(grad, ctx.group), None


class GatherFromHidden(_Function):
    """Gathers hidden-partitioned tensors along the last dimension
    in forward and scatters gradients in backward."""
    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return _gather_along_last_dim(x, group)

    @staticmethod
    def backward(ctx, grad):
        return _scatter_along_last_dim(grad, ctx.group), None


class ScatterToHidden(_Function):
    """Scatters tensors into hidden partitions in forward and
    gathers gradients from partitions in backward."""
    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return _scatter_along_last_dim(x, group)

    @staticmethod
    def backward(ctx, grad):
        return _gather_along_last_dim(grad, ctx.group), None


class ReduceFromAll(_Function):
    """Performs an all-reduce on forward;
    backward returns the upstream gradient unchanged."""
    @staticmethod
    def forward(_, x, group):
        return _reduce(x, group)

    @staticmethod
    def backward(_, grad):
        return grad, None


class GatherFromSequence(_Function):
    """Gathers sequence partitions along the first dimension in forward;
    backward either reduce-scatter or scatter based on a flag."""
    @staticmethod
    def forward(ctx, x, group, tensor_parallel_output_grad=True):
        ctx.group = group
        ctx.tensor_parallel_output_grad=tensor_parallel_output_grad
        return _gather_along_first_dim(x, group)

    @staticmethod
    def backward(ctx, grad):
        if ctx.tensor_parallel_output_grad:
            return _reduce_scatter_along_first_dim(grad, ctx.group), None, None
        return _scatter_along_first_dim(grad, ctx.group), None, None


class ReduceScatterToSequence(_Function):
    """Performs reduce-scatter across sequence partitions in forward and gathers in backward."""
    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return _reduce_scatter_along_first_dim(x, group)

    @staticmethod
    def backward(ctx, grad):
        return _gather_along_first_dim(grad, ctx.group), None


class ScatterToSequence(_Function):
    """Scatters tensors across the first dimension to form sequence partitions and gathers on backward."""
    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return _scatter_along_first_dim(x, group)

    @staticmethod
    def backward(ctx, grad):
        return _gather_along_first_dim(grad, ctx.group), None


class AllToAllFromHiddenToSequence(_Function):
    """Performs an all-to-all from hidden layout to sequence layout in forward and the inverse on backward."""
    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return _alltoall_h2s(x, group)

    @staticmethod
    def backward(ctx, grad):
        return _alltoall_s2h(grad, ctx.group), None


class AllToAllFromSequenceToHidden(_Function):
    """Performs an all-to-all from sequence layout to hidden layout in forward and the inverse on backward."""
    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return _alltoall_s2h(x, group)

    @staticmethod
    def backward(ctx, grad):
        return _alltoall_h2s(grad, ctx.group), None
