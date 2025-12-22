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
    """Forwards the input to all ranks and reduces gradients across the group in backward.

    This operation copies the input tensor to all ranks in the specified group during
    the forward pass and performs an all-reduce operation on the gradients during
    the backward pass.

    Args:
        x (Tensor): Input tensor to be copied to all ranks.
        group (Union[CommGroup, CommGroupBase]): Communication group for the operation.

    Returns:
        - Forward pass. The input tensor (copied to all ranks).
        - Backward pass. Reduced gradients across the group.
    """
    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return x

    @staticmethod
    def backward(ctx, grad):
        return _reduce(grad, ctx.group), None


class GatherFromHidden(_Function):
    """Gathers hidden-partitioned tensors along the last dimension in forward and scatters gradients in backward.

    This operation gathers tensors that are partitioned along the last dimension during
    the forward pass and scatters the gradients back to the corresponding partitions
    during the backward pass.

    Args:
        x (Tensor): Input tensor with hidden partitions along the last dimension.
        group (Union[CommGroup, CommGroupBase]): Communication group for the operation.

    Returns:
        - Forward pass. Tensor with all hidden partitions gathered along the last dimension.
        - Backward pass. Gradients scattered to respective partitions.
    """
    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return _gather_along_last_dim(x, group)

    @staticmethod
    def backward(ctx, grad):
        return _scatter_along_last_dim(grad, ctx.group), None


class ScatterToHidden(_Function):
    """Scatters tensors into hidden partitions in forward and gathers gradients from partitions in backward.

    This operation scatters the input tensor into hidden partitions along the last
    dimension during the forward pass and gathers the gradients from all partitions
    during the backward pass.

    Args:
        x (Tensor): Input tensor to be scattered into hidden partitions.
        group (Union[CommGroup, CommGroupBase]): Communication group for the operation.

    Returns:
        - Forward pass. Tensor partition corresponding to the current rank.
        - Backward pass. Gradients gathered from all partitions along the last dimension.
    """
    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return _scatter_along_last_dim(x, group)

    @staticmethod
    def backward(ctx, grad):
        return _gather_along_last_dim(grad, ctx.group), None


class ReduceFromAll(_Function):
    """Performs an all-reduce on forward; backward returns the upstream gradient unchanged.

    This operation performs an all-reduce operation across all ranks in the group
    during the forward pass and returns the upstream gradient unchanged during
    the backward pass.

    Args:
        x (Tensor): Input tensor to be reduced across all ranks.
        group (Union[CommGroup, CommGroupBase]): Communication group for the operation.

    Returns:
        - Forward pass. Reduced tensor from all ranks.
        - Backward pass. Unchanged upstream gradient.
    """
    @staticmethod
    def forward(_, x, group):
        return _reduce(x, group)

    @staticmethod
    def backward(_, grad):
        return grad, None


class GatherFromSequence(_Function):
    """Gathers sequence partitions along the first dimension in forward;
    backward either reduce-scatter or scatter based on a flag.

    This operation gathers sequence partitions along the first dimension during
    the forward pass and either performs reduce-scatter or scatter operation
    during the backward pass depending on the tensor_parallel_output_grad flag.

    Args:
        x (Tensor): Input tensor with sequence partitions along the first dimension.
        group (Union[CommGroup, CommGroupBase]): Communication group for the operation.
        tensor_parallel_output_grad (bool, optional): Flag to determine whether
            to use reduce-scatter (True) or scatter (False) in backward pass.
            Defaults to True.

    Returns:
        - Forward pass. Tensor with all sequence partitions gathered along the first dimension.
        - Backward pass. Either reduce-scattered or scattered gradients based on the flag.
    """
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
    """Performs reduce-scatter across sequence partitions in forward and gathers in backward.

    This operation performs a reduce-scatter operation across sequence partitions
    along the first dimension during the forward pass and gathers the results
    during the backward pass.

    Args:
        x (Tensor): Input tensor to be reduced and scattered across sequence partitions.
        group (Union[CommGroup, CommGroupBase]): Communication group for the operation.

    Returns:
        - Forward pass. Reduced and scattered tensor for the current rank.
        - Backward pass. Gradients gathered from all sequence partitions.
    """
    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return _reduce_scatter_along_first_dim(x, group)

    @staticmethod
    def backward(ctx, grad):
        return _gather_along_first_dim(grad, ctx.group), None


class ScatterToSequence(_Function):
    """Scatters tensors across the first dimension to form sequence partitions and gathers on backward.

    This operation scatters the input tensor across the first dimension to form
    sequence partitions during the forward pass and gathers the gradients from
    all partitions during the backward pass.

    Args:
        x (Tensor): Input tensor to be scattered into sequence partitions.
        group (Union[CommGroup, CommGroupBase]): Communication group for the operation.

    Returns:
        - Forward pass. Tensor partition corresponding to the current rank along the first dimension.
        - Backward pass. Gradients gathered from all sequence partitions.
    """
    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return _scatter_along_first_dim(x, group)

    @staticmethod
    def backward(ctx, grad):
        return _gather_along_first_dim(grad, ctx.group), None


class AllToAllFromHiddenToSequence(_Function):
    """Performs an all-to-all from hidden layout to sequence layout in forward and the inverse on backward.

    This operation performs an all-to-all communication to transform from hidden
    layout to sequence layout during the forward pass and performs the inverse
    transformation during the backward pass.

    Args:
        x (Tensor): Input tensor in hidden layout.
        group (Union[CommGroup, CommGroupBase]): Communication group for the operation.

    Returns:
        - Forward pass. Tensor transformed from hidden layout to sequence layout.
        - Backward pass. Gradients transformed from sequence layout back to hidden layout.
    """
    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return _alltoall_h2s(x, group)

    @staticmethod
    def backward(ctx, grad):
        return _alltoall_s2h(grad, ctx.group), None


class AllToAllFromSequenceToHidden(_Function):
    """Performs an all-to-all from sequence layout to hidden layout in forward and the inverse on backward.

    This operation performs an all-to-all communication to transform from sequence
    layout to hidden layout during the forward pass and performs the inverse
    transformation during the backward pass.

    Args:
        x (Tensor): Input tensor in sequence layout.
        group (Union[CommGroup, CommGroupBase]): Communication group for the operation.

    Returns:
        - Forward pass. Tensor transformed from sequence layout to hidden layout.
        - Backward pass. Gradients transformed from hidden layout back to sequence layout.
    """
    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return _alltoall_s2h(x, group)

    @staticmethod
    def backward(ctx, grad):
        return _alltoall_h2s(grad, ctx.group), None


def copy_to_all(x, group):
    """Forwards the input to all ranks in the specified group.

    Args:
        x (Tensor): Input tensor to be copied to all ranks.
        group (Union[CommGroup, CommGroupBase]): Communication group for the operation.

    Returns:
        The input tensor (copied to all ranks).
    """
    return CopyToAll.apply(x, group)


def gather_from_hidden(x, group):
    """Gathers hidden-partitioned tensors along the last dimension.

    Args:
        x (Tensor): Input tensor with hidden partitions along the last dimension.
        group (Union[CommGroup, CommGroupBase]): Communication group for the operation.

    Returns:
        Tensor with all hidden partitions gathered along the last dimension.
    """
    return GatherFromHidden.apply(x, group)


def scatter_to_hidden(x, group):
    """Scatters tensors into hidden partitions along the last dimension.

    Args:
        x (Tensor): Input tensor to be scattered into hidden partitions.
        group (Union[CommGroup, CommGroupBase]): Communication group for the operation.

    Returns:
        Tensor partition corresponding to the current rank.
    """
    return ScatterToHidden.apply(x, group)


def reduce_from_all(x, group):
    """Performs an all-reduce operation across all ranks in the group.

    Args:
        x (Tensor): Input tensor to be reduced across all ranks.
        group (Union[CommGroup, CommGroupBase]): Communication group for the operation.

    Returns:
        Reduced tensor from all ranks.
    """
    return ReduceFromAll.apply(x, group)


def gather_from_sequence(x, group, tensor_parallel_output_grad=True):
    """Gathers sequence partitions along the first dimension.

    Args:
        x (Tensor): Input tensor with sequence partitions along the first dimension.
        group (Union[CommGroup, CommGroupBase]): Communication group for the operation.
        tensor_parallel_output_grad (bool, optional): Flag to determine whether
            to use reduce-scatter (True) or scatter (False) in backward pass.
            Default: ``True``.

    Returns:
        Tensor with all sequence partitions gathered along the first dimension.
    """
    return GatherFromSequence.apply(x, group, tensor_parallel_output_grad)


def reduce_scatter_to_sequence(x, group):
    """Performs reduce-scatter across sequence partitions along the first dimension.

    Args:
        x (Tensor): Input tensor to be reduced and scattered across sequence partitions.
        group (Union[CommGroup, CommGroupBase]): Communication group for the operation.

    Returns:
        Reduced and scattered tensor for the current rank.
    """
    return ReduceScatterToSequence.apply(x, group)


def scatter_to_sequence(x, group):
    """Scatters tensors across the first dimension to form sequence partitions.

    Args:
        x (Tensor): Input tensor to be scattered into sequence partitions.
        group (Union[CommGroup, CommGroupBase]): Communication group for the operation.

    Returns:
        Tensor partition corresponding to the current rank along the first dimension.
    """
    return ScatterToSequence.apply(x, group)


def all_to_all_from_hidden_to_sequence(x, group):
    """Performs an all-to-all from hidden layout to sequence layout.

    Args:
        x (Tensor): Input tensor in hidden layout.
        group (Union[CommGroup, CommGroupBase]): Communication group for the operation.

    Returns:
        Tensor transformed from hidden layout to sequence layout.
    """
    return AllToAllFromHiddenToSequence.apply(x, group)


def all_to_all_from_sequence_to_hidden(x, group):
    """Performs an all-to-all from sequence layout to hidden layout.

    Args:
        x (Tensor): Input tensor in sequence layout.
        group (Union[CommGroup, CommGroupBase]): Communication group for the operation.

    Returns:
        Tensor transformed from sequence layout to hidden layout.
    """
    return AllToAllFromSequenceToHidden.apply(x, group)
