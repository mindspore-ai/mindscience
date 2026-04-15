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
functional
"""

import mindspore as ms
from mindspore import ops
from .scatter import scatter_add, scatter_mean, scatter_max, scatter_log_softmax, scatter_softmax


def masked_mean(inputs, mask, axis=None, keepdim=False):
    """
    Masked mean of a tensor.

    Args:
        inputs (Tensor): inputs tensor
        mask (BoolTensor): mask tensor
        axis (int or tuple of int, optional): dimension to reduce
        keepdim (bool, optional): whether retain ``dim`` or not

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    inputs = inputs.masked_scatter(~mask, ops.zeros_like(inputs))  # safe with nan
    if axis is None:
        return inputs.sum() / mask.sum().clip(1)
    return inputs.sum(axis, keepdim=keepdim) / mask.sum(axis, keepdim=keepdim).clip(1)


def mean_with_nan(inputs, axis=None, keepdims=False):
    """
    Mean of a tensor. Ignore all nan values.

    Args:
        inputs (Tensor): inputs tensor
        dim (int or tuple of int, optional): dimension to reduce
        keepdim (bool, optional): whether retain ``dim`` or not

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    mask = ~ops.isnan(inputs)
    return masked_mean(inputs, mask, axis, keepdims)


def shifted_softplus(inputs):
    """
    Shifted softplus function.

    Args:
        inputs (Tensor): inputs tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    return ops.softplus(inputs) - ops.softplus(ops.zeros(1))


def multi_slice(starts, ends):
    """
    Compute the union of indexes in multiple slices.

    Examples:
        >>> mask = multi_slice(ms.tensor([0, 1, 4]), ms.tensor([2, 3, 6]), 6)
        >>> assert (mask == ms.tensor([0, 1, 2, 4, 5]).all()

    Args:
        starts (LongTensor): start indexes of slices
        ends (LongTensor): end indexes of slices

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    values = ops.concat([ops.ones_like(starts), -ops.ones_like(ends)])
    slices = ops.concat([starts, ends])
    slices, order = slices.sort()
    values = values[order]
    depth = values.cumsum(0)
    valid = (values == 1 & depth == 0) | (values == -1 & depth == 1)
    slices = slices[valid]

    starts, ends = slices.view(-1, 2).t()
    size = ends - starts
    indexes = variadic_arange(size)
    indexes = indexes + starts.repeat(size)
    return indexes


def multi_slice_mask(starts, ends, length):
    """
    Compute the union of multiple slices into a binary mask.

    Examples:
        >>> mask = multi_slice_mask(ms.tensor([0, 1, 4]), ms.tensor([2, 3, 6]), 6)
        >>> assert (mask == ms.tensor([1, 1, 1, 0, 1, 1])).all()

    Args:
        starts (LongTensor): start indexes of slices
        ends (LongTensor): end indexes of slices
        length (int): length of mask

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    values = ops.concat([ops.ones_like(starts), -ops.ones_like(ends)])
    slices = ops.concat([starts, ends])
    if slices.numel():
        assert slices.min() >= 0 and slices.max() <= length
    mask = scatter_add(values, slices, axis=0, n_axis=length + 1)[:-1]
    mask = mask.cumsum(0).bool()
    return mask


def as_mask(indexes, length):
    """
    Convert indexes into a binary mask.

    Args:
        indexes (LongTensor): positive indexes
        length (int): maximal possible value of indexes

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    mask = ops.zeros(length)
    mask[indexes] = 1
    return mask


def size_to_index(size):
    """
    Convert sizes to variadic indexes.

    Examples:
        >>> index = _size_to_index(ms.tensor([3, 2, 1]))
        >>> assert (index == ms.tensor([0, 0, 0, 1, 1, 2])).all()

    Args:
        size (LongTensor): size of each sample

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    order = ops.arange(len(size))
    index2sample = order.repeat(size)
    return index2sample


def extend(data, size, inputs, input_size):
    """
    Extend variadic-sized data with variadic-sized inputs.
    This is a variadic variant of ``ops.concat([data, inputs], axis=-1)``.

    Examples:
        >>> data = ms.tensor([0, 1, 2, 3, 4])
        >>> size = ms.tensor([3, 2])
        >>> inputs = ms.tensor([-1, -2, -3])
        >>> input_size = ms.tensor([1, 2])
        >>> new_data, new_size = _extend(data, size, inputs, input_size)
        >>> assert (new_data == ms.tensor([0, 1, 2, -1, 3, 4, -2, -3])).all()
        >>> assert (new_size == ms.tensor([4, 4])).all()

    Args:
        data (Tensor): variadic data
        size (LongTensor): size of data
        inputs (Tensor): variadic inputs
        input_size (LongTensor): size of inputs

    Returns:
        (Tensor, LongTensor): output data, output size

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    new_size = size + input_size
    new_cum_size = new_size.cumsum(0)
    new_data = ops.zeros((int(new_cum_size[-1]), *data.shape[1:]), dtype=data.dtype)
    starts = new_cum_size - new_size
    ends = starts + size
    index = multi_slice_mask(starts, ends, new_cum_size[-1])
    new_data[index] = data
    new_data[~index] = inputs
    return new_data, new_size


def variadic_sum(inputs, size):
    """
    Compute sum over sets with variadic sizes.

    Suppose there are :math:`N` sets, and the sizes of all sets are summed to :math:`B`.

    Args:
        inputs (Tensor): inputs of shape :math:`(B, ...)`
        size (LongTensor): size of sets of shape :math:`(N,)`

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    index2sample = size_to_index(size)
    index2sample = index2sample.view([-1] + [1] * (inputs.ndim - 1))
    index2sample = index2sample.expand_as(inputs)

    value = scatter_add(inputs, index2sample, axis=0)
    return value


def variadic_mean(inputs, size):
    """
    Compute mean over sets with variadic sizes.

    Suppose there are :math:`N` sets, and the sizes of all sets are summed to :math:`B`.

    Args:
        inputs (Tensor): inputs of shape :math:`(B, ...)`
        size (LongTensor): size of sets of shape :math:`(N,)`

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    index2sample = size_to_index(size)
    index2sample = index2sample.view([-1] + [1] * (inputs.ndim - 1))
    index2sample = index2sample.expand_as(inputs)

    value = scatter_mean(inputs, index2sample, axis=0)
    return value


def variadic_max(inputs, size):
    """
    Compute max over sets with variadic sizes.

    Suppose there are :math:`N` sets, and the sizes of all sets are summed to :math:`B`.

    Args:
        inputs (Tensor): inputs of shape :math:`(B, ...)`
        size (LongTensor): size of sets of shape :math:`(N,)`

    Returns
        (Tensor, LongTensor): max values and indexes

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    index2sample = size_to_index(size)
    index2sample = index2sample.view([-1] + [1] * (inputs.ndim - 1))
    index2sample = index2sample.expand_as(inputs)

    value, index = scatter_max(inputs, index2sample, axis=0)
    index = index + (size - size.cumsum(0)).view([-1] + [1] * (index.ndim - 1))
    return value, index


def variadic_log_softmax(inputs, size):
    """
    Compute log softmax over categories with variadic sizes.

    Suppose there are :math:`N` samples, and the numbers of categories in all samples are summed to :math:`B`.

    Args:
        inputs (Tensor): inputs of shape :math:`(B, ...)`
        size (LongTensor): number of categories of shape :math:`(N,)`

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    index2sample = size_to_index(size)
    index2sample = index2sample.view([-1] + [1] * (inputs.ndim - 1))
    index2sample = index2sample.expand_as(inputs)

    log_likelihood = scatter_log_softmax(inputs, index2sample, axis=0)
    return log_likelihood


def variadic_softmax(inputs, size):
    """
    Compute softmax over categories with variadic sizes.

    Suppose there are :math:`N` samples, and the numbers of categories in all samples are summed to :math:`B`.

    Args:
        inputs (Tensor): inputs of shape :math:`(B, ...)`
        size (LongTensor): number of categories of shape :math:`(N,)`

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    index2sample = size_to_index(size)
    index2sample = index2sample.view([-1] + [1] * (inputs.ndim - 1))
    index2sample = index2sample.expand_as(inputs)

    log_likelihood = scatter_softmax(inputs, index2sample, axis=0)
    return log_likelihood


def variadic_cross_entropy(inputs, target, size, reduction="mean"):
    """
    Compute cross entropy loss over categories with variadic sizes.

    Suppose there are :math:`N` samples, and the numbers of categories in all samples are summed to :math:`B`.

    Args:
        inputs (Tensor): prediction of shape :math:`(B, ...)`
        target (Tensor): target of shape :math:`(N, ...)`. Each target is a relative index in a sample.
        size (LongTensor): number of categories of shape :math:`(N,)`
        reduction (string, optional): reduction to apply to the output.
            Available reductions are ``none``, ``sum`` and ``mean``.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    index2sample = size_to_index(size)
    index2sample = index2sample.view([-1] + [1] * (inputs.ndim - 1))
    index2sample = index2sample.expand_as(inputs)

    log_likelihood = scatter_log_softmax(inputs, index2sample, axis=0)
    size = size.view([-1] + [1] * (inputs.ndim - 1))
    assert (target >= 0).all() and (target < size).all()
    target_index = target + size.cumsum(0) - size
    loss = -log_likelihood.gather(0, target_index)
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError(f"Unknown reduction `{reduction}`")


def variadic_topk(inputs, size, k, largest=True):
    """
    Compute the :math:`k` largest elements over sets with variadic sizes.

    Suppose there are :math:`N` sets, and the sizes of all sets are summed to :math:`B`.

    If any set has less than than :math:`k` elements, the size-th largest element will be
    repeated to pad the output to :math:`k`.

    Args:
        inputs (Tensor): inputs of shape :math:`(B, ...)`
        size (LongTensor): size of sets of shape :math:`(N,)`
        k (int or LongTensor): the k in "top-k". Can be a fixed value for all sets,
            or different values for different sets of shape :math:`(N,)`.
        largest (bool, optional): return largest or smallest elements

    Returns
        (Tensor, LongTensor): top-k values and indexes

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    index2graph = size_to_index(size)
    index2graph = index2graph.view([-1] + [1] * (inputs.ndim - 1))

    mask = ~ops.isinf(inputs)
    dmax = inputs[mask].max()
    dmin = inputs[mask].min()
    abs_max = inputs[mask].abs().max()
    # special case: max = min
    gap = dmax - dmin + abs_max * 1e-6
    safe_inputs = inputs.clip(dmin - gap, dmax + gap)
    offset = gap * 4
    if largest:
        offset = -offset
    inputs_ext = safe_inputs + offset * index2graph
    index_ext = inputs_ext.argsort(axis=0, descending=largest)
    if isinstance(k, ms.Tensor) and k.shape == size.shape:
        num_actual = ops.min(size, k)
    else:
        num_actual = size.clip(max=k)
    num_padding = k - num_actual
    starts = size.cumsum(0) - size
    ends = starts + num_actual
    mask = multi_slice_mask(starts, ends, len(index_ext)).nonzero().flatten()

    if (num_padding > 0).any():
        # special case: size < k, pad with the last valid index
        padding = ends - 1
        padding2graph = size_to_index(num_padding)
        mask = extend(mask, num_actual, padding[padding2graph], num_padding)[0]

    index = index_ext[mask]  # (N * k, ...)
    value = inputs.gather(0, index)
    if isinstance(k, ms.Tensor) and k.shape == size.shape:
        value = value.view(-1, *inputs.shape[1:])
        index = index.view(-1, *inputs.shape[1:])
        index = index - (size.cumsum(0) - size).repeat(k).view([-1] + [1] * (index.ndim - 1))
    else:
        value = value.view(-1, k, *inputs.shape[1:])
        index = index.view(-1, k, *inputs.shape[1:])
        index = index - (size.cumsum(0) - size).view([-1] + [1] * (index.ndim - 1))

    return value, index


def variadic_sort(inputs, size, descending=False):
    """
    Sort elements in sets with variadic sizes.

    Suppose there are :math:`N` sets, and the sizes of all sets are summed to :math:`B`.

    Args:
        inputs (Tensor): inputs of shape :math:`(B, ...)`
        size (LongTensor): size of sets of shape :math:`(N,)`
        descending (bool, optional): return ascending or descending order

    Returns
        (Tensor, LongTensor): sorted values and indexes

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    index2sample = size_to_index(size)
    index2sample = index2sample.view([-1] + [1] * (inputs.ndim - 1))

    mask = ~ops.isinf(inputs)
    dmax = inputs[mask].max()
    dmin = inputs[mask].min()
    safe_input = inputs.clip(2 * dmin - dmax, 2 * dmax - dmin)
    offset = (dmax - dmin) * 4
    if descending:
        offset = -offset
    input_ext = safe_input + offset * index2sample
    index = input_ext.argsort(axis=0, descending=descending)
    value = inputs.gather(0, index)
    index = index - (size.cumsum(0) - size)[index2sample]
    return value, index


def variadic_arange(size):
    """
    Return a 1-D tensor that contains integer intervals of variadic sizes.
    This is a variadic variant of ``ops.arange(stop).expand(batch_size, -1)``.

    Suppose there are :math:`N` intervals.

    Args:
        size (LongTensor): size of intervals of shape :math:`(N,)`
    """
    starts = size.cumsum(0) - size

    order = ops.arange(size.sum())
    order = order - starts.repeat(size)
    return order


def variadic_randperm(size):
    """
    Return random permutations for sets with variadic sizes.
    The ``i``-th permutation contains integers from 0 to ``size[i] - 1``.

    Suppose there are :math:`N` sets.

    Args:
        size (LongTensor): size of sets of shape :math:`(N,)`
    """
    rand = ops.rand(size.sum())
    perm = variadic_sort(rand, size)[1]
    return perm


def variadic_sample(inputs, size, num_sample):
    """
    Draw samples with replacement from sets with variadic sizes.

    Suppose there are :math:`N` sets, and the sizes of all sets are summed to :math:`B`.

    Args:
        inputs (Tensor): inputs of shape :math:`(B, ...)`
        size (LongTensor): size of sets of shape :math:`(N,)`
        num_sample (int): number of samples to draw from each set

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    rand = ops.rand(len(size), num_sample)
    index = (rand * size.expand_dims(-1)).long()
    index = index + (size.cumsum(0) - size).expand_dims(-1)
    sample = inputs[index]
    return sample


def variadic_meshgrid(input1, size1, input2, size2):
    """
    Compute the Cartesian product for two batches of sets with variadic sizes.

    Suppose there are :math:`N` sets in each inputs,
    and the sizes of all sets are summed to :math:`B_1` and :math:`B_2` respectively.

    Args:
        input1 (Tensor): input of shape :math:`(B_1, ...)`
        size1 (LongTensor): size of :attr:`input1` of shape :math:`(N,)`
        input2 (Tensor): input of shape :math:`(B_2, ...)`
        size2 (LongTensor): size of :attr:`input2` of shape :math:`(N,)`

    Returns
        (Tensor, Tensor): the first and the second elements in the Cartesian product

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    grid_size = size1 * size2
    local_index = variadic_arange(grid_size)
    local_inner_size = size2.repeat(grid_size)
    offset1 = (size1.cumsum(0) - size1).repeat(grid_size)
    offset2 = (size2.cumsum(0) - size2).repeat(grid_size)
    index1 = local_index // local_inner_size + offset1
    index2 = local_index % local_inner_size + offset2
    return input1[index1], input2[index2]


def variadic_to_padded(inputs, size, value=0):
    """
    Convert a variadic tensor to a padded tensor.

    Suppose there are :math:`N` sets, and the sizes of all sets are summed to :math:`B`.

    Args:
        inputs (Tensor): inputs of shape :math:`(B, ...)`
        size (LongTensor): size of sets of shape :math:`(N,)`
        value (scalar): fill value for padding

    Returns:
        (Tensor, BoolTensor): padded tensor and mask

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    num_sample = len(size)
    max_size = size.max()
    starts = ops.arange(num_sample) * max_size
    ends = starts + size
    mask = multi_slice_mask(starts, ends, num_sample * max_size)
    mask = mask.view(num_sample, max_size)
    shape = (num_sample, max_size) + inputs.shape[1:]
    padded = ops.full(shape, value, dtype=inputs.dtype)
    padded[mask] = inputs
    return padded, mask


def padded_to_variadic(padded, size):
    """
    Convert a padded tensor to a variadic tensor.

    Args:
        padded (Tensor): padded tensor of shape :math:`(N, ...)`
        size (LongTensor): size of sets of shape :math:`(N,)`

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    num_sample, max_size = padded.shape[:2]
    starts = ops.arange(num_sample) * max_size
    ends = starts + size
    mask = multi_slice_mask(starts, ends, num_sample * max_size)
    mask = mask.view(num_sample, max_size)
    return padded[mask]


def clipped_policy_gradient_objective(policy, agent, reward, eps=0.2):
    """
    clipped policy gradient objective

    Args:
        policy (ms.Tensor): pobability calculated by policy network.
        agent (ms.Tensor): pobability calculated by agent network.
        reward (ms.Tensor): reward value.
        eps (float, optional): epsilon value.

    Returns:

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    ratio = (policy - agent).exp()
    ratio = ratio.clip(-10, 10)
    objective = ops.minimum(ratio * reward, ratio.clip(1 - eps, 1 + eps) * reward)
    return objective


def policy_gradient_objective(policy, reward):
    """pollicy gradient objective

    Args:
        policy (ms.Tensor): pobability calculated by policy network.
        reward (ms.Tensor): reward value.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    return policy * reward


def margin_ranking_loss(input1: ms.Tensor,
                        input2: ms.Tensor,
                        target: ms.Tensor,
                        margin: float = 0,
                        reduction: str = "mean",) -> ms.Tensor:
    """
    margin ranking loss

    Args:
        input1 (ms.Tensor): the first input
        input2 (ms.Tensor): the second input
        target (ms.Tensor): target
        margin (float):     margin
        reduction (str):    reduction method of output tensor.
                            Either ``mean`` or ``sum``.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    loss = ops.maximum(0, -target * (input1 - input2) + margin)
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    return loss
