# Copyright 2025 Huawei Technologies Co., Ltd
#
# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""Utils for geometry library."""

from collections.abc import Iterable
import numbers
from typing import Optional, Union

import mindspore as ms
from mindspore import ops
import mindspore.numpy as mnp


def safe_select(condition, true_fn, false_fn):
    """Safe version of selection (i.e. `where`).

    This applies the double-where trick.
    Like jnp.where, this function will still execute both branches and is
    expected to be more lightweight than lax.cond.  Other than NaN-semantics,
    safe_select(condition, true_fn, false_fn) is equivalent to

        utils.tree.map(lambda x, y: jnp.where(condition, x, y),
                    true_fn(),
                    false_fn()),

    Compared to the naive implementation above, safe_select provides the
    following guarantee: in either the forward or backward pass, a NaN produced
    *during the execution of true_fn()* will not propagate to the rest of the
    computation and similarly for false_fn.  It is very important to note that
    while true_fn and false_fn will typically close over other tensors (i.e. they
    use values computed prior to the safe_select function), there is no NaN-safety
    for the backward pass of closed over values.  It is important than any NaN's
    are produced within the branch functions and not before them.  For example,

        safe_select(x < eps, lambda: 0., lambda: jnp.sqrt(x))

    will not produce NaN on the backward pass even if x == 0. since sqrt happens
    within the false_fn, but the very similar

        y = jnp.sqrt(x)
        safe_select(x < eps, lambda: 0., lambda: y)

    will produce a NaN on the backward pass if x == 0 because the sqrt happens
    prior to the false_fn.

    Args:
        condition: Boolean array to use in where
        true_fn: Zero-argument function to construct the values used in the True
        condition.  Tensors that this function closes over will be extracted
        automatically to implement the double-where trick to suppress spurious NaN
        propagation.
        false_fn: False branch equivalent of true_fn

    Returns:
        Resulting PyTree equivalent to tree_map line above.
    """
    true_result = true_fn()
    false_result = false_fn()

    # Apply the double-where trick
    true_part = ops.select(condition, true_result,
                           ops.stop_gradient(true_result))
    false_part = ops.select(
        condition, ops.stop_gradient(false_result), false_result)

    return ops.select(condition, true_part, false_part)


def unstack(value: ms.Tensor, axis: int = -1) -> list[ms.Tensor]:
    """unstack"""
    split_tensors = []
    if len(value.shape) == 3:
        if axis == -1:
            split_tensors = [value[:, :, i] for i in range(value.shape[axis])]
        elif axis == -2:
            split_tensors = [value[:, i, :] for i in range(value.shape[axis])]
        else:
            split_tensors = [value[i, :, :] for i in range(value.shape[axis])]
    elif len(value.shape) == 2:
        if axis == -1:
            split_tensors = [value[:, i] for i in range(value.shape[axis])]
        else:
            split_tensors = [value[i, :] for i in range(value.shape[axis])]
    return split_tensors


def angdiff(alpha: ms.Tensor, beta: ms.Tensor) -> ms.Tensor:
    """Compute absolute difference between two angles."""
    d = alpha - beta
    d = (d + mnp.pi) % (2 * mnp.pi) - mnp.pi
    return d


def safe_arctan2(
        x1: ms.Tensor, x2: ms.Tensor, eps: float = 1e-8
) -> ms.Tensor:
    """Safe version of arctan2 that avoids NaN gradients when x1=x2=0."""

    return safe_select(
        ops.abs(x1) + ops.abs(x2) < eps,
        lambda: ops.zeros_like(ops.atan2(x1, x2)),
        lambda: ops.atan2(x1, x2),
    )


def weighted_mean(
        *,
        weights: ms.Tensor,
        value: ms.Tensor,
        axis: Optional[Union[int, Iterable[int]]] = None,
        eps: float = 1e-10,
) -> ms.Tensor:
    """Computes weighted mean in a safe way that avoids NaNs.

    This is equivalent to jnp.average for the case eps=0.0, but adds a small
    constant to the denominator of the weighted average to avoid NaNs.
    'weights' should be broadcastable to the shape of value.

    Args:
        weights: Weights to weight value by.
        value: Values to average
        axis: Axes to average over.
        eps: Epsilon to add to the denominator.

    Returns:
        Weighted average.
    """

    weights = ops.cast(weights, value.dtype)
    weights = ops.broadcast_to(weights, value.shape)

    weights_shape = weights.shape

    if isinstance(axis, numbers.Integral):
        axis = [axis]
    elif axis is None:
        axis = list(range(len(weights_shape)))

    numerator = ops.reduce_sum(weights * value, axis=tuple(axis))
    denominator = ops.reduce_sum(weights, axis=tuple(axis)) + eps

    return numerator / denominator
