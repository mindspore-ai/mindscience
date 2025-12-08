# Copyright 2024 DeepMind Technologies Limited
# Copyright (C) 2025 Huawei Technologies Co., Ltd
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md
#
# Modifications by Huawei Technologies Co., Ltd: Adapt to run by MindSpore on Ascend

"""Specialized mapping functions."""

from collections.abc import Callable, Sequence
import functools
from typing import Any, Union, Optional
import mindspore as ms


Pytree = Any
PytreeJaxArray = Any

partial = functools.partial
PROXY = object()


def _maybe_slice(array, i, slice_size, axis):
    "modified to mindspore"
    if axis is PROXY:
        return array
    start = [0]*array.ndim
    start[axis] = i
    size = list(array.shape)
    size[axis] = slice_size
    return ms.ops.slice(array, start, size)


def _maybe_get_size(array, axis):
    "modified to mindspore"
    if axis == PROXY:
        return -1
    return array.shape[axis]


def tree_flatten(tree):
    """tree flatten"""
    if isinstance(tree, (list, tuple)):
        flat, structure = [], []
        for item in tree:
            sub_flat, sub_struct = tree_flatten(item)
            flat.extend(sub_flat)
            structure.append(sub_struct)
        return flat, structure
    elif isinstance(tree, dict):
        flat, structure = [], {}
        for key, value in tree.items():
            sub_flat, sub_struct = tree_flatten(value)
            flat.extend(sub_flat)
            structure[key] = sub_struct
        return flat, structure
    else:
        return [tree], None


def tree_unflatten(flat, structure):
    """tree unflatten"""
    if isinstance(structure, list):
        result, idx = [], 0
        for sub_struct in structure:
            sub_tree, idx = tree_unflatten(flat[idx:], sub_struct)
            result.append(sub_tree)
        return result, idx
    elif isinstance(structure, dict):
        result, idx = {}, 0
        for key, sub_struct in structure.items():
            sub_tree, idx = tree_unflatten(flat[idx:], sub_struct)
            result[key] = sub_tree
        return result, idx
    else:
        return flat[0], 1


def _expand_axes(axes, values, name="sharded_apply"):
    values_tree_def = tree_flatten(values)[1]
    # flat_axes = tree_flatten(axes)[0]
    flat_axes = [PROXY if axes is None else axes for _ in values_tree_def]
    expanded_axes, _ = tree_unflatten(flat_axes, values_tree_def)
    return expanded_axes


def tree_map(fn, *trees):
    "Mindspore do not have the same function like Jax.tree.map, so try to write a mindspore version."
    tree_types = {type(tree) for tree in trees}
    tree_type = tree_types.pop()
    if tree_type in (list,):
        return tree_type(tree_map(fn, *subtrees) for subtrees in zip(*trees))
    if tree_type is dict:
        keys = trees[0].keys()
        if not all(tree.keys() == keys for tree in trees):
            raise ValueError("All input dictionaries must have the same keys")
        return {key: tree_map(fn, *(tree[key] for tree in trees)) for key in keys}
    return fn(*trees)


def tree_leaves(tree):
    "same as tree_map"
    if isinstance(tree, (list, tuple)):
        leaves = []
        for item in tree:
            leaves.extend(tree_leaves(item))
        return leaves
    if isinstance(tree, dict):
        leaves = []
        for key in tree:
            leaves.extend(tree_leaves(tree[key]))
        return leaves
    return [tree]


def eval_shape(fun, *args, **kwargs):
    fake_inputs = [ms.ops.zeros(arg.shape, dtype=arg.dtype) if isinstance(
        arg, ms.Tensor) else arg for arg in args]
    output = fun(*fake_inputs, **kwargs)
    return output


def sharded_apply(
        fun: Callable[..., PytreeJaxArray],
        shard_size: Optional[int] = 1,
        in_axes: Union[int, Pytree] = 0,
        out_axes: Union[int, Pytree] = 0,
        new_out_axes: bool = False,
) -> Callable[..., PytreeJaxArray]:
    """Sharded apply.

    Applies `fun` over shards to axes, in a way similar to vmap,
    but does so in shards of `shard_size`. Shards are stacked after.
    This allows a smooth trade-off between
    memory usage (as in a plain map) vs higher throughput (as in a vmap).

    Args:
        fun: Function to apply smap transform to.
        shard_size: Integer denoting shard size.
        in_axes: Either integer or pytree describing which axis to map over for each
            input to `fun`, None denotes broadcasting.
        out_axes: Integer or pytree denoting to what axis in the output the mapped
            over axis maps.
        new_out_axes: Whether to stack outputs on new axes. This assumes that the
            output sizes for each shard (including the possible remainder shard) are
            the same.

    Returns:
        Function with smap applied.
    """
    docstr = (
        "Mapped version of {fun}. Takes similar arguments to {fun} "
        "but with additional array axes over which {fun} is mapped."
    )
    if new_out_axes:
        raise NotImplementedError("New output axes not yet implemented.")

    # shard size None denotes no sharding
    if shard_size is None:
        return fun

    def mapped_fn(*args, **kwargs):
        # Expand in axes and determine loop range.
        in_axes_ = _expand_axes(ms.Tensor(in_axes), args)

        in_sizes = tree_map(_maybe_get_size, list(args), in_axes_)
        in_size = max(tree_leaves(in_sizes))

        num_extra_shards = (in_size - 1) // shard_size

        # Fix if necessary.
        last_shard_size = in_size % shard_size
        last_shard_size = shard_size if last_shard_size == 0 else last_shard_size

        def apply_fun_to_slice(slice_start, slice_size, args, in_axes_):
            input_slice = tree_map(
                lambda array, axis: _maybe_slice(
                    array, slice_start, slice_size, axis
                ),
                args,
                in_axes_,
            )
            return fun(input_slice, **kwargs)

        remainder_shape_dtype = eval_shape(
            lambda array, axis: apply_fun_to_slice(
                0, last_shard_size, array, axis),
            args, in_axes_
        )

        out_shapes = tree_map(lambda x: x.shape, remainder_shape_dtype)
        out_dtypes = tree_map(lambda x: x.dtype, remainder_shape_dtype)
        out_axes_ = _expand_axes(out_axes, out_shapes)

        if num_extra_shards > 0:
            regular_shard_shape_dtype = eval_shape(
                lambda array, axis: apply_fun_to_slice(
                    0, shard_size, array, axis),
                args, in_axes_
            )
            shard_shapes = tree_map(
                lambda x: x.shape, regular_shard_shape_dtype)

            def make_output_shape(axis, shard_shape, remainder_shape):
                axis = axis if isinstance(axis, int) else int(axis[0])
                shard_shape = tuple(shard_shape)
                remainder_shape = tuple(remainder_shape)
                return ms.ops.stack(
                    shard_shape[:axis]
                    + (shard_shape[axis] * num_extra_shards +
                       remainder_shape[axis],)
                    + shard_shape[axis + 1:]
                )

            out_shapes = tree_map(
                make_output_shape, out_axes_[0], ms.Tensor(
                    shard_shapes), ms.Tensor(out_shapes)
            )

        # Calls dynamic Update slice with different argument order.
        # This is here since tree_map only works with positional arguments.
        def dynamic_update_slice_in_dim(array, slice_size, axis, i):
            start = [0]*array.ndim
            start[axis] = int(i)
            size = list(array.shape)
            size[axis] = slice_size.shape[axis]
            # return ms.ops.slice(array, start, size)
            end = [x + y for x, y in zip(start, size)]
            array[start[0]: end[0]] = slice_size
            return array

        def compute_shard(outputs, slice_start, slice_size):
            def slice_op(array, axis):
                return apply_fun_to_slice(int(slice_start), shard_size, array, axis)
            slice_out = slice_op(args, in_axes_)
            update_slice = partial(dynamic_update_slice_in_dim, i=slice_start)
            # slice_out = (slice_out,) if not isinstance(slice, (int, float)) else [int(x) for x in slice_out]
            return tree_map(update_slice, outputs, slice_out, out_axes_[0])

        def scan_iteration(outputs, i):
            new_outputs = compute_shard(outputs, i, shard_size)
            return new_outputs

        slice_starts = ms.ops.arange(0, in_size - shard_size + 1, shard_size)

        def allocate_buffer(dtype, shape):
            return ms.ops.zeros(shape, dtype=dtype)

        outputs = tree_map(allocate_buffer, out_dtypes, out_shapes)

        if slice_starts.shape[0] > 0:
            for slice_start in slice_starts:
                outputs = scan_iteration(outputs, slice_start)
            # scan_op = ms.ops.Scan()
            # outputs, _ = scan_op(scan_iteration, outputs, slice_starts)

        if last_shard_size != shard_size:
            remainder_start = in_size - last_shard_size
            outputs = compute_shard(outputs, remainder_start, last_shard_size)

        return outputs

    return mapped_fn


def sharded_map(fun, shard_size=1, in_axes=0, out_axes=0):
    vmapped_fun = ms.vmap(fun, int(in_axes), int(out_axes))
    return sharded_apply(vmapped_fun, shard_size, in_axes, out_axes)


def reshape_partitioned_inputs(batched_args, partitioned_dim, subbatch_size):
    """Reshapes so subbatching doesn't happen on the partitioned dim."""
    subbatched_args = []
    for arg in batched_args:
        shape = arg.shape
        new_shape = (
            shape[:partitioned_dim]
            + (subbatch_size, shape[partitioned_dim] // subbatch_size)
            + shape[partitioned_dim + 1:]
        )
        subbatched_args.append(arg.reshape(new_shape))
    return subbatched_args


def reshape_partitioned_output(output, output_subbatch_dim):
    """Reshapes outputs as if reshape_partitioned_inputs were never applied."""
    out_shape = (
        output.shape[: output_subbatch_dim - 1]
        + (-1,)
        + output.shape[output_subbatch_dim + 1:]
    )
    return output.reshape(out_shape)


def inference_subbatch(module, subbatch_size, batched_args,
                       nonbatched_args, input_subbatch_dim=0, output_subbatch_dim=None,
                       input_subbatch_dim_is_partitioned=False):
    """Run through subbatches (like batch apply but with split and concat)."""
    if output_subbatch_dim is None:
        output_subbatch_dim = input_subbatch_dim
    if input_subbatch_dim_is_partitioned:
        # Subbatching along the partitioned axis would induce an all-gather that
        # undoes the partitioning. So instead we reshape such that
        # [..., partitioned_input_size, ...] becomes [..., subbatch_size,
        # partitioned_input_size // subbatch_size, ...] and then actually subbatch
        # along the partitioned_input_size // subbatch_size axis in slices of
        # size 1. Partitioning is then preserved on the partitioned axis, except
        # that dimension is now of size subbatch_size instead of
        # partitioned_input_size. Note that the module itself still sees inputs of
        # size [..., subbatch_size, ...], just as it would if this reshaping were
        # not applied.
        batched_args = reshape_partitioned_inputs(
            batched_args, input_subbatch_dim, subbatch_size
        )
        input_subbatch_dim += 1
        output_subbatch_dim += 1
        subbatch_size = 1

    def run_module(*batched_args):
        if input_subbatch_dim_is_partitioned:
            # Squeeze off the singleton dimension (otherwise the module would see
            # [..., subbatch_size, 1, ...]).
            batched_args = [b.squeeze(axis=input_subbatch_dim)
                            for b in batched_args]
        args = list(batched_args)[0] + list(nonbatched_args)
        res = module(*args)
        if input_subbatch_dim_is_partitioned:
            # Add back in the singleton dimension so the outputs are stacked on the
            # axis we are actually subbatching over (i.e stacked back to
            # [..., subbatch_size, partitioned_input_size // subbatch_size, ...]),
            # rather than on the partitioned axis, which would again induce an
            # all-gather that breaks partitioning.
            res = ms.ops.expand_dims(res, axis=output_subbatch_dim)
        return res
    sharded_module = sharded_apply(
        run_module,
        shard_size=subbatch_size,
        in_axes=input_subbatch_dim,
        out_axes=output_subbatch_dim,
    )
    output = sharded_module(*batched_args)
    if input_subbatch_dim_is_partitioned:
        # The is of the same shape as the inputs [..., subbatch_size,
        # partitioned_input_size // subbatch_size, ...]. Reshape to
        # [..., partitioned_input_size, ...] as if the reshaping due to partitioning
        # had never been applied.
        output = reshape_partitioned_output(output, output_subbatch_dim)

    return output
