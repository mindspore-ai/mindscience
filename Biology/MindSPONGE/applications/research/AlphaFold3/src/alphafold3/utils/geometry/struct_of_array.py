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

"""Class decorator to represent (nested) struct of arrays."""

import dataclasses
import mindspore as ms

def get_item(instance, key):
    sliced = {}
    for field in get_array_fields(instance):
        num_trailing_dims = field.metadata.get('num_trailing_dims', 0)
        this_key = key
        if isinstance(key, tuple) and Ellipsis in this_key:
            this_key += (slice(None),) * num_trailing_dims

        def apply_slice(x):
            if isinstance(x, ms.Tensor):
                return x[this_key]
            elif isinstance(x, dict):
                return {k: apply_slice(v) for k, v in x.items()}
            elif isinstance(x, list):
                return [apply_slice(item) for item in x]
            else:
                return x

        sliced[field.name] = apply_slice(getattr(instance, field.name))

    return dataclasses.replace(instance, **sliced)


@property
def get_shape(instance):
    """Returns Shape for given instance of dataclass."""
    first_field = dataclasses.fields(instance)[0]
    num_trailing_dims = first_field.metadata.get('num_trailing_dims', None)
    value = getattr(instance, first_field.name)
    if num_trailing_dims:
        return value.shape[:-num_trailing_dims]
    else:
        return value.shape


def get_len(instance):
    """Returns length for given instance of dataclass."""
    shape = instance.shape
    if shape:
        return shape[0]
    else:
        # Match utils.numpy behavior.
        raise TypeError('len() of unsized object')


@property
def get_dtype(instance):
    """Returns Dtype for given instance of dataclass."""
    fields = dataclasses.fields(instance)
    sets_dtype = [
        field.name for field in fields if field.metadata.get('sets_dtype', False)
    ]
    if sets_dtype:
        assert len(sets_dtype) == 1, 'at most one field can set dtype'
        field_value = getattr(instance, sets_dtype[0])
    elif instance.same_dtype:
        field_value = getattr(instance, fields[0].name)
    else:
        raise AttributeError(
            'Trying to access Dtype on Struct of Array without'
            'either "same_dtype" or field setting dtype'
        )

    if hasattr(field_value, 'dtype'):
        return field_value.dtype
    else:
        raise AttributeError(f'field_value {field_value} does not have dtype')


def replace(instance, **kwargs):
    return dataclasses.replace(instance, **kwargs)


def post_init(instance):
    """Validate instance has same shapes & dtypes."""
    array_fields = get_array_fields(instance)
    arrays = list(get_array_fields(instance, return_values=True).values())
    first_field = array_fields[0]
    try:
        dtype = instance.dtype
    except AttributeError:
        dtype = None
    if dtype is not None:
        first_shape = instance.shape
        for array, field in zip(arrays, array_fields, strict=True):
            num_trailing_dims = field.metadata.get('num_trailing_dims', None)
            if num_trailing_dims:
                array_shape = array.shape
                field_shape = array_shape[:-num_trailing_dims]
                msg = (
                    f'field {field} should have number of trailing dims'
                    ' {num_trailing_dims}'
                )
                assert len(array_shape) == len(
                    first_shape) + num_trailing_dims, msg
            else:

                field_shape = array.shape

            shape_msg = (
                f"Stripped Shape {field_shape} of field {field} doesn't "
                f'match shape {first_shape} of field {first_field}'
            )

            assert field_shape == first_shape, shape_msg

            field_dtype = array.dtype

            allowed_metadata_dtypes = field.metadata.get('allowed_dtypes', [])
            if allowed_metadata_dtypes:
                msg = f'Dtype is {field_dtype} but must be in {allowed_metadata_dtypes}'
                assert field_dtype in allowed_metadata_dtypes, msg

            if 'dtype' in field.metadata:
                target_dtype = field.metadata['dtype']
            else:
                target_dtype = dtype

            msg = f'Dtype is {field_dtype} but must be {target_dtype}'
            assert field_dtype == target_dtype, msg


def flatten(instance):
    """Flatten Struct Of Array instance."""
    array_likes = get_array_fields(instance, return_values=True).values()
    flat_array_likes = []
    inner_treedefs = []
    num_arrays = []
    for array_like in array_likes:
        flat_array_like, inner_treedef = tree_flatten(array_like)
        inner_treedefs.append(inner_treedef)
        flat_array_likes += flat_array_like
        num_arrays.append(len(flat_array_like))
    metadata = get_metadata_fields(instance, return_values=True)
    metadata = type(instance).metadata_cls(**metadata)
    return flat_array_likes, (inner_treedefs, metadata, num_arrays)


def make_metadata_class(cls):
    metadata_fields = get_fields(
        cls, lambda x: x.metadata.get('is_metadata', False)
    )
    metadata_cls = dataclasses.make_dataclass(
        cls_name='Meta' + cls.__name__,
        fields=[(field.name, field.type, field) for field in metadata_fields],
        frozen=True,
        eq=True,
    )
    return metadata_cls


def get_fields(cls_or_instance, filterfn, return_values=False):
    fields = dataclasses.fields(cls_or_instance)
    fields = [field for field in fields if filterfn(field)]
    if return_values:
        return {
            field.name: getattr(cls_or_instance, field.name) for field in fields
        }
    else:
        return fields


def get_array_fields(cls, return_values=False):
    return get_fields(
        cls,
        lambda x: not x.metadata.get('is_metadata', False),
        return_values=return_values,
    )


def get_metadata_fields(cls, return_values=False):
    return get_fields(
        cls,
        lambda x: x.metadata.get('is_metadata', False),
        return_values=return_values,
    )


def tree_flatten(pytree):
    """Custom tree flattening function for MindSpore tensors."""
    if isinstance(pytree, ms.Tensor):
        return [pytree], None
    elif isinstance(pytree, dict):
        keys, values = zip(*pytree.items())
        flat_values, treedefs = zip(*(tree_flatten(v) for v in values))
        return sum(flat_values, []), {'keys': keys, 'treedefs': treedefs}
    elif isinstance(pytree, list):
        flat_items, treedefs = zip(*(tree_flatten(item) for item in pytree))
        return sum(flat_items, []), {'treedefs': treedefs}
    else:
        return [], None


def tree_unflatten(treedef, leaves):
    """Custom tree unflattening function for MindSpore tensors."""
    if treedef is None:
        return leaves[0]
    elif isinstance(treedef, dict):
        if 'keys' in treedef:
            keys = treedef['keys']
            treedefs = treedef['treedefs']
            items = [tree_unflatten(td, leaves[i:i+1])
                     for i, td in enumerate(treedefs)]
            return dict(zip(keys, items))
        else:
            treedefs = treedef['treedefs']
            start = 0
            items = []
            for td in treedefs:
                size = len(tree_flatten(tree_unflatten(
                    td, leaves[start:start+1]))[0])
                items.append(tree_unflatten(td, leaves[start:start+size]))
                start += size
            return items
    else:
        return []


class StructOfArray:
    """Class Decorator for Struct Of Arrays."""

    def __init__(self, same_dtype=True):
        self.same_dtype = same_dtype

    def __call__(self, cls):
        cls.__array_ufunc__ = None
        cls.replace = replace
        cls.same_dtype = self.same_dtype
        cls.dtype = get_dtype
        cls.shape = get_shape
        cls.__len__ = get_len
        cls.__getitem__ = get_item
        cls.__post_init__ = post_init
        new_cls = dataclasses.dataclass(cls, frozen=True, eq=False)
        # pytree claims to require metadata to be hashable, not sure why,
        # But making derived dataclass that can just hold metadata
        new_cls.metadata_cls = make_metadata_class(new_cls)

        def unflatten(cls, params):
            aux, data = params
            inner_treedefs, metadata, num_arrays = aux
            array_fields = [field.name for field in get_array_fields(new_cls)]
            value_dict = {}
            array_start = 0
            for num_array, inner_treedef, array_field in zip(
                    num_arrays, inner_treedefs, array_fields, strict=True
            ):
                value_dict[array_field] = tree_unflatten(
                    inner_treedef, data[array_start: array_start + num_array]
                )
                array_start += num_array
            metadata_fields = get_metadata_fields(new_cls)
            for field in metadata_fields:
                value_dict[field.name] = getattr(metadata, field.name)

            return new_cls(**value_dict)

        # Override __flatten__ and __unflatten__ methods
        new_cls.__flatten__ = flatten
        new_cls.__unflatten__ = unflatten

        return new_cls
