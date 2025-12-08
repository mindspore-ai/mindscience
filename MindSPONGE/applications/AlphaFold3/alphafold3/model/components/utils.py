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

"""Utils for components"""
import numbers

import numpy as np
import mindspore as ms

VALID_DTYPES = [np.float32, np.float64, np.int8, np.int32, np.int32, bool]


def remove_invalidly_typed_feats(batch):
    """Remove features of types we don't want to send to the TPU e.g. strings."""
    return {
        k: v
        for k, v in batch.items()
        if hasattr(v, 'dtype') and v.dtype in VALID_DTYPES
    }


def mask_mean(mask, value, axis=None, keepdims=False, eps=1e-10):
    """Masked mean."""

    mask_shape = mask.shape
    value_shape = value.shape

    if isinstance(axis, numbers.Integral):
        axis = [axis]
    elif axis is None:
        axis = list(range(len(mask_shape)))

    broadcast_factor = 1.0
    for axis_ in axis:
        value_size = value_shape[axis_]
        mask_size = mask_shape[axis_]
        if mask_size == 1:
            broadcast_factor *= value_size
        else:
            error = f'Shapes are not compatible, shapes: {mask_shape}, {value_shape}'
            if mask_size != value_size:
                raise ValueError(error)

    return ms.ops.sum(mask * value, keepdim=keepdims, dim=axis) / (
        ms.ops.maximum(
            ms.ops.sum(mask, keepdim=keepdims, dim=axis) *
            broadcast_factor, eps
        )
    )
