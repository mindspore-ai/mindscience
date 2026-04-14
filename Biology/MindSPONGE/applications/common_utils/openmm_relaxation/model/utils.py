# Copyright 2023 Huawei Technologies Co., Ltd & CPL YiQin GAO Research Group
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
'''utils'''
import collections
import numbers
import numpy as np


def batched_gather(params, indices, axis=0):
    """batched_gather."""
    take_fn = lambda p, i: np.take(p, i, axis=axis)
    return take_fn(params, indices)


def mask_mean(mask, value, axis=None, drop_mask_channel=False, eps=1e-10):
    """Masked mean."""
    if drop_mask_channel:
        mask = mask[..., 0]

    mask_shape = mask.shape
    value_shape = value.shape

    assert len(mask_shape) == len(value_shape)

    if isinstance(axis, numbers.Integral):
        axis = [axis]
    elif axis is None:
        axis = list(range(len(mask_shape)))
    assert isinstance(axis, collections.Iterable), (
        'axis needs to be either an iterable, integer or "None"')

    broadcast_factor = 1.
    for axis_ in axis:
        value_size = value_shape[axis_]
        mask_size = mask_shape[axis_]
        if mask_size == 1:
            broadcast_factor *= value_size
        else:
            assert mask_size == value_size
    return (mask * value).sum(axis=tuple(axis)) / (mask.sum(axis=tuple(axis)) * broadcast_factor + eps)
