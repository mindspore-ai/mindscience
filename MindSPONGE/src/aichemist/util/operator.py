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
External
"""
from typing import Union
import mindspore as ms
import mindspore.numpy as mnp
import numpy as np


def clip(x: ms.Tensor or np.ndarray,
         xmin: float or int = None,
         xmax: float or int = None) -> ms.Tensor or np.ndarray:
    """
    For the original API of clip() in Mindspore, both xmin and xmax are required.
    Here, the extended clip() is implemented in which either xmin or xmax are allowed to be None.
    """
    if xmin is not None:
        x[x < xmin] = xmin
    if xmax is not None:
        x[x > xmax] = xmax
    return x


def where(condition: ms.Tensor, x: ms.Tensor = None, y: ms.Tensor = None):
    """
    For the original API of where() in Mindspore, both x and y are required.
    Here, the extended where() is implemented to be as same as Numpy-style API.
    When only condition is provided, this function is a shorthand for ms.Tensor(condition).nonzero().
    """
    ix = condition.nonzero()
    if x is None and y is None:
        return ix
    return mnp.where(condition, x, y)


def argwhere(condition: ms.Tensor):
    """
    Find the indices of array elements that are non-zero, grouped by element.
    """
    ix = condition.nonzero()
    return ms.Tensor(ix.asnumpy(), dtype=ix.dtype)


def bincount(x: ms.Tensor, weights: ms.Tensor = None, minlength: int = None):
    """_summary_

    Args:
        x (ms.Tensor): _description_
        weights (ms.Tensor, optional): _description_. Defaults to None.
        minlength (int, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    if weights is None:
        y = mnp.bincount(x, minlength=minlength).astype(x.dtype)
        return y
    return mnp.bincount(x, weights=weights, minlength=minlength)


def repeat(x: ms.Tensor, repeats, axis=0):
    """_summary_

    Args:
        x (ms.Tensor): _description_
        repeats (_type_): _description_
        axis (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """
    if isinstance(repeats, ms.Tensor):
        repeats = repeats.asnumpy().tolist()
    return mnp.repeat(x, repeats, axis=axis)


def flatten(x: np.ndarray or ms.Tensor, start: int = 0, end: int = -1) -> np.ndarray or ms.Tensor:
    """_summary_

    Args:
        x (np.ndarrayorms.Tensor): _description_
        start (int, optional): _description_. Defaults to 0.
        end (int, optional): _description_. Defaults to -1.

    Returns:
        np.ndarray or ms.Tensor: _description_
    """
    shape = list(x.shape)
    n_dim = x.ndim
    start = n_dim + start if start < 0 else start
    end = n_dim + end + 1 if end < 0 else end + 1
    assert start < end
    new_shape = [np.prod(shape[start:end])]
    if start is not None and start != 0:
        new_shape = shape[:start] + new_shape
    if end is not None and end != len(shape):
        new_shape += shape[end:]
    return x.reshape(new_shape)


def to_tensor(x: Union[list, np.ndarray, tuple], dtype=None):
    """_summary_

    Args:
        x (Union[list, np.ndarray, tuple]): _description_
        dtype (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    if isinstance(x, (list, np.ndarray, tuple)):
        x = ms.Tensor(x, dtype=dtype)
        if x.dtype == ms.float64:
            x = x.astype(ms.float32)
        elif x.dtype == ms.int64:
            x = x.astype(ms.int32)
    return x


def to_array(x: ms.Tensor):
    """_summary_

    Args:
        x (ms.Tensor): _description_

    Returns:
        _type_: _description_
    """
    if isinstance(x, ms.Tensor):
        x = x.asnumpy()
    return x


def is_numeric(value):
    types = (np.ndarray, ms.Tensor, int, float, bool)
    return isinstance(value, types)


def to_device(data, detach=True):
    """_summary_

    Args:
        data (_type_): _description_
        detach (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    if not detach:
        data = data.to_tensor() if hasattr(data, 'to_tensor') else to_tensor(data)
    else:
        data = data.to_array() if hasattr(data, 'to_array') else to_array(data)
    return data


def batch_to_device(data: Union[tuple, list, dict], detach=True):
    """_summary_

    Args:
        data (Union[tuple, list, dict]): _description_
        detach (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    if isinstance(data, (tuple, list)):
        data = list(data)
        for i, d in enumerate(data):
            if isinstance(d, (tuple, list, dict)):
                data[i] = batch_to_device(d, detach=detach)
            else:
                data[i] = to_device(d, detach=detach)
    elif isinstance(data, dict):
        for k, d in data.items():
            if isinstance(d, (tuple, list, dict)):
                data[k] = batch_to_device(d, detach=detach)
            else:
                data[k] = to_device(d, detach=detach)
    else:
        raise ValueError("Input data has not supported format!")
    return data
