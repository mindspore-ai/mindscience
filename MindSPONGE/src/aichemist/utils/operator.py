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
Operator
"""
import mindspore as ms
import numpy as np


def flatten(x: np.ndarray or ms.Tensor, start: int = 0, end: int = -1) -> np.ndarray or ms.Tensor:
    """
    flatten the given axes of the given tensor of ndarray

    Args:
        x (np.ndarray or ms.Tensor): given matrix
        start (int, optional): the start axis. Defaults to 0.
        end (int, optional): the end axis. Defaults to -1.

    Returns:
        x (np.ndarray or ms.Tensor): flatten matrix.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
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


def to_tensor(x: np.ndarray):
    """
    Cast numpy array to mindspore tensor

    Args:
        x (np.ndarray): array like matrix

    Returns:
        x (ms.Tensor): tensor like matrix

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    if not isinstance(x, np.ndarray):
        return x
    if 0 in x.shape:
        x = ms.Tensor([])
    else:
        x = ms.Tensor.from_numpy(x)
    return x


def to_array(x: ms.Tensor):
    """
    Cast mindspore tensor to numpy array

    Args:
        x (np.ndarray): tensor like matrix

    Returns:
        x (ms.Tensor): array like matrix

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    if isinstance(x, ms.Tensor):
        x = x.asnumpy()
    return x


def is_numeric(x):
    """
    Check if the value is numeric or not.

    Args:
        x (Generic): the given value

    Returns:
        y (bool): if the value is numeric or not

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    types = (np.ndarray, ms.Tensor, int, float, bool, np.integer, np.floating)
    return isinstance(x, types)


def to_device(data, detach=True):
    """_summary_

    Args:
        data (np.ndarray, ms.Tensor):   The input matrix
        detach (bool, optional):        If true, the data will be cast to np.ndarray, otherwise ms.Tensor
                                        Defaults to True.

    Returns:
        data: The output matrix

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    if not detach:
        data = data.to_tensor() if hasattr(data, 'to_tensor') else to_tensor(data)
    else:
        data = data.to_array() if hasattr(data, 'to_array') else to_array(data)
    return data
