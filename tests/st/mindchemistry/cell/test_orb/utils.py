# ============================================================================
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
"""Utils"""
import os
import sys
from typing import Any

import numpy as np
from mindspore import Tensor

# pylint: disable=C0413
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(PROJECT_ROOT)
from common.cell import compare_output, FP32_ATOL, FP32_RTOL


def tensor_to_numpy(data: Any) -> Any:
    """Convert MindSpore Tensors to NumPy arrays recursively.
    This function traverses the input data structure and converts all MindSpore Tensors
    to NumPy arrays, while leaving other data types unchanged.
    Args:
        data (Any): Input data which can be a MindSpore Tensor, dict, list, tuple, or other types.
    Returns:
        Any: Data structure with MindSpore Tensors converted to NumPy arrays.
    """
    if isinstance(data, Tensor):
        return data.numpy()
    if isinstance(data, dict):
        return {k: tensor_to_numpy(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return type(data)(tensor_to_numpy(v) for v in data)
    return data


def numpy_to_tensor(data: Any) -> Any:
    """Convert NumPy arrays to MindSpore Tensors recursively.
    This function traverses the input data structure and converts all NumPy arrays
    to MindSpore Tensors, while leaving other data types unchanged.
    Args:
        data (Any): Input data which can be a NumPy array, dict, list, tuple, or other types.
    Returns:
        Any: Data structure with NumPy arrays converted to MindSpore Tensors.
    """
    if isinstance(data, np.ndarray):
        return Tensor(data)
    if isinstance(data, dict):
        return {k: numpy_to_tensor(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return type(data)(numpy_to_tensor(v) for v in data)
    return data


def is_equal(a: Any, b: Any) -> bool:
    """Compare two objects for equality with special handling for different types.

    This function performs a deep comparison between two objects, supporting:
    - NumPy arrays (using tolerance-based comparison)
    - Dictionaries (recursive comparison of values)
    - Lists and tuples (element-wise comparison)
    - NamedTuples (field-wise comparison)
    - Other types (using standard equality comparison)

    Args:
        a (Any): First object to compare
        b (Any): Second object to compare

    Returns:
        bool: True if objects are considered equal, False otherwise

    Examples:
        >>> is_equal(np.array([1.0]), np.array([1.0]))
        True
        >>> is_equal({'a': 1, 'b': 2}, {'a': 1, 'b': 2})
        True
        >>> is_equal([1, 2, 3], [1, 2, 3])
        True
    """
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return compare_output(a, b, FP32_ATOL, FP32_RTOL)
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return False
        return all(is_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(is_equal(x, y) for x, y in zip(a, b))
    if hasattr(a, "_fields") and hasattr(b, "_fields"):
        if a._fields != b._fields:
            return False
        return all(is_equal(getattr(a, f), getattr(b, f)) for f in a._fields)
    return a == b
