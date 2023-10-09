# Copyright 2023 Huawei Technologies Co., Ltd
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
# ==============================================================================
"""ms utils"""
import numbers
import random

import numpy as np
import mindspore as ms
from mindspore import nn
from mindspore._c_expression import typing

from sciai.utils.check_utils import to_tuple, _check_type, _recursive_type_check


def str2datatype(type_str):
    """
    Map from float data type string to MindSpore data type.

    Args:
        type_str (str): Float data type string.

    Returns:
        dtype, MindSpore Tensor data type.
    """
    type_dict = {"float16": ms.float16, "float32": ms.float32, "float64": ms.float64}
    return type_dict.get(type_str, ms.float32)


def amp2datatype(type_str):
    """
    Map from auto mixed precision level string to MindSpore data type.
    Support amp level from `O0` to `O3`.

    Args:
        type_str (str): Auto mixed precision level string.

    Returns:
        dtype, MindSpore Tensor data type.
    """
    type_dict = {"O0": ms.float32, "O1": ms.float16, "O2": ms.float16, "O3": ms.float16}
    return type_dict.get(type_str)


def datatype2np(ms_type):
    """
    Map from MindSpore data type to numpy data type.

    Args:
        ms_type (dtype): Mindspore Tensor data type.

    Returns:
        numpy.dtype, NumPy data type.
    """
    type_dict = {ms.float16: np.float16, ms.float32: np.float32, ms.float64: np.float64}
    return type_dict.get(ms_type)


def to_tensor(tensors, dtype=ms.float32):  # pylint: disable=W0621
    """
    Cast array(ies)/tensor(s) to a given MindSpore data type.

    Args:
        tensors (Union[Tensor, ndarray, Number, np.floating, tuple[Tensor, ndarray]]): Tensor(s) to cast.
        dtype (type): MindSpore Tensor data type. Default: ms.float32.

    Returns:
        Union(Tensor, tuple(Tensor)), Single one or tuple of cast tensor(s).

    Raises:
        TypeError: If input types are not correct.

    Examples:
        >>> import numpy as np
        >>> from sciai.utils import to_tensor
        >>> tensors = to_tensor((np.array([1]), np.array([2])))
        >>> print(tensors)
        (Tensor(shape=[1], dtype=Float32, value= [ 1.00000000e+00]),
        Tensor(shape=[1], dtype=Float32, value= [ 2.00000000e+00]))
        >>> print(tensors[0].dtype)
        Float32
    """
    _check_type(dtype, "dtype", (typing.Number, np.floating))
    _check_type(tensors, "tensors",
                (typing.Number, np.floating, numbers.Number, np.ndarray, ms.Tensor, np.ndarray, tuple))
    tensors = to_tuple(tensors)
    for tensor in tensors:
        _check_type(tensor, "single tensor", (typing.Number, np.floating, numbers.Number, ms.Tensor, np.ndarray))
    np_type = datatype2np(dtype)
    dtype_tensors = []
    for tensor in tensors:
        if isinstance(tensor, np.ndarray):
            dtype_tensors.append(ms.Tensor(tensor.astype(np_type), dtype))
        elif isinstance(tensor, ms.Tensor):
            dtype_tensors.append(tensor.astype(dtype))
        else:
            dtype_tensors.append(ms.Tensor(tensor, dtype))
    return dtype_tensors[0] if len(dtype_tensors) == 1 else tuple(dtype_tensors)


def set_seed(seed=1234):
    """
    Set random seed everywhere.

    Args:
        seed (int): Seed number.

    Raises:
        TypeError: if `seed` is not an integer.
    """
    _check_type(seed, "seed", int)
    ms.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def to_float(cells, target_type=ms.float32):
    """
    Set cell(s) to a given data type.

    Args:
        cells (Union[Cell, list[Cell], tuple[Cell]]): Cells to cast.
        target_type (dtype): Target MindSpore data type that the cell(s) would be converted to.
    """
    _recursive_type_check(cells, nn.Cell)
    _check_type(target_type, "target_type", typing.Number)
    cells = to_tuple(cells)
    for cell in cells:
        cell.to_float(target_type)


def flatten_add_dim(*data):
    r"""
    flatten data and add an extra dimension at the end.

    Args:
        \*data (np.array): data to flatten and add dimension.

    Returns:
        tuple[np.array], converted data.
    """
    res = tuple(_.flatten()[:, None] for _ in data)
    return res[0] if len(res) == 1 else res


def calc_ckpt_name(args):
    """
    Concatenate optimal checkpoint name according to existing namespace arguments.

    Args:
        args (Namespace): Argument namespace.

    Returns:
        str, The concatenated checkpoint filename.
    """
    components = ["Optim"]
    if hasattr(args, "model_name"):
        components.append(args.model_name)
    if hasattr(args, "problem"):
        components.append(args.problem)
    if hasattr(args, "amp_level"):
        components.append(args.amp_level)
    return "_".join(components) + ".ckpt"
