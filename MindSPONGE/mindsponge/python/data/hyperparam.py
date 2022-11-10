# Copyright 2021-2022 @ Shenzhen Bay Laboratory &
#                       Peking University &
#                       Huawei Technologies Co., Ltd
#
# This code is a part of MindSPONGE:
# MindSpore Simulation Package tOwards Next Generation molecular modelling.
#
# MindSPONGE is open-source software based on the AI-framework:
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
Functions for read and write hyperparameters in checkpoint file
"""

import numpy as np
from mindspore import Tensor
from mindspore.nn import Cell, CellList
from mindspore.train import load_checkpoint
from ..function.functions import get_integer


def str_to_tensor(string: str) -> Tensor:
    """
    encode string to Tensor[int]

    Args:
        string (str):    The input string.

    Returns:
        Tensor[int].

    Supported Platforms:
        ``Ascend`` ``GPU``
    """
    if isinstance(string, (list, tuple)):
        string = ' '.join(string)
    return Tensor(np.fromstring(string, dtype=np.int8))


def tensor_to_str(tensor: Tensor) -> str:
    """
    decode to Tensor[int] to string

    Args:
        tensor (Tensor[int]):   The input tensor.

    Returns:
        string(str).

    Supported Platforms:
        ``Ascend`` ``GPU``
    """
    tensor = Tensor(tensor).asnumpy()
    string = tensor.tostring().decode()
    string = string.split()
    if len(string) == 1:
        string = string[0]
    return string


def get_class_parameters(hyper_param: dict, prefix: str, num_class: int = 1) -> dict:
    """
    get hyperparameter from Cell class.

    Args:
        hyper_param (dict): A dict of hyperparameters.
        prefix (str):       Only parameters starting with the prefix will be loaded.
        num_class (int):    The number of the class. Default: 1

    Returns:
        hyperparameters, dict.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """
    def _get_class_parameters(hyper_param: dict, prefix: str) -> dict:
        new_params = {}
        idx = len(prefix) + 1
        for name, param in hyper_param.items():
            if name.find(prefix) == 0 \
                    and (name == prefix or name[len(prefix)] == "." or (prefix and prefix[-1] == ".")):
                new_params[name[idx:]] = param
        if 'name' in new_params.keys():
            new_params['name'] = get_hyper_string(new_params, 'name')
            if len(new_params) == 1:
                new_params = new_params.get('name')

        if new_params:
            return new_params
        return None

    if num_class == 1:
        return _get_class_parameters(hyper_param, prefix)

    param_list = []
    for i in range(num_class):
        param_list.append(_get_class_parameters(
            hyper_param, prefix+'.'+str(i)))
    return param_list


def get_hyper_parameter(hyper_param: dict, prefix: str):
    """
    get hyperparameter.

    Args:
        hyper_param (dict): A dict of hyperparameters.
        prefix (str):       Only parameters starting with the prefix will be loaded.

    Returns:
        hyper_param[prefix], Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """
    if prefix in hyper_param.keys():
        return Tensor(hyper_param[prefix])
    return None


def get_hyper_string(hyper_param: dict, prefix: str):
    """
    get string type hyperparameter.

    Args:
        hyper_param (dict): A dict of hyperparameters.
        prefix (str):       Only parameters starting with the prefix will be loaded.

    Returns:
        str. String type hyperparameter.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """
    if prefix in hyper_param.keys():
        string = hyper_param[prefix]
        if isinstance(string, str):
            return string
        return tensor_to_str(string)
    return None


def set_hyper_parameter(hyper_param: dict, prefix: str, param: None):
    """
    put param into hyper_param.

    Args:
        hyper_param (dict):         A dict of hyperparameters.
        prefix (str):               Only parameters starting with the prefix will be loaded.
        param (Union[str, Tensor]): Parameters need to be put into the hyperparameter dict. Default: None

    Supported Platforms:
        ``Ascend`` ``GPU``
    """
    if param is None:
        if prefix in hyper_param.keys():
            hyper_param.pop(prefix)
    else:
        if isinstance(param, str):
            hyper_param[prefix] = str_to_tensor(param)
        else:
            hyper_param[prefix] = param


def set_class_parameters(hyper_param: list, prefix: str, cell: Cell):
    """
    put hyperparameters into Cell class.

    Args:
        hyper_param (dict): A dict of hyperparameters.
        prefix (str):       Only parameters starting with the prefix will be loaded.
        cell (Cell):        A neural network cell.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """
    def _set_class_parameters(hyper_param: dict, prefix: str, cell: Cell):
        if isinstance(cell, Cell):
            if 'hyper_param' in cell.__dict__.keys():
                for key, param in cell.hyper_param.items():
                    set_hyper_parameter(hyper_param, prefix+'.'+key, param)
            else:
                set_hyper_parameter(hyper_param, prefix +
                                    '.name', cell.__class__.__name__)
        elif isinstance(cell, str):
            set_hyper_parameter(hyper_param, prefix, cell)
        elif cell is not None:
            raise TypeError('The type of "cls" must be "Cell", "str" or list of them, but got "' +
                            str(type(cell))+'".')

    if isinstance(cell, (CellList, list)):
        for i, c in enumerate(cell):
            _set_class_parameters(hyper_param, prefix+'.'+str(i), c)
    else:
        _set_class_parameters(hyper_param, prefix, cell)


def load_hyper_param_into_class(cls_dict: dict, hyper_param: dict, types: dict, prefix: str = ''):
    """
    load hyperparameter into Cell class.

    Args:
        cls_dict (dict):    A dict of cls.
        hyper_param (dict): A dict of hyperparameters.
        types (dict):       A dict of types of values.
        prefix (str):       Only parameters starting with the prefix will be loaded. Default: ''

    Supported Platforms:
        ``Ascend`` ``GPU``
    """
    if prefix:
        prefix = prefix + '.'
    for key, value_type in types.items():
        if value_type == 'str':
            cls_dict[key] = get_hyper_string(hyper_param, prefix+key)
        elif value_type == 'int':
            cls_dict[key] = get_integer(hyper_param[prefix+key])
        elif value_type == 'class':
            num_class = 1
            num_key = 'num_' + key
            if num_key in cls_dict.keys():
                num_class = get_integer(cls_dict[prefix+num_key])
                cls_dict[key] = num_class
            cls_dict[key] = get_class_parameters(
                hyper_param, prefix+key, num_class)
        else:
            cls_dict[key] = get_hyper_parameter(hyper_param, prefix+key)


def set_class_into_hyper_param(hyper_param: dict, types: dict, cls: Cell, prefix: str = ''):
    """
    take hyperparameter from Cell class.

    Args:
        hyper_param (dict): A dict of hyperparameters.
        types (dict):       A dict of types of values.
        cls (Cell):         A neural network cell.
        prefix (str):       Only parameters starting with the prefix will be loaded. Default: ''

    Supported Platforms:
        ``Ascend`` ``GPU``
    """
    #pylint: disable=protected-access
    if prefix:
        prefix = prefix + '.'
    for key, value_type in types.items():
        if value_type == 'Cell':
            if key in cls._cells.keys():
                if cls._cells[key] is not None:
                    set_class_parameters(
                        hyper_param, prefix+key, cls._cells[key])
        else:
            if key in cls.__dict__.keys():
                set_hyper_parameter(hyper_param, prefix+key, cls.__dict__[key])
            elif key in cls._tensor_list.keys():
                set_hyper_parameter(hyper_param, prefix +
                                    key, cls._tensor_list[key])


def load_hyperparam(ckpt_file_name, prefix='hyperparam', dec_key=None, dec_mode="AES-GCM"):
    """
    Load hyperparam from checkpoint file (.ckpt).

    Args:
        ckpt_file_name (str):                       Checkpoint file name.
        prefix (Union[str, list[str], tuple[str]]): Only parameters starting with the prefix
                                                    will be loaded. Default: 'hyperparam'
        dec_key (Union[None, bytes]):               Byte type key used for decryption. If the value is None,
                                                    the decryption is not required. Default: None
        dec_mode (str):                             This parameter is valid only when dec_key is not set to None.
                                                    Specifies the decryption mode, currently supports 'AES-GCM'
                                                    and 'AES-CBC'. Default: 'AES-GCM'

    Returns:
        Dict, key is parameter name, value is a Parameter.

    Raises:
        ValueError: Checkpoint file is incorrect.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> from mindspore import load_hyperparam
        >>>
        >>> ckpt_file_name = "molct.ckpt"
        >>> hyper_dict = load_hyperparam(ckpt_file_name, prefix="hyper")
        >>> print(hyper_dict["hyper.dim_feature"])
        Tensor(shape=[1], dtype=Int8, value= [128])
    """

    return load_checkpoint(ckpt_file_name, dec_key=dec_key, dec_mode=dec_mode, specify_prefix=prefix)
