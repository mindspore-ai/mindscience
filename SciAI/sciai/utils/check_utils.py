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
"""check utils"""
from __future__ import absolute_import

_SPACE = " "


def to_tuple(data):
    """
    Convert a given param into tuple if it's not, otherwise keep it unchanged.

    Args:
        data (Union[tuple, list, any]): A tuple or not tuple data.

    Returns:
        Tuple, Converted tuple representation of input data.

    Examples:
        >>> from sciai.utils import to_tuple
        >>> a = 1
        >>> a_tuple = to_tuple(a)
        >>> print(a_tuple)
        (1,)
    """
    if isinstance(data, tuple):
        return data
    if isinstance(data, list):
        return tuple(data)
    return (data,)


def _batch_check_type(type_dict):
    """
    Check types of a batch of dict.

    Args:
        type_dict (dict): Type dictionary, where the key is the param name, and value is a tuple of the parameter
            to check and its target type(s), e.g., {"network": (network, nn.Cell)}.

    Raises:
        TypeError: If any check in the dict fails.
    """
    for param_name, (param, target_type) in type_dict.items():
        _check_type(param, param_name, target_type)


def _check_type(input_param, param_name, target_type=None, exclude_type=None):
    """
    Check types of a given parameter.

    Args:
        input_param (any): Parameter to check type.
        param_name (str): Parameter name.
        target_type (Union[type, None, tuple[type|None]]): Target type(s) for the parameter. Single `None` would skip
            target type check.
        exclude_type (Union[type, tuple]): Exclude type(s) for the parameter. Single `None` would skip exclude type
            check.

    Raises:
        TypeError: If check fails.
    """
    target_type = to_tuple(target_type)
    if None in target_type:
        if len(target_type) > 1 and input_param is not None:
            target_type = tuple(_ for _ in target_type if _ is not None)
            if not isinstance(input_param, target_type):
                raise TypeError(f"The input parameter '{param_name}' should be of type(s) {target_type}, "
                                f"but got {type(input_param)}")
    elif not isinstance(input_param, target_type):
        raise TypeError(f"The input parameter '{param_name}' should be of type(s) {target_type}, "
                        f"but got {type(input_param)}")
    if exclude_type and isinstance(input_param, exclude_type):
        raise TypeError(f"The input parameter {param_name} should not be instance of {exclude_type},"
                        f" but {type(input_param)}")


def _check_value_in(input_param, param_name, target_values):
    """
    Check whether the input_param is in given collection.

    Args:
        input_param (any): Parameter to be checked.
        param_name (str): Parameter name.
        target_values (Union[iterable, object]): Iterable collection to check whether the parameter is in, or single
            value to check equal.

    Raises:
        ValueError: If the input parameter is not in the collection.
        TypeError: If collection is not iterable.
    """
    input_param = to_tuple(input_param)
    target_values = to_tuple(target_values)
    if not all(single in target_values for single in input_param):
        raise ValueError(f"input argument {param_name} should all be among {target_values}, but got {input_param}")


def _recursive_type_check(data, dtype, collection_types=(list, tuple)):
    """
    Check whether the data is of given dtype(s) or of a bunch of given dtype(s).

    Args:
        data (Collection): Data object to check type.
        dtype (type): Types that every element in the collection should be.
        collection_types: Supported collection types. Default: (list, tuple).

    Returns:
        boolean, Whether the check succeeds.

    Raises:
        TypeError If elements in `data` are not of type `dtype`.
    """
    if isinstance(data, dtype):
        return True
    if not (isinstance(data, collection_types) and all(isinstance(single, dtype) for single in data)):
        raise TypeError(f"elements in {data} are not of type {dtype}.")
    return True


def _check_param_no_greater(param, param_name, compared_value):
    """ Check whether the param less than the given compared_value"""
    if param > compared_value:
        raise ValueError("The value of {} should be no greater than {}, but got {}".format(
            param_name, compared_value, param))


def _check_param_even(param, param_name):
    """ Check whether the param is an even number"""
    for value in param:
        if value % 2 != 0:
            raise ValueError("The value of {} should be an even number, but got {}".format(
                param_name, param))


class _Attr:
    """
    Abstract attribute getter and setter. Only dict and Namespace concerned recently.
    """

    @staticmethod
    def setattr(args, key, value):
        if isinstance(args, dict):
            args[key] = value
        else:
            setattr(args, key, value)

    @staticmethod
    def hasattr(args, key):
        if isinstance(args, dict):
            return key in args
        return hasattr(args, key)

    @staticmethod
    def getattr(args, key, default=None):
        if isinstance(args, dict):
            return args.get(key, default)
        return getattr(args, key, default)

    @staticmethod
    def all_attr(args):
        if isinstance(args, dict):
            return args.keys()
        return args.__dict__.keys()

    @staticmethod
    def all_items(args):
        if isinstance(args, dict):
            return args.items()
        return args.__dict__.items()
