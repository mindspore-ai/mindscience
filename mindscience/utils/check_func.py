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
# ==============================================================================
"""
Parameter Checking Functions


This module provides utility functions for checking parameters in the MindScience toolkit.
It includes functions for validating parameter types, values, and dictionary structures.
"""
# pylint: disable=C0123
from __future__ import absolute_import

from mindspore import context

_SPACE = " "
__all__ = ["check_param_type", "check_param_type_value",
           "check_param_value", "check_dict_type_value", "check_dict_type"]


def _convert_to_tuple(params):
    if params is None:
        return params
    if not isinstance(params, (list, tuple)):
        params = (params,)
    if isinstance(params, list):
        params = tuple(params)
    return params


def check_param_type(param, param_name, data_type=None, exclude_type=None):
    r"""
    Check parameter's data type.
    
    Args:
        param (any): The parameter to check.
        param_name (str): The name of the parameter.
        data_type (Union[type, tuple[type], list[type], None], optional): The allowed data types. Default: ``None``.
        exclude_type (Union[type, tuple[type], list[type], None], optional): The excluded data types. Default: ``None``.

    Raises:
        TypeError: If the data type of the `param` is not in the allowed data types.
        TypeError: If the data type of the `param` is in the excluded data types.
    """
    data_type = _convert_to_tuple(data_type)
    exclude_type = _convert_to_tuple(exclude_type)

    if data_type and not isinstance(param, data_type):
        raise TypeError(
            f"The type of {param_name} should be instance of {data_type}, but got {param} "
            f"with type {type(param)}"
        )
    if exclude_type and type(param) in exclude_type:
        raise TypeError(
            f"The type of {param_name} should not be instance of {exclude_type}, but got {param} "
            f"with type {type(param)}"
        )


def check_param_value(param, param_name, valid_value):
    r"""
    Check parameter's value.
    
    Args:
        param (any): The parameter to check.
        param_name (str): The name of the parameter.
        valid_value (Union[any, tuple, list]): The allowed values.

    Raises:
        ValueError: If the value of the `param` is not in the allowed values.
    """
    valid_value = _convert_to_tuple(valid_value)
    if param not in valid_value:
        raise ValueError(
            f"The value of {param_name} should be in {valid_value}, but got {param}"
        )


def check_param_type_value(param, param_name, valid_value, data_type=None, exclude_type=None):
    r"""
    Check both data type and value.
    
    Args:
        param (any): The parameter to check.
        param_name (str): The name of the parameter.
        valid_value (Union[any, tuple, list]): The allowed values.
        data_type (Union[type, tuple[type], list[type], None], optional): The allowed data types. Default: ``None``.
        exclude_type (Union[type, tuple[type], list[type], None], optional): The excluded data types. Default: ``None``.

    Raises:
        TypeError: If the data type of the `param` is not in the allowed data types.
        ValueError: If the value of the `param` is not in the allowed values.
    """
    check_param_type(param, param_name, data_type=data_type,
                     exclude_type=exclude_type)
    check_param_value(param, param_name, valid_value)


def check_dict_type(param_dict, param_name, key_type=None, value_type=None):
    r"""
    Check data type for key and value of the specified dict.
    
    Args:
        param_dict (dict): The dictionary to check.
        param_name (str): The name of the parameter. Used for error messages.
        key_type (Union[type, tuple[type], list[type], None], optional): The allowed key types. Default: ``None``.
        value_type (Union[type, tuple[type], list[type], None], optional): The allowed value types. Default: ``None``.

    """
    check_param_type(param_dict, param_name, data_type=dict)

    for key in param_dict.keys():
        if key_type:
            check_param_type(
                key, _SPACE.join(("key of", param_name)),
                data_type=key_type
            )
        if value_type:
            values = _convert_to_tuple(param_dict[key])
            for value in values:
                check_param_type(
                    value, _SPACE.join(("value of", param_name)), data_type=value_type
                )


def check_dict_value(param_dict, param_name, key_value=None, value_value=None):
    r"""
    Check values for key and value of specified dict.
    
    Args:
        param_dict (dict): The dictionary to check.
        param_name (str): The name of the parameter.
        key_value (Union[any, tuple, list, None], optional): The allowed key values. Default: ``None``.
        value_value (Union[any, tuple, list, None], optional): The allowed value values. Default: ``None``.

    Raises:
        TypeError: If the type of the `param_dict` is not dict.
        ValueError: If the value of the key of `param_dict` is not in the allowed key values.
        ValueError: If the value of the value of `param_dict` is not in the allowed value values.
    """
    check_param_type(param_dict, param_name, data_type=dict)

    for key in param_dict.keys():
        if key_value:
            check_param_value(
                key, _SPACE.join(("key of", param_name)),
                key_value
            )
        if value_value:
            values = _convert_to_tuple(param_dict[key])
            for value in values:
                check_param_value(
                    value, _SPACE.join(("value of", param_name)), value_value
                )


def check_dict_type_value(param_dict, param_name, key_type=None, value_type=None, key_value=None, value_value=None):
    r"""
    Check values for key and value of specified dict.
    
    Args:
        param_dict (dict): The dictionary to check.
        param_name (str): The name of the parameter.
        key_type (Union[type, tuple[type], list[type], None], optional): The allowed key types. Default: ``None``.
        value_type (Union[type, tuple[type], list[type], None], optional): The allowed value types. Default: ``None``.
        key_value (Union[any, tuple, list, None], optional): The allowed key values. Default: ``None``.
        value_value (Union[any, tuple, list, None], optional): The allowed value values. Default: ``None``.

    Raises:
        TypeError: If the type of the `param_dict` is not dict, or if the type of the key/value is not in the allowed types.
        ValueError: If the value of the key or value of `param_dict` is not in the allowed values.
    """
    check_dict_type(param_dict, param_name,
                    key_type=key_type, value_type=value_type)
    check_dict_value(param_dict, param_name,
                     key_value=key_value, value_value=value_value)


def check_mode(api_name):
    """check running mode"""
    if context.get_context("mode") == context.PYNATIVE_MODE:
        raise RuntimeError(
            f"{api_name} is only supported GRAPH_MODE now but got PYNATIVE_MODE")


def check_param_no_greater(param, param_name, compared_value):
    """ Check whether the param less than the given compared_value"""
    if param > compared_value:
        raise ValueError(
            f"The value of {param_name} should be no greater than {compared_value}, but got {param}"
        )


def check_param_odd(param, param_name):
    """ Check whether the param is an odd number"""
    if param % 2 == 0:
        raise ValueError(
            f"The value of {param_name} should be an odd number, but got {param}"
        )


def check_param_even(param, param_name):
    """ Check whether the param is an even number"""
    for value in param:
        if value % 2 != 0:
            raise ValueError(
                f"The value of {param_name} should be an even number, but got {param}"
            )


def check_lr_param_type_value(param, param_name, param_type, thresh_hold=0,
                              restrict=False, exclude=None):
    """Check the type and value of the learning rate parameter."""
    if (exclude and isinstance(param, exclude)) or not isinstance(param, param_type):
        raise TypeError(
            f"the type of {param_name} should be {param_type}, but got {type(param)}"
        )
    if restrict:
        if param <= thresh_hold:
            raise ValueError(
                f"the value of {param_name} should be > {thresh_hold}, but got: {param}"
            )
    else:
        if param < thresh_hold:
            raise ValueError(
                f"the value of {param_name} should be >= {thresh_hold}, but got: {param}"
            )
