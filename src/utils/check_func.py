# Copyright 2021 Huawei Technologies Co., Ltd
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
"""functions"""
from __future__ import absolute_import

from mindspore import context

_SPACE = " "


def _convert_to_tuple(params):
    if params is None:
        return params
    if not isinstance(params, (list, tuple)):
        params = (params,)
    if isinstance(params, list):
        params = tuple(params)
    return params


def check_param_type(param, param_name, data_type=None, exclude_type=None):
    """Check parameter's data type"""
    data_type = _convert_to_tuple(data_type)
    exclude_type = _convert_to_tuple(exclude_type)

    if data_type and not isinstance(param, data_type):
        raise TypeError("The type of {} should be instance of {}, but got {} with type {}".format(
            param_name, data_type, param, type(param)))
    if exclude_type and type(param) in exclude_type:
        raise TypeError("The type of {} should not be instance of {}, but got {} with type {}".format(
            param_name, exclude_type, param, type(param)))
    return None


def check_param_value(param, param_name, valid_value):
    """check parameter's value"""
    valid_value = _convert_to_tuple(valid_value)
    if param not in valid_value:
        raise ValueError("The value of {} should be in {}, but got {}".format(
            param_name, valid_value, param))


def check_param_type_value(param, param_name, valid_value, data_type=None, exclude_type=None):
    """check both data type and value"""
    check_param_type(param, param_name, data_type=data_type, exclude_type=exclude_type)
    check_param_value(param, param_name, valid_value)


def check_dict_type(param_dict, param_name, key_type=None, value_type=None):
    """check data type for key and value of the specified dict"""
    check_param_type(param_dict, param_name, data_type=dict)

    for key in param_dict.keys():
        if key_type:
            check_param_type(key, _SPACE.join(("key of", param_name)), data_type=key_type)
        if value_type:
            values = _convert_to_tuple(param_dict[key])
            for value in values:
                check_param_type(value, _SPACE.join(("value of", param_name)), data_type=value_type)
    return None


def check_dict_value(param_dict, param_name, key_value=None, value_value=None):
    """check values for key and value of specified dict"""
    check_param_type(param_dict, param_name, data_type=dict)

    for key in param_dict.keys():
        if key_value:
            check_param_value(key, _SPACE.join(("key of", param_name)), key_value)
        if value_value:
            values = _convert_to_tuple(param_dict[key])
            for value in values:
                check_param_value(value, _SPACE.join(("value of", param_name)), value_value)
    return None


def check_dict_type_value(param_dict, param_name, key_type=None, value_type=None, key_value=None, value_value=None):
    """check values for key and value of specified dict"""
    check_dict_type(param_dict, param_name, key_type=key_type, value_type=value_type)
    check_dict_value(param_dict, param_name, key_value=key_value, value_value=value_value)
    return None


def check_mode(api_name):
    """check running mode"""
    if context.get_context("mode") == context.PYNATIVE_MODE:
        raise RuntimeError("{} is only supported GRAPH_MODE now but got PYNATIVE_MODE".format(api_name))


def check_param_no_greater(param, param_name, compared_value):
    """ Check whether the param less than the given compared_value"""
    if param > compared_value:
        raise ValueError("The value of {} should be no greater than {}, but got {}".format(
            param_name, compared_value, param))


def check_param_odd(param, param_name):
    """ Check whether the param is an odd number"""
    if param % 2 == 0:
        raise ValueError("The value of {} should be an odd number, but got {}".format(
            param_name, param))


def check_param_even(param, param_name):
    """ Check whether the param is an even number"""
    for value in param:
        if value % 2 != 0:
            raise ValueError("The value of {} should be an even number, but got {}".format(
                param_name, param))


def check_lr_param_type_value(param, param_name, param_type, thresh_hold=0, restrict=False, exclude=None):
    if (exclude and isinstance(param, exclude)) or not isinstance(param, param_type):
        raise TypeError("the type of {} should be {}, but got {}".format(param_name, param_type, type(param)))
    if restrict:
        if param <= thresh_hold:
            raise ValueError("the value of {} should be > {}, but got: {}".format(param_name, thresh_hold, param))
    else:
        if param < thresh_hold:
            raise ValueError("the value of {} should be >= {}, but got: {}".format(param_name, thresh_hold, param))
