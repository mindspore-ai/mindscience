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
# pylint: disable-all
from functools import wraps

import numpy as np


class dotdict(dict):
    """
    dot.notation access to dictionary attributes.
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def list_to_str(nums, precision=3):
    """
    list to str for displaying errors and metrics.
    """
    if nums is None:
        return ""
    if not isinstance(nums, (list, tuple, np.ndarray)):
        return "{:.{}e}".format(nums, precision)
    return "[{:s}]".format(", ".join(["{:.{}e}".format(x, precision) for x in nums]))


def make_config(model_params):
    dynamic = dotdict()
    dynamic.num_IRK_stages = model_params['num_IRK_stages']
    dynamic.state_dim = 4
    dynamic.activation = model_params['dyn_activation']
    dynamic.initializer = "Glorot normal"
    dynamic.dropout_rate = 0
    dynamic.batch_normalization = None if model_params['dyn_bn'] == "no-bn" else model_params['dyn_bn']
    dynamic.layer_normalization = None if model_params['dyn_ln'] == "no-ln" else model_params['dyn_ln']
    dynamic.type = model_params['dyn_type']

    if model_params['unstacked']:
        dim_out = dynamic.state_dim * (dynamic.num_IRK_stages + 1)
    else:
        dim_out = dynamic.num_IRK_stages + 1

    if model_params['use_input_layer']:
        dynamic.layer_size = [dynamic.state_dim * 5] + \
            [model_params['dyn_width']] * model_params['dyn_depth'] + [dim_out]
    else:
        dynamic.layer_size = [dynamic.state_dim] + \
            [model_params['dyn_width']] * model_params['dyn_depth'] + [dim_out]

    algebraic = dotdict()
    algebraic.num_IRK_stages = model_params['num_IRK_stages']
    dim_out_alg = algebraic.num_IRK_stages + 1
    algebraic.layer_size = [dynamic.state_dim] + \
        [model_params['alg_width']] * model_params['alg_depth'] + [dim_out_alg]
    algebraic.activation = model_params['alg_activation']
    algebraic.initializer = "Glorot normal"
    algebraic.dropout_rate = 0
    algebraic.batch_normalization = None if model_params['alg_bn'] == "no-bn" else model_params['alg_bn']
    algebraic.layer_normalization = None if model_params['alg_ln'] == "no-ln" else model_params['alg_ln']
    algebraic.type = model_params['alg_type']
    return dynamic, algebraic
