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
Force field parameters
"""
import os
from typing import Union, Tuple

from ..data import read_yaml, update_dict
from ..template import get_template


def get_forcefield(forcefield: Union[str, dict, list]) -> Tuple[dict, dict]:
    """
    Get force field parameters from YAML file.

    Args:
        forcefield (str, dict or list): The file name of force field parameters.

    Returns:
        parameters (dict), Force field parameters.
        template (dict), Molecular template.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """

    if forcefield is None:
        return None, None

    if isinstance(forcefield, str):
        if os.path.exists(forcefield):
            filename = forcefield
        else:
            filename = forcefield.lower()
            if os.path.splitext(forcefield)[-1] != '.yaml':
                filename += '.yaml'

            directory, _ = os.path.split(os.path.realpath(__file__))
            filename = os.path.join(directory, filename)
        if not os.path.exists(filename):
            raise ValueError('Cannot find force field parameters file: "'+forcefield+'".')

        forcefield: dict = read_yaml(filename)
    elif isinstance(forcefield, (list, tuple)):
        parameters = {}
        template = []
        for ff in forcefield:
            params, temp = get_forcefield(ff)
            template.append(temp)
            parameters = update_dict(parameters, params)
        template = get_template(template)
    elif not isinstance(forcefield, dict):
        raise TypeError('The type of forcefield must be str or dict but got: '+str(type(forcefield)))

    template = None
    if 'template' in forcefield.keys():
        template = get_template(forcefield.pop('template'))

    if 'parameters' in forcefield.keys():
        parameters = forcefield.get('parameters')
    else:
        parameters = forcefield

    return parameters, template
