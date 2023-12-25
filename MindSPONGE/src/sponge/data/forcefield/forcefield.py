# Copyright 2021-2023 @ Shenzhen Bay Laboratory &
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


def get_forcefield(forcefield: Union[str, dict, list, tuple]) -> Tuple[dict, dict]:
    """ Get force field parameters from YAML file.

    Args:
        forcefield (dict, str, list or tuple):
                            Force field parameters. It can be a `dict` of force field parameters,
                            a `str` of filename of a force field file in MindSPONGE YAML format,
                            or a `list` or `tuple` containing multiple `dict` or `str`.
                            If a filename is given, it will first look for a file with the same name
                            in the current directory. If the file does not exist, it will search
                            in MindSPONGE's built-in force field.
                            If multiple sets of parameters are given and the same atom type
                            is present in different parameters, then the atom type in the parameter
                            at the back of the array will replace the one at the front.

    Returns:
        parameters (dict), Force field parameters.
        template (dict), Molecular template.

    """

    if forcefield is None:
        return None, None

    if isinstance(forcefield, (list, tuple)):
        parameters = {}
        template = []
        for ff in forcefield:
            params, temp = get_forcefield(ff)
            template.append(temp)
            parameters = update_dict(parameters, params)
        template = get_template(template)
    else:
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
                raise ValueError(f'Cannot find force field parameters file: {forcefield}.')

            forcefield = read_yaml(filename)

        if not isinstance(forcefield, dict):
            raise TypeError(f'The type of forcefield must be str or dict but got: {type(forcefield)}')

        template = None
        if 'template' in forcefield.keys():
            template = get_template(forcefield.pop('template'))

        if 'parameters' in forcefield.keys():
            parameters = forcefield.get('parameters')
        else:
            parameters = forcefield

    return parameters, template
