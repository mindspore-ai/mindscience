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
import yaml


def get_forcefield_parameters(forcefield: str) -> dict:
    """ Get force field parameters from YAML file.

    Args:

        forcefield (str or dict):   The file name of force field parameters.

    Returns:

        parameters (dict):  Force field parameters

    """

    if isinstance(forcefield, dict):
        return forcefield

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

        with open(filename, 'r', encoding="utf-8") as file:
            parameters = yaml.safe_load(file.read())

        return parameters

    raise TypeError('The type of forcefield must be str or dict but got: '+str(type(forcefield)))
