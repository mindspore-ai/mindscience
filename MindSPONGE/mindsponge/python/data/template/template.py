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
Read template
"""

import os
import yaml
import numpy as np
from numpy import ndarray


def get_template(template: str) -> dict:
    """ Get molecular template.

    Args:

        template (str or dict): The file name of template.

    Returns:

        template (dict):  Force field parameters

    """

    if isinstance(template, dict):
        return template

    if not isinstance(template, str):
        raise TypeError('The type of template must be dict or str but got: ' +
                        str(type(template)))

    if os.path.exists(template):
        filename = template
    else:
        directory, _ = os.path.split(os.path.realpath(__file__))
        filename = os.path.join(directory, template)
        if not os.path.exists(filename):
            raise ValueError('Cannot find template file: "'+template+'".')

    with open(filename, 'r', encoding="utf-8") as file:
        template = yaml.safe_load(file.read())

    return template

def get_template_index(template: dict, names: ndarray, key: str = 'atom_name') -> ndarray:
    """get atom index of system according to atom names"""
    reference: list = template.get(key)
    index = [reference.index(name) for name in names.reshape(-1).tolist()]
    index = np.array(index, np.int32).reshape(names.shape)
    unknown_name = (index >= len(reference))
    if unknown_name.any():
        raise ValueError('Unknown name: ' + str(names[unknown_name]))
    return index
