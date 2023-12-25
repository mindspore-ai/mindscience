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
Read template
"""

import os
from typing import Union, List, Tuple
import numpy as np
from numpy import ndarray

from ..data import update_dict, read_yaml


def get_template(template: Union[dict, str, List[Union[dict, str]], Tuple[Union[dict, str]]],
                 residue_name: str = None) -> dict:
    """
    Get template for molecule or residue.

    Args:
        template (Union[dict, str, List[Union[dict, str]], Tuple[Union[dict, str]]]):
                            Template for molecule or residue. It can be a `dict` of template,
                            a `str` of filename of a template file in MindSPONGE YAML format,
                            or a `list` or `tuple` containing multiple `dict` or `str`.
                            If a filename is given, it will first look for a file with the same name
                            in the current directory. If the file does not exist, it will search
                            in MindSPONGE's built-in templates.
                            If multiple sets of templates are given and the same atomic perproties
                            is present in different templates, then the atomic perproties in the template
                            at the back of the array will replace the one at the front.

        residue_name (str): Name of the residue to be extracted.
                            If `None` is given, the full template dict will be return.
                            Default: ``None``.

    Returns:
        template (dict), Template for molecule or residue

    """

    if template is None or not template:
        return None

    if isinstance(template, str):
        if os.path.exists(template):
            filename = template
        else:
            directory, _ = os.path.split(os.path.realpath(__file__))
            filename = os.path.join(directory, template)
            if not os.path.exists(filename):
                raise ValueError('Cannot find template file: "'+template+'".')
        template: dict = read_yaml(filename)
    elif isinstance(template, (list, tuple)):
        template_ = {}
        for temp in template:
            temp = get_template(temp)
            template_ = update_dict(template_, temp)
        template = template_
    elif not isinstance(template, dict):
        raise TypeError(f'The type of template must be str, dict or list but got: {type(template)}')

    if 'template' in template.keys():
        template = get_template(template.get('template'))

    if 'base' in template.keys():
        base = get_template(template.pop('base'))
        template = update_dict(base, template)

    if residue_name is not None:
        if residue_name in template.keys():
            template = template.get(residue_name)
        else:
            raise ValueError(f'Cannot find the residue name "{residue_name}" in template.')

    if template:
        return template

    return None


def get_template_index(template: dict, names: ndarray, key: str = 'atom_name') -> ndarray:
    """get atom index of system according to atom names"""
    reference: list = template.get(key)
    index = [reference.index(name) for name in names.reshape(-1).tolist()]
    index = np.array(index, np.int32).reshape(names.shape)
    unknown_name = (index >= len(reference))
    if unknown_name.any():
        raise ValueError(f'Unknown name: {names[unknown_name]}')
    return index


def get_molecule(template: Union[dict, str]) -> Tuple[dict, dict]:
    """
    Get molecular template.

    Args:
        template (str or dict): The file name of template.

    Returns:
        template (dict), Molecular template.

    """

    if isinstance(template, str):
        if os.path.exists(template):
            filename = template
        else:
            directory, _ = os.path.split(os.path.realpath(__file__))
            filename = os.path.join(directory, template)
            if not os.path.exists(filename):
                raise ValueError('Cannot find template file: "'+template+'".')
        template: dict = read_yaml(filename)
    elif not isinstance(template, dict):
        raise TypeError(f'The type of template must be str or dict but got: {type(template)}')

    if 'molecule' in template.keys():
        molecule: dict = template.get('molecule')
        template: dict = get_template(template)
    else:
        raise ValueError('Cannot find "molecule" in template')

    for res in molecule.get('residue'):
        if res not in template.keys():
            raise ValueError(f'Cannot find residue name "{res}" in template!')

    return molecule, template
