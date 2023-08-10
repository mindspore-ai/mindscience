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
Base function for yaml
"""

import os
import stat
from itertools import permutations
import yaml
import numpy as np
from numpy import ndarray

from mindspore.train._utils import _make_directory


_cur_dir = os.getcwd()


__all__ = [
    'update_dict',
    'read_yaml',
    'write_yaml',
    'get_bonded_types',
    'get_dihedral_types',
    'get_improper_types',
    'get_parameter_terms',
]


def update_dict(origin: dict, addition: dict = None) -> dict:
    """
    update complex dict.

    Args:
        origin (dict):      Original dict to be updated.
        addition (dict):    Dict that need to be added to the original dict.

    Returns:
        dict, new dict.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    """
    if addition is None:
        return origin
    dictionary = origin.copy()
    origin.update()
    for k, v in addition.items():
        if k in dictionary.keys() and isinstance(dictionary.get(k), dict) and isinstance(v, dict):
            dictionary[k] = update_dict(dictionary[k], v)
        else:
            dictionary[k] = v
    return dictionary


def write_yaml(data: dict, filename: str, directory: str = None):
    """
    write YAML file.

    Args:
        data(dict):     Dict for output.
        filename(str):  Name of YAML file.


    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    """

    if directory is None:
        directory = _cur_dir
    else:
        directory = _make_directory(directory)

    filename = os.path.join(directory, filename)

    with os.fdopen(os.open(filename, os.O_WRONLY | os.O_CREAT, stat.S_IRWXU), 'w', encoding='utf-8') as file:
        yaml.dump(data, file, sort_keys=False)


def read_yaml(filename: str) -> dict:
    """
    read YAML file.

    Args:
        filename(str):  Name of YAML file.

    Returns:
        data(dict):     Data read from the YAML file.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    """
    with open(filename, 'r', encoding="utf-8") as file:
        data = yaml.safe_load(file.read())
    return data


def get_bonded_types(atom_type: ndarray, symbol: str = '-'):
    """
    get the types of bonded terms including bond, angle and dihedral.

    Args:
        atom_type(ndarray): types of atoms.
        symbol(str): a symbol.

    Returns:
        types(ndarray), types of bonded terms including bond, angle and dihedral.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    num_atoms = atom_type.shape[-1]

    if num_atoms == 1:
        return atom_type

    types = atom_type[..., 0]
    for i in range(1, num_atoms):
        types = np.char.add(types, symbol)
        types = np.char.add(types, atom_type[..., i])

    return types


def get_dihedral_types(atom_type: ndarray, symbol: str = '-'):
    """
    The multi atom name constructor.

    Args:
        atom_type(ndarray): types of atoms.
        symbol(str): a symbol.

    Returns:
        - types, ndarray.
        - inverse_types, ndarray.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    num_atoms = atom_type.shape[-1]

    if num_atoms == 1:
        return atom_type

    types = atom_type[..., 0]
    for i in range(1, num_atoms):
        types = np.char.add(types, symbol)
        types = np.char.add(types, atom_type[..., i])

    inverse_types = atom_type[..., -1]
    for i in range(1, num_atoms):
        inverse_types = np.char.add(inverse_types, symbol)
        inverse_types = np.char.add(inverse_types, atom_type[..., -1-i])

    return types, inverse_types


def get_improper_types(atom_type: ndarray, symbol: str = '-'):
    """
    The multi atom name constructor.

    Args:
        atom_type(ndarray): types of atoms.
        symbol(str): a symbol.

    Returns:
        - permuation_types, tuple.
        - orders, tuple.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    num_atoms = atom_type.shape[-1]

    if num_atoms == 1:
        return atom_type

    permuation_types = ()
    orders = ()
    for combination in permutations(range(num_atoms)):
        types = atom_type[..., combination[0]]
        for i in range(1, num_atoms):
            types = np.char.add(types, symbol)
            types = np.char.add(types, atom_type[..., combination[i]])
        permuation_types += (types,)
        orders += (combination,)

    return permuation_types, orders

def get_parameter_terms(parameters: dict,
                        name_key: str = 'parameter_names',
                        param_key: str = 'parameters',
                        pattern_key: str = 'pattern',
                        ) -> dict:
    """
    get the parameter terms of force field

    Args:
        parameters (dict): force field parameters.
        name_key (str): key of the parameter names. Default: 'parameter_names'
        param_key (str): key of the parameters. Default: 'parameters'
        pattern_key (str): key of the parameter pattern. Default: 'pattern'

    Returns:
        - parameter_terms, dict.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    parameter_names: dict = parameters.get(name_key)
    if parameter_names is None:
        raise ValueError(f'Cannot find the key "{name_key}" in parameters!')
    parameter_names: list = parameter_names.get(pattern_key)
    if parameter_names is None:
        raise ValueError(f'Cannot find the key "{name_key}" in parameter_names!')
    params: dict = parameters.get(param_key)

    terms = {}
    for i, key in enumerate(parameter_names):
        terms[key] = {k: v[i] for k, v in params.items()}

    return terms
