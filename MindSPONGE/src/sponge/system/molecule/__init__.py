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
Molecules
"""

from typing import Union, List, Tuple
from numpy import ndarray
from mindspore import Tensor
from .molecule import Molecule, _MoleculeFromPDB, MoleculeFromMol2
from .protein import Protein


def get_molecule(pdb_name: str, pbc_box: Union[Tensor, ndarray, List[float]] = None, length_unit: str = None,
                 template: Union[dict, str, List[Union[dict, str]], Tuple[Union[dict, str]]] = None,
                 rebuild_hydrogen: bool = False):
    r"""
    Base function for get molecular system, used as the "system module" in MindSPONGE.
    The `Molecule` Cell can represent a molecule or a system consisting of multiple molecules.
    The major components of the `Molecule` Cell is the `Residue` Cell. A `Molecule` Cell can
    contain multiple `Residue` Cells.

    Args:
        pdb_name(str): The string format pdb file name.
        pbc_box(Tensor): The periodic boundary box of given system.
        length_unit(str): The length unit for input and output.
        template(list, str): The template for build the system.
        rebuild_hydrogen(bool): Decide to rebuild all the hydrogen atoms or not.

    Outputs:
        - coordinate, Tensor of shape `(B, A, D)`. Data type is float.
        - pbc_box, Tensor of shape `(B, D)`. Data type is float.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Note:
        B:  Batchsize, i.e. number of walkers in simulation
        A:  Number of atoms.
        b:  Number of bonds.
        D:  Spatial dimension of the simulation system. Usually is 3.
    """
    if pdb_name.endswith('.pdb'):
        return _MoleculeFromPDB(pdb_name,
                                pbc_box=pbc_box,
                                length_unit=length_unit,
                                template=template,
                                rebuild_hydrogen=rebuild_hydrogen)
    else:
        raise ValueError('Only pdb format is supported in this function, but got {}.'.format(pdb_name.split()[-1]))


__all__ = ['Molecule', 'Protein', 'MoleculeFromMol2', 'get_molecule']
