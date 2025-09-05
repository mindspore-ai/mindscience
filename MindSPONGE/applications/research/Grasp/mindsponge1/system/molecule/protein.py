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
Protein modeling.
"""

import numpy as np
from mindspore.common import Tensor
from .molecule import Molecule
from ..residue.amino import AminoAcid
from ..modeling.hadder import ReadPdbByMindsponge as read_pdb
from ...data.template import get_template


backbone_atoms = np.array(['N', 'CA', 'C', 'O'], np.str_)
include_backbone_atoms = np.array(['OXT'], np.str_)


class Protein(Molecule):
    r"""
    Protein molecule.

    Args:
        pdb (str):                         Atoms in system. Can be list of str or int. Default: None.
        sequence (list):                   Atom type. Can be ndarray or list of str. Default: None.
        coordinate (Tensor):               Tensor of shape (B, A, D) or (1, A, D). Data type is float.
                                           Position coordinates of atoms. Default: None.
        pbc_box (Tensor):                  Tensor of shape (B, D) or (1, D). Data type is float.
                                           Box of periodic boundary condition. Default: None.
        template (Union[dict, str]):       Template of residue.
                                           The key of the dict are base, template, the name of molecule and so on.
                                           The value of the dict is file name.
                                           Default: 'protein0.yaml'
        ignore_hydrogen (bool, optional):  Ignore hydrogen. Default: True.
        length_unit (str):                 Length unit for position coordinates. Default: None.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Symbols:
        B:  Batchsize, i.e. number of walkers in simulation.
        A:  Number of atoms.
        D:  Dimension of the simulation system. Usually is 3.
    """

    def __init__(self,
                 pdb: str = None,
                 sequence: list = None,
                 coordinate: Tensor = None,
                 pbc_box: Tensor = None,
                 template: dict = 'protein0.yaml',
                 ignore_hydrogen: bool = True,
                 length_unit: str = None,
                 ):

        super().__init__(length_unit=length_unit)

        if pdb is None:
            #TODO
            if sequence is None:
                raise ValueError('At least 1 of pdb name and residue sequence should be given.')
        else:
            _, residue_name, _, coordinate, residue_pointer, flatten_atoms, flatten_crds, init_res_names,\
                init_res_ids, \
                _, _, _, _, _ = read_pdb(
                    pdb, ignore_hydrogen)

            if len(residue_name) > 1:
                if residue_name[0] != 'ACE' and residue_name[0] != 'NME':
                    residue_name[0] = 'N' + residue_name[0]
                if residue_name[-1] != 'ACE' and residue_name[-1] != 'NME':
                    residue_name[-1] = 'C' + residue_name[-1]

            self.init_resname = init_res_names
            self.init_resid = init_res_ids
            num_residue = len(residue_name)
            residue_pointer = np.append(residue_pointer, len(flatten_atoms))
            template = get_template(template)

            self.residue = []
            for i in range(num_residue):
                name = residue_name[i]
                atom_name = flatten_atoms[residue_pointer[i]: residue_pointer[i + 1]][None, :]
                residue = AminoAcid(name=name, template=template, atom_name=atom_name)
                self.residue.append(residue)

            coordinate = flatten_crds * self.units.convert_length_from('A')

        self.build_system()
        self.build_space(coordinate, pbc_box)

    def get_head_atom(self, residue_index, this_atom_names):
        if residue_index == 0:
            return None
        for index, atom in enumerate(this_atom_names[0]):
            if atom == 'N':
                return np.array([index], np.int32)
        return self

    def get_tail_atom(self, this_atom_names):
        for index, atom in enumerate(this_atom_names[0]):
            if atom == 'C':
                return np.array([index], np.int32)
        return self
