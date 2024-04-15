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
Protein modeling.
"""

from typing import Union, List, Tuple
import numpy as np
from numpy import ndarray
from mindspore.common import Tensor
from .molecule import Molecule
from ..residue.amino import AminoAcid
from ..residue import Residue
from ..modelling.hadder import read_pdb
from ...data.template import get_template
from ...function import get_arguments


RESIDUE_NAMES = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HID', 'HIS',
                 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
backbone_atoms = np.array(['N', 'CA', 'C', 'O'], np.str_)
include_backbone_atoms = np.array(['OXT'], np.str_)


class Protein(Molecule):
    r"""Protein molecule, which is a subclass of `Molecule` Cell.

        Protein Cell can be initialized by accepting a PDB file, thus creating a `Molecule` Cell for the protein.

    Args:
        pdb (str):                  Filename of the PDB (Protein Data Bank) file. Default: ``None``.

        sequence (List[str]):       Sequence of the protein moleulce. Default: ``None``.

        coordinate (Union[Tensor, ndarray, List[float]]):
                                    Array of the position coordinates of atoms of the simulation system.
                                    The shape of the array is (A, D) or (B, A, D), and the data type is float.
                                    Default: ``None``.

        pbc_box (Union[Tensor, ndarray, List[float]]):
                                    Array of the Box of periodic boundary condition.
                                    The shape of the array is (D) or (B, D), and the data type is float.
                                    Default: ``None``.

        template (Union[dict, str]):
                                    Template for protein molecule. It can be a `dict` of template,
                                    a `str` of filename of a template file in MindSPONGE YAML format,
                                    If a filename is given, it will first look for a file with the same name
                                    in the current directory. If the file does not exist, it will search
                                    in MindSPONGE's built-in templates.

        rebuild_hydrogen (bool):    Whether to rebuild the hydrogen atoms of the protein molecule from PDB file.
                                    Default: ``False``.

        rebuild_suffix (str):       The suffix of the PDB file of the protetin module with rebuilt hydrogen.
                                    Default: '_addH'

        length_unit (str):          Length unit for position coordinates. Default: ``None``.

    Note:

        B:  Batchsize, i.e. number of walkers in simulation

        A:  Number of atoms.

        D:  Spatial dimension of the simulation system. Usually is 3.

    Examples:
        >>> from sponge import Protein
        >>> system = Protein('case1.pdb', rebuild_hydrogen=True)
        [MindSPONGE] Adding 57 hydrogen atoms for the protein molecule in 0.007 seconds.
        >>> print ('The number of atoms in the system is: ', system.num_atoms)
        The number of atoms in the system is:  57
        >>> print ('All the atom names in the system are: ', system.atom_name)
        All the atom names in the system are:  [['N' 'CA' 'CB' 'C' 'O' 'H1' 'H2' 'H3' 'HA' 'HB1' 'HB2' 'HB3' 'N' 'CA'
        'CB' 'CG' 'CD' 'NE' 'CZ' 'NH1' 'NH2' 'C' 'O' 'H' 'HA' 'HB2' 'HB3' 'HG2'
        'HG3' 'HD2' 'HD3' 'HE' 'HH11' 'HH12' 'HH21' 'HH22' 'N' 'CA' 'CB' 'C'
        'O' 'H' 'HA' 'HB1' 'HB2' 'HB3' 'N' 'CA' 'CB' 'C' 'O' 'OXT' 'H' 'HA'
        'HB1' 'HB2' 'HB3']]

    """

    def __init__(self,
                 pdb: str = None,
                 sequence: List[str] = None,
                 coordinate: Union[Tensor, ndarray, List[float]] = None,
                 pbc_box: Union[Tensor, ndarray, List[float]] = None,
                 template: Union[dict, str, List[Union[dict, str]], Tuple[Union[dict, str]]] = 'protein0.yaml',
                 rebuild_hydrogen: bool = False,
                 rebuild_suffix: str = '_addH',
                 length_unit: str = None,
                 **kwargs
                 ):

        super().__init__(length_unit=length_unit)
        self._kwargs = get_arguments(locals(), kwargs)

        if pdb is None:
            #TODO
            if sequence is None:
                raise ValueError('At least 1 of pdb name and residue sequence should be given.')
        else:
            pdb_obj = read_pdb(pdb, rebuild_hydrogen, rebuild_suffix=rebuild_suffix)
            residue_name = pdb_obj.res_names
            residue_pointer = pdb_obj.res_pointer
            flatten_atoms = pdb_obj.flatten_atoms
            flatten_crds = pdb_obj.flatten_crds
            init_res_names = pdb_obj.init_res_names
            init_res_ids = pdb_obj.init_res_ids
            chain_id = pdb_obj.chain_id
            self.chain_id = chain_id

            residue_names = np.array(RESIDUE_NAMES, np.str_)
            is_amino = np.isin(residue_name, residue_names)

            for i, res in enumerate(residue_name):
                if res == 'HIE':
                    residue_name[i] = 'HIS'
                if res == 'HOH':
                    residue_name[i] = 'WAT'
                if not is_amino[i]:
                    continue
                if i == 0:
                    residue_name[i] = 'N' * (res != 'ACE') + res
                    continue
                elif i == len(residue_name) - 1:
                    residue_name[i] = 'C' * (res != 'NME') + res
                    break
                if chain_id[i] < chain_id[i + 1]:
                    residue_name[i] = 'C' * (res != 'ACE') + res
                if chain_id[i] > chain_id[i - 1]:
                    residue_name[i] = 'N' * (res != 'ACE') + res

            self.init_resname = init_res_names
            self.init_resid = init_res_ids
            num_residue = len(residue_name)
            residue_pointer = np.append(residue_pointer, len(flatten_atoms))
            self.template = template
            template = get_template(template)

            self.residue = []
            for i in range(num_residue):
                name = residue_name[i]
                if name == 'HIE':
                    name = 'HIS'
                atom_name = flatten_atoms[residue_pointer[i]: residue_pointer[i + 1]][None, :]
                if name in RESIDUE_NAMES:
                    residue = AminoAcid(name=name, template=template, atom_name=atom_name)
                else:
                    residue = Residue(name=name, template=template, atom_name=atom_name,
                                      length_unit=self.units.length_unit)
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
