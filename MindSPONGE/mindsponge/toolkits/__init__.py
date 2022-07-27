# Copyright 2021 The AIMM Group at Shenzhen Bay Laboratory & Peking University & Huawei Technologies Co., Ltd
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
Xponge, a lightweight and easy-customizing python package for pre- and post- process of molecular modelling
"""
__version__ = "stable-1.2.6"

import os
import time
import sys
from collections import OrderedDict, deque
from itertools import product, permutations

import numpy as np

from . import assign
from .helper import GlobalSetting, Type, ResidueType, Entity, Atom, Residue, ResidueLink, Molecule, AtomType, \
    set_global_alternative_names, generate_new_pairwise_force_type, generate_new_bonded_force_type
from .load import load_ffitp, load_mol2, load_rst7, load_frcmod, load_pdb, load_parmdat
from .build import save_mol2, save_pdb, save_sponge_input, save_gro
from .process import impose_bond, impose_angle, impose_dihedral, add_solvent_box, h_mass_repartition, solvent_replace, \
    main_axis_rotate


def _initialize():
    """

    :return:
    """
    set_global_alternative_names(globals(), True)
    AtomType.New_From_String("name\nUNKNOWN")

    def write_residue(self):
        towrite = "%d %d\n" % (len(self.atoms), len(self.residues))
        towrite += "\n".join([str(len(res.atoms)) for res in self.residues])
        return towrite

    Molecule.Set_Save_SPONGE_Input("residue")(write_residue)

    def write_coordinate(self):
        towrite = "%d\n" % (len(self.atoms))
        boxlength = [0, 0, 0, self.box_angle[0], self.box_angle[1], self.box_angle[2]]
        maxi = [-float("inf"), -float("inf"), -float("inf")]
        mini = [float("inf"), float("inf"), float("inf")]
        for atom in self.atoms:
            if atom.x > maxi[0]:
                maxi[0] = atom.x
            if atom.y > maxi[1]:
                maxi[1] = atom.y
            if atom.z > maxi[2]:
                maxi[2] = atom.z
            if atom.x < mini[0]:
                mini[0] = atom.x
            if atom.y < mini[1]:
                mini[1] = atom.y
            if atom.z < mini[2]:
                mini[2] = atom.z
        if not GlobalSetting.nocenter and self.box_length is None:
            towrite += "\n".join(
                ["%f %f %f" % (atom.x - mini[0] + 3, atom.y - mini[1] + 3, atom.z - mini[2] + 3) for atom in
                 self.atoms])
        else:
            towrite += "\n".join(["%f %f %f" % (atom.x, atom.y, atom.z) for atom in self.atoms])
        if self.box_length is None:
            boxlength[0] = maxi[0] - mini[0] + 6
            boxlength[1] = maxi[1] - mini[1] + 6
            boxlength[2] = maxi[2] - mini[2] + 6
            self.box_length = [boxlength[0], boxlength[1], boxlength[2]]
        else:
            boxlength[0] = self.box_length[0]
            boxlength[1] = self.box_length[1]
            boxlength[2] = self.box_length[2]
        towrite += "\n%f %f %f %f %f %f" % (
            boxlength[0], boxlength[1], boxlength[2], boxlength[3], boxlength[4], boxlength[5])
        return towrite

    Molecule.Set_Save_SPONGE_Input("coordinate")(write_coordinate)


_initialize()
