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

.. TIP::

    When you use this **package** as an independent package, this package is called ``Xponge``; when you use it as \
a part of mindsponge, it is called ``mindsponge.toolkits``. The documatation is all written as ``Xponge``, but \
remember to change your codes for importing according to your environment!

name style
============

All modules follow the lower snake case such as ``forcefield.amber.ff14sb``

All classes follow the upper camel case such as ``ResidueType``, ``Molecule``, ``GromacsTopologyIterator``.

All functions can be used in four name styles. Take the function ``impose_bond`` as example::

    impose_bond == Impose_Bond == ImposeBond == imposeBond

For some abbreviations, all upper letters are also acceptable. Take the function ``load_pdb`` as example::

    load_pdb == Load_Pdb == Load_PDB == LoadPdb == LoadPDB == loadPdb == loadPDB

Functions in a Python class, no matter it is called a class method, a instance method or a static method follow the \
same name styles as the usual functions do. Take the function ``Residue.add_atom`` as  example::

    Residue.add_atom == Residue.Add_Atom == Residue.addAtom == Residue.AddAtom

name space
============

The submodules in ``forcefield`` will do some global configure settings after loading. For example, after the code::

    import Xponge.forcefield.amber.ff14sb

you can use all ``ResidueType`` instances in ff14sb then.

The names of all ``ResidueType`` instances will be loaded into the main dict, which means you can directly use it::

    import Xponge.forcefield.amber.ff14sb
    print(ALA)
    # output:
    # Type of Residue: ALA

All functions in the base module is also loaded into the main dict, \
which means you can directly use it without the module name. For example, you can use::

    import Xponge
    load_mol2("example.mol2")
    # This is the same as
    # Xponge.load_mol2("example.mol2")

The atoms in a ``Residue`` or a ``ResidueType`` can be obtained by their names. For example::

    import Xponge.forcefield.amber.ff14sb
    print(ALA.CA)

"""
__version__ = "1.2.6.5"

import os
import time
import sys
from collections import OrderedDict, deque
from itertools import product, permutations

import numpy as np

from . import assign
from .assign import Assign, get_assignment_from_pdb, get_assignment_from_mol2, get_assignment_from_pubchem, \
    get_assignment_from_residuetype
from .helper import GlobalSetting, Type, ResidueType, Entity, Atom, Residue, ResidueLink, Molecule, AtomType, \
    set_global_alternative_names, generate_new_pairwise_force_type, generate_new_bonded_force_type, source
from .load import load_ffitp, load_mol2, load_rst7, load_frcmod, load_pdb, load_parmdat, load_coordinate
from .build import save_mol2, save_pdb, save_sponge_input, save_gro, build_bonded_force, get_mindsponge_system_energy
from .process import impose_bond, impose_angle, impose_dihedral, add_solvent_box, h_mass_repartition, solvent_replace, \
    main_axis_rotate, get_peptide_from_sequence, optimize, Region, UnionRegion, IntersectRegion, \
    BlockRegion, SphereRegion, FrustumRegion, PrismRegion, Lattice


def _initialize():
    """

    :return:
    """
    set_global_alternative_names(True)
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

    #pylint: disable=unused-argument
    def _do_initial(self, sys_kwarg, ene_kwarg, use_pbc):
        if "coordinate" not in sys_kwarg:
            sys_kwarg["coordinate"] = [self.get_atom_coordinates().tolist()]
            sys_kwarg["atoms"] = [[atom.name for atom in self.atoms]]
        else:
            sys_kwarg["coordinate"].append(self.get_atom_coordinates().tolist())
            sys_kwarg["atoms"].append([atom.name for atom in self.atoms])

    Molecule.Set_MindSponge_Todo("coordinate")(_do_initial)


_initialize()
