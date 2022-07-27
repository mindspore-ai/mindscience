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
"""data loader"""
import math
import parmed as pmd
import numpy as np

from ..common.constants import TIME_UNIT, CHARGE_UNIT, SUPPORT_TYPE, KB


class system:
    """system"""

    def __init__(self, config):
        if config.file_type not in SUPPORT_TYPE:
            return None
        self.mode = config.mode
        self.dt = config.dt
        self.target_temperature = config.target_temperature
        self.gamma_ln = 1.0
        if 'gamma_ln' in vars(config):
            self.gamma_ln = config.gamma_ln
        elif 'langevin_gamma' in vars(config):
            self.gamma_ln = config.langevin_gamma
        self.gamma_ln = self.gamma_ln / TIME_UNIT
        self.exp_gamma = math.exp(-1 * self.gamma_ln * self.dt)
        self.sqrt_gamma = math.sqrt((1. - self.exp_gamma * self.exp_gamma) * self.target_temperature * KB)

        if "coordinates_file" in vars(config):
            self._system = pmd.load_file(config.topology_file, xyz=config.coordinates_file)
            self._crd = pmd.load_file(config.coordinates_file)
        else:
            self._system = pmd.load_file(config.topology_file, xyz=config.coordinates_file)
        self.atom_numbers = len(self._system.atoms)
        self.system_freedom = 3 * self.atom_numbers
        self.coordinates = self._system.coordinates.astype(np.float32)
        self.box_len = self._crd.box.astype(np.float32)[:3]
        self.velocities = None
        if self._crd.hasvels:
            self.velocities = self._crd.velocities
            self.velocities = self.velocities / TIME_UNIT
        else:
            self.velocities = np.zeros([self.atom_numbers, 3]).astype(np.float32)
        self._residue()
        self._atom_info()
        self._virtual_atom()
        self._bond()
        self._angle()
        self._dihedral_nb14()
        self._lennard_jones()
        self._exclude()
        self._simple_constrain(config)

    def _residue(self):
        self.residue_numbers = len(self._system.residues)
        self.residues = []
        for residue in self._system.residues:
            self.residues.append(len(residue))

    def _atom_info(self):
        self.mass = []
        self.charge = []
        for atom in self._system.atoms:
            self.mass.append(atom.mass)
            self.charge.append(atom.charge * CHARGE_UNIT)
        self.mass = np.array(self.mass).astype(np.float32)
        self.inverse_mass = 1.0 / self.mass
        self.charge = np.array(self.charge).astype(np.float32)

    def _virtual_atom(self):
        virtual_atoms = []
        for residue in self._system.residues:
            if residue.name in ("WAT", "HOH"):
                if len(residue.atoms) == 4:
                    virtual_bond = residue.atoms[3].bonds[0]
                    if residue.atoms[1].bonds[0].atom1 == residue.atoms[0] or residue.atoms[1].bonds[0].atom2 == \
                            residue.atoms[0]:
                        ho_bond = residue.atoms[1].bonds[0].type.req
                        hh_bond = residue.atoms[1].bonds[1].type.req
                    else:
                        ho_bond = residue.atoms[1].bonds[1].type.req
                        hh_bond = residue.atoms[1].bonds[0].type.req
                    factor = virtual_bond.type.req / (4 * ho_bond * ho_bond - hh_bond * hh_bond) ** 0.5
                    virtual_atoms.append([2, residue.atoms[3].idx, residue.atoms[0].idx,
                                          residue.atoms[1].idx, residue.atoms[2].idx, factor, factor])
                    self._system.bonds.remove(virtual_bond)
        if virtual_atoms:
            self.virtual_atoms = virtual_atoms

    def _bond(self):
        self.bond_numbers = len(self._system.bonds)
        bond = []
        for _bond in self._system.bonds:
            bond.append([_bond.atom1.idx, _bond.atom2.idx, _bond.type.k, _bond.type.req])
        bond = np.array(bond)
        self.bond_atom_a = bond[:, 0].astype(np.int32)
        self.bond_atom_b = bond[:, 1].astype(np.int32)
        self.bond_k = bond[:, 2].astype(np.float32)
        self.bond_r0 = bond[:, 3].astype(np.float32)

    def _angle(self):
        self.angle_numbers = len(self._system.angles)
        angle = []
        for _angle in self._system.angles:
            angle.append([_angle.atom1.idx, _angle.atom2.idx, _angle.atom3.idx, _angle.type.k,
                          _angle.type.utheteq.value_in_unit(pmd.unit.radian)])
        angle = np.array(angle)
        self.angle_atom_a = angle[:, 0].astype(np.int32)
        self.angle_atom_b = angle[:, 1].astype(np.int32)
        self.angle_atom_c = angle[:, 2].astype(np.int32)
        self.angle_k = angle[:, 3].astype(np.float32)
        self.angle_theta0 = angle[:, 4].astype(np.float32)

    def _dihedral_nb14(self):
        self.dihedral_numbers = len(self._system.dihedrals)
        if self.dihedral_numbers:
            self.dihedral_numbers = 0
            dihedral = []
            towrite = ""
            for _dihedral in self._system.dihedrals:
                if _dihedral.type.phi_k != 0:
                    dihedral.append([_dihedral.atom1.idx, _dihedral.atom2.idx, _dihedral.atom3.idx, _dihedral.atom4.idx,
                                     _dihedral.type.per, _dihedral.type.phi_k,
                                     _dihedral.type.uphase.value_in_unit(pmd.unit.radian)])
                    self.dihedral_numbers += 1
            dihedral = np.array(dihedral)
            self.dihedral_atom_a = dihedral[:, 0].astype(np.int32)
            self.dihedral_atom_b = dihedral[:, 1].astype(np.int32)
            self.dihedral_atom_c = dihedral[:, 2].astype(np.int32)
            self.dihedral_atom_d = dihedral[:, 3].astype(np.int32)
            self.dihedral_ipn = dihedral[:, 4].astype(np.float32)
            self.dihedral_pn = dihedral[:, 4].astype(np.float32)
            self.dihedral_pk = dihedral[:, 5].astype(np.float32)
            self.dihedral_gamc = np.float32(np.cos(dihedral[:, 6]) * dihedral[:, 5])
            self.dihedral_gams = np.float32(np.sin(dihedral[:, 6]) * dihedral[:, 5])

            self.nb14_numbers = 0
            nb14 = []
            for _dihedral in self._system.dihedrals:
                if _dihedral.type.scnb != 0 and _dihedral.type.scee != 0 and not _dihedral.ignore_end:
                    nb14.append([_dihedral.atom1.idx, _dihedral.atom4.idx,
                                 1.0 / _dihedral.type.scnb, 1.0 / _dihedral.type.scee])
                    self.nb14_numbers += 1
            self.nb14_atom_a = np.array(nb14)[:, 0].astype(np.int32)
            self.nb14_atom_b = np.array(nb14)[:, 1].astype(np.int32)
            self.nb14_lj_scale_factor = np.array(nb14)[:, 2].astype(np.float32)
            self.nb14_cf_scale_factor = np.array(nb14)[:, 3].astype(np.float32)

    def _lennard_jones(self):
        LJ_depth = self._system.LJ_depth
        LJ_radius = self._system.LJ_radius
        self.atom_type_numbers = len(LJ_depth)
        LJ_idx = []
        for atom in self._system.atoms:
            LJ_idx.append(atom.nb_idx - 1)

        def getLJ_A(i, j):
            return (np.sqrt(LJ_depth[i] * LJ_depth[j]) * ((LJ_radius[i] + LJ_radius[j])) ** 12)

        def getLJ_B(i, j):
            return (np.sqrt(LJ_depth[i] * LJ_depth[j]) * 2 * ((LJ_radius[i] + LJ_radius[j])) ** 6)

        self.LJ_A = []
        for i in range(self.atom_type_numbers):
            temp = []
            for j in range(i + 1):
                temp.append(getLJ_A(i, j))
            self.LJ_A.extend(temp)
        self.LJ_B = []
        for i in range(self.atom_type_numbers):
            temp = []
            for j in range(i + 1):
                temp.append(getLJ_B(i, j))
            self.LJ_B.extend(temp)
        _LJ_idx = []
        for i in range(self.atom_numbers):
            _LJ_idx.append(LJ_idx[i])

        self.LJ_A = np.float32(np.array(self.LJ_A) * 12.0)
        self.LJ_B = np.float32(np.array(self.LJ_B) * 6.0)
        self.LJ_type = np.array(_LJ_idx).astype(np.float32)

    def _exclude(self):
        self.excluded_atom_numbers = 0
        self.excluded_atom = []
        for atom in self._system.atoms:
            temp = 0
            exclusions_temp = atom.bond_partners + atom.angle_partners + \
                              atom.dihedral_partners + atom.tortor_partners + atom.exclusion_partners
            exclusions = []
            for atom_e in exclusions_temp:
                if (atom_e > atom):
                    exclusions.append("%d" % atom_e.idx)
                    temp += 1
            exclusions.sort(key=lambda x: int(x))
            self.excluded_atom.append(exclusions)

    def _simple_constrain(self, config):
        def add_bond_to_constrain_pair(constrain_mass, atom_mass, atom_a, atom_b, bond_r):
            bond_pair = []
            s = 0
            for i in range(self.bond_numbers):
                mass_a = atom_mass[atom_a[i]]
                mass_b = atom_mass[atom_b[i]]
                if (float(mass_a) < constrain_mass and mass_a > 0) or \
                        (float(mass_b) < constrain_mass and mass_b > 0):
                    constrain_k = \
                        atom_mass[atom_a[i]] * atom_mass[atom_b[i]] / (atom_mass[atom_a[i]] + atom_mass[atom_b[i]])
                    bond_pair.append([atom_a[i], atom_b[i], bond_r[i], constrain_k])
                    s += 1
            return bond_pair

        def add_angle_to_constrain_pair(agnle_num, atom_a, atom_b, atom_c, theta, mass):
            raise NotImplementedError

        constrain_mass = 3.0 if "constrain_mass" not in vars(config) else float(config.constrain_mass)
        bond_pair = add_bond_to_constrain_pair(constrain_mass, self.mass, self.bond_atom_a, self.bond_atom_b,
                                               self.bond_r0)
        angle_pair = []
        self.volume = float(self.box_len[0] * self.box_len[1] * self.box_len[2])
        half_exp_gamma_plus_half = 0.5 * (1. + self.exp_gamma)
        if config.mode == "Minimization":
            exp_gamma = 0.0
        self.iteration_numbers = 25 if "iteration_numbers" not in vars(config) else config.iteration_numbers
        step_length = 1 if "step_length" not in vars(config) else float(config.iteration_numbers)
        extra_numbers = 0

        self.dt_inverse = 1 / self.dt
        constrain_pair_numbers = len(bond_pair) + len(angle_pair) + extra_numbers
        self.system_freedom -= constrain_pair_numbers
        self.constrain_pair = []
        for i in range(len(bond_pair)):
            self.constrain_pair.append(bond_pair[i])
            self.constrain_pair[i][-1] = step_length / half_exp_gamma_plus_half \
                                         * self.constrain_pair[i][-1]
