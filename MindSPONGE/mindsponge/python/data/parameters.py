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
Modeling Module.
"""
import itertools
from itertools import product
from pathlib import Path
from typing import NamedTuple
import numpy as np

this_directory = Path(__file__).parent

backbone_atoms = np.array(["N", "CA", "C", "O"], np.str_)
include_backbone_atoms = np.array(["OXT"], np.str_)


class ForceConstants(NamedTuple):
    """ The structured object for return force field parameters.
    """
    bond_params: np.ndarray
    angle_params: np.ndarray
    dihedral_params: np.ndarray
    improper_dihedral_params: np.ndarray
    angles: np.ndarray
    dihedrals: np.ndarray
    idihedrals: np.ndarray
    excludes: np.ndarray
    vdw_param: np.ndarray
    hbonds: np.ndarray
    non_hbonds: np.ndarray
    pair_params: np.ndarray


class ForceFieldParameters:
    """ Getting parameters for given bonds and atom types.
    Args:
        atom_types(np.str_): The atom types defined in forcefields.
        force_constants(dict): A dictionary stores all force field constants.
        atom_names(np.str_): Unique atom names in an amino acid.
    Parameters:
        bonds(np.int32): The bond pairs defined for a given molecule.
    """

    def __init__(self, atom_types, force_constants, atom_names=None, atom_charges=None):
        self.atom_types = atom_types
        self.atom_names = atom_names
        atom_nums = atom_types.shape[-1]
        assert atom_nums > 0
        self.atom_charges = atom_charges
        self.atom_nums = atom_nums

        # Load force field parameters.
        self.vdw_params = force_constants['parameters']["vdw_energy"]
        for key in self.vdw_params:
            self.vdw_params[key] = np.array(self.vdw_params[key])
        self._bonds = force_constants['parameters']["bond_energy"]
        for key in self._bonds:
            self._bonds[key] = np.array(self._bonds[key])
        self._angles = force_constants['parameters']["angle_energy"]
        for key in self._angles:
            self._angles[key] = np.array(self._angles[key])
        self._dihedrals = force_constants['parameters']["dihedral_energy"]['dihedral']
        for key in self._dihedrals:
            self._dihedrals[key] = np.array(self._dihedrals[key])
        self._idihedrals = force_constants['parameters']["dihedral_energy"]['idihedral']
        for key in self._idihedrals:
            self._idihedrals[key] = np.array(self._idihedrals[key])
        self._wildcard = np.array(["X"], dtype=np.str_)

        self.htypes = np.array(
            ["H", "HC", "H1", "HS", "H5", "H4", "HP", "HA", "HO"], np.str_
        )

        self.bond_params = None
        self.angle_params = None
        self.dihedral_params = None
        self.improper_dihedral_params = None
        self.excludes = np.empty(atom_nums)[:, None]
        self.vdw_param = np.empty((atom_nums, 2))
        self.pair_index = None

    def get_bond_params(self, bonds, atom_types):
        """ Get the force field bond parameters. """
        names = atom_types[bonds]
        bond_types = np.append(
            np.char.add(np.char.add(names[:, 0], "-"), names[:, 1])[None, :],
            np.char.add(np.char.add(names[:, 1], "-"), names[:, 0])[None, :],
            axis=0,
        )
        bond_id = -1 * np.ones(bonds.shape[0], dtype=np.int32)
        mask_id = np.where(
            bond_types.reshape(bonds.shape[-2] * 2, 1) == self._bonds["atoms"]
        )

        if mask_id[0].shape[0] < bonds.shape[0]:
            raise ValueError("Elements in atom types not recognized!")

        left_id = np.where(mask_id[0] < bonds.shape[0])[0]
        bond_id[mask_id[0][left_id]] = mask_id[1][left_id]
        right_id = np.where(mask_id[0] >= bonds.shape[0])[0]
        bond_id[mask_id[0][right_id] - bonds.shape[0]] = mask_id[1][right_id]
        return np.append(
            self._bonds["force_constant"][bond_id][None, :],
            self._bonds["bond_length"][bond_id][None, :],
            axis=0,
        ).T

    def get_angle_params(self, angles, atom_types):
        """ Get the force field angle parameters. """
        names = atom_types[angles]
        angle_types = np.append(
            np.char.add(
                np.char.add(
                    np.char.add(np.char.add(
                        names[:, 0], "-"), names[:, 1]), "-"
                ),
                names[:, 2],
            )[None, :],
            np.char.add(
                np.char.add(
                    np.char.add(np.char.add(
                        names[:, 2], "-"), names[:, 1]), "-"
                ),
                names[:, 0],
            )[None, :],
            axis=0,
        )
        angle_id = -1 * np.ones(angles.shape[0], dtype=np.int32)
        mask_id = np.where(
            angle_types.reshape(angles.shape[0] * 2, 1) == self._angles["atoms"]
        )

        left_id = np.where(mask_id[0] < angles.shape[0])[0]
        angle_id[mask_id[0][left_id]] = mask_id[1][left_id]
        right_id = np.where(mask_id[0] >= angles.shape[0])[0]
        angle_id[mask_id[0][right_id] - angles.shape[0]] = mask_id[1][right_id]
        return np.append(
            self._angles["force_constant"][angle_id][None, :],
            self._angles["bond_angle"][angle_id][None, :],
            axis=0,
        ).T

    def addchar(self, n0, n1, n2, n3):
        """ The multi atom name constructor. """
        return np.append(
            np.char.add(
                np.char.add(
                    np.char.add(
                        np.char.add(
                            np.char.add(np.char.add(
                                n0, "-"), n1), "-"
                        ),
                        n2,
                    ),
                    "-",
                ),
                n3,
            )[None, :],
            np.char.add(
                np.char.add(
                    np.char.add(
                        np.char.add(
                            np.char.add(np.char.add(
                                n3, "-"), n2), "-"
                        ),
                        n1,
                    ),
                    "-",
                ),
                n0,
            )[None, :],
            axis=0,
        )

    def get_dihedral_params(self, dihedrals_in, atom_types, return_includes=False):
        """ Get the force field dihedral parameters. """
        # pylint: disable=redefined-outer-name
        dihedrals = dihedrals_in.copy()
        standar_dihedrals = dihedrals.copy()
        names = atom_types[dihedrals]
        dihedral_types = self.addchar(names[:, 0], names[:, 1], names[:, 2], names[:, 3])
        dihedral_id = -1 * np.ones(dihedrals.shape[0], dtype=np.int32)
        mask_id = np.where(dihedral_types.reshape(dihedrals.shape[0] * 2, 1) == self._dihedrals["atoms"])

        # Constructing A-B-C-D
        left_id = np.where(mask_id[0] < dihedrals.shape[0])[0]
        dihedral_id[mask_id[0][left_id]] = mask_id[1][left_id]
        right_id = np.where(mask_id[0] >= dihedrals.shape[0])[0]
        dihedral_id[mask_id[0][right_id] - dihedrals.shape[0]] = mask_id[1][right_id]
        include_1 = np.pad(np.where(dihedral_id > -1, 1, 0)[None, :],
                           ((0, self._dihedrals["force_constant"].shape[1] - 1), (0, 0)),
                           mode="edge").T.flatten()[:, None]
        exclude_1 = include_1 - 1 < 0

        dihedral_params = np.pad(
            dihedrals[:, None, :],
            ((0, 0), (0, self._dihedrals["force_constant"].shape[1] - 1), (0, 0)),
            mode="edge",
        ).reshape(dihedrals.shape[0] * 4, self._dihedrals["force_constant"].shape[1])
        dihedral_params = np.concatenate((dihedral_params,
                                          self._dihedrals["periodicity"][dihedral_id].flatten()[:, None],
                                          self._dihedrals["force_constant"][dihedral_id].flatten()[:, None],
                                          self._dihedrals["phase"][dihedral_id].flatten()[:, None]), axis=1)
        dihedral_params *= include_1

        # Constructing X-B-C-D and D-C-B-X
        dihedrals[:, 0] = -1 * np.ones_like(dihedrals[:, 0])
        names = atom_types[dihedrals]
        dihedral_types = self.addchar(names[:, 0], names[:, 1], names[:, 2], names[:, 3])
        mask_id = np.where(dihedral_types.reshape(dihedrals.shape[0] * 2, 1) == self._dihedrals["atoms"])
        left_id = np.where(mask_id[0] < dihedrals.shape[0])[0]
        dihedral_id[mask_id[0][left_id]] = mask_id[1][left_id]
        right_id = np.where(mask_id[0] >= dihedrals.shape[0])[0]
        dihedral_id[mask_id[0][right_id] - dihedrals.shape[0]] = mask_id[1][right_id]
        include_2 = np.pad(np.where(dihedral_id > -1, 1, 0)[None, :],
                           ((0, self._dihedrals["force_constant"].shape[1] - 1), (0, 0)),
                           mode="edge").T.flatten()[:, None]
        exclude_2 = include_2 - 1 < 0
        dihedral_params_1 = np.pad(standar_dihedrals[:, None, :],
                                   ((0, 0), (0, self._dihedrals["force_constant"].shape[1] - 1), (0, 0)),
                                   mode="edge",
                                   ).reshape(dihedrals.shape[0] * 4, self._dihedrals["force_constant"].shape[1])
        dihedral_params_1 = np.concatenate((dihedral_params_1,
                                            self._dihedrals["periodicity"][dihedral_id].flatten()[:, None],
                                            self._dihedrals["force_constant"][dihedral_id].flatten()[:, None],
                                            self._dihedrals["phase"][dihedral_id].flatten()[:, None]), axis=1)
        dihedral_params_1 *= include_2 * exclude_1

        # Constructing A-B-C-X and X-C-B-A
        dihedrals = dihedrals_in.copy()
        dihedrals[:, -1] = -1 * np.ones_like(dihedrals[:, -1])
        names = atom_types[dihedrals]
        dihedral_types = self.addchar(names[:, 0], names[:, 1], names[:, 2], names[:, 3])
        mask_id = np.where(dihedral_types.reshape(dihedrals.shape[0] * 2, 1) == self._dihedrals["atoms"])
        left_id = np.where(mask_id[0] < dihedrals.shape[0])[0]
        dihedral_id[mask_id[0][left_id]] = mask_id[1][left_id]
        right_id = np.where(mask_id[0] >= dihedrals.shape[0])[0]
        dihedral_id[mask_id[0][right_id] - dihedrals.shape[0]] = mask_id[1][right_id]
        include_4 = np.pad(np.where(dihedral_id > -1, 1, 0)[None, :],
                           ((0, self._dihedrals["force_constant"].shape[1] - 1), (0, 0)),
                           mode="edge").T.flatten()[:, None]
        exclude_4 = include_4 - 1 < 0
        dihedral_params_3 = np.pad(standar_dihedrals[:, None, :],
                                   ((0, 0), (0, self._dihedrals["force_constant"].shape[1] - 1), (0, 0)),
                                   mode="edge",
                                   ).reshape(dihedrals.shape[0] * 4, self._dihedrals["force_constant"].shape[1])
        dihedral_params_3 = np.concatenate((dihedral_params_3,
                                            self._dihedrals["periodicity"][dihedral_id].flatten()[:, None],
                                            self._dihedrals["force_constant"][dihedral_id].flatten()[:, None],
                                            self._dihedrals["phase"][dihedral_id].flatten()[:, None]), axis=1)
        dihedral_params_3 *= include_2 * exclude_1 * exclude_4

        # Constructing X-A-B-X
        dihedrals[:, 0] = -1 * np.ones_like(dihedrals[:, 0])
        dihedrals[:, -1] = -1 * np.ones_like(dihedrals[:, -1])
        names = atom_types[dihedrals]
        dihedral_types = self.addchar(names[:, 0], names[:, 1], names[:, 2], names[:, 3])
        mask_id = np.where(dihedral_types.reshape(dihedrals.shape[0] * 2, 1) == self._dihedrals["atoms"])
        left_id = np.where(mask_id[0] < dihedrals.shape[0])[0]
        dihedral_id[mask_id[0][left_id]] = mask_id[1][left_id]
        right_id = np.where(mask_id[0] >= dihedrals.shape[0])[0]
        dihedral_id[mask_id[0][right_id] -
                    dihedrals.shape[0]] = mask_id[1][right_id]
        include_3 = np.pad(
            np.where(dihedral_id > -1, 1, 0)[None, :],
            ((0, self._dihedrals["force_constant"].shape[1] - 1), (0, 0)),
            mode="edge",
        ).T.flatten()[:, None]
        dihedral_params_2 = np.pad(
            standar_dihedrals[:, None, :],
            ((0, 0), (0, self._dihedrals["force_constant"].shape[1] - 1), (0, 0)),
            mode="edge",
        ).reshape(dihedrals.shape[0] * 4, self._dihedrals["force_constant"].shape[1])
        dihedral_params_2 = np.concatenate((dihedral_params_2,
                                            self._dihedrals["periodicity"][dihedral_id].flatten()[:, None],
                                            self._dihedrals["force_constant"][dihedral_id].flatten()[:, None],
                                            self._dihedrals["phase"][dihedral_id].flatten()[:, None]), axis=1)
        dihedral_params_2 *= include_3 * exclude_1 * exclude_2 * exclude_4

        dihedral_params += dihedral_params_1 + dihedral_params_2 + dihedral_params_3
        ks0_condition = dihedral_params[:, -2] != 0

        if not return_includes:
            return dihedral_params[np.where(ks0_condition)[0]]

        return dihedral_params, (include_1, include_2, include_4, include_3)

    def _get_idihedral_params(self, dihedrals_in, atom_types, return_includes=False):
        """ Pre-processing of getting improper dihedrals. """
        # pylint: disable=redefined-outer-name
        dihedrals = dihedrals_in.copy()
        standar_dihedrals = dihedrals.copy()
        names = atom_types[dihedrals]
        dihedral_types = self.addchar(names[:, 0], names[:, 1], names[:, 2], names[:, 3])
        dihedral_id = -1 * np.ones(dihedrals.shape[0], dtype=np.int32)
        mask_id = np.where(dihedral_types.reshape(dihedrals.shape[0] * 2, 1)
                           == self._idihedrals["atoms"])

        # Constructing A-B-C-D
        left_id = np.where(mask_id[0] < dihedrals.shape[0])[0]
        dihedral_id[mask_id[0][left_id]] = mask_id[1][left_id]
        right_id = np.where(mask_id[0] >= dihedrals.shape[0])[0]
        dihedral_id[mask_id[0][right_id] - dihedrals.shape[0]] = mask_id[1][right_id]
        include_1 = np.where(dihedral_id > -1, 1, 0)
        exclude_1 = include_1 - 1 < 0

        dihedral_params = standar_dihedrals
        dihedral_params = np.concatenate((dihedral_params,
                                          self._idihedrals["periodicity"][dihedral_id].flatten()[:, None],
                                          self._idihedrals["force_constant"][dihedral_id].flatten()[:, None],
                                          self._idihedrals["phase"][dihedral_id].flatten()[:, None]), axis=1)
        dihedral_params *= include_1[:, None]

        # Constructing X-B-C-D and D-C-B-X
        dihedrals[:, 0] = -1 * np.ones_like(dihedrals[:, 0])
        names = atom_types[dihedrals]
        dihedral_types = self.addchar(names[:, 0], names[:, 1], names[:, 2], names[:, 3])
        mask_id = np.where(dihedral_types.reshape(dihedrals.shape[0] * 2, 1)
                           == self._idihedrals["atoms"])
        left_id = np.where(mask_id[0] < dihedrals.shape[0])[0]
        dihedral_id[mask_id[0][left_id]] = mask_id[1][left_id]
        right_id = np.where(mask_id[0] >= dihedrals.shape[0])[0]
        dihedral_id[mask_id[0][right_id] - dihedrals.shape[0]] = mask_id[1][right_id]
        include_2 = np.where(dihedral_id > -1, 1, 0)
        exclude_2 = include_2 - 1 < 0
        dihedral_params_1 = standar_dihedrals
        dihedral_params_1 = np.concatenate((dihedral_params_1,
                                            self._idihedrals["periodicity"][dihedral_id].flatten()[:, None],
                                            self._idihedrals["force_constant"][dihedral_id].flatten()[:, None],
                                            self._idihedrals["phase"][dihedral_id].flatten()[:, None]), axis=1)
        dihedral_params_1 *= include_2[:, None] * exclude_1[:, None]

        # Constructing A-B-C-X and X-C-B-A
        dihedrals = dihedrals_in.copy()
        dihedrals[:, -1] = -1 * np.ones_like(dihedrals[:, -1])
        names = atom_types[dihedrals]
        dihedral_types = self.addchar(names[:, 0], names[:, 1], names[:, 2], names[:, 3])
        mask_id = np.where(dihedral_types.reshape(dihedrals.shape[0] * 2, 1) == self._idihedrals["atoms"])
        left_id = np.where(mask_id[0] < dihedrals.shape[0])[0]
        dihedral_id[mask_id[0][left_id]] = mask_id[1][left_id]
        right_id = np.where(mask_id[0] >= dihedrals.shape[0])[0]
        dihedral_id[mask_id[0][right_id] - dihedrals.shape[0]] = mask_id[1][right_id]
        include_4 = np.where(dihedral_id > -1, 1, 0)
        exclude_4 = include_4 - 1 < 0
        dihedral_params_3 = standar_dihedrals
        dihedral_params_3 = np.concatenate((dihedral_params_3,
                                            self._idihedrals["periodicity"][dihedral_id].flatten()[:, None],
                                            self._idihedrals["force_constant"][dihedral_id].flatten()[:, None],
                                            self._idihedrals["phase"][dihedral_id].flatten()[:, None]), axis=1)
        dihedral_params_3 *= (include_2[:, None] * exclude_1[:, None] * exclude_4[:, None])

        # Constructing X-A-B-X
        dihedrals = dihedrals_in.copy()
        dihedrals[:, 0] = -1 * np.ones_like(dihedrals[:, 0])
        dihedrals[:, -1] = -1 * np.ones_like(dihedrals[:, -1])
        names = atom_types[dihedrals]
        dihedral_types = self.addchar(names[:, 0], names[:, 1], names[:, 2], names[:, 3])
        mask_id = np.where(dihedral_types.reshape(dihedrals.shape[0] * 2, 1) == self._idihedrals["atoms"])
        left_id = np.where(mask_id[0] < dihedrals.shape[0])[0]
        dihedral_id[mask_id[0][left_id]] = mask_id[1][left_id]
        right_id = np.where(mask_id[0] >= dihedrals.shape[0])[0]
        dihedral_id[mask_id[0][right_id] - dihedrals.shape[0]] = mask_id[1][right_id]
        include_3 = np.where(dihedral_id > -1, 1, 0)
        exclude_3 = include_3 - 1 < 0
        dihedral_params_2 = standar_dihedrals
        dihedral_params_2 = np.concatenate((dihedral_params_2,
                                            self._idihedrals["periodicity"][dihedral_id].flatten()[:, None],
                                            self._idihedrals["force_constant"][dihedral_id].flatten()[:, None],
                                            self._idihedrals["phase"][dihedral_id].flatten()[:, None]), axis=1)
        dihedral_params_2 *= (include_3[:, None] * exclude_1[:, None] * exclude_2[:, None] * exclude_4[:, None])

        # Constructing X-X-C-D and D-C-X-X
        dihedrals = dihedrals_in.copy()
        dihedrals[:, 0] = -1 * np.ones_like(dihedrals[:, 0])
        dihedrals[:, 1] = -1 * np.ones_like(dihedrals[:, 1])
        names = atom_types[dihedrals]
        dihedral_types = self.addchar(names[:, 0], names[:, 1], names[:, 2], names[:, 3])
        mask_id = np.where(dihedral_types.reshape(dihedrals.shape[0] * 2, 1) == self._idihedrals["atoms"])
        left_id = np.where(mask_id[0] < dihedrals.shape[0])[0]
        dihedral_id[mask_id[0][left_id]] = mask_id[1][left_id]
        right_id = np.where(mask_id[0] >= dihedrals.shape[0])[0]
        dihedral_id[mask_id[0][right_id] - dihedrals.shape[0]] = mask_id[1][right_id]
        include_5 = np.where(dihedral_id > -1, 1, 0)
        dihedral_params_4 = standar_dihedrals
        dihedral_params_4 = np.concatenate((dihedral_params_4,
                                            self._idihedrals["periodicity"][dihedral_id].flatten()[:, None],
                                            self._idihedrals["force_constant"][dihedral_id].flatten()[:, None],
                                            self._idihedrals["phase"][dihedral_id].flatten()[:, None]), axis=1)
        dihedral_params_4 *= (include_5[:, None]
                              * exclude_1[:, None]
                              * exclude_2[:, None]
                              * exclude_4[:, None]
                              * exclude_3[:, None])

        dihedral_params += (dihedral_params_1 + dihedral_params_2 + dihedral_params_3 + dihedral_params_4)
        ks0_condition = dihedral_params[:, -2] != 0

        if not return_includes:
            return dihedral_params[np.where(ks0_condition)[0]]

        return dihedral_params, (include_1, include_2, include_4, include_3, include_5)

    def get_idihedral_params(self, idihedrals_in, atom_types, third_id):
        """ Get the force field improper dihedral parameters. """
        try:
            idihedral_params, includes = self._get_idihedral_params(
                idihedrals_in.copy(), atom_types, return_includes=True
            )
        except AttributeError:
            return None

        priorities = (
            includes[0] * 16
            + includes[1] * 8
            + includes[2] * 4
            + includes[3] * 2
            + includes[4] * 1
        )
        for i, j, k, l in itertools.permutations(range(4), 4):
            idihedrals = np.ones_like(idihedrals_in.copy())
            idihedrals[:, 0] = idihedrals_in[:, i].copy()
            idihedrals[:, 1] = idihedrals_in[:, j].copy()
            idihedrals[:, 2] = idihedrals_in[:, k].copy()
            idihedrals[:, 3] = idihedrals_in[:, l].copy()
            this_idihedral_params, includes = self._get_idihedral_params(
                idihedrals, atom_types, return_includes=True
            )
            this_priorities = (
                includes[0] * 16
                + includes[1] * 8
                + includes[2] * 4
                + includes[3] * 2
                + includes[4] * 1
            )
            this_priorities *= idihedrals[:, 2] == third_id
            this_id = np.where(this_priorities >= priorities)[0]
            idihedral_params[this_id] = this_idihedral_params[this_id]
            priorities[this_id] = this_priorities[this_id]
        ks0_id = np.where(idihedral_params[:, -2] != 0)[0]
        return idihedral_params[ks0_id]

    def construct_angles(self, bonds, bonds_for_angle, middle_id):
        for idx in middle_id:
            this_bonds = bonds[np.where(bonds_for_angle == idx)[0]]
            flatten_bonds = this_bonds.flatten()
            this_idx = np.delete(flatten_bonds, np.where(flatten_bonds == idx))
            yield this_idx

    def combinations(self, bonds, bonds_for_angle, middle_id):
        """ Get all the combinations of 3 atoms. """
        this_idx = self.construct_angles(bonds, bonds_for_angle, middle_id)
        id_selections = [
            [[0, 1]],
            [[0, 1], [1, 2], [0, 2]],
            [[0, 1], [1, 2], [2, 3], [0, 2], [0, 3], [1, 3]],
        ]
        angles = None
        counter = 0
        for idx in this_idx:
            selections = id_selections[idx.size - 2]
            for selection in selections:
                if angles is None:
                    angles = np.insert(idx[selection], 1, middle_id[counter])[
                        None, :]
                else:
                    angles = np.append(
                        angles,
                        np.insert(idx[selection], 1, middle_id[counter])[
                            None, :],
                        axis=0,
                    )
            counter += 1
        return angles

    def construct_hash(self, bonds):
        hash_map = {}
        for i, b in enumerate(bonds):
            bond = tuple(b)
            hash_map[hash(bond)] = i
        return hash_map

    def trans_dangles(self, dangles, middle_id):
        """ Construct the dihedrals. """
        left_id = np.isin(dangles[:, 0], middle_id[0])
        left_ele = dangles[:, 2][left_id]
        left_id = np.isin(dangles[:, 2], middle_id[0])
        left_ele = np.append(left_ele, dangles[:, 0][left_id])
        right_id = np.isin(dangles[:, 1], middle_id[0])
        right_ele = np.unique(dangles[right_id])
        right_ele = right_ele[np.where(
            np.isin(right_ele, middle_id, invert=True))[0]]
        sides = product(right_ele, left_ele)
        sides_array = np.array(list(sides))

        if sides_array.size == 0:
            return sides_array

        sides = sides_array[np.where(
            sides_array[:, 0] != sides_array[:, 1])[0]]
        left = np.append(
            sides[:, 0].reshape(sides.shape[0], 1),
            np.broadcast_to(middle_id, (sides.shape[0],) + middle_id.shape),
            axis=1,
        )
        dihedrals = np.append(
            left, sides[:, 1].reshape(sides.shape[0], 1), axis=1)
        return dihedrals

    def get_dihedrals(self, angles, dihedral_middle_id):
        """ Get the dihedrals indexes. """
        dihedrals = None
        for i in range(dihedral_middle_id.shape[0]):
            dangles = angles[
                np.where(
                    (
                        np.isin(angles, dihedral_middle_id[i]).sum(axis=1)
                        * np.isin(angles[:, 1], dihedral_middle_id[i])
                    )
                    > 1
                )[0]
            ]
            this_sides = self.trans_dangles(dangles, dihedral_middle_id[i])
            if this_sides.size == 0:
                continue
            if dihedrals is None:
                dihedrals = this_sides
            else:
                dihedrals = np.append(dihedrals, this_sides, axis=0)
        return dihedrals

    def check_idihedral(self, bonds, core_id):
        """ Check if there are same improper dihedrals. """
        # pylint: disable=pointless-statement
        checked_core_id = core_id.copy()
        bonds_hash = [hash(tuple(x)) for x in bonds]
        for i in range(core_id.shape[0]):
            ids_for_idihedral = np.where(
                np.sum(np.isin(bonds, core_id[i]), axis=1) > 0
            )[0]
            bonds_for_idihedral = bonds[ids_for_idihedral]
            uniques = np.unique(bonds_for_idihedral.flatten())
            uniques = np.delete(uniques, np.where(uniques == core_id[i])[0])
            uniques_product = np.array(list(product(uniques, uniques)))
            uniques_hash = np.array([hash(tuple(x)) for x in product(uniques, uniques)])
            excludes = np.isin(uniques_hash, bonds_hash)
            exclude_size = np.unique(uniques_product[excludes]).size
            # Exclude condition
            if uniques.shape[0] - excludes.sum() <= 2 or exclude_size > 3:
                checked_core_id[i] == -1
        return checked_core_id[np.where(checked_core_id > -1)[0]]

    def get_idihedrals(self, bonds, core_id):
        """ Get the improper dihedrals indexes. """
        idihedrals = None
        new_id = None
        for i in range(core_id.shape[0]):
            ids_for_idihedral = np.where(
                np.sum(np.isin(bonds, core_id[i]), axis=1) > 0
            )[0]
            bonds_for_idihedral = bonds[ids_for_idihedral]
            if bonds_for_idihedral.shape[0] == 3:
                idihedral = np.unique(bonds_for_idihedral.flatten())[None, :]
                if idihedrals is None:
                    idihedrals = idihedral
                    new_id = core_id[i]
                else:
                    idihedrals = np.append(idihedrals, idihedral, axis=0)
                    new_id = np.append(new_id, core_id[i])
            else:
                # Only SP2 is considered.
                continue
        return idihedrals, new_id

    def get_excludes(self, bonds, angles, dihedrals, idihedrals):
        """ Get the exclude atoms index. """
        excludes = []
        for i in range(self.atom_nums):
            bond_excludes = bonds[np.where(
                np.isin(bonds, i).sum(axis=1))[0]].flatten()
            angle_excludes = angles[
                np.where(np.isin(angles, i).sum(axis=1))[0]
            ].flatten()
            dihedral_excludes = dihedrals[
                np.where(np.isin(dihedrals, i).sum(axis=1))[0]
            ].flatten()
            if idihedrals is not None:
                idihedral_excludes = idihedrals[
                    np.where(np.isin(idihedrals, i).sum(axis=1))[0]
                ].flatten()
            this_excludes = np.append(bond_excludes, angle_excludes)
            this_excludes = np.append(this_excludes, dihedral_excludes)
            if idihedrals is not None:
                this_excludes = np.append(this_excludes, idihedral_excludes)
            this_excludes = np.unique(this_excludes)
            excludes.append(this_excludes[np.where(
                this_excludes != i)[0]].tolist())
        padding_length = 0
        for i in range(self.atom_nums):
            padding_length = max(padding_length, len(excludes[i]))
        self.excludes = np.empty((self.atom_nums, padding_length))
        for i in range(self.atom_nums):
            self.excludes[i] = np.pad(
                np.array(excludes[i]),
                (0, padding_length - len(excludes[i])),
                mode="constant",
                constant_values=self.atom_nums,
            )
        return self.excludes

    def get_vdw_params(self, atom_names):
        """
        ['H','HO','HS','HC','H1','H2','H3','HP','HA','H4',
         'H5','HZ','O','O2','OH','OS','OP','C*','CI','C5',
         'C4','CT','CX','C','N','N3','S','SH','P','MG',
         'C0','F','Cl','Br','I','2C','3C','C8','CO']
        """
        atom_names_count = np.zeros(39)
        for i in range(self.atom_nums):
            this_id = np.where(
                np.isin(self.vdw_params["atoms"], atom_names[i]))[0]
            if atom_names[i] in ["N2", "NA", "NB"]:
                this_id = [24]
            if atom_names[i] in ["CA", "CC", "CR", "CW", "CN", "CB", "CV"]:
                this_id = [17]
            self.vdw_param[i][0] = self.vdw_params["epsilon"][this_id]
            self.vdw_param[i][1] = self.vdw_params["sigma"][this_id]
            atom_names_count[this_id] += 1

    def get_pairwise_c6(self, e0, e1, r0, r1):
        """ Calculate the B coefficient in vdw potential. """
        e01 = np.sqrt(e0 * e1)
        r01 = r0 + r1
        return 2 * e01 * r01 ** 6

    def get_hbonds(self, bonds):
        """ Get hydrogen bonds. """
        hatoms = np.where(np.isin(self.atom_types, self.htypes))[0]
        bonds_with_h = np.where(np.isin(bonds, hatoms).sum(axis=-1))[0]
        non_hbonds = np.where(np.isin(bonds, hatoms).sum(axis=-1) == 0)[0]
        return bonds[bonds_with_h], bonds[non_hbonds]

    def get_pair_index(self, dihedrals, angles, bonds):
        """ Get the non-bonded atom pairs index. """
        pairs = dihedrals[:, [0, -1]]
        pairs.sort()
        pair_index = np.unique(pairs, axis=0)
        pair_hash = []
        for pair in pair_index:
            if pair[0] < pair[1]:
                pair_hash.append(hash((pair[0], pair[1])))
            else:
                pair_hash.append(hash((pair[1], pair[0])))
        pair_hash = np.array(pair_hash)
        angle_hash = []
        for angle in angles:
            if angle[0] < angle[-1]:
                angle_hash.append(hash((angle[0], angle[-1])))
            else:
                angle_hash.append(hash((angle[-1], angle[0])))
        angle_hash = np.array(angle_hash)
        bond_hash = []
        for bond in bonds:
            b = sorted(bond)
            bond_hash.append(hash(tuple(b)))
        bond_hash = np.array(bond_hash)
        include_index = np.where(
            np.isin(pair_hash, angle_hash) + np.isin(pair_hash, bond_hash) == 0
        )[0]
        return pair_index[include_index]

    def get_pair_params(self, pair_index, epsilon, sigma):
        """ Return all the pair parameters. """
        qiqj = np.take_along_axis(self.atom_charges, pair_index, axis=1)
        qiqj = np.prod(qiqj, -1)[:, None]

        epsilon_ij = np.take_along_axis(epsilon, pair_index, axis=1)
        epsilon_ij = np.sqrt(np.prod(epsilon_ij, -1)[:, None])

        sigma_ij = np.take_along_axis(sigma, pair_index, axis=1)
        sigma_ij = np.mean(sigma_ij, -1)[:, None]

        pair_params = np.concatenate((qiqj, epsilon_ij, sigma_ij), axis=-1)
        return pair_params

    def __call__(self, bonds):
        # pylint: disable=unused-argument
        hbonds, non_hbonds = self.get_hbonds(bonds)
        atoms_types = self.atom_types
        self.get_vdw_params(atoms_types)
        atom_types = np.append(atoms_types, self._wildcard)
        this_bond_params = self.get_bond_params(bonds, atoms_types)
        self.bond_params = np.append(bonds, this_bond_params, axis=1)
        middle_id = np.where(np.bincount(bonds.flatten()) > 1)[0]
        ids_for_angle = np.where(
            np.sum(np.isin(bonds, middle_id), axis=1) > 0)[0]
        bonds_for_angle = bonds[ids_for_angle]
        angles = self.combinations(bonds, bonds_for_angle, middle_id)
        this_angle_params = self.get_angle_params(angles, atoms_types)
        self.angle_params = np.append(angles, this_angle_params, axis=1)
        dihedral_middle_id = bonds[
            np.where(np.isin(bonds, middle_id).sum(axis=1) == 2)[0]
        ]
        dihedrals = self.get_dihedrals(angles, dihedral_middle_id)
        self.dihedral_params = self.get_dihedral_params(dihedrals, atom_types)
        core_id = np.where(np.bincount(bonds.flatten()) > 2)[0]
        checked_core_id = self.check_idihedral(bonds, core_id)
        idihedrals, third_id = self.get_idihedrals(bonds, checked_core_id)
        self.improper_dihedral_params = self.get_idihedral_params(
            idihedrals, atom_types, third_id
        )
        self.pair_index = self.get_pair_index(dihedrals, angles, bonds)
        pair_params = self.get_pair_params(self.pair_index, self.vdw_param[:, 0][None, :],
                                           self.vdw_param[:, 1][None, :])
        self.excludes = self.get_excludes(bonds, angles, dihedrals, idihedrals)
        return ForceConstants(
            self.bond_params,
            self.angle_params,
            self.dihedral_params,
            self.improper_dihedral_params,
            angles,
            dihedrals,
            idihedrals,
            self.excludes,
            self.vdw_param,
            hbonds,
            non_hbonds,
            pair_params,
        )
