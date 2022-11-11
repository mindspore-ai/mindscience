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
import re
from operator import itemgetter
from itertools import product
from pathlib import Path
from typing import NamedTuple
import numpy as np
from numpy import ndarray

from .data import get_bonded_types, get_dihedral_types, get_improper_types

this_directory = Path(__file__).parent

backbone_atoms = np.array(["N", "CA", "C", "O"], np.str_)
include_backbone_atoms = np.array(["OXT"], np.str_)


class ForceConstants(NamedTuple):
    """ The structured object for return force field parameters.
    """
    bond_params: dict = None
    angle_params: dict = None
    dihedral_params: dict = None
    improper_params: dict = None
    angles: np.ndarray = None
    dihedrals: np.ndarray = None
    improper: np.ndarray = None
    excludes: np.ndarray = None
    vdw_param: dict = None
    hbonds: np.ndarray = None
    non_hbonds: np.ndarray = None
    pair_params: dict = None


class ForceFieldParameters:
    r"""
    Getting parameters for given bonds and atom types.

    Args:
        atom_types(str):        The atom types defined in forcefields.
        parameters(dict):       A dictionary stores all force field constants.
        atom_names(str):        Unique atom names in an amino acid. Default: None
        atom_charges(ndarray):  The charge of the atoms. Default: None

    Supported Platforms:
        ``Ascend`` ``GPU``
    """

    def __init__(self, atom_types, parameters, atom_names=None, atom_charges=None):
        self.atom_types = atom_types[0]
        self.atom_names = atom_names[0]
        atom_nums = atom_types.shape[-1]
        assert atom_nums > 0
        self.atom_charges = atom_charges
        self.atom_nums = atom_nums

        # Load force field parameters.
        self.vdw_params = None
        if 'vdw_energy' in parameters.keys():
            self.vdw_params = parameters["vdw_energy"]

        self.bond_params = None
        if 'bond_energy' in parameters.keys():
            self.bond_params = parameters["bond_energy"]

        self.angle_params = None
        if 'angle_energy' in parameters.keys():
            self.angle_params = parameters["angle_energy"]

        self._dihedrals = None
        if 'dihedral_energy' in parameters.keys():
            self._dihedrals = parameters["dihedral_energy"]

        self._improper = None
        if 'improper_energy' in parameters.keys():
            self._improper = parameters["improper_energy"]

        self.pair_params = None
        if 'nb_pair_energy' in parameters.keys():
            self.pair_params = parameters["nb_pair_energy"]

        self._wildcard = np.array(["X"], dtype=np.str_)

        self.htypes = np.array(
            ["H", "HC", "H1", "HS", "H5", "H4", "HP", "HA", "HO"], np.str_
        )

        self.dihedral_params = None
        self.improper_params = None
        self.excludes = np.empty(atom_nums)[:, None]
        self.vdw_param = {}
        self.pair_index = None

    def get_bond_params(self, bonds, atom_type):
        """
        Get the force field bond parameters.

        Args:
            bonds (ndarray):        Array of bonds between two atoms.
            atom_type (ndarray):    Array of the types of atoms.

        Returns:
            dict, params.
        """
        bond_atoms = np.take(atom_type, bonds, -1)

        k_index = self.bond_params['parameter_names']["pattern"].index('force_constant')
        r_index = self.bond_params['parameter_names']["pattern"].index('bond_length')

        bond_params: dict = self.bond_params['parameters']
        params = {}
        for k, v in bond_params.items():
            [a, b] = k.split('-')
            if a != b:
                params[b + '-' + a] = v
        bond_params.update(params)

        bond_type = get_bonded_types(bond_atoms)
        type_list: list = bond_type.reshape(-1).tolist()

        if len(type_list) == 1:
            bond_length = [bond_params[type_list[0]][r_index]]
            force_constant = [bond_params[type_list[0]][k_index]]
        else:
            bond_length = []
            force_constant = []
            for params in itemgetter(*type_list)(bond_params):
                bond_length.append(params[r_index])
                force_constant.append(params[k_index])

        params = {'bond_index': bonds}
        params['force_constant'] = np.array(force_constant, np.float32).reshape(bond_type.shape)
        params['bond_length'] = np.array(bond_length, np.float32).reshape(bond_type.shape)

        return params

    def get_angle_params(self, angles, atom_type):
        """
        Get the force field angle parameters.

        Args:
            angles (ndarray):       Array of angles.
            atom_type (ndarray):    Array of the types of atoms.

        Returns:
            dict, params.
        """
        angle_atoms = np.take(atom_type, angles, -1)

        k_index = self.angle_params['parameter_names']["pattern"].index('force_constant')
        t_index = self.angle_params['parameter_names']["pattern"].index('bond_angle')

        angle_params: dict = self.angle_params['parameters']
        params = {}
        for k, v in angle_params.items():
            [a, b, c] = k.split('-')
            if a != c:
                params[c + '-' + b + '-' + a] = v
        angle_params.update(params)

        angle_type = get_bonded_types(angle_atoms)
        type_list: list = angle_type.reshape(-1).tolist()

        if len(type_list) == 1:
            bond_angle = [angle_params[type_list[0]][t_index]]
            force_constant = [angle_params[type_list[0]][k_index]]
        else:
            bond_angle = []
            force_constant = []
            for params in itemgetter(*type_list)(angle_params):
                bond_angle.append(params[t_index])
                force_constant.append(params[k_index])

        params = {'angle_index': angles}
        params['force_constant'] = np.array(force_constant, np.float32).reshape(angle_type.shape)
        params['bond_angle'] = np.array(bond_angle, np.float32).reshape(angle_type.shape) / 180 * np.pi

        return params

    def get_dihedral_params(self, dihedrals_in, atom_types):
        """
        Get the force field dihedral parameters.

        Args:
            dihedrals_in (ndarray): Array of input dihedrals.
            atom_type (ndarray):    Array of the types of atoms.

        Returns:
            dict, params.
        """
        dihedral_atoms = np.take(atom_types, dihedrals_in, -1)

        k_index = self._dihedrals['parameter_names']["pattern"][0].index('force_constant')
        phi_index = self._dihedrals['parameter_names']["pattern"][0].index('phase')
        t_index = self._dihedrals['parameter_names']["pattern"][0].index('periodicity')

        dihedral_params: dict = self._dihedrals['parameters']

        key_types_ndarray = np.array([specific_name.split('-') for specific_name in dihedral_params.keys()], np.str_)
        types_sorted_args = np.argsort((key_types_ndarray == '?').sum(axis=-1))
        sorted_key_types = key_types_ndarray[types_sorted_args]
        transformed_key_types = ['-'.join(specific_name).replace('?', '.+').replace('*', '\\*') for
                                 specific_name in sorted_key_types]

        dihedral_types, inverse_dihedral_types = get_dihedral_types(dihedral_atoms)
        type_list: list = dihedral_types.reshape(-1).tolist()
        inverse_type_list: list = inverse_dihedral_types.reshape(-1).tolist()

        for i, _ in enumerate(type_list):
            for key_type in transformed_key_types:
                if re.match('^'+key_type+'$', type_list[i]) or re.match('^'+key_type+'$', inverse_type_list[i]):
                    type_list[i] = key_type.replace('.+', '?').replace('\\', '')
                    break

        force_constant = []
        phase = []
        periodicity = []
        dihedral_index = []
        for i, params in enumerate(itemgetter(*type_list)(dihedral_params)):
            for _, lastd_params in enumerate(params):
                dihedral_index.append(dihedrals_in[i])
                force_constant.append(lastd_params[k_index])
                phase.append(lastd_params[phi_index])
                periodicity.append(lastd_params[t_index])

        params = {}
        params['force_constant'] = np.array(force_constant, np.float32)
        ks0_filter = np.where(params['force_constant'] != 0)[0]
        params['force_constant'] = params['force_constant'][ks0_filter]
        params['dihedral_index'] = np.array(dihedral_index, np.int32)[ks0_filter]
        params['phase'] = np.array(phase, np.float32)[ks0_filter] / 180 * np.pi
        params['periodicity'] = np.array(periodicity, np.float32)[ks0_filter]

        return params

    def get_improper_params(self, improper_in, atom_types, third_id):
        """
        Pre-processing of getting improper dihedrals.

        Args:
            improper_in (ndarray):  Array of input improper dihedrals.
            atom_types (ndarray):   Array of the types of atoms.
            third_id (ndarray):     Array of the third IDs.

        Returns:
            dict, params.
        """
        improper_atoms = np.take(atom_types, improper_in, -1)

        k_index = self._improper['parameter_names']["pattern"][0].index('force_constant')
        phi_index = self._improper['parameter_names']["pattern"][0].index('phase')
        t_index = self._improper['parameter_names']["pattern"][0].index('periodicity')

        improper_params: dict = self._improper['parameters']

        key_types_ndarray = np.array([specific_name.split('-') for specific_name in improper_params.keys()], np.str_)
        types_sorted_args = np.argsort((key_types_ndarray == '?').sum(axis=-1))
        sorted_key_types = key_types_ndarray[types_sorted_args]
        transformed_key_types = ['-'.join(specific_name).replace('?', '.+').replace('*', '\\*') for specific_name in
                                 sorted_key_types]

        improper_types, orders = get_improper_types(improper_atoms)
        type_list = improper_types[0].reshape(-1)

        not_defined_mask = np.zeros(type_list.shape).astype(np.int32)
        for i, _ in enumerate(type_list):
            for key_type in transformed_key_types:
                for j, itypes in enumerate(improper_types):
                    if re.match('^'+key_type+'$', itypes[i]):
                        this_improper = improper_in[i][np.array(list(orders[j]))]
                        if this_improper[2] != third_id[i]:
                            continue
                        improper_in[i] = this_improper
                        not_defined_mask[i] = 1
                        type_list[i] = key_type.replace('.+', '?').replace('\\', '')
                        break
                else:
                    continue
                break

        type_list = type_list[np.where(not_defined_mask > 0)[0]]

        force_constant = []
        phase = []
        periodicity = []
        improper_index = []
        improper = improper_in[np.where(not_defined_mask > 0)[0]]
        for i, params in enumerate(itemgetter(*type_list)(improper_params)):
            for _, lastd_params in enumerate(params):
                improper_index.append(improper[i])
                force_constant.append(lastd_params[k_index])
                phase.append(lastd_params[phi_index])
                periodicity.append(lastd_params[t_index])

        params = {'improper_index': np.array(improper_index, np.int32)}
        params['force_constant'] = np.array(force_constant, np.float32)
        params['phase'] = np.array(phase, np.float32) / 180 * np.pi
        params['periodicity'] = np.array(periodicity, np.float32)

        return params

    def construct_angles(self, bonds, bonds_for_angle, middle_id):
        for idx in middle_id:
            this_bonds = bonds[np.where(bonds_for_angle == idx)[0]]
            flatten_bonds = this_bonds.flatten()
            this_idx = np.delete(flatten_bonds, np.where(flatten_bonds == idx))
            yield this_idx

    def combinations(self, bonds, bonds_for_angle, middle_id):
        """
        Get all the combinations of 3 atoms.

        Args:
            bonds (ndarray):            Array of bonds.
            bonds_for_angle (ndarray):  Array of bonds for angles.
            middle_id (ndarray):        Array of middle IDs.

        Returns:
            np.ndarray, angles.
        """
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
        """
        Args:
            bonds (ndarray):    Array of bonds.

        Returns:
            dict, hash map.
        """
        hash_map = {}
        for i, b in enumerate(bonds):
            bond = tuple(b)
            hash_map[hash(bond)] = i
        return hash_map

    def trans_dangles(self, dangles, middle_id):
        """
        Construct the dihedrals.

        Args:
            dangles (ndarray):      Array of dangles.
            middle_id (ndarray):    Array of middle IDs.

        Returns:
            np.ndarray, dihedrals.
        """
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
        """
        Get the dihedrals indexes.

        Args:
            angles (ndarray):               Array of angles.
            dihedral_middle_id (ndarray):   Array of dihedrals middle indexes.

        Returns:
            np.ndarray, dihedrals.
        """
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

    def check_improper(self, bonds, core_id):
        """
        Check if there are same improper dihedrals.

        Args:
            bonds (ndarray):    Array of bonds.
            core_id (ndarray):  Array of core indexes.

        Returns:
            int, core id of same improper dihedrals.
        """
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

    def get_improper(self, bonds, core_id):
        """
        Get the improper dihedrals indexes.

        Args:
            bonds (ndarray):    Array of bonds.
            core_id (ndarray):  Array of core indexes.

        Returns:
            - improper (np.ndarray).
            - new_id (np.ndarray).
        """
        improper = None
        new_id = None
        for i in range(core_id.shape[0]):
            ids_for_idihedral = np.where(
                np.sum(np.isin(bonds, core_id[i]), axis=1) > 0
            )[0]
            bonds_for_idihedral = bonds[ids_for_idihedral]
            if bonds_for_idihedral.shape[0] == 3:
                idihedral = np.unique(bonds_for_idihedral.flatten())[None, :]
                if improper is None:
                    improper = idihedral
                    new_id = core_id[i]
                else:
                    improper = np.append(improper, idihedral, axis=0)
                    new_id = np.append(new_id, core_id[i])
            else:
                # Only SP2 is considered.
                continue
        return improper, new_id

    def get_excludes(self, bonds, angles, dihedrals, improper):
        """
        Get the exclude atoms index.

        Args:
            bonds (ndarray):        Array of bonds.
            angles (ndarray):       Array of angles.
            dihedrals (ndarray):    Array of dihedrals.
            improper (ndarray):     Array of improper.

        Returns:
            np.ndarray, the index of exclude atoms.
        """
        excludes = []
        for i in range(self.atom_nums):
            bond_excludes = bonds[np.where(
                np.isin(bonds, i).sum(axis=1))[0]].flatten()
            this_excludes = bond_excludes

            if angles is not None:
                angle_excludes = angles[
                    np.where(np.isin(angles, i).sum(axis=1))[0]
                ].flatten()
                this_excludes = np.append(this_excludes, angle_excludes)

            if dihedrals is not None:
                dihedral_excludes = dihedrals[
                    np.where(np.isin(dihedrals, i).sum(axis=1))[0]
                ].flatten()
                this_excludes = np.append(this_excludes, dihedral_excludes)
            if improper is not None:
                idihedral_excludes = improper[
                    np.where(np.isin(improper, i).sum(axis=1))[0]
                ].flatten()
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

    def get_vdw_params(self, atom_type: ndarray):
        """
        ['H','HO','HS','HC','H1','H2','H3','HP','HA','H4',
         'H5','HZ','O','O2','OH','OS','OP','C*','CI','C5',
         'C4','CT','CX','C','N','N3','S','SH','P','MG',
         'C0','F','Cl','Br','I','2C','3C','C8','CO']

        Args:
            atom_type (ndarray):    Array of atoms.

        Returns:
            dict, parameters.
        """

        sigma_index = self.vdw_params['parameter_names']["pattern"].index('sigma')
        eps_index = self.vdw_params['parameter_names']["pattern"].index('epsilon')

        vdw_params = self.vdw_params['parameters']
        type_list: list = atom_type.reshape(-1).tolist()
        sigma = []
        epsilon = []
        for params in itemgetter(*type_list)(vdw_params):
            sigma.append(params[sigma_index])
            epsilon.append(params[eps_index])

        if atom_type.ndim == 2 and atom_type.shape[0] > 1:
            #TODO
            type_list: list = atom_type[0].tolist()

        type_set = list(set(type_list))
        count = np.array([type_list.count(i) for i in type_set], np.int32)

        sigma_set = []
        eps_set = []
        for params in itemgetter(*type_set)(vdw_params):
            sigma_set.append(params[sigma_index])
            eps_set.append(params[eps_index])

        sigma_set = np.array(sigma_set)
        eps_set = np.array(eps_set)
        c6_set = 4 * eps_set * np.power(sigma_set, 6)
        param_count = count.reshape(1, -1) * count.reshape(-1, 1) - np.diag(count)
        mean_c6 = np.sum(c6_set * param_count) / param_count.sum()

        params = {}
        params['sigma'] = np.array(sigma, np.float32).reshape(atom_type.shape)
        params['epsilon'] = np.array(epsilon, np.float32).reshape(atom_type.shape)
        params['mean_c6'] = mean_c6.astype(np.float32)

        return params

    def get_pairwise_c6(self, e0, e1, r0, r1):
        """
        Calculate the B coefficient in vdw potential.

        Args:
            e0 (ndarray):   Coefficient one.
            e1 (ndarray):   Coefficient two.
            r0 (ndarray):   Coefficient three.
            r1 (ndarray):   Coefficient four.

        Returns:
            np.ndarray, the B coefficient in vdw potential.
        """
        e01 = np.sqrt(e0 * e1)
        r01 = r0 + r1
        return 2 * e01 * r01 ** 6

    def get_hbonds(self, bonds):
        """
        Get hydrogen bonds.

        Args:
            atom_type (ndarray):    Array of atoms.

        Returns:
            - bonds (np.ndarray), bonds with H.
            - bonds (np.ndarray), non H bonds.
        """
        hatoms = np.where(np.isin(self.atom_types, self.htypes))[0]
        bonds_with_h = np.where(np.isin(bonds, hatoms).sum(axis=-1))[0]
        non_hbonds = np.where(np.isin(bonds, hatoms).sum(axis=-1) == 0)[0]
        return bonds[bonds_with_h], bonds[non_hbonds]

    def get_pair_index(self, dihedrals, angles, bonds):
        """
        Get the non-bonded atom pairs index.

        Args:
            dihedrals (ndarray):    Array of dihedrals.
            angles (ndarray):       Array of angles.
            bonds (ndarray):        Array of bonds.

        Returns:
            np.ndarray, non-bonded atom pairs index.
        """
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
        """
        Return all the pair parameters.

        Args:
            pair_index (ndarray):   Array of pair indexes.
            epsilon (ndarray):      Array of epsilon.
            sigma (ndarray):        Array of sigma.

        Returns:
            dict, pair parameters.
        """

        r_index = self.pair_params['parameter_names']["pattern"].index('r_scale')
        r6_index = self.pair_params['parameter_names']["pattern"].index('r6_scale')
        r12_index = self.pair_params['parameter_names']["pattern"].index('r12_scale')

        pair_params = self.pair_params['parameters']
        if len(pair_params) == 1 and '?' in pair_params.keys():
            r_scale = pair_params['?'][r_index]
            r6_scale = pair_params['?'][r6_index]
            r12_scale = pair_params['?'][r12_index]
        else:
            #TODO
            r_scale = 0
            r6_scale = 0
            r12_scale = 0

        qiqj = np.take_along_axis(self.atom_charges, pair_index, axis=1)
        qiqj = np.prod(qiqj, -1)

        epsilon_ij = epsilon[pair_index]
        epsilon_ij = np.sqrt(np.prod(epsilon_ij, -1))

        sigma_ij = sigma[pair_index]
        sigma_ij = np.mean(sigma_ij, -1)

        pair_params = {}
        pair_params['qiqj'] = qiqj
        pair_params['epsilon_ij'] = epsilon_ij
        pair_params['sigma_ij'] = sigma_ij
        pair_params['r_scale'] = r_scale
        pair_params['r6_scale'] = r6_scale
        pair_params['r12_scale'] = r12_scale

        return pair_params

    def __call__(self, bonds):
        # pylint: disable=unused-argument
        bonds = bonds[0]
        atoms_types = self.atom_types.copy()
        vdw_params = self.get_vdw_params(atoms_types)
        atom_types = np.append(atoms_types, self._wildcard)

        bond_params = None
        angle_params = None
        if bonds is not None:
            hbonds, non_hbonds = self.get_hbonds(bonds)
            bond_params = self.get_bond_params(bonds, atoms_types)
            middle_id = np.where(np.bincount(bonds.flatten()) > 1)[0]
            ids_for_angle = np.where(
                np.sum(np.isin(bonds, middle_id), axis=1) > 0)[0]
            bonds_for_angle = bonds[ids_for_angle]
            angles = self.combinations(bonds, bonds_for_angle, middle_id)

            if angles is not None:
                angle_params = self.get_angle_params(angles, atoms_types)
            dihedral_middle_id = bonds[
                np.where(np.isin(bonds, middle_id).sum(axis=1) == 2)[0]
            ]
            dihedrals = self.get_dihedrals(angles, dihedral_middle_id)
            dihedral_params = None
            if dihedrals is not None:
                dihedral_params = self.get_dihedral_params(dihedrals, atom_types)
            core_id = np.where(np.bincount(bonds.flatten()) > 2)[0]
            improper = None
            improper_params = None
            if self._improper is not None:
                checked_core_id = self.check_improper(bonds, core_id)
                improper, third_id = self.get_improper(bonds, checked_core_id)
                improper_params = self.get_improper_params(improper, atom_types, third_id)
            if dihedrals is not None:
                self.pair_index = self.get_pair_index(dihedrals, angles, bonds)
                pair_params = self.get_pair_params(self.pair_index, vdw_params['epsilon'],
                                                   vdw_params['sigma'])
            else:
                pair_params = None
            self.excludes = self.get_excludes(bonds, angles, dihedrals, improper)

            return ForceConstants(
                bond_params,
                angle_params,
                dihedral_params,
                improper_params,
                angles,
                dihedrals,
                improper,
                self.excludes,
                vdw_params,
                hbonds,
                non_hbonds,
                pair_params,
            )

        return ForceConstants(excludes=self.excludes, vdw_param=self.vdw_param)
