# Copyright 2021-2024 @ Shenzhen Bay Laboratory &
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
Rigid Body System
"""

import sys
import math
import numpy as np

import mindspore as ms
from mindspore import Tensor, jit, ops, vmap, Parameter
from mindspore import numpy as msnp

from sponge import Molecule
from sponge.function import hamiltonian_product
from sponge.function.units import GLOBAL_UNITS, Units
from sponge.system.modelling.mol2_parser import mol2parser

sys.path.append('../../../')

PI = 3.1415927
ATOM_MASS = {'C': 12.01, 'H': 1.008}


class BenzRigidBody(Molecule):
    """ Class for Benzene toy model system.
    Args:
        mol2(str): The input mol2 format file name.
        length_unit(str): The input length unit.
        residues: TODO
        index(ndarray): The rigid atom indexes in the system.
    """
    def __init__(self, mol2, length_unit=None, residues=None, index=None):
        super().__init__()
        if length_unit is None:
            length_unit = GLOBAL_UNITS.length_unit
        self.units = Units(length_unit, GLOBAL_UNITS.energy_unit)
        self.einsum_0 = ops.Einsum('ij,ij->i')
        self.concat = ops.Concat(axis=-2)
        self.hamiltonian_product = vmap(hamiltonian_product, in_axes=(1, 1))
        self.hamiltonian_product_2 = vmap(self.hamiltonian_product, in_axes=(None, 2))
        self.l1 = 1.4*math.sqrt(3)
        self.l2 = 1.4*math.sqrt(3)
        self.angle = PI / 3
        self.height = self.l2 * math.sin(self.angle)
        self.floor = math.sqrt(self.l2 ** 2 - self.height ** 2)
        mol2_obj = mol2parser(mol2)
        # (A, )
        atom_names = mol2_obj.atom_names
        self.atom_name = atom_names[None]
        # (B, A)
        self.atom_type = np.array([[atype.replace('.AR', 'A').replace('H', 'HA') for atype in mol2_obj.atom_types]])
        charges = mol2_obj.charges
        self.atom_charge = Tensor(charges, ms.float32)
        res_ids = mol2_obj.res_ids
        self.atom_resids = Tensor(res_ids, ms.int32)
        self.atom_mass = Tensor([ATOM_MASS[name] for name in atom_names], ms.float32)
        self.atom_mask = msnp.ones(self.atom_mass.shape).astype(ms.int32)[None]
        self._atom_mask = self.atom_mask.asnumpy()
        # (A, 3)
        crds = mol2_obj.crds
        self.full_crds = Tensor(crds, ms.float32)
        # (B, 2)
        bond_index = mol2_obj.bond_indexes
        self.bonds = Tensor(bond_index[None], ms.int32)
        # residue_atoms = mol2_obj.residue_atoms
        if index is None:
            raise ValueError('The rigid atom indexes should be set.')

        if residues is None:
            if isinstance(index, Tensor):
                self.rigid_index = index
            else:
                self.rigid_index = Tensor(index, ms.int32)
            self.tri_mask = self.rigid_index.reshape(-1)
            self.delete_mask = msnp.setdiff1d(msnp.arange(self.full_crds.shape[-2]), self.tri_mask)
            # (G, 3)
            group_rigid_index = self.rigid_index.reshape((self.rigid_index.shape[0], 3))
            # (G, 3)
            self.origins = (self.full_crds[group_rigid_index[:, 0]] + self.full_crds[group_rigid_index[:, 1]]) / 2
            group_atoms = int(self.full_crds.shape[-2] / (self.tri_mask.shape[0] / 3))
            # (G, g-3, 3)
            group_crds = self.full_crds[self.delete_mask].reshape((-1, group_atoms-3, 3))
            self.group_vec = group_crds - self.origins[:, None]
        else:
            #TODO
            raise ValueError("To be implemented.")

        self.num_rigids = self.rigid_index.shape[-2]
        self.total_atoms = crds.shape[-2]
        self.num_atoms = self.total_atoms
        # (G, 3, 3)
        rcrd = self.full_crds[self.rigid_index]
        # (1, 4G, 3)
        self.rt = self.rigid_from_points(rcrd).reshape((1, self.num_rigids*4, 3))
        # (1, G, 6)
        self.quat = Parameter(Tensor(np.zeros((1, self.num_rigids, 6)), ms.float32), name='quat', requires_grad=True)
        # (1, G, 3)
        self.standard_shape = (1, self.num_rigids, 3)
        self.build_system()
        self.build_space(self.full_crds, pbc_box=None)

    # @jit
    def full_atoms(self):
        """ full atoms """
        rigid_crds = self.get_rotate_rigid()
        full_crd = self.coordinate[0]
        # (G, 3, 3)
        rigid_xyz = msnp.squeeze(self.get_xyz(rigid_crds))
        full_crd[self.tri_mask] = rigid_xyz
        group_xyz = rigid_xyz.reshape((self.group_vec.shape[0], -1, 3))
        # (G, 3)
        new_origins = (group_xyz[:, 0] + group_xyz[:, 1]) / 2
        last_xyz = (self.calc_full_rotate() + new_origins[None, :, None]).reshape((-1, 3))
        full_crd[self.delete_mask] = last_xyz
        return full_crd[None]

    # @jit
    def get_coordinate(self, atoms: AtomsBase = None):
        _ = atoms
        return self.full_atoms()

    @jit
    def rigid_from_points(self, crd):
        """ Transform the coordinates formulation. """
        # (N, 3)
        v1 = crd[:, 2] - crd[:, 1]
        v2 = crd[:, 0] - crd[:, 1]
        e1 = v1 / msnp.norm(v1, axis=-1, keepdims=True)
        u2 = v2 - e1 * self.einsum_0((e1, v2))[:, None]
        e2 = u2 / msnp.norm(u2, axis=-1, keepdims=True)
        e3 = msnp.cross(e1, e2, axisc=-1)
        # (N, 3, 3)
        r = self.concat((e1[:, None], e2[:, None], e3[:, None]))
        t = crd[:, 1][:, None]
        # (N, 4, 3)
        new_crd = self.concat((r, t))
        return new_crd

    @jit
    def get_xyz(self, crd):
        """ Transform the (R,T) into cartesian coordinate system. """
        batches = crd.shape[0]
        # (B, N, 4, 3)
        rigid_crd = crd.reshape((batches, -1, 4, 3))
        # (B, N, 3)
        ca = rigid_crd[:, :, -1]
        ca_c = rigid_crd[:, :, 0] * self.l1
        c = ca + ca_c
        ca_c_n = rigid_crd[:, :, 1] * self.height
        # N = Ca + NC
        n = c - rigid_crd[:, :, 0] * self.floor + ca_c_n
        # (B, N, 3, 3)
        xyz = msnp.vstack((n, ca, c)).swapaxes(0, 1)
        # (B, 3N, 3)
        return xyz.reshape((batches, -1, 3))

    @jit
    def get_rotate_rigid(self):
        """ Apply hamiltonian product. """
        # (B, N, 4, 3)
        triangle = self.rt.reshape((self.coordinate.shape[0], -1, 4, 3))
        # (B, N, 4)
        e1 = msnp.pad(triangle[:, :, 0], ((0, 0), (0, 0), (1, 0)), mode='constant', constant_values=0)
        e2 = msnp.pad(triangle[:, :, 1], ((0, 0), (0, 0), (1, 0)), mode='constant', constant_values=0)
        e3 = msnp.pad(triangle[:, :, 2], ((0, 0), (0, 0), (1, 0)), mode='constant', constant_values=0)
        # (B, N, 3)
        t = triangle[:, :, 3]
        # (B, N, 4)
        r = msnp.pad(self.quat[:, :, :3], ((0, 0), (0, 0), (1, 0)), mode='constant', constant_values=1)
        r /= msnp.norm(r, axis=-1, keepdims=True)
        # (B, N, 3)
        dt = self.quat[:, :, 3:]
        t += dt
        # (B, N, 3)
        new_e1 = self.hamiltonian_product(r, e1)[:, :, 1:].reshape(self.standard_shape)
        new_e2 = self.hamiltonian_product(r, e2)[:, :, 1:].reshape(self.standard_shape)
        new_e3 = self.hamiltonian_product(r, e3)[:, :, 1:].reshape(self.standard_shape)
        # (B, N, 4, 3)
        new_triangle = msnp.vstack((new_e1, new_e2, new_e3, t)).swapaxes(0, 1).reshape(self.rt.shape)
        return new_triangle

    @jit
    def calc_full_rotate(self):
        """ Calculate the full rotation"""
        # (B, G, g-3, 4)
        crds = msnp.pad(self.group_vec, ((0, 0), (0, 0), (1, 0)), mode='constant', constant_values=0)[None]
        # (B, G, 4)
        r = msnp.pad(self.quat[:, :, :3], ((0, 0), (0, 0), (1, 0)), mode='constant', constant_values=1)
        r /= msnp.norm(r, axis=-1, keepdims=True)
        # (B, G, g-3, 3)
        new_crds = self.hamiltonian_product_2(r, crds).swapaxes(0, 1).reshape(
            (self.coordinate.shape[0], self.quat.shape[1], crds.shape[-2], crds.shape[-1]))[..., 1:]
        return new_crds

    def construct(self):
        pbc_box = None
        if self.pbc_box is not None:
            pbc_box = self.identity(self.pbc_box)
        return self.full_atoms(), pbc_box
