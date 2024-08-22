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
"""Coupling dihedral energy"""

from itertools import product
import numpy as np

import mindspore as ms
from mindspore import Tensor, ops, vmap
from mindspore import numpy as msnp

from ...colvar import Torsion
from .energy import EnergyCell, _energy_register
from ...system import Molecule
from ...function import get_arguments


INDEX_CONSTANT_16 = Tensor(list(product([- 1, 0, 1, 2], [- 1, 0, 1, 2])), ms.int32)
EINSUM = ops.Einsum('ij,bj->bi')
POW = ops.Pow()
PI = 3.14159265359
TI = Tensor([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]], ms.float32)
TJ = Tensor([[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]], ms.float32)
DM = Tensor([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
             [-3., -0., 3., -0., -0., -0., -0., -0., -2., -0., -1., -0., -0., -0., -0., -0.],
             [2., 0., -2., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
             [-0., -0., -0., -0., -3., -0., 3., -0., -0., -0., -0., -0., -2., -0., -1., -0.],
             [0., 0., 0., 0., 2., 0., -2., 0., 0., 0., 0., 0., 1., 0., 1., 0.],
             [-3., 3., 0., 0., -2., -1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., -3., 3., 0., 0., -2., -1., 0., 0.],
             [9., -9., -9., 9., 6., 3., -6., -3., 6., -6., 3., -3., 4., 2., 2., 1.],
             [-6., 6., 6., -6., -4., -2., 4., 2., -3., 3., -3., 3., -2., -1., -2., -1.],
             [2., -2., -0., -0., 1., 1., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 2., -2., 0., 0., 1., 1., 0., 0.],
             [-6., 6., 6., -6., -3., -3., 3., 3., -4., 4., -2., 2., -2., -2., -1., -1.],
             [4., -4., -4., 4., 2., 2., -2., -2., 2., -2., 2., -2., 1., 1., 1., 1.]], dtype=ms.float32)
EPSILON = 1e-08


@ms.jit
def get_es(phi, psi, resolutions):
    """ Gather sub resolution table from origin resolutions.
    Args:
        phi: The dihedral of N-C-CA-N.
        psi: The dihedral of C-CA-N-C.
        resolutions: The complete resolution table.
    Returns:
        es: The sub resolution table with size (P, 16).
    """
    # (16, )
    idx_i = (phi + INDEX_CONSTANT_16[:, 0]) % 24
    idx_j = (psi + INDEX_CONSTANT_16[:, 1]) % 24
    # (16, 2)
    idx = msnp.vstack((idx_i, idx_j)).T
    # (16, )
    es = ops.gather_nd(resolutions, idx)
    return es


batch_es = vmap(get_es, in_axes=(0, 0, 0))


@ms.jit
def get_c1(e_sub):
    """ Bicubic interpolation.
    Args:
        e_sub: The input 16 * 16 sub resolution table.
    Returns:
        c_1: Interpolation coefficient tensor.
    """
    de = msnp.vstack([e_sub[:, 1, 1],
                      e_sub[:, 2, 1],
                      e_sub[:, 1, 2],
                      e_sub[:, 2, 2],
                      0.5 * (e_sub[:, 2, 1] - e_sub[:, 0, 1]),
                      0.5 * (e_sub[:, 3, 1] - e_sub[:, 1, 1]),
                      0.5 * (e_sub[:, 2, 2] - e_sub[:, 0, 2]),
                      0.5 * (e_sub[:, 3, 2] - e_sub[:, 1, 2]),
                      0.5 * (e_sub[:, 1, 2] - e_sub[:, 1, 0]),
                      0.5 * (e_sub[:, 2, 2] - e_sub[:, 2, 0]),
                      0.5 * (e_sub[:, 1, 3] - e_sub[:, 1, 1]),
                      0.5 * (e_sub[:, 2, 3] - e_sub[:, 2, 1]),
                      0.25 * (e_sub[:, 2, 2] + e_sub[:, 0, 0] - e_sub[:, 2, 0] - e_sub[:, 0, 2]),
                      0.25 * (e_sub[:, 3, 2] + e_sub[:, 1, 0] - e_sub[:, 3, 0] - e_sub[:, 1, 2]),
                      0.25 * (e_sub[:, 2, 2] + e_sub[:, 0, 2] - e_sub[:, 2, 1] - e_sub[:, 0, 3]),
                      0.25 * (e_sub[:, 3, 3] + e_sub[:, 1, 1] - e_sub[:, 3, 1] - e_sub[:, 1, 3])]).T
    # (P, 16)
    c_1 = EINSUM((DM, de))
    return c_1.reshape((-1, 4, 4))


@ms.jit
def e_interp(c, phi, psi):
    """ Calculate one-site energy.
    Args:
        c: Interpolation coefficient tensor.
        phi: The dihedral of N-C-CA-N.
        psi: The dihedral of C-CA-N-C.
    Returns:
        One-site energy for each phi-psi pairs.
    """
    # (P, 4, 4)
    ei = c * POW(phi[..., None], TI) * POW(psi[..., None], TJ)
    return ei.sum((1, 2))


@_energy_register('cmap_energy')
class CmapEnergy(EnergyCell):
    r"""Energy term of coupling dihedral (torsion) angles.

    Math:

    .. math::

        E_{cmap}(\phi, \psi) = \sum_{i=1}^{4}\sum_{j=1}^{4}c_{ij}\left(\frac{\phi-\phi_L}{\Delta\phi}\right)^i\left(
            \frac{\psi-\psi_L}{\Delta\psi}\right)^j

    Args:

        index (Union[Tensor, ndarray, List[int]]):
                            Array of the indices of the atoms forming the dihedral angles.
                            The shape of array is `(B, d, 4)`, and the data type is int.

        parameters (dict):  Force field parameters. Default: None

        use_pbc (bool):     Whether to use periodic boundary condition.

        length_unit (str):  Length unit. If None is given, it will be assigned with the global length unit.
                            Default: 'nm'

        energy_unit (str):  Energy unit. If None is given, it will be assigned with the global energy unit.
                            Default: 'kj/mol'

        name (str):         Name of the energy. Default: 'dihedral'

    Supported Platforms:

        ``Ascend`` ``GPU``

    Symbols:

        B:  Batchsize, i.e. number of walkers in simulation

        d:  Number of dihedral angles.

        D:  Spatial dimension of the simulation system. Usually is 3.

    """
    def __init__(self,
                 system: Molecule = None,
                 parameters: dict = None,
                 use_pbc: bool = None,
                 length_unit: str = 'nm',
                 energy_unit: str = 'kj/mol',
                 name: str = 'cmap_energy',
                 **kwargs,
                 ):

        super().__init__(
            name=name,
            use_pbc=use_pbc,
            length_unit=length_unit,
            energy_unit=energy_unit,
        )
        self._kwargs = get_arguments(locals(), kwargs)
        if 'exclude_index' in self._kwargs:
            self._kwargs.pop('exclude_index')

        self.resolutions = None
        if parameters is not None:
            if system is None:
                raise ValueError('`system` cannot be None when using `parameters`!')
            length_unit = parameters.get('length_unit')
            energy_unit = parameters.get('energy_unit')
            self.units.set_units(length_unit, energy_unit)
            self._use_pbc = system.use_pbc
            res_index, torsion_index_0, torsion_index_1, resolutions = self._get_parameters(system, parameters)
            res_index = Tensor(res_index, ms.int32)
            torsion_index_0 = Tensor(torsion_index_0, ms.int32)
            torsion_index_1 = Tensor(torsion_index_1, ms.int32)
            # (d)
            self.get_phi = Torsion(atoms=torsion_index_0[res_index], use_pbc=use_pbc, batched=True)
            self.get_psi = Torsion(atoms=torsion_index_1[res_index], use_pbc=use_pbc, batched=True)
            # (B,24,24)
            self.resolutions = Tensor(resolutions.reshape((resolutions.shape[0], 24, 24)), ms.float32)
        self.cast = ops.Cast()
        self.concat = ops.Concat()

    @staticmethod
    def _get_parameters(system: Molecule, parameters: dict):
        """ Get the parameters from force field param file for the given system.
        Args:
            system: Input system.
            parameters: Force field parameters.
        Returns:
            res_index: Indicates of residues with coupling dihedrals.
            torsion_index_0: The phi dihedral atom indexes.
            torsion_index_1: The psi dihedral atom indexes.
            params: The correspond resolution tables.
        """
        res = system.residue_name
        atom_name = np.array(system.atom_name[0], np.str_)
        n_index = np.where(atom_name == 'N')[0]
        c_index = np.where(atom_name == 'C')[0]
        ca_index = np.where(atom_name == 'CA')[0]
        last_c_index = np.roll(c_index, 1)
        next_n_index = np.roll(n_index, -1)
        torsion_index_0 = np.vstack((last_c_index, n_index, ca_index, c_index)).T
        torsion_index_1 = np.vstack((n_index, ca_index, c_index, next_n_index)).T
        resolutions: dict = parameters['parameters']
        params = []
        res_index = []
        for i, r in enumerate(res):
            if r not in resolutions.keys():
                continue
            params.append(resolutions[r])
            res_index.append(i)
        res_index = np.array(res_index, np.int32)
        params = np.array(params, np.float32)
        return res_index, torsion_index_0, torsion_index_1, params

    @ms.jit
    def get_e_sub(self, phi, psi):
        """ Get sub resolution from given phi and psi dihedrals.
        Args:
            phi: The phi angles.
            psi: The psi angles.
        Returns:
            Esub: Sub resolution with size (P, 4, 4).
        """
        # (P, )
        phi_index = self.cast(((phi + PI) // (2 * PI / 24)) % 24, ms.int32).reshape(-1)
        psi_index = self.cast(((psi + PI) // (2 * PI / 24)) % 24, ms.int32).reshape(-1)
        # (P, 16)
        e_sub = batch_es(phi_index, psi_index, self.resolutions)
        return e_sub.reshape((-1, 4, 4))

    @ms.jit
    def construct(self,
                  coordinate: Tensor,
                  neighbour_index: Tensor = None,
                  neighbour_mask: Tensor = None,
                  neighbour_vector: Tensor = None,
                  neighbour_distance: Tensor = None,
                  pbc_box: Tensor = None
                  ):
        r"""Calculate energy term.

        Args:
            coordinate (Tensor):            Tensor of shape (B, A, D). Data type is float.
                                            Position coordinate of atoms in system
            neighbour_index (Tensor):       Tensor of shape (B, A, N). Data type is int.
                                            Index of neighbour atoms.
            neighbour_mask (Tensor):        Tensor of shape (B, A, N). Data type is bool.
                                            Mask for neighbour index.
            neighbour_vector (Tensor):       Tensor of shape (B, A, N). Data type is bool.
                                            Vectors from central atom to neighbouring atoms.
            neighbour_distance (Tensor):    Tensor of shape (B, A, N). Data type is float.
                                            Distance between neighbours atoms.
            pbc_box (Tensor):               Tensor of shape (B, D). Data type is float.
                                            Tensor of PBC box. Default: None

        Returns:
            energy (Tensor):    Tensor of shape (B, 1). Data type is float.

        Symbols:
            B:  Batch size, i.e. number of walkers in simulation
            A:  Number of atoms.
            D:  Spatial dimension of the simulation system. Usually is 3.

        """
        # (P, 1)
        phi = self.get_phi(coordinate, pbc_box)
        psi = self.get_psi(coordinate, pbc_box)
        # (P, 4, 4)
        e_sub = self.get_e_sub(phi, psi)
        # (P, 4, 4)
        c = get_c1(e_sub)
        delta_phi = 2 * PI / 24
        delta_psi = 2 * PI / 24
        # (1, )
        energy = e_interp(c, phi % delta_phi / delta_phi,
                          psi % delta_psi / delta_psi)
        # (B,1) <- (1, )
        return energy.sum(keepdims=True)[None]
