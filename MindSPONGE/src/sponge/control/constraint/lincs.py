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
LINCS Constraint algorithm
"""

from typing import Union, Tuple
import numpy as np

import mindspore as ms
import mindspore.numpy as msnp
from mindspore import Tensor
from mindspore import ops
from mindspore.ops import functional as F

from . import Constraint
from ...system import Molecule
from ...potential import PotentialCell
from ...function.operations import GetShiftGrad
from ...function import get_arguments, get_ms_array

class EinsumWrapper(ms.nn.Cell):
    r"""
    Implement particular Einsum operation

    Args:
        equation (str): an equation representing the operation.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """
    def __init__(self, equation: str):
        super().__init__(auto_prefix=False)
        self.equation = equation

    def construct(self, xy: Tuple[Tensor, Tensor]) -> Tensor:
        """Calculation for Einsum operation"""

        result = None
        if self.equation == 'ijk,ilkm->iljm':
            ijk, ilkm = xy
            iljk = ops.expand_dims(ijk, 1).broadcast_to(ijk.shape[:1] + ilkm.shape[1:2] + ijk.shape[1:])
            iljm = ops.BatchMatMul()(iljk, ilkm)
            result = iljm
        elif self.equation == 'ijkl,imkl->ijm':
            ijkl, imkl = xy
            ijmkl1 = ops.expand_dims(ijkl, 2).broadcast_to(ijkl.shape[:2] + imkl.shape[1:2] + ijkl.shape[2:])
            ijmkl2 = ops.expand_dims(imkl, 1).broadcast_to(imkl.shape[:1] + ijkl.shape[1:2] + imkl.shape[1:])
            ijm = ops.ReduceSum()(ijmkl1 * ijmkl2, [-1, -2])
            result = ijm
        elif self.equation == 'ijkl,ikl->ij':
            ijkl, ikl = xy
            ijkl2 = ops.expand_dims(ikl, 1).broadcast_to(ikl.shape[:1] + ijkl.shape[1:2] + ikl.shape[1:])
            ij = ops.ReduceSum()(ijkl * ijkl2, [-1, -2])
            result = ij
        elif self.equation == 'ijk,ik->ij':
            ijk, ik = xy
            ijk2 = ops.expand_dims(ik, 1).broadcast_to(ik.shape[:1] + ijk.shape[1:2] + ik.shape[1:])
            ij = ops.ReduceSum()(ijk * ijk2, -1)
            result = ij
        elif self.equation == 'ijkl,ij->ikl':
            ijkl, ij = xy
            ijkl2 = ij.reshape(ij.shape + (1, 1)).broadcast_to(ij.shape + ijkl.shape[-2:])
            ikl = ops.ReduceSum()(ijkl * ijkl2, 1)
            result = ikl
        elif self.equation == 'ijk,ikl->ijl':
            ijk, ikl = xy
            ijl = ops.BatchMatMul()(ijk, ikl)
            result = ijl
        elif self.equation == 'ijk,ijl->ikl':
            ijk, ijl = xy
            ijkl1 = ijk[..., None].broadcast_to(ijk.shape + ijl.shape[-1:])
            ijkl2 = ijl[..., None, :].broadcast_to(ijl.shape[:2] + ijk.shape[-1:] + ijl.shape[-1:])
            result = (ijkl1 * ijkl2).sum(axis=1)
        else:
            raise NotImplementedError("This equation is not implemented")
        return result


class Lincs(Constraint):
    r"""
    A LINCS (LINear Constraint Solver) constraint module,
    which is a subclass of :class:`sponge.control.Constraint`.

    Args:
        system ( :class:`sponge.system.Molecule`): Simulation system.
        bonds (Union[Tensor, str], optional): Bonds to be constrained.
          This arguments accept a Tensor of shape :math:`(K, 2)` with data type int,
          or a string with value "h-bonds" or "all-bonds".
          Default: "h-bonds".
        potential ( :class:`sponge.potential.PotentialCell`, optional):
          Potential Cell. Default: ``None``.

    Inputs:
        - **coordinate** (Tensor) - Coordinate. Tensor of shape :math:`(B, A, D)`.
          Data type is float.
          Here :math:`B` is the number of walkers in simulation,
          :math:`A` is the number of atoms and
          :math:`D` is the spatial dimension of the simulation system, which is usually 3.
        - **velocity** (Tensor) - Velocity. Tensor of shape :math:`(B, A, D)`. Data type is float.
        - **force** (Tensor) - Force. Tensor of shape :math:`(B, A, D)`. Data type is float.
        - **energy** (Tensor) - Energy. Tensor of shape :math:`(B, 1)`. Data type is float.
        - **kinetics** (Tensor) - Kinetics. Tensor of shape :math:`(B, D)`. Data type is float.
        - **virial** (Tensor) - Virial. Tensor of shape :math:`(B, D)`. Data type is float.
        - **pbc_box** (Tensor) - Pressure boundary condition box. Tensor of shape :math:`(B, D)`.
          Data type is float.
        - **step** (int) - Simulation step. Default: ``0``.

    Outputs:
        - coordinate, Tensor of shape :math:`(B, A, D)`. Coordinate. Data type is float.
        - velocity, Tensor of shape :math:`(B, A, D)`. Velocity. Data type is float.
        - force, Tensor of shape :math:`(B, A, D)`. Force. Data type is float.
        - energy, Tensor of shape :math:`(B, 1)`. Energy. Data type is float.
        - kinetics, Tensor of shape :math:`(B, D)`. Kinetics. Data type is float.
        - virial, Tensor of shape :math:`(B, D)`. Virial. Data type is float.
        - pbc_box, Tensor of shape :math:`(B, D)`. Periodic boundary condition box.
          Data type is float.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> from sponge import Molecule
        >>> from sponge.control import Lincs
        >>> system = Molecule(template='water.tip3p.yaml')
        >>> controller = Lincs(system)
    """

    def __init__(self,
                 system: Molecule,
                 bonds: Union[Tensor, str] = 'h-bonds',
                 potential: PotentialCell = None,
                 **kwargs
                 ):

        super().__init__(
            system=system,
            bonds=bonds,
            potential=potential,
        )
        print('[MindSPONGE] The lincs constraint is used for the molecule system.')

        self._kwargs = get_arguments(locals(), kwargs)

        if isinstance(bonds, str):
            if bonds.lower() == 'h-bonds':
                if system.remaining_index is None:
                    self.bonds = ops.gather(system.bonds, system.h_bonds, axis=-2)
                else:
                    take_index = ops.nonzero((system.remaining_index == system.h_bonds[..., None]).sum(-2)).reshape(-1)
                    self.bonds = ops.gather(system.bonds, system.remaining_index[take_index], axis=-2)
            elif bonds.lower() == 'all-bonds':
                if system.remaining_index is None:
                    self.bonds = system.bonds
                else:
                    self.bonds = ops.gather(system.bonds, system.remaining_index, axis=-2)
            else:
                raise ValueError(f'"bonds" must be "h-bonds" or "all-bonds" but got: {bonds}')
        else:
            try:
                self.bonds = get_ms_array(bonds, ms.int32)
            except TypeError:
                raise TypeError(f'The type of "bonds" must be Tensor or str, but got: {type(bonds)}')

        if self.bonds.ndim != 2:
            if self.bonds.ndim != 3:
                raise ValueError(f'The rank of "bonds" must be 2 or 3 but got: {self.bonds.ndim}')

            if self.bonds.shape[0] != 1:
                raise ValueError(f'For constraint, the batch size of "bonds" must be 1 but got: {self.bonds[0]}')
            self.bonds = self.bonds[0]

        if self.bonds.shape[-1] != 2:
            raise ValueError(f'The last dimension of "bonds" but got: {self.bonds.shape[-1]}')

        self.num_constraints = self.bonds.shape[-2]

        #pylint: disable=invalid-name
        flatten_bonds = self.bonds.reshape(-1).asnumpy()
        remaining_atoms = Tensor(np.sort(np.unique(flatten_bonds)), ms.int32)

        self.remaining_atoms = remaining_atoms
        self.bs_index = ops.broadcast_to(self.remaining_atoms[None, ..., None],
                                         (1, self.remaining_atoms.shape[0], 3))
        mapping_atoms = msnp.arange(remaining_atoms.shape[-1])

        mapping = dict(zip(remaining_atoms.asnumpy(), mapping_atoms.asnumpy()))
        self.bonds = Tensor(np.vectorize(mapping.get)(self.bonds.asnumpy()), ms.int32)

        self.num_atoms = remaining_atoms.shape[0]
        # (R,R) <- (R,R)
        iinvM = msnp.identity(self.num_atoms)
        self.inv_mass = ops.gather(self.inv_mass, remaining_atoms, axis=-1)

        # (B,R,R) = (1,R,R) * (B,1,R)
        self.Mii = msnp.broadcast_to(
            iinvM, (1,) + iinvM.shape) * self.inv_mass[:, None, :]

        self.BMatrix = GetShiftGrad(
            num_atoms=self.num_atoms,
            bonds=self.bonds,
            num_walkers=self.num_walker,
            dimension=self.dimension,
            use_pbc=self.use_pbc
        )
        # (B,C,R,D)
        shape = (self.num_walker,
                 self.bonds.shape[-2], self.num_atoms, self.dimension)

        self.broadcast = ops.BroadcastTo(shape)
        self.inv = ops.MatrixInverse(adjoint=False)
        self.squeeze = ops.Squeeze()
        self.einsum0 = EinsumWrapper('ijk,ilkm->iljm')
        self.einsum1 = EinsumWrapper('ijkl,imkl->ijm')
        self.einsum2 = EinsumWrapper('ijkl,ikl->ij')
        self.einsum3 = EinsumWrapper('ijk,ik->ij')
        self.einsum4 = EinsumWrapper('ijkl,ij->ikl')
        self.einsum5 = EinsumWrapper('ijk,ikl->ijl')
        self.einsum6 = EinsumWrapper('ijk,ijl->ikl')

        # (B,C,R)
        shape = (self.num_walker, self.num_constraints, self.num_atoms)

        # (1,C,1)
        bond0 = self.bonds[..., 0].reshape(1, -1, 1).asnumpy()
        # (B,C,R) <- (B,A,1)
        mask0 = np.zeros(shape)
        np.put_along_axis(mask0, bond0, 1, axis=-1)
        # (B,C,R,1)
        self.mask0 = F.expand_dims(Tensor(mask0, ms.int32), -1)

        # (1,C,1)
        bond1 = self.bonds[..., 1].reshape(1, -1, 1).asnumpy()
        # (B,C,R) <- (B,R,1)
        mask1 = np.zeros(shape)
        np.put_along_axis(mask1, bond1, 1, axis=-1)
        # (B,C,R,1)
        self.mask1 = F.expand_dims(Tensor(mask1, ms.int32), -1)
        self.scatter_update = ops.tensor_scatter_elements

    def construct(self,
                  coordinate: Tensor,
                  velocity: Tensor,
                  force: Tensor,
                  energy: Tensor,
                  kinetics: Tensor,
                  virial: Tensor = None,
                  pbc_box: Tensor = None,
                  step: int = 0,
                  ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        r"""
        Constraint the bonds.

        Args:
            coordinate (Tensor): Tensor of shape :math:`(B, A, D)`. Data type is float.
            velocity (Tensor): Tensor of shape :math:`(B, A, D)`. Data type is float.
            force (Tensor): Tensor of shape :math:`(B, A, D)`. Data type is float.
            energy (Tensor): Tensor of shape :math:`(B, 1)`. Data type is float.
            kinetics (Tensor): Tensor of shape :math:`(B, D)`. Data type is float.
            virial (Tensor): Tensor of shape :math:`(B, D)`. Data type is float.
            pbc_box (Tensor): Tensor of shape :math:`(B, D)`. Data type is float.
            step (int): Simulation step. Default: ``0``.

        Returns:
            - **coordinate** (Tensor) - Tensor of shape :math:`(B, A, D)`. Data type is float.
            - **velocity** (Tensor) - Tensor of shape :math:`(B, A, D)`. Data type is float.
            - **force** (Tensor) - Tensor of shape :math:`(B, A, D)`. Data type is float.
            - **energy** (Tensor) - Tensor of shape :math:`(B, 1)`. Data type is float.
            - **kinetics** (Tensor) - Tensor of shape :math:`(B, D)`. Data type is float.
            - **virial** (Tensor) - Tensor of shape :math:`(B, D)`. Data type is float.
            - **pbc_box** (Tensor) - Tensor of shape :math:`(B, D)`. Data type is float.

        Note:
            :math:`B` is the number of walkers in simulation.
            :math:`A` is the number of atoms.
            :math:`D` is the spatial dimension of the simulation system. Usually is 3.
        """

        #pylint: disable=invalid-name

        # (B,A,D)
        last_crd = coordinate.copy()

        # (B,R,D)
        coordinate_old = msnp.take_along_axis(self._coordinate.copy(), self.remaining_atoms[None, ..., None], axis=-2)
        coordinate_new = msnp.take_along_axis(coordinate.copy(), self.remaining_atoms[None, ..., None], axis=-2)

        # (B,C,R,D)
        BMatrix = self.BMatrix(coordinate_new, coordinate_old, pbc_box)

        # ijk,ilkm->iljm
        # (B,C,R,D)<-(B,R,R),(B,C,R,D)
        tmp0 = self.einsum0((self.Mii, BMatrix))

        # ijkl,imkl->ijm
        # (B,C,C)<-(B,C,R,D),(B,C,R,D)
        tmp1 = self.einsum1((BMatrix, tmp0))

        # (B,C,C)
        tmp2 = self.inv(tmp1)

        # (B,C,R,D) <- (B,R,D)
        pos_old = self.broadcast(F.expand_dims(coordinate_old, -3))

        # (B,C,D) <- (B,C,R,D) = (B,C,R,1) * (B,C,R,D)
        pos_old_0 = F.reduce_sum(self.mask0 * pos_old, -2)
        pos_old_1 = F.reduce_sum(self.mask1 * pos_old, -2)

        # (B,C)
        di = self.get_distance(pos_old_0, pos_old_1, pbc_box)

        # ijkl,ikl->ij
        # (B,C)<-(B,C,R,D),(B,R,D)
        tmp3 = self.einsum2((BMatrix, coordinate_new)) - di

        # ijk,ik->ij
        # (B,C)<-(B,C,C),(B,C)
        tmp4 = self.einsum3((tmp2, tmp3))

        # ijkl,ij->ikl
        # (B,R,D)<-(B,C,R,D),(B,C)
        tmp5 = self.einsum4((BMatrix, tmp4))

        # ijk,ikl->ijl
        # (B,R,D)<-(B,R,R),(B,R,D)
        dr = -self.einsum5((self.Mii, tmp5))

        # (B,R,D)
        update_crd = msnp.take_along_axis(coordinate.copy(), self.remaining_atoms[None, ..., None], axis=-2)
        # (B,A,D)
        coordinate = self.scatter_update(coordinate, self.bs_index, update_crd + dr, axis=-2)

        # (B,A,D)
        velocity += (coordinate - last_crd) / self.time_step

        # Constraint force = m * dR / dt^2
        # (B,A,D)<-(B,A,1),(B,A,D)
        constraint_force = self._atom_mass * (coordinate - last_crd) / (self.time_step ** 2)
        force += constraint_force

        if self._pbc_box is not None:
            # (B,D)<-(B,A,D)<-(B,A,D),(B,A,D)
            virial += -0.5 * (last_crd * constraint_force).sum(-2)

        return coordinate, velocity, force, energy, kinetics, virial, pbc_box
