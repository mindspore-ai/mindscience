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
LINCS Constraint algorithm
"""

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


class Lincs(Constraint):
    """
    LINCS (LINear Constraint Solver) constraint controller.

    Args:
        system (Molecule):          Simulation system.
        bonds (Tensor):             Bonds to be constraint.
                                    Tensor of shape (B, 2). Data type is int.
                                    Default: "h-bonds".
        potential (PotentialCell):  Potential Cell. Default: None

    Inputs:
        - **coordinate** (Tensor) - The coordinates of the system.
        - **velocity** (Tensor) - The velocity of the system.
        - **force** (Tensor) - The force of the system.
        - **energy** (Tensor) - The energy of the system.
        - **kinetics** (Tensor) - The kinetics of the system.
        - **virial** (Tensor) - The virial of the system. Default: None
        - **pbc_box** (Tensor) - PBC box of the system. Default: None
        - **step** (int) - The step of the system. Default: 0

    Return:
        - coordinate (Tensor), Tensor of shape (B, A, D). Data type is float.
        - velocity (Tensor), Tensor of shape (B, A, D). Data type is float.
        - force (Tensor), Tensor of shape (B, A, D). Data type is float.
        - energy (Tensor), Tensor of shape (B, 1). Data type is float.
        - kinetics (Tensor), Tensor of shape (B, D). Data type is float.
        - virial (Tensor), Tensor of shape (B, D). Data type is float.
        - pbc_box (Tensor), Tensor of shape (B, D). Data type is float.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """

    def __init__(self,
                 system: Molecule,
                 bonds: Tensor = 'h-bonds',
                 potential: PotentialCell = None,
                 ):

        super().__init__(
            system=system,
            bonds=bonds,
            potential=potential,
        )
        #pylint: disable=invalid-name

        # (A,A) <- (A,A)
        iinvM = msnp.identity(self.num_atoms)

        # (B,A,A) = (1,A,A) * (B,1,A)
        self.Mii = msnp.broadcast_to(
            iinvM, (1,) + iinvM.shape) * self.inv_mass[:, None, :]

        self.BMatrix = GetShiftGrad(
            num_atoms=self.num_atoms,
            bonds=self.bonds,
            num_walkers=self.num_walker,
            dimension=self.dimension,
            use_pbc=self.use_pbc
        )
        # (B,C,A,D)
        shape = (self.num_walker,
                 self.bonds.shape[-2], self.num_atoms, self.dimension)

        self.broadcast = ops.BroadcastTo(shape)
        self.inv = ops.MatrixInverse(adjoint=False)
        self.squeeze = ops.Squeeze()
        self.einsum0 = ops.Einsum('ijk,ilkm->iljm')
        self.einsum1 = ops.Einsum('ijkl,imkl->ijm')
        self.einsum2 = ops.Einsum('ijkl,ikl->ij')
        self.einsum3 = ops.Einsum('ijk,ik->ij')
        self.einsum4 = ops.Einsum('ijkl,ij->ikl')
        self.einsum5 = ops.Einsum('ijk,ikl->ijl')

        # (B,C,A)
        shape = (self.num_walker, self.num_constraints, self.num_atoms)

        # (1,C,1)
        bond0 = self.bonds[..., 0].reshape(1, -1, 1).asnumpy()
        # (B,C,A) <- (B,A,1)
        mask0 = np.zeros(shape)
        np.put_along_axis(mask0, bond0, 1, axis=-1)
        # (B,C,A,1)
        self.mask0 = F.expand_dims(Tensor(mask0, ms.int32), -1)

        # (1,C,1)
        bond1 = self.bonds[..., 1].reshape(1, -1, 1).asnumpy()
        # (B,C,A) <- (B,A,1)
        mask1 = np.zeros(shape)
        np.put_along_axis(mask1, bond1, 1, axis=-1)
        # (B,C,A,1)
        self.mask1 = F.expand_dims(Tensor(mask1, ms.int32), -1)

    def construct(self,
                  coordinate: Tensor,
                  velocity: Tensor,
                  force: Tensor,
                  energy: Tensor,
                  kinetics: Tensor,
                  virial: Tensor = None,
                  pbc_box: Tensor = None,
                  step: int = 0,
                  ):
        """ Construct function of Lincs"""
        #pylint: disable=invalid-name

        # (B,A,D)
        coordinate_old = self._coordinate
        coordinate_new = coordinate

        # (B,C,A,D)
        BMatrix = self.BMatrix(coordinate_new, coordinate_old, pbc_box)

        # ijk,ilkm->iljm
        # (B,A,A),(B,C,A,D)->(B,C,A,D)
        # (B,1,A,A,1),(B,C,1,A,D)->(B,C,A,'A',D)->(B,C,A,D)
        tmp0 = self.einsum0((self.Mii, BMatrix))

        # ijkl,imkl->ijm
        # (B,C,A,D),(B,C,A,D)->(B,C,C)
        # (B,C,A,D),(B,A,C,D)->(B,C,A,1,D),(B,1,A,C,D)->(B,C,'A',C,'D')->(B,C,C)
        tmp1 = self.einsum1((BMatrix, tmp0))
        # (B,C,C)
        tmp2 = self.inv(tmp1)

        # (B,1,A,D) <- (B,A,D)
        pos_old = self.broadcast(F.expand_dims(coordinate_old, -3))
        # (B,C,D) <- (B,C,A,D) = (B,C,A,1) * (B,1,A,D)
        pos_old_0 = F.reduce_sum(self.mask0 * pos_old, -2)
        pos_old_1 = F.reduce_sum(self.mask1 * pos_old, -2)
        # (B,C)
        di = self.get_distance(pos_old_0, pos_old_1, pbc_box)

        # ijkl,ikl->ij
        # (B,C,A,D),(B,A,D)->(B,C)
        # (B,C,A,D),(B,1,A,D)->(B,C,A,D)->(B,C)
        tmp3 = self.einsum2((BMatrix, coordinate_new)) - di

        # ijk,ik->ij
        # (B,C,C),(B,C)->(B,C)
        # (B,C,C),(B,1,C)->(B,C,'C')->(B,C)
        tmp4 = self.einsum3((tmp2, tmp3))

        # ijkl,ij->ikl
        # (B,C,A,D),(B,C)->(B,A,D)
        # (B,A,C,D),(B,1,C,1)->(B,A,C,D)->(B,A,D)
        tmp5 = self.einsum4((BMatrix, tmp4))

        # ijk,ikl->ijl
        # (B,A,A),(B,A,D)->(B,A,D)
        # (B,A,A,1),(B,1,A,D)->(B,A,'A',D)->(B,A,D)
        dr = -self.einsum5((self.Mii, tmp5))
        coordinate = coordinate_new + dr

        # (B,A,D)
        velocity += dr / self.time_step
        # Constraint force = m * dR / dt^2
        # (B,A,1) * (B,A,D)
        constraint_force = self._atom_mass * dr / (self.time_step**2)
        force += constraint_force
        if self._pbc_box is not None:
            # (B,D) <- (B,A,D)
            virial += F.reduce_sum(-0.5 * coordinate * constraint_force, -2)

        return coordinate, velocity, force, energy, kinetics, virial, pbc_box
