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
Constraint
"""

from typing import Union, Tuple

import numpy as np
import mindspore as ms
from mindspore import Tensor, Parameter

from .. import Controller
from ...system import Molecule
from ...potential import PotentialCell
from ...function.operations import GetVector, GetDistance
from ...function import get_arguments


class Constraint(Controller):
    r"""Base class for constraint module in MindSPONGE, which is a subclass of `Controller`.

        The `Constraint` module is used for bond constraint. It controls the atomic coordinates, the atomic velocities,
        the atomic forces and the virial of the system during the simulation process.

    Args:

        system (Molecule):          Simulation system.

        bonds (Union[Tensor, str]): Bonds to be constraint.
                                    Tensor of shape (K, 2). Data type is int.
                                    Alternative: "h-bonds" or "all-bonds".

        potential (PotentialCell):  Potential Cell. Default: None

    Supported Platforms:

        ``Ascend`` ``GPU``

    """

    def __init__(self,
                 system: Molecule,
                 bonds: Union[Tensor, str] = 'h-bonds',
                 potential: PotentialCell = None,
                 **kwargs,
                 ):

        super().__init__(
            system=system,
            control_step=1,
        )
        self._kwargs = get_arguments(locals(), kwargs)

        if potential is None:
            self.all_bonds = system.bond
            self.h_bonds = system.hydrogen_bond
        else:
            self.all_bonds = potential.bond
            self.h_bonds = potential.hydrogen_bond

        if isinstance(bonds, (Tensor, Parameter, np.ndarray)):
            self.bonds = Tensor(bonds, ms.int32)
        elif isinstance(bonds, str):
            if bonds.lower() == 'h-bonds':
                self.bonds = self.h_bonds
            elif bonds.lower() == 'all-bonds':
                self.bonds = self.all_bonds
            else:
                raise ValueError(f'"bonds" must be "h-bonds" or "all-bonds" but got: {bonds}')
        else:
            raise TypeError(f'The type of "bonds" must be Tensor or str, but got: {type(bonds)}')

        if self.bonds.ndim != 2:
            if self.bonds.ndim != 3:
                raise ValueError(f'The rank of "bonds" must be 2 or 3 but got: {self.bonds.ndim}')

            if self.bonds.shape[0] != 1:
                raise ValueError(f'For constraint, the batch size of "bonds" must be 1 but got: {self.bonds[0]}')
            self.bonds = self.bonds[0]

        if self.bonds.shape[-1] != 2:
            raise ValueError(f'The last dimension of "bonds" but got: {self.bonds.shape[-1]}')

        # C
        self.num_constraints = self.bonds.shape[-2]

        self.use_pbc = self._pbc_box is not None

        self.get_vector = GetVector(self.use_pbc)
        self.get_distance = GetDistance(use_pbc=self.use_pbc)

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
        r""" constraint the bonds.

        Args:
            coordinate (Tensor):    Tensor of shape `(B, A, D)`. Data type is float.
            velocity (Tensor):      Tensor of shape `(B, A, D)`. Data type is float.
            force (Tensor):         Tensor of shape `(B, A, D)`. Data type is float.
            energy (Tensor):        Tensor of shape `(B, 1)`. Data type is float.
            kinetics (Tensor):      Tensor of shape `(B, D)`. Data type is float.
            virial (Tensor):        Tensor of shape `(B, D)`. Data type is float.
            pbc_box (Tensor):       Tensor of shape `(B, D)`. Data type is float.
            step (int):             Simulation step. Default: 0

        Returns:
            coordinate (Tensor):    Tensor of shape `(B, A, D)`. Data type is float.
            velocity (Tensor):      Tensor of shape `(B, A, D)`. Data type is float.
            force (Tensor):         Tensor of shape `(B, A, D)`. Data type is float.
            energy (Tensor):        Tensor of shape `(B, 1)`. Data type is float.
            kinetics (Tensor):      Tensor of shape `(B, D)`. Data type is float.
            virial (Tensor):        Tensor of shape `(B, D)`. Data type is float.
            pbc_box (Tensor):       Tensor of shape `(B, D)`. Data type is float.

        Symbols:
            B:  Number of walkers in simulation.
            A:  Number of atoms.
            D:  Spatial dimension of the simulation system. Usually is 3.

        """

        raise NotImplementedError
