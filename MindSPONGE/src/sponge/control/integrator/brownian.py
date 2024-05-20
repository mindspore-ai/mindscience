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
Brownian integrator
"""

from typing import Tuple

import mindspore as ms
import mindspore.numpy as msnp
from mindspore import Tensor
from mindspore import ops
from mindspore.ops import functional as F

from .integrator import Integrator, _integrator_register
from ...system import Molecule
from ...function import get_arguments


@_integrator_register('brownian')
class Brownian(Integrator):
    r"""
    A Brownian integrator module, which is a subclass of :class:`sponge.control.Integrator`.

    Args:
        system (:class:`sponge.system.Molecule`): Simulation system
        temperature (float, optional): Simulation temperature T (K). Default: ``300.0``.
        friction_coefficient (float, optional): Friction coefficient g (amu/ps). Default: ``1e3``.

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
        >>> from sponge.control import Brownian
        >>> system = Molecule(template='water.tip3p.yaml')
        >>> controller = Brownian(system)
    """
    def __init__(self,
                 system: Molecule,
                 temperature: float = 300,
                 friction_coefficient: float = 1e3,
                 **kwargs,
                 ):

        super().__init__(
            system=system,
            thermostat=None,
            barostat=None,
            constraint=None,
        )
        self._kwargs = get_arguments(locals(), kwargs)

        self.ref_temp = Tensor(temperature, ms.float32)

        self.inv_sqrt_mass = F.sqrt(self._inv_mass)

        self.friction_coefficient = Tensor(friction_coefficient, ms.float32)
        # \gamma = 1.0 / \tau_t
        self.inv_gamma = msnp.reciprocal(self.friction_coefficient) * self._inv_mass

        # k = \sqrt(2 * k_B * T * dt / \gamma)
        self.random_scale = F.sqrt(2 * self.boltzmann * self.ref_temp * self.time_step
                                   * self.inv_gamma / self.kinetic_unit_scale)

        self.normal = ops.StandardNormal()

    @property
    def temperature(self) -> Tensor:
        return self.ref_temp

    def set_thermostat(self, thermostat: None = None):
        r"""
        Set thermostat algorithm for integrator.

        Args:
            thermostat (None): Thermostat algorithm,
              which needs to be ``None`` for Brownian integrator. Default: ``None``.
        """
        if thermostat is not None:
            raise ValueError('The Brownian integrator cannot accept thermostat')
        return self

    def set_barostat(self, barostat: None = None):
        r"""
        Set barostat algorithm for integrator.

        Args:
            barostat (None): Barostat algorithm,
              which needs to be ``None`` for Brownian integrator. Default: ``None``.
        """
        if barostat is not None:
            raise ValueError('The Brownian integrator cannot accept barostat')
        return self

    def set_constraint(self, constraint: None = None, num_constraints: int = 0):
        r"""
        Set constraint algorithm for integrator.

        Args:
            constraint (None): Constraint algorithm,
              which needs to be ``None`` for Brownian integrator. Default: ``None``.
            num_constraints (int, optional): Number of constraints. Default: ``0.0``.
        """
        if constraint is not None:
            raise ValueError('The Brownian integrator cannot accept constraint')
        return self

    def set_time_step(self, dt: float):
        r"""
        Set simulation time step.

        Args:
            dt (float): Simulation time step.
        """
        self.time_step = Tensor(dt, ms.float32)
        self.random_scale = F.sqrt(2 * self.boltzmann * self.ref_temp * self.time_step * self.inv_gamma)
        return self

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
        Update simulation step.

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

        coordinate += self.acc_unit_scale * force * self.inv_gamma * self.time_step
        coordinate += self.normal(coordinate.shape) * self.random_scale

        return coordinate, velocity, force, energy, kinetics, virial, pbc_box
