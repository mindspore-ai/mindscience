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
Velocity verlet integrator
"""

from typing import Tuple

import mindspore.numpy as msnp
from mindspore.ops import functional as F
from mindspore import Tensor, Parameter

from .integrator import Integrator, _integrator_register
from ..thermostat import Thermostat
from ..barostat import Barostat
from ..constraint import Constraint
from ...system import Molecule
from ...function import get_arguments


@_integrator_register('velocity_verlet')
class VelocityVerlet(Integrator):
    r"""
    A velocity verlet integrator based on "middle scheme" developed by Jian Liu, et al.
    It is a subclass of :class:`sponge.control.Integrator`.

    Reference Zhang, Z.; Liu, X.; Chen, Z.; Zheng, H.; Yan, K.; Liu, J.
    A Unified Thermostat Scheme for Efficient Configurational Sampling for
    Classical/Quantum Canonical Ensembles via Molecular Dynamics [J].
    The Journal of Chemical Physics, 2017, 147(3).

    Args:
        system ( :class:`sponge.system.Molecule`): Simulation system.
        thermostat ( :class:`sponge.control.Thermostat`): Thermostat for temperature coupling.
          Default: ``None``.
        barostat ( :class:`sponge.control.Barostat`): Barostat for pressure coupling.
          Default: ``None``.
        constraint ( :class:`sponge.control.Constraint`): Constraint algorithm.
          Default: ``None``.

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
        >>> from sponge.control import VelocityVerlet
        >>> system = Molecule(template='water.tip3p.yaml')
        >>> controller = VelocityVerlet(system)
    """

    def __init__(self,
                 system: Molecule,
                 thermostat: Thermostat = None,
                 barostat: Barostat = None,
                 constraint: Constraint = None,
                 **kwargs,
                 ):

        super().__init__(
            system=system,
            thermostat=thermostat,
            barostat=barostat,
            constraint=constraint,
        )
        self._kwargs = get_arguments(locals(), kwargs)

        # v(t+0.5) = v(t) + 0.5 * a(t) * dt
        velocity_half = msnp.zeros_like(self.system.coordinate)
        self.velocity_half = Parameter(velocity_half, name='velocity_half')

    def set_velocity_half(self, velocity_half: Tensor) -> bool:
        """set the veloctiy before half step"""
        return F.assign(self.velocity_half, velocity_half)

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

        acceleration = self.acc_unit_scale * force * self._inv_mass

        # if t > 0: v(t) = v(t-0.5) + 0.5 * a(t) * dt
        velocity = msnp.where(step > 0, self.velocity_half +
                              0.5 * acceleration * self.time_step, velocity)

        # v(t+0.5) = v(t) + 0.5 * a(t) * dt
        velocity_half = velocity + 0.5 * acceleration * self.time_step
        # (B,A,D) = (B,A,D) - (B,1,D)
        velocity_half -= self.get_com_velocity(velocity_half)
        kinetics = self.get_kinetics(velocity_half)

        # R(t+0.5) = R(t) + 0.5 * v(t+0.5) * dt
        coordinate_half = coordinate + velocity_half * self.time_step * 0.5

        if self.thermostat is not None:
            # v'(t) = f_T[v(t)]
            coordinate_half, velocity_half, force, energy, kinetics, virial, pbc_box = \
                self.thermostat(coordinate_half, velocity_half,
                                force, energy, kinetics, virial, pbc_box, step)

        # R(t+1) = R(t+0.5) + 0.5 * v'(t) * dt
        coordinate_new = coordinate_half + velocity_half * self.time_step * 0.5

        if self.constraint is not None:
            for i in range(self.num_constraint_controller):
                coordinate_new, velocity_half, force, energy, kinetics, virial, pbc_box = \
                    self.constraint[i](
                        coordinate_new, velocity_half, force, energy, kinetics, virial, pbc_box, step)

        if self.barostat is not None:
            coordinate_new, velocity_half, force, energy, kinetics, virial, pbc_box = \
                self.barostat(coordinate_new, velocity_half, force,
                              energy, kinetics, virial, pbc_box, step)

        velocity = F.depend(velocity, F.assign(self.velocity_half, velocity_half))

        return coordinate_new, velocity, force, energy, kinetics, virial, pbc_box
