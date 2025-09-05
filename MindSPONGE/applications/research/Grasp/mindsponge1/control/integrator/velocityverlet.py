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
Velocity verlet integrator
"""

import mindspore.numpy as msnp
from mindspore.ops import functional as F
from mindspore import Tensor, Parameter

from .integrator import Integrator
from ..thermostat import Thermostat
from ..barostat import Barostat
from ..constraint import Constraint
from ...system import Molecule


class VelocityVerlet(Integrator):
    r"""
    A velocity verlet integrator based on "middle scheme" developed by Jian Liu, et al.

    Reference:
        `Zhang, Z.; Liu, X.; Chen, Z.; Zheng, H.; Yan, K.; Liu, J.
        A Unified Thermostat Scheme for Efficient Configurational Sampling for
        Classical/Quantum Canonical Ensembles via Molecular Dynamics [J].
        The Journal of Chemical Physics, 2017, 147(3): 034109.
        <https://aip.scitation.org/doi/abs/10.1063/1.4991621>`_.

    Args:
        system (Molecule):          Simulation system.
        thermostat (Thermostat):    Thermostat for temperature coupling. Default: None
        barostat (Barostat):        Barostat for pressure coupling. Default: None
        constraint (Constraint):    Constraint algorithm. Default: None

    Returns:
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
                 thermostat: Thermostat = None,
                 barostat: Barostat = None,
                 constraint: Constraint = None,
                 ):

        super().__init__(
            system=system,
            thermostat=thermostat,
            barostat=barostat,
            constraint=constraint,
        )

        # v(t+0.5) = v(t) + 0.5 * a(t) * dt
        velocity_half = msnp.zeros_like(self.system.coordinate)
        self.velocity_half = Parameter(velocity_half, name='velocity_half')

    def set_velocity_half(self, velocity_half: Tensor, success: bool = True) -> bool:
        """
        set the veloctiy before half step.

        Args:
            velocity_half (Tensor): Tensor of velocity before half step.
            success (Tensor):       Whether the velocity has been set successfully.
        """
        return F.depend(success, F.assign(self.velocity_half, velocity_half))

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

        acceleration = self.acc_unit_scale * force * self._inv_mass

        # if t > 0: v(t) = v(t-0.5) + 0.5 * a(t) * dt
        velocity = msnp.where(step > 0, self.velocity_half +
                              0.5 * acceleration * self.time_step, velocity)
        # (B,A,D) = (B,A,D) - (B,1,D)
        velocity -= self.get_com_velocity(velocity)

        # v(t+0.5) = v(t) + 0.5 * a(t) * dt
        velocity_half = velocity + 0.5 * acceleration * self.time_step

        # R(t+0.5) = R(t) + 0.5 * v(t+0.5) * dt
        coordinate_half = coordinate + velocity_half * self.time_step * 0.5

        if self.thermostat is not None:
            # v'(t) = f_T[v(t)]
            kinetics = self.get_kinetics(velocity_half)
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

        F.depend(True, F.assign(self.velocity_half, velocity_half))

        kinetics = self.get_kinetics(velocity)

        return coordinate_new, velocity, force, energy, kinetics, virial, pbc_box
