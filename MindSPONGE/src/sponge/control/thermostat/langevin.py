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
"""Langevin thermostat"""

from typing import Tuple

import mindspore.numpy as msnp
from mindspore import Tensor
from mindspore import ops
from mindspore.ops import functional as F

from .thermostat import Thermostat
from ...system import Molecule
from ...function import get_arguments


class Langevin(Thermostat):
    r"""A Langevin thermostat module, which is a subclass of `Thermostat`.

    Reference:

        Goga, N.; Rzepiela, A. J.; de Vries, A. H.; Marrink, S. J.; Berendsen, H. J. C..
        Efficient Algorithms for Langevin and DPD Dynamics [J].
        Journal of Chemical Theory and Computation, 2012, 8(10): 3637-3649.

    Args:

        system (Molecule):      Simulation system

        temperature (float):    Reference temperature :math:`T_{ref}` in unit Kelvin for temperature coupling.
                                Default: 300

        control_step (int):     Step interval for controller execution. Default: 1

        time_constant (float)   Time constant :math:`\tau_T` in unit picosecond for temperature coupling.
                                Default: 0.2

        seed (int):             Random seed for standard normal. Default: 0

        seed2 (int):            Random seed2 for standard normal. Default: 0


    Supported Platforms:

        ``Ascend`` ``GPU``

    """

    def __init__(self,
                 system: Molecule,
                 temperature: float = 300,
                 control_step: int = 1,
                 time_constant: float = 0.1,
                 seed: int = 0,
                 seed2: int = 0,
                 **kwargs,
                 ):

        super().__init__(
            system=system,
            temperature=temperature,
            control_step=control_step,
            time_constant=time_constant,
        )
        self._kwargs = get_arguments(locals(), kwargs)

        # (B,A,1)
        self._inv_sqrt_mass = F.sqrt(self._inv_mass)

        # (B,1,1)
        # \gamma = 1.0 / \tau_t
        self.effective_friction_rate = msnp.reciprocal(self.time_constant)
        # \f = 1 - exp(-\gamma * dt)
        self.friction = 1.0 - \
            msnp.exp(-self.effective_friction_rate*self.time_step)
        # k = \sqrt(f * (2 - f) * k_B * T)
        self.random_scale = F.sqrt(self.friction * (2 - self.friction) * self.boltzmann *
                                   self.ref_temp / self.kinetic_unit_scale)

        self.standard_normal = ops.StandardNormal(seed, seed2)

    def set_time_step(self, dt):
        """set simulation time step"""
        self.time_step = dt
        # \f = 1 - exp(-\gamma * dt)
        self.friction = 1.0 - \
            msnp.exp(-self.effective_friction_rate*self.time_step)
        # k = \sqrt(f * (2 - f) * k_B * T)
        self.random_scale = F.sqrt(self.friction * (2 - self.friction) * self.boltzmann *
                                   self.ref_temp / self.kinetic_unit_scale)
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

        if self.control_step == 1 or step % self.control_step == 0:
            velocity += -self.friction * velocity + self.random_scale * \
                self._inv_sqrt_mass * self.standard_normal(velocity.shape)

        return coordinate, velocity, force, energy, kinetics, virial, pbc_box
