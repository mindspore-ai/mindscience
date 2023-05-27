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

from .integrator import Integrator
from ...system import Molecule
from ...function import get_arguments


class Brownian(Integrator):
    r"""A Brownian integrator module, which is a subclass of `Integrator`.

    Args:

        system (Molecule):              Simulation system

        temperature (float):            Simulation temperature T (K). Default: 300

        friction_coefficient (float):   Friction coefficient g (amu/ps). Default: 1e3


    Supported Platforms:

        ``Ascend`` ``GPU``

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
        """set thermostat algorithm for integrator"""
        if thermostat is not None:
            raise ValueError('The Brownian integrator cannot accept thermostat')
        return self

    def set_barostat(self, barostat: None = None):
        """set barostat algorithm for integrator"""
        if barostat is not None:
            raise ValueError('The Brownian integrator cannot accept barostat')
        return self

    def set_constraint(self, constraint: None = None, num_constraints: int = 0):
        """set constraint algorithm for integrator"""
        if constraint is not None:
            raise ValueError('The Brownian integrator cannot accept constraint')
        return self

    def set_time_step(self, dt: float):
        """set simulation time step"""
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

        coordinate += self.acc_unit_scale * force * self.inv_gamma * self.time_step
        coordinate += self.normal(coordinate.shape) * self.random_scale

        return coordinate, velocity, force, energy, kinetics, virial, pbc_box
