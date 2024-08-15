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
Leap-frog integrator
"""

from typing import Dict

from mindspore import Tensor

from .integrator import Integrator, _integrator_register
from ..thermostat import Thermostat
from ..barostat import Barostat
from ..constraint import Constraint
from ...system import Molecule
from ...function import get_arguments


@_integrator_register('leap_frog')
class LeapFrog(Integrator):
    r"""A leap-frog integrator based on "middle scheme" developed by Jian Liu, et al. It is a subclass of `Integrator`.

    Reference:

        Zhang, Z.; Yan, K; Liu, X.; Liu, J.
        A Leap-Frog Algorithm-based Efficient Unified Thermostat Scheme for Molecular Dynamics [J].
        Chinese Science Bulletin, 2018, 63(33): 3467-3483.

    Args:

        system (Molecule):          Simulation system

        thermostat (Thermostat):    Thermostat for temperature coupling. Default: None

        barostat (Barostat):        Barostat for pressure coupling. Default: None

        constraint (Constraint):    Constraint algorithm. Default: None


    Supported Platforms:

        ``Ascend`` ``GPU``

    """
    def __init__(self,
                 system: Molecule,
                 thermostat: Thermostat = None,
                 barostat: Barostat = None,
                 constraint: Constraint = None,
                 **kwargs
                 ):

        super().__init__(
            system=system,
            thermostat=thermostat,
            barostat=barostat,
            constraint=constraint,
        )
        self._kwargs = get_arguments(locals(), kwargs)

    def construct(self,
                  coordinate: Tensor,
                  velocity: Tensor,
                  force: Tensor,
                  energy: Tensor,
                  virial: Tensor = None,
                  pbc_box: Tensor = None,
                  step: int = 0,
                  **kwargs
                  ) -> Dict[str, Tensor]:

        # (B,A,D) = (B,A,D) * (B,A,1)
        acceleration = self.acc_unit_scale * force * self._inv_mass

        # v(t+0.5) = v(t-0.5) + a(t) * dt
        velocity_half = velocity + acceleration * self.time_step
        # (B,A,D) = (B,A,D) - (B,1,D)
        velocity_half -= self.get_com_velocity(velocity_half)

        # R(t+0.5) = R(t) + v(t+0.5) * dt
        coordinate_half = coordinate + velocity_half * self.time_step * 0.5

        variables = {'coordinate': coordinate_half,
                     'velocity': velocity_half,
                     'force': force,
                     'energy': energy,
                     'virial': virial,
                     'pbc_box': pbc_box,
                     }

        if self.thermostat is not None:
            # v'(t+0.5) = f_T[v(t+0.5)]
            variables = self.thermostat(**variables, step=step)

        # R(t+1) = R(t+0.5) + v'(t+0.5) * dt
        variables['coordinate'] = variables['coordinate'] + variables['velocity'] * self.time_step * 0.5

        if self.constraint is not None:
            for i in range(self.num_constraint_controller):
                variables = self.constraint[i](**variables, step=step)

        if self.barostat is not None:
            variables = self.barostat(**variables, step=step)

        return variables
