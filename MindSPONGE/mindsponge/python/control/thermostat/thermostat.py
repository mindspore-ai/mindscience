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
Thermostat
"""

import mindspore as ms
from mindspore import Tensor
from mindspore.ops import functional as F

from .. import Controller
from ...system import Molecule
from ...function import functions as func


class Thermostat(Controller):
    r"""
    Thermostat controller for temperature coupling.

    Args:
        system (Molecule):      Simulation system.
        temperature (float):    Reference temperature T_ref (K) for temperature coupling.
                                Default: 300
        control_step (int):     Step interval for controller execution. Default: 1
        time_constant (float)   Time constant \tau_T (ps) for temperature coupling.
                                Default: 4

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
                 temperature: float = 300,
                 control_step: int = 1,
                 time_constant: float = 4.,
                 ):

        super().__init__(
            system=system,
            control_step=control_step,
        )

        self.boltzmann = self.units.boltzmann
        self.kinetic_unit_scale = self.units.kinetic_ref

        self.ref_temp = Tensor(temperature, ms.float32).reshape(-1, 1)
        self.ref_kinetics = 0.5 * self.degrees_of_freedom * self.boltzmann * self.ref_temp

        # \tau_t
        self.time_constant = Tensor(time_constant, ms.float32).reshape(-1, 1)
        if self.time_constant.shape[0] != self.num_walker and self.time_constant.shape[0] != 1:
            raise ValueError(
                'The first shape of time_constant must equal to 1 or num_walker')

    @property
    def temperature(self):
        """reference temperature."""
        return self.ref_temp

    @property
    def kinetics(self):
        """reference kinetics"""
        return self.ref_kinetics

    def set_degrees_of_freedom(self, dofs: int):
        """
        set degrees of freedom (DOFs).

        Args:
            dofs (int): Degrees of freedom.
        """
        self.degrees_of_freedom = dofs
        self.ref_kinetics = 0.5 * self.degrees_of_freedom * self.boltzmann * self.ref_temp
        return self

    def velocity_scale(self, sim_kinetics: Tensor, ref_kinetics: Tensor, ratio: float = 1) -> Tensor:
        r"""
        calculate the velocity scale factor for temperature coupling.

        Args:
            sim_kinetics (Tensor):  Tensor of simulation kinetics.
            ref_kinetics (Tensor):  Tensor of reference kinetics.
            ratio (float):          The degree of change lambda\_.

        Returns:
            Tensor, teh velocity scale factor.
        """
        sim_kinetics = func.keepdim_sum(sim_kinetics, -1)
        lambda_ = 1. + ratio * (ref_kinetics / sim_kinetics - 1)
        return F.sqrt(lambda_)

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

        r"""
        Control the temperature of the simulation system.

        Args:
            coordinate (Tensor):    Tensor of shape (B, A, D). Data type is float.
            velocity (Tensor):      Tensor of shape (B, A, D). Data type is float.
            force (Tensor):         Tensor of shape (B, A, D). Data type is float.
            energy (Tensor):        Tensor of shape (B, 1). Data type is float.
            kinetics (Tensor):      Tensor of shape (B, D). Data type is float.
            virial (Tensor):        Tensor of shape (B, D). Data type is float.
            pbc_box (Tensor):       Tensor of shape (B, D). Data type is float.
            step (int):             Simulation step. Default: 0

        Returns:
            coordinate (Tensor), Tensor of shape (B, A, D). Data type is float.
            velocity (Tensor), Tensor of shape (B, A, D). Data type is float.
            force (Tensor), Tensor of shape (B, A, D). Data type is float.
            energy (Tensor), Tensor of shape (B, 1). Data type is float.
            kinetics (Tensor), Tensor of shape (B, D). Data type is float.
            virial (Tensor), Tensor of shape (B, D). Data type is float.
            pbc_box (Tensor), Tensor of shape (B, D). Data type is float.

        Symbols:
            B:  Number of walkers in simulation.
            A:  Number of atoms.
            D:  Dimension of the simulation system. Usually is 3.
        """

        raise NotImplementedError
