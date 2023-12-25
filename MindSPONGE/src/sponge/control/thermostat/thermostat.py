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
Thermostat
"""

from typing import Union, Tuple, List
from numpy import ndarray

from mindspore import Tensor, Parameter
from mindspore.ops import functional as F

from .. import Controller
from ...system import Molecule
from ...function import get_arguments

_THERMOSTAT_BY_KEY = dict()


def _thermostat_register(*aliases):
    """Return the alias register."""
    def alias_reg(cls):
        name = cls.__name__
        name = name.lower()
        if name not in _THERMOSTAT_BY_KEY:
            _THERMOSTAT_BY_KEY[name] = cls

        for alias in aliases:
            if alias not in _THERMOSTAT_BY_KEY:
                _THERMOSTAT_BY_KEY[alias] = cls

        return cls

    return alias_reg


class Thermostat(Controller):
    r"""Base class for thermostat module in MindSPONGE, which is a subclass of `Controller`.

        The `Thermostat` module is used for temperature coupling. It controls the atomic velocities and the kinetics
        of the system during the simulation process.

    Args:
        system (Molecule): Simulation system

        temperature (Union[float, ndarray, Tensor]): Reference temperature :math:`T_{ref}` in unit Kelvin
            for temperature coupling. Default: 300

        control_step (int): Step interval for controller execution. Default: 1

        time_constant (float): Time constant :math:`\tau_T` in unit picosecond for temperature coupling.
            Default: 0.5

    Supported Platforms:
        ``Ascend`` ``GPU``

    """
    def __init__(self,
                 system: Molecule,
                 temperature: Union[float, ndarray, Tensor, List[float]] = 300,
                 control_step: int = 1,
                 time_constant: float = 0.5,
                 **kwargs,
                 ):

        super().__init__(
            system=system,
            control_step=control_step,
        )
        self._kwargs = get_arguments(locals(), kwargs)

        temperature = self._get_mw_tensor(temperature, 'temperature')
        self.ref_temp = Parameter(temperature, name='ref_temp', requires_grad=False)

        # \tau_t
        self.time_constant = self._get_mw_tensor(time_constant, 'time_constant')

    @property
    def temperature(self) -> Tensor:
        """reference temperature"""
        return self.identity(self.ref_temp)

    @property
    def ref_kinetics(self) -> Tensor:
        """reference kinetics"""
        return self.get_ref_kinetics()

    def get_ref_kinetics(self) -> Tensor:
        """get reference kinetics"""
        return 0.5 * self.degrees_of_freedom * self.boltzmann * self.ref_temp

    def set_temperature(self, temperature: Union[float, ndarray, Tensor, List[float]]) -> Tensor:
        r"""set the value of reference temperature.
            The size of the temperature array must be equal to current temperature.
        """
        return F.assign(self.ref_temp, self._get_mw_tensor(temperature, 'temperature'))

    def reconstruct_temperature(self, temperature: Union[float, ndarray, Tensor, List[float]]):
        r"""reset the reference temperature"""
        temperature = self._get_mw_tensor(temperature, 'temperature')
        self.ref_temp = Parameter(temperature, name='ref_temp', requires_grad=False)
        return self

    def set_degrees_of_freedom(self, dofs: int):
        """set degrees of freedom (DOFs)"""
        self.degrees_of_freedom = dofs
        return self

    def velocity_scale(self, sim_kinetics: Tensor, ref_kinetics: Tensor, ratio: float = 1) -> Tensor:
        """calculate the velocity scale factor for temperature coupling"""
        sim_kinetics = self.keepdims_sum(sim_kinetics, -1)
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
                  ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        r"""Control the temperature of the simulation system

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

        Note:
            B:  Number of walkers in simulation.
            A:  Number of atoms.
            D:  Spatial dimension of the simulation system. Usually is 3.

        """

        raise NotImplementedError
