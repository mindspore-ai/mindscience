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
"""Berendsen thermostat"""

from typing import Tuple

from mindspore import Tensor
from mindspore import ops

from .thermostat import Thermostat, _thermostat_register
from ...system import Molecule
from ...function import get_arguments


@_thermostat_register('berendsen')
class BerendsenThermostat(Thermostat):
    r"""
    A Berendsen (weak coupling) thermostat module, which is a subclass of :class:`sponge.control.Thermostat`.

    Reference Berendsen, H. J. C.; Postma, J. P. M.; van Gunsteren, W. F.; DiNola, A.; Haak, J. R..
    Molecular Dynamics with Coupling to an External Bath [J].
    The Journal of Chemical Physics, 1984, 81(8).

    Args:
        system (:class: `sponge.system.Molecule`): Simulation system.
        temperature (float, optional): Reference temperature :math:`T_{ref}`
          in unit Kelvin for temperature coupling.
          Default: ``300.0``.
        control_step (int, optional): Step interval for controller execution.
          Default: ``1``.
        time_constant (float, optional): Time constant :math:`\tau_T`
          in unit picosecond for temperature coupling.
          Default: ``0.5``.
        scale_min (float, optional): The minimum value to clip the velocity scale factor.
          Default: ``0.8``.
        scale_max (float, optional): The maximum value to clip the velocity scale factor.
          Default: ``1.25``.

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
        >>> from sponge.control import BerendsenThermostat
        >>> system = Molecule(template='water.tip3p.yaml')
        >>> controller = BerendsenThermostat(system)
    """
    def __init__(self,
                 system: Molecule,
                 temperature: float = 300,
                 control_step: int = 1,
                 time_constant: float = 0.5,
                 scale_min: float = 0.8,
                 scale_max: float = 1.25,
                 **kwargs,
                 ):

        super().__init__(
            system=system,
            temperature=temperature,
            control_step=control_step,
            time_constant=time_constant,
        )
        self._kwargs = get_arguments(locals(), kwargs)

        self.scale_min = scale_min
        self.scale_max = scale_max

        self.ratio = self.control_step * self.time_step / self.time_constant

    def set_time_step(self, dt):
        r"""
        Set simulation time step

        Args:
            dt (float): Simulation time step
        """
        self.time_step = dt
        self.ratio = self.control_step * self.time_step / self.time_constant
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
        Control the temperature of the simulation system using Berendsen (weak coupling) thermostat.

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
        if self.control_step == 1 or step % self.control_step == 0:
            scale = self.velocity_scale(kinetics, self.get_ref_kinetics(), self.ratio)
            scale = ops.clip_by_value(scale, self.scale_min, self.scale_max)
            velocity *= scale

        return coordinate, velocity, force, energy, kinetics, virial, pbc_box
