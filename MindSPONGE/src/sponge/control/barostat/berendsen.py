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
"""Berendsen barostat"""

from typing import Tuple

import mindspore as ms
import mindspore.numpy as msnp
from mindspore import Tensor
from mindspore.ops import functional as F

from .barostat import Barostat, _barostat_register
from ...system import Molecule
from ...function import get_arguments


@_barostat_register('berendsen')
class BerendsenBarostat(Barostat):
    r"""
    A Berendsen (weak coupling) barostat module,
    which is a subclass of :class:`sponge.control.Barostat`.

    Reference Berendsen, H. J. C.; Postma, J. P. M.; van Gunsteren, W. F.; DiNola, A.; Haak, J. R.
    Molecular Dynamics with Coupling to an External Bath [J].
    The Journal of Chemical Physics, 1984, 81(8).

    Args:
        system (:class:`sponge.system.Molecule`): Simulation system.
        pressure (float, optional): Reference pressure :math:`P_{ref}`
            in unit :math:`bar` for pressure coupling.
            Default: ``1.0``.
        anisotropic (bool, optional): Whether to perform anisotropic pressure control.
            Default: ``False``.
        control_step (int, optional): Step interval for controller execution. Default: ``1``.
        compressibility (float, optional): Isothermal compressibility :math:`\beta`
            in unit :math:`bar^{-1}`.
            Default: ``4.6e-5``.
        time_constant (float, optional): Time constant :math:`\tau_p` in
            unit picosecond for pressure coupling.
            Default: ``1``.

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
        >>> from sponge.control import BerendsenBarostat
        >>> system = Molecule(template='water.tip3p.yaml')
        >>> controller = BerendsenBarostat(system)
    """
    def __init__(self,
                 system: Molecule,
                 pressure: float = 1,
                 anisotropic: bool = False,
                 control_step: int = 1,
                 compressibility: float = 4.6e-5,
                 time_constant: float = 1.,
                 **kwargs
                 ):

        super().__init__(
            system=system,
            pressure=pressure,
            anisotropic=anisotropic,
            control_step=control_step,
            compressibility=compressibility,
            time_constant=time_constant,
        )
        self._kwargs = get_arguments(locals(), kwargs)

        self.ratio = self.control_step * self.time_step / self.time_constant / 3.

    def set_time_step(self, dt: float):
        r"""
        Set simulation time step.

        Args:
            dt (float): Simulation time step.

        Returns:
            :class:`sponge.control.BerendsenBarostat`, current object of :class:`sponge.control.BerendsenBarostat`.
        """
        self.time_step = Tensor(dt, ms.float32)
        self.ratio = self.control_step * self.time_step / self.time_constant / 3.
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
        Control the pressure of the simulation system.

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
            # (B, D)
            pressure = self.get_pressure(kinetics, virial, pbc_box)
            if not self.anisotropic:
                # (B, 1) <- (B, D)
                pressure = msnp.mean(pressure, axis=-1, keepdims=True)
                # (B, D) <- (B, 1)
                pressure = msnp.broadcast_to(pressure, self.shape)

            # (B, D)
            scale = self.pressure_scale(pressure, self.ref_press, self.ratio)

            # (B, A, D) * (B, 1, D)
            coordinate *= F.expand_dims(scale, -2)
            # (B, D)
            pbc_box *= scale

        return coordinate, velocity, force, energy, kinetics, virial, pbc_box
