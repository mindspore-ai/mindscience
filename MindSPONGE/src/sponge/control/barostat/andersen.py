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

import mindspore.numpy as msnp
from mindspore import Tensor
from mindspore.ops import functional as F

from .barostat import Barostat, _barostat_register
from ...system import Molecule
from ...function import get_arguments


@_barostat_register('andersen')
class AndersenBarostat(Barostat):
    r"""A Andersen barostat module, which is a subclass of `Barostat`.

    Reference:

        Andersen, Hans Christian.
        Molecular dynamics simulations at constant pressure and/or temperature [J].
        Journal of Chemical Physics, 1980, 72: 2384-2393.

    Args:
        system (Molecule):          Simulation system

        pressure (float):           Reference pressure :math:`P_{ref}` in unit bar for pressure coupling.
                                    Default: 1

        anisotropic (bool):         Whether to perform anisotropic pressure control.
                                    Default: ``False``.

        control_step (int):         Step interval for controller execution. Default: 1

        compressibility (float):    Isothermal compressibility :math:`\beta` in unit bar^-1.
                                    Default: 4.6e-5

        time_constant (float)       Time constant :math:`\tau_p` in unit picosecond for pressure coupling.
                                    Default: 1

    Supported Platforms:
        ``Ascend`` ``GPU``

    """
    def __init__(self,
                 system: Molecule,
                 pressure: float = 1,
                 anisotropic: bool = False,
                 control_step: int = 1,
                 compressibility: float = 4.6e-5,
                 time_constant: float = 1.,
                 **kwargs,
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

        self.h_mass_inverse_0 = F.square(self.time_constant) / self.compressibility

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
            crd_scale_factor = 0
            # (B, D)
            pressure = self.get_pressure(kinetics, virial, pbc_box)
            volume0 = self.get_volume(pbc_box)

            dv_dt = F.reduce_sum(pressure - self.ref_press) / self.h_mass_inverse_0 * volume0
            volume = volume0 + dv_dt * self.time_step
            crd_scale_factor = msnp.cbrt(volume / volume0)
            coordinate *= crd_scale_factor
            pbc_box *= crd_scale_factor
            velocity *= msnp.reciprocal(crd_scale_factor)

        return coordinate, velocity, force, energy, kinetics, virial, pbc_box
