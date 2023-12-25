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
    r"""A Berendsen (weak coupling) barostat module, which is a subclass of `Barostat`.

    Reference:

        Berendsen, H. J. C.; Postma, J. P. M.; van Gunsteren, W. F.; DiNola, A.; Haak, J. R.
        Molecular Dynamics with Coupling to an External Bath [J].
        The Journal of Chemical Physics, 1984, 81(8): 3684.

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
        """set simulation time step"""
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
