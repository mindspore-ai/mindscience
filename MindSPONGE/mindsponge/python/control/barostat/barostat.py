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
Barostat
"""

from typing import Tuple

import mindspore as ms
import mindspore.numpy as msnp
from mindspore import Tensor, Parameter
from mindspore.ops import functional as F

from .. import Controller
from ...system import Molecule
from ...function import get_ms_array, get_arguments


class Barostat(Controller):
    r"""Base class for barostat module in MindSPONGE, which is a subclass of `Controller`.

        The `Barostat` module is used for pressure coupling. It controls the atomic coordinates and the size of
        the PBC box of the system during the simulation process.

    Args:

        system (Molecule):          Simulation system

        pressure (float):           Reference pressure :math:`P_{ref}` in unit bar for pressure coupling.
                                    Default: 1

        anisotropic (bool):         Whether to perform anisotropic pressure control.
                                    Default: False

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
            control_step=control_step,
        )
        self._kwargs = get_arguments(locals(), kwargs)

        self.anisotropic = anisotropic

        self.sens = Tensor(1e8, ms.float32)
        self.inv_sens = msnp.reciprocal(self.sens)

        #(B,1)
        self.ref_press = get_ms_array(pressure, ms.float32).reshape(-1, 1)
        if self.ref_press.shape[0] != 1 and self.ref_press.shape[0] != self.num_walker:
            raise ValueError(f'The first dimension of "pressure" ({self.ref_press.shape[0]})'
                             f'does not match the number of multiple walkers ({self.num_walker})!')

        # isothermal compressibility
        self.beta = get_ms_array(compressibility, ms.float32)

        # \tau_t
        self.time_constant = get_ms_array(time_constant, ms.float32).reshape(-1, 1)
        if self.time_constant.shape[0] != self.num_walker and self.time_constant.shape[0] != 1:
            raise ValueError('The first shape of self.time_constant must equal to 1 or num_walker')

        self.shape = (self.num_walker, self.dimension)
        self.change_accumulation = Parameter(msnp.zeros(self.shape), name='change_accumulation', requires_grad=False)

        self.critical_change = 1e-6

    @property
    def pressure(self):
        """reference pressure"""
        return self.ref_press

    @property
    def compressibility(self):
        """isothermal compressibility"""
        return self.beta

    def pressure_scale(self, sim_press: Tensor, ref_press: Tensor, ratio: float = 1) -> Tensor:
        """calculate the coordinate scale factor for pressure coupling"""
        delta_p = ref_press - sim_press
        change = - ratio * self.beta * delta_p

        # If the change is too small, the float32 data will not be able to represent the scale.
        # Therefore, the small changes will be accumulated:
        # (1 + x) ^ n \approx 1 + nx, when x << 1
        # When the total change accumulates to a critical value, then the coordinate and PBC box will be scaled.
        change += self.change_accumulation
        mask = msnp.abs(change) > self.critical_change
        scale = msnp.where(mask, 1+change, 1.)
        change = msnp.where(mask, 0., change)
        scale = F.depend(scale, F.assign(self.change_accumulation, change))

        return scale

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
        r"""Control the pressure of the simulation system.

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

        Symbols:
            B:  Number of walkers in simulation.
            A:  Number of atoms.
            D:  Spatial dimension of the simulation system. Usually is 3.

        """

        raise NotImplementedError
