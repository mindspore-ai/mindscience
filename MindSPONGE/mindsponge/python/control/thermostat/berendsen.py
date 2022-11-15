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
"""Berendsen thermostat"""

from mindspore import Tensor
from mindspore import ops

from . import Thermostat
from ...system import Molecule


class BerendsenThermostat(Thermostat):
    r"""
    A Berendsen (weak coupling) thermostat controller.

    Reference:
        `Berendsen, H. J. C.; Postma, J. P. M.; van Gunsteren, W. F.; DiNola, A.; Haak, J. R..
        Molecular Dynamics with Coupling to an External Bath [J].
        The Journal of Chemical Physics, 1984, 81(8): 3684.
        <https://pure.rug.nl/ws/portalfiles/portal/64380902/1.448118>`_.

    Args:
        system (Molecule):      Simulation system.
        temperature (float):    Reference temperature T_ref (K) for temperature coupling.
                                Default: 300
        control_step (int):     Step interval for controller execution. Default: 1
        time_constant (float)   Time constant \tau_T (ps) for temperature coupling.
                                Default: 4
        scale_min (float):      The minimum value to clip the velocity scale factor. Default: 0.8
        scale_max (float):      The maximum value to clip the velocity scale factor. Default: 1.25

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
                 time_constant: float = 4,
                 scale_min: float = 0.8,
                 scale_max: float = 1.25,
                 ):

        super().__init__(
            system=system,
            temperature=temperature,
            control_step=control_step,
            time_constant=time_constant,
        )

        self.scale_min = scale_min
        self.scale_max = scale_max

        self.ratio = self.control_step * self.time_step / self.time_constant

    def set_time_step(self, dt):
        """
        set simulation time step.

        Args:
            dt (float): Time of a time step.
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
                  ):

        if self.control_step == 1 or step % self.control_step == 0:
            scale = self.velocity_scale(kinetics, self.ref_kinetics, self.ratio)
            scale = ops.clip_by_value(scale, self.scale_min, self.scale_max)
            velocity *= scale

        return coordinate, velocity, force, energy, kinetics, virial, pbc_box
