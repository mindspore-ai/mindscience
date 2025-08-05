# Copyright 2022 Huawei Technologies Co., Ltd
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
# ==============================================================================
"""Rusanov (Local Lax-Friedrichs) Riemann computer"""
from mindspore import numpy as mnp
from mindspore import jit_class

from ...utils import cal_flux, cal_pri_var
from .base import RiemannComputer


@jit_class
class Rusanov(RiemannComputer):
    r"""
    Rusanov (Local Lax-Friedrichs) Riemann computer

    Args:
        material (Material): The information container of the fluid material.

    Supported Platforms:
        ``GPU``

    """

    def __init__(self, material, net_dict=None):
        super(Rusanov, self).__init__(material)

    def compute_riemann_flux(self, con_var_left, con_var_right, axis):
        """
        Compute Riemann flux on face.

        Inputs:
            - **con_var_left** (Tensor) - Conservative variables on left side face.
            - **con_var_right** (Tensor) - Conservative variables on right side face.
            - **axis** (int) - 0, 1, 2 indicate x-dimension, y-dimension and z-dimension respectively.

        Outputs:
            Tensor, calculated riemann flux.
        """
        pri_var_left = cal_pri_var(con_var_left, self.material)
        pri_var_right = cal_pri_var(con_var_right, self.material)

        flux_left = cal_flux(con_var_left, pri_var_left, axis)
        flux_right = cal_flux(con_var_right, pri_var_right, axis)

        sound_speed_left = self.material.sound_speed(pri_var_left)
        sound_speed_right = self.material.sound_speed(pri_var_right)

        alpha = mnp.maximum(mnp.abs(pri_var_left[axis + 1]) + sound_speed_left,
                            mnp.abs(pri_var_right[axis + 1]) + sound_speed_right)

        flux = 0.5 * (flux_left + flux_right) - 0.5 * alpha * (con_var_right - con_var_left)

        return flux
