# Copyright 2023 Huawei Technologies Co., Ltd
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
"""HLLC (Harten-Lax-van Leer-Contact) Riemann Solver"""
from mindspore import jit_class
from mindspore import numpy as mnp

from ...utils import cal_flux, cal_pri_var
from .base import RiemannComputer


@jit_class
class HLLC(RiemannComputer):
    r"""
    HLLC (Harten-Lax-van Leer-Contact) Riemann Solver based on Toro et al. 2009

    Args:
        material (Material): The information container of the fluid material.

    Supported Platforms:
        ``GPU``

    """

    def __init__(self, material, net_dict=None):
        self.minor = [
            [2, 3],
            [3, 1],
            [1, 2],
        ]
        super().__init__(material)

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

        # Step 1: pressure estimate
        rho_bar = 0.5 * (pri_var_left[0] + pri_var_right[0])
        sound_speed_mean = 0.5 * (sound_speed_left + sound_speed_right)
        pressure_pvrs = (
            0.5 * (pri_var_left[4] + pri_var_right[4])
            - 0.5 * (pri_var_left[axis + 1] - pri_var_right[axis + 1]) * rho_bar * sound_speed_mean
        )
        pressure_star = mnp.maximum(0.0, pressure_pvrs)

        # Step 2.1: left and right wave speed estimate
        gamma_ = (self.material.gamma + 1) * 0.5 / self.material.gamma
        q_left = 1.0 * (pressure_star <= pri_var_left[4]) + mnp.sqrt(
            1 + gamma_ * (pressure_star / pri_var_left[4] - 1)
        ) * (pressure_star > pri_var_left[4])
        q_right = 1.0 * (pressure_star <= pri_var_right[4]) + mnp.sqrt(
            1 + gamma_ * (pressure_star / pri_var_right[4] - 1)
        ) * (pressure_star > pri_var_right[4])
        wave_speed_left = pri_var_left[axis + 1] - sound_speed_left * q_left
        wave_speed_right = pri_var_right[axis + 1] + sound_speed_right * q_right
        wave_speed_left = mnp.minimum(wave_speed_left, 0.0)
        wave_speed_right = mnp.maximum(wave_speed_right, 0.0)

        # Step 2.2: wave speed estimate
        delta_u_left = wave_speed_left - pri_var_left[axis + 1]
        delta_u_right = wave_speed_right - pri_var_right[axis + 1]
        delta_rho_su = pri_var_left[0] * delta_u_left - pri_var_right[0] * delta_u_right
        wave_speed_star = (
            1.0
            / delta_rho_su
            * (
                pri_var_right[4]
                - pri_var_left[4]
                + pri_var_left[0] * pri_var_left[axis + 1] * delta_u_left
                - pri_var_right[0] * pri_var_right[axis + 1] * delta_u_right
            )
        )

        # Step 3: Compute the HLLC flux

        # Compute pre-factors for left and right states
        pre_factor_left = (
            (wave_speed_left - pri_var_left[axis + 1]) / (wave_speed_left - wave_speed_star) * pri_var_left[0]
        )
        pre_factor_right = (
            (wave_speed_right - pri_var_right[axis + 1]) / (wave_speed_right - wave_speed_star) * pri_var_right[0]
        )

        # Compute the star state for left and right states
        u_star_left = [
            pre_factor_left,
            pre_factor_left,
            pre_factor_left,
            pre_factor_left,
            pre_factor_left
            * (
                con_var_left[4] / con_var_left[0]
                + (wave_speed_star - pri_var_left[axis + 1])
                * (wave_speed_star + pri_var_left[4] / pri_var_left[0] / (wave_speed_left - pri_var_left[axis + 1]))
            ),
        ]
        u_star_left[axis + 1] *= wave_speed_star
        u_star_left[self.minor[axis][0]] *= pri_var_left[self.minor[axis][0]]
        u_star_left[self.minor[axis][1]] *= pri_var_left[self.minor[axis][1]]
        u_star_left = mnp.stack(u_star_left)

        u_star_right = [
            pre_factor_right,
            pre_factor_right,
            pre_factor_right,
            pre_factor_right,
            pre_factor_right
            * (
                con_var_right[4] / con_var_right[0]
                + (wave_speed_star - pri_var_right[axis + 1])
                * (wave_speed_star + pri_var_right[4] / pri_var_right[0] / (wave_speed_right - pri_var_right[axis + 1]))
            ),
        ]
        u_star_right[axis + 1] *= wave_speed_star
        u_star_right[self.minor[axis][0]] *= pri_var_right[self.minor[axis][0]]
        u_star_right[self.minor[axis][1]] *= pri_var_right[self.minor[axis][1]]
        u_star_right = mnp.stack(u_star_right)

        # Compute the flux at the star state for left and right states
        flux_star_left = flux_left + wave_speed_left * (u_star_left - con_var_left)
        flux_star_right = flux_right + wave_speed_right * (u_star_right - con_var_right)

        # Compute the final flux
        fluxes = (
            0.5 * (1 + mnp.sign(wave_speed_star)) * flux_star_left
            + 0.5 * (1 - mnp.sign(wave_speed_star)) * flux_star_right
        )

        return fluxes
