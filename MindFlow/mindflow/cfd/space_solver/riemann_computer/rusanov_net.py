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
"""Rusanov Riemann computer with network."""
from mindspore import numpy as mnp
from mindspore import jit_class, ops

from ...utils import cal_flux, cal_pri_var
from .base import RiemannComputer


@jit_class
class RusanovNet(RiemannComputer):
    r"""
    Rusanov Riemann computer with network. The network is inspired by Rusanov_Net from paper "JAX-FLUIDS: A
    fully-differentiable high-order computational fluid dynamics solver for compressible two-phase flows"
    https://arxiv.org/pdf/2203.13760.pdf

    Args:
        material (Material): The information container of the fluid material.
        net_dict (dict): The dict that contains the networks to be trained.

    Inputs:
        con_var_left (Tensor): Conservative variables on left side face.
        con_var_right (Tensor): Conservative variables on right side face.
        axis (int): 0, 1, 2 indicate x-dimension, y-dimension and z-dimension respectively.

    Outputs:
        Tensor, calculated riemann flux.

    Raises:
        ValueError: If `net_dict` do not have 'rusanov_net'.

    Supported Platforms:
        ``GPU``
    """

    def __init__(self, material, net_dict):
        super(RusanovNet, self).__init__(material)
        self.transpose = ops.Transpose()
        self.net = net_dict.get('rusanov_net')
        if self.net is None:
            raise ValueError('Can not find rusanov net.')
        self.concat = ops.Concat(axis=0)

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
        mean_sound_speed = 0.5 * (sound_speed_left + sound_speed_right)

        delta_vel = mnp.abs(pri_var_right[axis + 1] - pri_var_left[axis + 1])
        mean_vel = 0.5 * (mnp.abs(pri_var_right[axis + 1]) + mnp.abs(pri_var_left[axis + 1]))
        delta_sound_speed = mnp.abs(sound_speed_left - sound_speed_right)

        var = mnp.stack([delta_vel, mean_vel, mean_sound_speed, delta_sound_speed], axis=0)

        var = self.transpose(var, (3, 1, 2, 0))
        net_out = mnp.exp(self.net(var))
        net_out = self.transpose(net_out, (3, 1, 2, 0))

        flux = 0.5 * (flux_left + flux_right) - net_out * (con_var_right - con_var_left)

        return flux
