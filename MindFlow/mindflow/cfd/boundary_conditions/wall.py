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
"""no slip wall boundary condition"""
import mindspore.numpy as mnp
from mindspore import jit_class

from .base import Boundary


@jit_class
class Wall(Boundary):
    """No slip wall boundary condition"""

    def __init__(self, config):
        super(Wall, self).__init__(config)
        velocity_x = config.get("velocity_x", 0.0)
        velocity_y = config.get("velocity_y", 0.0)
        velocity_z = config.get("velocity_z", 0.0)

        self.velocities = [velocity_x, velocity_y, velocity_z]

    def fill_values_head(self, pri_var, axis, pad_size):
        val = pri_var.copy()[:, :pad_size, :, :]
        val[1: 2, ...] = 2 * self.velocities[0] - val[1: 2, ...]
        val[2: 3, ...] = 2 * self.velocities[1] - val[2: 3, ...]
        val[3: 4, ...] = 2 * self.velocities[2] - val[3: 4, ...]
        return mnp.flip(val, 1)

    def fill_values_tail(self, pri_var, axis, pad_size):
        val = pri_var.copy()[:, -pad_size:, :, :]
        val[1: 2, ...] = 2 * self.velocities[0] - val[1: 2, ...]
        val[2: 3, ...] = 2 * self.velocities[1] - val[2: 3, ...]
        val[3: 4, ...] = 2 * self.velocities[2] - val[3: 4, ...]
        return mnp.flip(val, 1)
