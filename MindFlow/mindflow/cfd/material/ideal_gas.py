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
"""the ideal gas law."""
from mindspore import numpy as mnp
from mindspore import jit_class
from .base import Material


@jit_class
class IdealGas(Material):
    """The ideal gas law."""

    def __init__(self, config):
        super(IdealGas, self).__init__(config)
        self.gamma = config.get('heat_ratio', None)
        self.gas_constant = config.get('gas_constant', 0.0)
        self.cp = self.gamma / (self.gamma - 1) * self.gas_constant

    def psi(self, pri_var):
        return pri_var[4, ...] / pri_var[0, ...]

    def grueneisen(self, pri_var):
        return self.gamma - 1

    def sound_speed(self, pri_var):
        return mnp.sqrt(self.gamma * pri_var[4, ...] / pri_var[0, ...])

    def temperature(self, pri_var):
        return pri_var[4, ...] / (pri_var[0, ...] * self.gas_constant)

    def energy(self, pri_var):
        return pri_var[4, ...] / (pri_var[0, ...] * (self.gamma - 1))

    def total_energy(self, pri_var):
        return pri_var[4, ...] / (self.gamma - 1) + 0.5 * pri_var[0, ...] * (
            (pri_var[1, ...] ** 2 + pri_var[2, ...] ** 2 + pri_var[3, ...] ** 2))

    def total_enthalpy(self, pri_var):
        return (self.total_energy(pri_var) + pri_var[4, ...]) / pri_var[0, ...]
