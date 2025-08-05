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
"""4th order stencil for 1st derivative at the cell center"""
from mindspore import jit_class

from .base import DerivativeComputer


@jit_class
class FourthOrderCentralDerivativeComputer(DerivativeComputer):
    """
    4th order stencil for 1st derivative at the cell center

                   central_derivative
    |        |        |        |        |        |
    | var_b2 | var_b1 |        | var_f1 | var_f2 |
    |        |        |        |        |        |
    """

    def __init__(self, mesh_info):
        super(FourthOrderCentralDerivativeComputer, self).__init__(mesh_info)
        if self.pad < 2:
            raise ValueError('pad should be not smaller than 3 for Fourth Order Central Derivative Computer')

    def derivative(self, var, dxi, axis):
        var_b2, var_b1, var_f1, var_f2 = self._get_var(var, axis)
        central_derivative = (1.0 / 24.0 / dxi) * (var_b2 - 27.0 * var_b1 + 27.0 * var_f1 - var_f2)
        return central_derivative

    def _get_var(self, var, axis):
        """get variables for derivative computer."""
        var_b2 = None
        var_b1 = None
        var_f1 = None
        var_f2 = None

        if axis == 0:
            var_b2 = var[:, self.pad - 2: - self.pad - 2, :, :]
            var_b1 = var[:, self.pad - 1: - self.pad - 1, :, :]
            var_f1 = var[:, self.pad: - self.pad, :, :]
            var_f2 = var[:, self.pad + 1: - self.pad + 1, :, :]

        if axis == 1:
            var_b2 = var[:, :, self.pad - 2: - self.pad - 2, :]
            var_b1 = var[:, :, self.pad - 1: - self.pad - 1, :]
            var_f1 = var[:, :, self.pad: - self.pad, :]
            var_f2 = var[:, :, self.pad + 1: - self.pad + 1, :]

        if axis == 2:
            var_b2 = var[:, :, :, self.pad - 2: - self.pad - 2]
            var_b1 = var[:, :, :, self.pad - 1: - self.pad - 1]
            var_f1 = var[:, :, :, self.pad: - self.pad]
            var_f2 = var[:, :, :, self.pad + 1: - self.pad + 1]

        return var_b2, var_b1, var_f1, var_f2
