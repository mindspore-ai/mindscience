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
"""4th order stencil for reconstruction at the cell face."""
from mindspore import jit_class

from .base import Interpolator


@jit_class
class CentralFourthOrderInterpolator(Interpolator):
    """
    4th order stencil for reconstruction at the cell face

                   face_var
    |        |        |        |        |
    | var_b2 | var_b1 | var_f1 | var_f2 |
    |        |        |        |        |
    """

    def __init__(self, mesh_info):
        super(CentralFourthOrderInterpolator, self).__init__(mesh_info)
        if self.pad < 2:
            raise ValueError('pad should be not smaller than 3 for Central Fourth Order Interpolator')

    def interpolate(self, var, axis):
        var_b2, var_b1, var_f1, var_f2 = self._get_var(var, axis)
        face_var = (1.0 / 16.0) * (-1.0 * var_b2 + 9.0 * var_b1 + 9.0 * var_f1 - 1 * var_f2)
        return face_var

    def _get_var(self, var, axis):
        """get variables for interpolator."""
        var_b2 = None
        var_b1 = None
        var_f1 = None
        var_f2 = None

        if axis == 0:
            var_b2 = var[:, self.pad - 2: - self.pad - 1, :, :]
            var_b1 = var[:, self.pad - 1: - self.pad, :, :]
            var_f1 = var[:, self.pad: - self.pad + 1, :, :]
            var_f2 = var[:, self.pad + 1: - self.pad + 2, :, :]

        if axis == 1:
            var_b2 = var[:, :, self.pad - 2: - self.pad - 1, :]
            var_b1 = var[:, :, self.pad - 1: - self.pad, :]
            var_f1 = var[:, :, self.pad: - self.pad + 1, :]
            var_f2 = var[:, :, self.pad + 1: - self.pad + 2, :]

        if axis == 2:
            var_b2 = var[:, :, :, self.pad - 2: - self.pad - 1]
            var_b1 = var[:, :, :, self.pad - 1: - self.pad]
            var_f1 = var[:, :, :, self.pad: - self.pad + 1]
            var_f2 = var[:, :, :, self.pad + 1: - self.pad + 2]

        return var_b2, var_b1, var_f1, var_f2
