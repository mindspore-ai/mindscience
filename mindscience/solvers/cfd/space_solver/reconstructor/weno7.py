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
"""7th order Weighted Essentially Non-Oscillatory (WENO) scheme"""
from mindspore import jit_class

from .base import Reconstructor


@jit_class
class WENO7(Reconstructor):
    r"""
    7th order Weighted Essentially Non-Oscillatory (WENO) scheme

    The basic idea of the WENO scheme is to replace the ENO scheme which only uses the smoothest interpolation region to
    provide an approximation of the numerical fluxes at mesh sections with a weighted average of the numerical fluxes
    at mesh interfaces provided by each possible interpolation region. Specifically, each possible interpolation region
    is assigned a weight that determines its contribution to the numerical flux at the final grid interface,

    For more details, please refers to the paper:  `D.S. Balsara, C.W. Shu, J. Comput. Phys. 160 (2) (2000) 405–452,
    https://doi.org/10.1006/jcph.2000.6443.`

    Args:
        mesh_info (MeshInfo): The information container of the computing mesh.

    Raises:
        ValueError: If `mesh_info.pad` is less than 4.

    Supported Platforms:
        ``GPU``

    """

    def __init__(self, mesh_info):
        super().__init__(mesh_info)
        self._coe1 = [
            [1.0 / 35.0, 12.0 / 35.0, 18.0 / 35.0, 4.0 / 35.0],
            [4.0 / 35.0, 18.0 / 35.0, 12.0 / 35.0, 1.0 / 35.0],
        ]
        self._coe2 = [
            [
                [-1.0 / 4.0, 13.0 / 12.0, -23.0 / 12.0, 25.0 / 12.0],
                [1.0 / 12.0, -5.0 / 12.0, 13.0 / 12.0, 1.0 / 4.0],
                [-1.0 / 12.0, 7.0 / 12.0, 7.0 / 12.0, -1.0 / 12.0],
                [1.0 / 4.0, 13.0 / 12.0, -5.0 / 12.0, 1.0 / 12.0],
            ],
            [
                [1.0 / 12.0, -5.0 / 12.0, 13.0 / 12.0, 1.0 / 4.0],
                [-1.0 / 12.0, 7.0 / 12.0, 7.0 / 12.0, -1.0 / 12.0],
                [1.0 / 4.0, 13.0 / 12.0, -5.0 / 12.0, 1.0 / 12.0],
                [25.0 / 12.0, -23.0 / 12.0, 13.0 / 12.0, -1.0 / 4.0],
            ],
        ]
        if self.pad < 4:
            raise ValueError('pad should be not smaller than 4 for WENO7 reconstructor')
        self.eps = 1e-5

    def _reconstruct_on_face(self, var, axis, j):
        """
        Calculate the recunstructed variables on faces.

        Inputs:
        - **var** (Tensor) - Input tensor.
        - **axis** (int) - 0, 1, 2 indicate x-dimension, y-dimension and z-dimension respectively.
        - **j** (int) - reconstruct direction, 0, 1 indicate reconstruct from left and right respectively.

        Outputs:
        Tensor, output tensor.
        """
        var_0, var_1, var_2, var_3, var_4, var_5, var_6 = self._get_var(var, axis, j)

        beta_0 = (
            var_0 * (547 * var_0 - 3882 * var_1 + 4642 * var_2 - 1854 * var_3)
            + var_1 * (7043 * var_1 - 17246 * var_2 + 7042 * var_3)
            + var_2 * (11003 * var_2 - 9402 * var_3)
            + var_3 * (2107 * var_3)
        )

        beta_1 = (
            var_1 * (267 * var_1 - 1642 * var_2 + 1602 * var_3 - 494 * var_4)
            + var_2 * (2843 * var_2 - 5966 * var_3 + 1922 * var_4)
            + var_3 * (3443 * var_3 - 2522 * var_4)
            + var_4 * (547 * var_4)
        )

        beta_2 = (
            var_2 * (547 * var_2 - 2522 * var_3 + 1922 * var_4 - 494 * var_5)
            + var_3 * (3443 * var_3 - 5966 * var_4 + 1602 * var_5)
            + var_4 * (2843 * var_4 - 1642 * var_5)
            + var_5 * (267 * var_5)
        )

        beta_3 = (
            var_3 * (2107 * var_3 - 9402 * var_4 + 7042 * var_5 - 1854 * var_6)
            + var_4 * (11003 * var_4 - 17246 * var_5 + 4642 * var_6)
            + var_5 * (7043 * var_5 - 3882 * var_6)
            + var_6 * (547 * var_6)
        )

        one_beta_0_sq = 1.0 / (self.eps + beta_0) ** 2
        one_beta_1_sq = 1.0 / (self.eps + beta_1) ** 2
        one_beta_2_sq = 1.0 / (self.eps + beta_2) ** 2
        one_beta_3_sq = 1.0 / (self.eps + beta_3) ** 2

        alpha_0 = self._coe1[j][0] * one_beta_0_sq
        alpha_1 = self._coe1[j][1] * one_beta_1_sq
        alpha_2 = self._coe1[j][2] * one_beta_2_sq
        alpha_3 = self._coe1[j][3] * one_beta_3_sq

        one_alpha = 1.0 / (alpha_0 + alpha_1 + alpha_2 + alpha_3)

        omega_0 = alpha_0 * one_alpha
        omega_1 = alpha_1 * one_alpha
        omega_2 = alpha_2 * one_alpha
        omega_3 = alpha_3 * one_alpha

        p_0 = (
            self._coe2[j][0][0] * var_0
            + self._coe2[j][0][1] * var_1
            + self._coe2[j][0][2] * var_2
            + self._coe2[j][0][3] * var_3
        )
        p_1 = (
            self._coe2[j][1][0] * var_1
            + self._coe2[j][1][1] * var_2
            + self._coe2[j][1][2] * var_3
            + self._coe2[j][1][3] * var_4
        )
        p_2 = (
            self._coe2[j][2][0] * var_2
            + self._coe2[j][2][1] * var_3
            + self._coe2[j][2][2] * var_4
            + self._coe2[j][2][3] * var_5
        )
        p_3 = (
            self._coe2[j][3][0] * var_3
            + self._coe2[j][3][1] * var_4
            + self._coe2[j][3][2] * var_5
            + self._coe2[j][3][3] * var_6
        )

        var_on_face = omega_0 * p_0 + omega_1 * p_1 + omega_2 * p_2 + omega_3 * p_3

        output_size = [
            7,
        ] + self.mesh_info.number_of_cells
        output_size[axis + 1] += 1

        return self._slice(var_on_face, output_size)

    def _get_var(self, inputs, axis, j):
        """get variables for reconstructor."""
        var_0 = None
        var_1 = None
        var_2 = None
        var_3 = None
        var_4 = None
        var_5 = None
        var_6 = None

        if axis == 0:
            var_0 = inputs[:, self.pad - 4 + j : self.pad - 3 + j + self.mesh_info.number_of_cells[0], :, :]
            var_1 = inputs[:, self.pad - 3 + j : self.pad - 2 + j + self.mesh_info.number_of_cells[0], :, :]
            var_2 = inputs[:, self.pad - 2 + j : self.pad - 1 + j + self.mesh_info.number_of_cells[0], :, :]
            var_3 = inputs[:, self.pad - 1 + j : self.pad + j + self.mesh_info.number_of_cells[0], :, :]
            var_4 = inputs[:, self.pad + j : self.pad + 1 + j + self.mesh_info.number_of_cells[0], :, :]
            var_5 = inputs[:, self.pad + j + 1 : self.pad + j + 2 + self.mesh_info.number_of_cells[0], :, :]
            var_6 = inputs[:, self.pad + j + 2 : self.pad + j + 3 + self.mesh_info.number_of_cells[0], :, :]

        if axis == 1:
            var_0 = inputs[:, :, self.pad - 4 + j : self.pad - 3 + j + self.mesh_info.number_of_cells[1], :]
            var_1 = inputs[:, :, self.pad - 3 + j : self.pad - 2 + j + self.mesh_info.number_of_cells[1], :]
            var_2 = inputs[:, :, self.pad - 2 + j : self.pad - 1 + j + self.mesh_info.number_of_cells[1], :]
            var_3 = inputs[:, :, self.pad - 1 + j : self.pad + j + self.mesh_info.number_of_cells[1], :]
            var_4 = inputs[:, :, self.pad + j : self.pad + 1 + j + self.mesh_info.number_of_cells[1], :]
            var_5 = inputs[:, :, self.pad + j + 1 : self.pad + j + 2 + self.mesh_info.number_of_cells[1], :]
            var_6 = inputs[:, :, self.pad + j + 2 : self.pad + j + 3 + self.mesh_info.number_of_cells[1], :]

        if axis == 2:
            var_0 = inputs[:, :, : self.pad - 4 + j : self.pad - 3 + j + self.mesh_info.number_of_cells[2]]
            var_1 = inputs[:, :, : self.pad - 3 + j : self.pad - 2 + j + self.mesh_info.number_of_cells[2]]
            var_2 = inputs[:, :, : self.pad - 2 + j : self.pad - 1 + j + self.mesh_info.number_of_cells[2]]
            var_3 = inputs[:, :, : self.pad - 1 + j : self.pad + j + self.mesh_info.number_of_cells[2]]
            var_4 = inputs[:, :, : self.pad + j : self.pad + 1 + j + self.mesh_info.number_of_cells[2]]
            var_5 = inputs[:, :, : self.pad + j + 1 : self.pad + j + 2 + self.mesh_info.number_of_cells[2]]
            var_6 = inputs[:, :, : self.pad + j + 2 : self.pad + j + 3 + self.mesh_info.number_of_cells[2]]

        return [var_0, var_1, var_2, var_3, var_4, var_5, var_6]
