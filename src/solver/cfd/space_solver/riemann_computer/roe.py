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
"""Roe Riemann Solver"""
from mindspore import jit_class, ops
from mindspore import numpy as mnp

from ...utils import cal_flux, cal_pri_var
from .base import RiemannComputer


@jit_class
class Roe(RiemannComputer):
    r"""
    ROE Riemann Solver based on Toro et al. 2009

    Args:
        material (Material): The information container of the fluid material.

    Supported Platforms:
        ``GPU``

    """

    def __init__(self, material, net_dict=None):
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

        flux = 0.5 * (flux_left + flux_right)
        right_eigen, eigen_vals, left_eigen = self.eigen_composition(pri_var_left, pri_var_right, axis)
        einsum = ops.Einsum('ij...,jk...,kl...,l...->i...')
        result = 0.5 * einsum((right_eigen, eigen_vals, left_eigen, (con_var_right - con_var_left)))
        flux -= result

        return flux

    def roe_avg(self, pri_var_left, pri_var_right):
        """
        Compute the average Roe variables and related quantities.

        Inputs:
            - pri_var_left (Tensor): Primitive variables of the left state.
            - pri_var_right (Tensor): Primitive variables of the right state.

        Outputs:
            list: A list containing the computed average Roe variables and related quantities in the following order:
            - primes_ave (Tensor): The average Roe variables computed using the left and right primitive variables.
            - c_ave (Tensor): The average speed of sound.
            - grueneisen (Tensor): The Grueneisen coefficient computed using the average Roe variables.
            - enthalpy_ave (Tensor): The average total enthalpy.
            - velocity_square (Tensor): The square of the average velocity magnitude.
        """
        enthalpy_left = self.material.total_enthalpy(pri_var_left)
        enthalpy_right = self.material.total_enthalpy(pri_var_right)
        alpha = 1 / (mnp.sqrt(abs(pri_var_left[0])) + mnp.sqrt(abs(pri_var_right[0])))
        primes_ave = (
            mnp.sqrt(abs(pri_var_left[0])) * pri_var_left + mnp.sqrt(abs(pri_var_right[0])) * pri_var_right
        ) * alpha
        enthalpy_ave = (
            mnp.sqrt(abs(pri_var_left[0])) * enthalpy_left + mnp.sqrt(abs(pri_var_right[0])) * enthalpy_right
        ) * alpha
        velocity_square = primes_ave[1] * primes_ave[1] + primes_ave[2] * primes_ave[2] + primes_ave[3] * primes_ave[3]
        c_ave = mnp.sqrt(abs((self.material.gamma - 1) * (enthalpy_ave - 0.5 * velocity_square)))
        grueneisen = self.material.grueneisen(primes_ave)
        return [primes_ave, c_ave, grueneisen, enthalpy_ave, velocity_square]

    def eigen_composition(self, pri_var_left, pri_var_right, axis):
        """
        Performs eigen composition for a given axis.

        Inputs:
            pri_var_left (Tensor): Array of primary variables on the left side of the interface.
            pri_var_right (Tensor): Array of primary variables on the right side of the interface.
            axis (int): Axis along which the eigen composition is performed.

        Outputs:
            tuple: A tuple containing the following elements:
                - right_eigen (Tensor): Right eigenvector matrix.
                - eigen_vals (Tensor): Eigenvalue matrix.
                - left_eigen (Tensor): Left eigenvector matrix.
        """
        primes_ave, c_ave, grueneisen, enthalpy_ave, velocity_square = self.roe_avg(pri_var_left, pri_var_right)
        ek = 0.5 * velocity_square
        zeros = mnp.zeros_like(primes_ave[0])
        ones = mnp.ones_like(primes_ave[0])

        right_eigen, eigen_vals, left_eigen = None, None, None
        gamma_1 = mnp.abs(primes_ave[axis + 1] - c_ave)
        gamma_234 = mnp.abs(primes_ave[axis + 1])
        gamma_5 = mnp.abs(primes_ave[axis + 1] + c_ave)

        eigen_vals = mnp.stack(
            (
                mnp.stack((gamma_1, zeros, zeros, zeros, zeros)),
                mnp.stack((zeros, gamma_234, zeros, zeros, zeros)),
                mnp.stack((zeros, zeros, gamma_234, zeros, zeros)),
                mnp.stack((zeros, zeros, zeros, gamma_234, zeros)),
                mnp.stack((zeros, zeros, zeros, zeros, gamma_5)),
            )
        )

        if axis == 0:
            right_eigen = mnp.stack(
                (
                    mnp.stack((ones, ones, zeros, zeros, ones)),
                    mnp.stack(
                        (
                            primes_ave[1] - c_ave,
                            primes_ave[1],
                            zeros,
                            zeros,
                            primes_ave[1] + c_ave,
                        )
                    ),
                    mnp.stack((primes_ave[2], primes_ave[2], -ones, zeros, primes_ave[2])),
                    mnp.stack((primes_ave[3], primes_ave[3], zeros, ones, primes_ave[3])),
                    mnp.stack(
                        (
                            enthalpy_ave - primes_ave[1] * c_ave,
                            ek,
                            -primes_ave[2],
                            primes_ave[3],
                            enthalpy_ave + primes_ave[1] * c_ave,
                        )
                    ),
                )
            )

            left_eigen = (
                grueneisen
                / 2
                / c_ave**2
                * mnp.stack(
                    (
                        mnp.stack(
                            (
                                ek + c_ave / grueneisen * primes_ave[1],
                                -primes_ave[1] - c_ave / grueneisen,
                                -primes_ave[2],
                                -primes_ave[3],
                                ones,
                            )
                        ),
                        mnp.stack(
                            (
                                2 / grueneisen * c_ave**2 - velocity_square,
                                2 * primes_ave[1],
                                2 * primes_ave[2],
                                2 * primes_ave[3],
                                -2 * ones,
                            )
                        ),
                        mnp.stack(
                            (
                                2 * c_ave**2 / grueneisen * primes_ave[2],
                                zeros,
                                -2 * c_ave**2 / grueneisen,
                                zeros,
                                zeros,
                            )
                        ),
                        mnp.stack(
                            (
                                -2 * c_ave**2 / grueneisen * primes_ave[3],
                                zeros,
                                zeros,
                                2 * c_ave**2 / grueneisen,
                                zeros,
                            )
                        ),
                        mnp.stack(
                            (
                                ek - c_ave / grueneisen * primes_ave[1],
                                -primes_ave[1] + c_ave / grueneisen,
                                -primes_ave[2],
                                -primes_ave[3],
                                ones,
                            )
                        ),
                    )
                )
            )

        # # Y - DIRECTION
        elif axis == 1:
            right_eigen = mnp.stack(
                (
                    mnp.stack((ones, zeros, ones, zeros, ones)),
                    mnp.stack((primes_ave[1], ones, primes_ave[1], zeros, primes_ave[1])),
                    mnp.stack(
                        (
                            primes_ave[2] - c_ave,
                            zeros,
                            primes_ave[2],
                            zeros,
                            primes_ave[2] + c_ave,
                        )
                    ),
                    mnp.stack((primes_ave[3], zeros, primes_ave[3], -ones, primes_ave[3])),
                    mnp.stack(
                        (
                            enthalpy_ave - primes_ave[2] * c_ave,
                            primes_ave[1],
                            ek,
                            -primes_ave[3],
                            enthalpy_ave + primes_ave[2] * c_ave,
                        )
                    ),
                )
            )

            left_eigen = (
                grueneisen
                / 2
                / c_ave**2
                * mnp.stack(
                    (
                        mnp.stack(
                            (
                                ek + c_ave / grueneisen * primes_ave[2],
                                -primes_ave[1],
                                -primes_ave[2] - c_ave / grueneisen,
                                -primes_ave[3],
                                ones,
                            )
                        ),
                        mnp.stack(
                            (
                                -2 * c_ave**2 / grueneisen * primes_ave[1],
                                2 * c_ave**2 / grueneisen,
                                zeros,
                                zeros,
                                zeros,
                            )
                        ),
                        mnp.stack(
                            (
                                2 / grueneisen * c_ave**2 - velocity_square,
                                2 * primes_ave[1],
                                2 * primes_ave[2],
                                2 * primes_ave[3],
                                -2 * ones,
                            )
                        ),
                        mnp.stack(
                            (
                                2 * c_ave**2 / grueneisen * primes_ave[3],
                                zeros,
                                zeros,
                                -2 * c_ave**2 / grueneisen,
                                zeros,
                            )
                        ),
                        mnp.stack(
                            (
                                ek - c_ave / grueneisen * primes_ave[2],
                                -primes_ave[1],
                                -primes_ave[2] + c_ave / grueneisen,
                                -primes_ave[3],
                                ones,
                            )
                        ),
                    )
                )
            )

        # # Z - DIRECTION
        elif axis == 2:
            right_eigen = mnp.stack(
                (
                    mnp.stack((ones, zeros, zeros, ones, ones)),
                    mnp.stack((primes_ave[1], -ones, zeros, primes_ave[1], primes_ave[1])),
                    mnp.stack((primes_ave[2], zeros, ones, primes_ave[2], primes_ave[2])),
                    mnp.stack(
                        (
                            primes_ave[3] - c_ave,
                            zeros,
                            zeros,
                            primes_ave[3],
                            primes_ave[3] + c_ave,
                        )
                    ),
                    mnp.stack(
                        (
                            enthalpy_ave - primes_ave[3] * c_ave,
                            -primes_ave[1],
                            primes_ave[2],
                            ek,
                            enthalpy_ave + primes_ave[3] * c_ave,
                        )
                    ),
                )
            )

            left_eigen = (
                grueneisen
                / 2
                / c_ave**2
                * mnp.stack(
                    (
                        mnp.stack(
                            (
                                ek + c_ave / grueneisen * primes_ave[3],
                                -primes_ave[1],
                                -primes_ave[2],
                                -primes_ave[3] - c_ave / grueneisen,
                                ones,
                            )
                        ),
                        mnp.stack(
                            (
                                2 * c_ave**2 / grueneisen * primes_ave[1],
                                -2 * c_ave**2 / grueneisen,
                                zeros,
                                zeros,
                                zeros,
                            )
                        ),
                        mnp.stack(
                            (
                                -2 * c_ave**2 / grueneisen * primes_ave[2],
                                zeros,
                                2 * c_ave**2 / grueneisen,
                                zeros,
                                zeros,
                            )
                        ),
                        mnp.stack(
                            (
                                2 / grueneisen * c_ave**2 - velocity_square,
                                2 * primes_ave[1],
                                2 * primes_ave[2],
                                2 * primes_ave[3],
                                -2 * ones,
                            ),
                        ),
                        mnp.stack(
                            (
                                ek - c_ave / grueneisen * primes_ave[3],
                                -primes_ave[1],
                                -primes_ave[2],
                                -primes_ave[3] + c_ave / grueneisen,
                                ones,
                            )
                        ),
                    )
                )
            )
        return right_eigen, eigen_vals, left_eigen
