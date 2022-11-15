# Copyright 2021 Huawei Technologies Co., Ltd
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
#pylint: disable=W0613
"""
Updating coefficients in CFS-PML.
"""
import numpy as np
from .constants import epsilon0, mu0, eta0


class CFSParameters:
    """
    Computes the updating coefficients in the CFS-PML.

    Args:
        npml (int): Thickness of the PML.
        order (int): Order of the PML.
        sigma_factor (float): Sigma parameter, sigma_max = sigma_factor * sigma_opt.
        alpha_min (float): Minimum of alpha.
        alpha_max (float): Maximum of alpha.
        kappa_max (int): Maximum of kappa.
    """

    def __init__(self, npml,
                 order=4, sigma_factor=1.5,
                 alpha_min=0., alpha_max=0.0,
                 kappa_max=11,
                 ):
        self.npml = npml
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.kappa_max = kappa_max
        self.sigma_factor = sigma_factor
        self.order = order

    @staticmethod
    def _compute_scaled_distance(npml):
        """compute distance between field points and pml inner boundary"""
        rho = (np.arange(0, 2 * npml) + 0.5) / (2 * npml)
        rho_e = np.vstack([rho[-2::-2], rho[0::2]])
        rho_m = np.vstack([rho[-1::-2], rho[1::2]])
        return rho_e, rho_m

    def get_update_coefficients(self, nx, dx, dt, epsr):
        """Computes updating coefficients.

        Args:
            nx(int): Number of cells in this direction.
            dx(float): Space step.
            dt(float): Time interval.
            epsr(float): Background relative permittivity.

        Returns:
            numpy.ndarray, shape=(3, nx + 1)
            numpy.ndarray, shape=(3, nx)
        """
        sigma_opt = (self.order + 1) / ((150. * np.pi * np.sqrt(epsr) * dx))
        sigma_max = self.sigma_factor * sigma_opt

        # compute PML material parameters
        factor = eta0 ** 2
        rho_e, rho_m = CFSParameters._compute_scaled_distance(self.npml)

        sigma_e = sigma_max * rho_e**self.order
        sigma_m = (factor * sigma_max) * (rho_m**self.order)

        kappa_e = 1 + (self.kappa_max - 1) * (rho_e**self.order)
        kappa_m = 1 + (self.kappa_max - 1) * (rho_m**self.order)

        alpha_e = self.alpha_min + \
            (self.alpha_max - self.alpha_min) * (1 - rho_e)
        alpha_m = factor * (self.alpha_min +
                            (self.alpha_max - self.alpha_min) * (1 - rho_m))

        # compute updaing coefficients
        be = np.exp(-(alpha_e + sigma_e / kappa_e) * (dt / epsilon0))
        ae = (be - 1.) * (sigma_e / (sigma_e + alpha_e * kappa_e))
        bm = np.exp(-(alpha_m + sigma_m / kappa_m) * (dt / mu0))
        am = (bm - 1.) * (sigma_m / (sigma_m + alpha_m * kappa_m))

        return self._extend_dims(np.vstack([be, ae, kappa_e, bm, am, kappa_m]), nx)

    def _extend_dims(self, paras, ncells):
        """extend dimensions"""
        ce = np.zeros((3, ncells + 1,))  # be, ae, ke
        cm = np.zeros((3, ncells,))  # bm, am, km
        npml = self.npml

        ce[2, :] = 1.  # k = 1 outside of the pml
        ce[:, 1:npml + 1] = paras[0:6:2, :]
        ce[:, -1 - npml:-1] = paras[1:6:2, :]

        cm[2, :] = 1.  # k = 1 outside of the pml
        cm[:, 0:npml] = paras[6::2, :]
        cm[:, -npml:] = paras[7::2, :]

        return ce, cm
