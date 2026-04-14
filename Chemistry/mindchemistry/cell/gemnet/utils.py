# Copyright 2024 Huawei Technologies Co., Ltd
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
"""
utils
"""
import math
import numpy as np
from scipy.optimize import brentq
from scipy import special as sp


def jn(r, n):
    """
    Return a normalized version of the Bessel function of half-integer order
    Args:
        r (int): order of the function
        n (int): radial distance
    Returns:
        (numpy.ndarray) Bessel function of half-integer order
    """
    return np.sqrt(np.pi / (2 * r)) * sp.jv(n + 0.5, r)


def jn_zeros(n, k):
    """
    Compute the zeros of the Bessel function
    Args:
        n (int): order of the function
        k (int): number of zeros to compute
    Returns:
        zerosj (numpy.ndarray): zeros of the Bessel function
    """

    zerosj = np.zeros((n, k), dtype=np.float32)
    zerosj[0] = np.arange(1, k + 1) * np.pi
    points = np.arange(1, k + n) * np.pi
    racines = np.zeros(k + n - 1, dtype=np.float32)
    for i in range(1, n):
        for j in range(k + n - 1 - i):
            foo = brentq(jn, points[j], points[j + 1], (i,))
            racines[j] = foo
        points = racines
        zerosj[i][:k] = racines[:k]

    return zerosj


def bessel_basis(n, k):
    """
    Compute the Bessel basis
    Args:
        n (int): order of the function
        k (int): number of zeros to compute
    Returns:
        normalizer (numpy.ndarray): normalizer of the Bessel basis
        zeros (numpy.ndarray): zeros of the Bessel function
    """
    zeros = jn_zeros(n, k)
    normalizer = []
    for order in range(n):
        normalizer_tmp = []
        for i in range(k):
            normalizer_tmp += [0.5 * jn(zeros[order, i], order + 1)**2]
        normalizer_tmp = 1 / np.array(normalizer_tmp)**0.5
        normalizer += [normalizer_tmp]
    return normalizer, zeros


def spherical_bessel_formulas(x, n):
    """
    compute spherical bessel formulas
    Args:
        x (int): input
        n (int): order of the function
    Returns:
        res (numpy.ndarray): spherical bessel formulas
    """

    if n == 0:
        res = np.sin(x)/x
    elif n == 1:
        res = (-x * np.cos(x) + np.sin(x))/x**2
    elif n == 2:
        res = ((-x**2 * np.sin(x))
               - (x * 3 * np.cos(x))
               + 3 * np.sin(x))/x**3
    elif n == 3:
        res = ((x**3 * np.cos(x))
               - (6 * x**2 * np.sin(x))
               - (15 * x * np.cos(x))
               + 15 * np.sin(x))/x**4
    elif n == 4:
        res = ((x**4 * np.sin(x))
               + (10 * x**3 * np.cos(x))
               - (45 * x**2 * np.sin(x))
               - (105 * x * np.cos(x))
               + 105 * np.sin(x))/x**5
    elif n == 5:
        res = ((-x**5 * np.cos(x))
               + (15 * x**4 * np.sin(x))
               + (105 * x**3 * np.cos(x))
               - (420 * x**2 * np.sin(x))
               - (945 * x * np.cos(x))
               + 945 * np.sin(x))/x**6
    elif n == 6:
        res = ((-x**6 * np.sin(x))
               - (21 * x**5 * np.cos(x))
               + (210 * x**4 * np.sin(x))
               + (1260 * x**3 * np.cos(x))
               - (4725 * x**2 * np.sin(x))
               - (10395 * x * np.cos(x))
               + 10395 * np.sin(x))/x**7
    else:
        raise ValueError("Order too high")
    return res


def bessel_basis_result(x, normalizer, n, k, zeros):
    """
    bessel basis result
    Args:
        x (int): input
        normalizer (numpy.ndarray): normalizer of the Bessel basis
        n (int): order of the function
        k (int): number of zeros to compute
        zeros (numpy.ndarray): zeros of the Bessel function
    Returns:
        bess_basis (numpy.ndarray): bessel basis result
    """
    bess_basis = np.zeros((n, k, x.shape[0]))
    for order in range(n):
        for i in range(k):
            x_1 = zeros[order, i] * x
            sbf_res = spherical_bessel_formulas(x_1, order)
            bess_basis_tmp = normalizer[order][i] * sbf_res
            bess_basis[order, i] = bess_basis_tmp.reshape(1, -1)
    return bess_basis


def sph_harm_prefactor_np(l_degree, m_order):
    """Computes the constant pre-factor for the spherical harmonic of degree l and order m.

    Args:
        l_degree (int): Degree of the spherical harmonic :math:`l >= 0`
        m_order (int): Order of the spherical harmonic :math:`-l <= m <= l`

    Returns:
        res (float): factor for the spherical harmonic
    """

    res = ((2 * l_degree + 1) / (4 * np.pi) * math.factorial(l_degree - abs(m_order)) /
           math.factorial(l_degree + abs(m_order))) ** 0.5
    return res


def associated_legendre_polynomials_np(z, l_maxdegree):
    """
    Computes string formulas of the associated legendre polynomials up to degree L (excluded).
       zero_m_only: only calculate the polynomials for the polynomials where m=0.
    calculations from http://web.cmb.usc.edu/people/alber/Software/tomominer/docs/cpp/group__legendre__polynomials.html

    Args:
        z (numpy.ndarray): input
        l_maxdegree (int): Degree up to which to calculate the associated legendre polynomials.

    Returns:
        p_l_m (numpy.ndarray): associated legendre polynomials
    """

    p_l_m = [[0] * (2 * l_degree + 1)
             for l_degree in range(l_maxdegree)]  # for order l: -l <= m <= l
    p_l_m[0][0] = np.ones_like(z)
    p_l_m[1][0] = z
    for l_degree in range(2, l_maxdegree):
        p_l_m[l_degree][0] = (
            ((2 * l_degree - 1) * z * p_l_m[l_degree - 1][0] -
             (l_degree - 1) * p_l_m[l_degree - 2][0]) / l_degree
        )
    return p_l_m


def real_sph_harm_np(z, l_maxdegree, use_theta, zero_m_only=True):
    """
    Computes formula strings of the the real part of the spherical harmonics up to degree L (excluded).
    Variables are either spherical coordinates phi and theta (or cartesian coordinates x,y,z) on the UNIT SPHERE.
    Args:
        z (numpy.ndarray): input
        l_maxdegree (int): Degree up to which to calculate the spherical harmonics (degree L is excluded).
        use_theta (bool): True: Expects the input of the formula strings to contain theta.
            False: Expects the input of the formula strings to contain z.
        zero_m_only (bool): If True only calculate the harmonics where m=0.
    Returns:
        y_l_m (numpy.ndarray): real part of the spherical harmonics
    """

    if zero_m_only:
        y_l_m = [[0] for l_degree in range(l_maxdegree)]
    else:
        y_l_m = [
            [0] * (2 * l_degree + 1)
            for l_degree in range(l_maxdegree)
        ]

    if use_theta:
        p_l_m = associated_legendre_polynomials_np(np.cos(z), l_maxdegree)
    else:
        p_l_m = associated_legendre_polynomials_np(z, l_maxdegree)

    for l_degree in range(l_maxdegree):
        y_l_m[l_degree] = (
            sph_harm_prefactor_np(l_degree, 0) * p_l_m[l_degree][0]
        )
    y_l_m = np.stack(y_l_m, axis=0)
    return y_l_m


class SphericalBasisLayer:
    r"""
    Spherical Basis Layer

    Args:
        num_spherical (int): Number of spherical harmonics.
        num_radial (int): Number of radial basis functions.
        cutoff (float): Cutoff distance.
        envelope_exponent (int): Exponent of the envelope function.

    Inputs:
        - **dist** (numpy.ndarray) - The shape of array is :math:`(total\_edges,)`.
        - **angle** (numpy.ndarray) - The shape of array is :math:`(total\_triplets,)`.
        - **idx_kj** (numpy.ndarray) - The shape of array is :math:`(total\_triplets,)`.

    Outputs:
        - **out** (numpy.ndarray) - The shape of array is :math:`(total\_edges, num\_spherical * num\_radial)`.
    """

    def __init__(self, num_spherical, num_radial, cutoff=5.0,
                 envelope_exponent=5):
        super().__init__()
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.cutoff = cutoff
        self.p = envelope_exponent + 1
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

        self.normalizer, self.zeros = bessel_basis(
            self.num_spherical, self.num_radial)

    def sbf(self, dist, angle, idx_kj):
        """Spherical Basis Layer Construct"""
        dist_np = dist / self.cutoff

        cbf = real_sph_harm_np(angle, self.num_spherical,
                               use_theta=True, zero_m_only=True).T
        bessel_forms = bessel_basis_result(
            dist_np, self.normalizer, self.num_spherical, self.num_radial, self.zeros)
        rbf = bessel_forms.reshape(
            (self.num_spherical * self.num_radial, -1)).T

        rbf = np.expand_dims(self.envelope_np(dist_np), axis=-1) * rbf
        n, k = self.num_spherical, self.num_radial

        reshaped_rbf = rbf[idx_kj].reshape(-1, n, k)
        reshaped_cbf = cbf.reshape(-1, n, 1)
        multiplied = reshaped_rbf * reshaped_cbf
        out = multiplied.reshape(-1, n * k)
        return out

    def envelope_np(self, x):
        x_pow_p0 = x ** (self.p - 1)
        x_pow_p1 = x_pow_p0 * x
        x_pow_p2 = x_pow_p1 * x
        return 1 / x + self.a * x_pow_p0 + self.b * x_pow_p1 + self.c * x_pow_p2
