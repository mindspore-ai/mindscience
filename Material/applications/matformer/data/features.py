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
"""file to get features"""

import sympy as sym
import numpy as np
from scipy.optimize import brentq
from scipy import special as sp


def jn(r, n):
    """jn"""
    return np.sqrt(np.pi / (2 * r)) * sp.jv(n + 0.5, r)


def jn_zeros(n, k):
    """jn_zeros"""
    zerosj = np.zeros((n, k), dtype='float32')
    zerosj[0] = np.arange(1, k + 1) * np.pi
    points = np.arange(1, k + n) * np.pi
    racines = np.zeros(k + n - 1, dtype='float32')
    for i in range(1, n):
        for j in range(k + n - 1 - i):
            foos = brentq(jn, points[j], points[j + 1], (i,))
            racines[j] = foos
        points = racines
        zerosj[i][:k] = racines[:k]

    return zerosj


def spherical_bessel_formulas(n):
    """spherical_bessel_formulas"""
    x = sym.symbols('x')

    f = [sym.sin(x) / x]
    a = sym.sin(x) / x
    for i in range(1, n):
        b = sym.diff(a, x) / x
        f += [sym.simplify(b * (-x) ** i)]
        a = sym.simplify(b)
    return f


def bessel_basis(n, k):
    """bessel_basis"""
    zeros = jn_zeros(n, k)
    normalizer = []
    for order in range(n):
        normalizer_tmp = []
        for i in range(k):
            normalizer_tmp += [0.5 * jn(zeros[order, i], order + 1) ** 2]
        normalizer_tmp = 1 / np.array(normalizer_tmp) ** 0.5
        normalizer += [normalizer_tmp]

    f = spherical_bessel_formulas(n)
    x = sym.symbols('x')
    bess_basis = []
    for order in range(n):
        bess_basis_tmp = []
        for i in range(k):
            bess_basis_tmp += [
                sym.simplify(normalizer[order][i] *
                             f[order].subs(x, zeros[order, i] * x))
            ]
        bess_basis += [bess_basis_tmp]
    return bess_basis


def sph_harm_prefactor(k, m):
    """sph_harm_prefactor"""
    return ((2 * k + 1) * np.math.factorial(k - abs(m)) /
            (4 * np.pi * np.math.factorial(k + abs(m)))) ** 0.5


def associated_legendre_polynomials(k, zero_m_only=True):
    """associated_legendre_polynomials"""
    z = sym.symbols('z')
    p_l_m = [[0] * (j + 1) for j in range(k)]

    p_l_m[0][0] = 1
    if k > 0:
        p_l_m[1][0] = z

        for j in range(2, k):
            p_l_m[j][0] = sym.simplify(((2 * j - 1) * z * p_l_m[j - 1][0] -
                                        (j - 1) * p_l_m[j - 2][0]) / j)
        if not zero_m_only:
            for i in range(1, k):
                p_l_m[i][i] = sym.simplify((1 - 2 * i) * p_l_m[i - 1][i - 1])
                if i + 1 < k:
                    p_l_m[i + 1][i] = sym.simplify(
                        (2 * i + 1) * z * p_l_m[i][i])
                for j in range(i + 2, k):
                    p_l_m[j][i] = sym.simplify(
                        ((2 * j - 1) * z * p_l_m[j - 1][i] -
                         (i + j - 1) * p_l_m[j - 2][i]) / (j - i))

    return p_l_m
