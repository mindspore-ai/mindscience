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
# ============================================================================
import functools
import math
from fractions import Fraction
from math import factorial

import numpy as np

from mindspore import Tensor, ops, float32, float64, complex64, complex128

from ..utils.func import _ndexpm, broadcast_args, _expand_last_dims

PI = Tensor(math.pi)


def change_basis_real_to_complex(l, dtype=float32):
    r"""
    Convert a real basis of spherical harmonics in term of complex.

    Arg:
        l (int): degree of spherical harmonics.
        dtype (dtype):{float32, float64} data type of the real basis. Default: float32.

    Returns:
        Tensor, the complex basis with dtpye complex64 for `dtype`=float32 and complex128 for `dtype`=float64.
    """
    q = np.zeros((2 * l + 1, 2 * l + 1), np.complex128)
    for m in range(-l, 0):
        q[l + m, l + abs(m)] = 1 / 2 ** 0.5
        q[l + m, l - abs(m)] = -1j / 2 ** 0.5
    q[l, l] = 1
    for m in range(1, l + 1):
        q[l + m, l + abs(m)] = (-1) ** m / 2 ** 0.5
        q[l + m, l - abs(m)] = 1j * (-1) ** m / 2 ** 0.5
    q = (-1j) ** l * q

    dtype = {
        float32: complex64,
        float64: complex128,
    }[dtype]

    q_new = Tensor(q, dtype=dtype)
    return q_new


def su2_generators(j, dtype=complex64):
    r"""
    Compute the su(2) Lie algebra generators.

    Args:
        j (int): degree of generators.
        dtype (dtype): {complex64, complex128} data type of generators. Default: complex64.

    Returns:
        Tensor, su(2) generators with the dtype is `dtype`.

    Raise:
        TypeError: If `j` is not int.
    """
    if not isinstance(j, int):
        raise TypeError
    m = np.arange(-j, j)
    raising = np.diag(-np.sqrt(j * (j + 1) - m * (m + 1)), k=-1)

    m = np.arange(-j + 1, j + 1)
    lowering = np.diag(np.sqrt(j * (j + 1) - m * (m - 1)), k=1)

    m = np.arange(-j, j + 1)
    res = np.stack([
        0.5 * (raising + lowering),  # x (usually)
        np.diag(1j * m),  # z (usually)
        -0.5j * (raising - lowering),  # -y (usually)
    ], axis=0)
    return Tensor(res, dtype=dtype)


def so3_generators(l, dtype=float32):
    r"""
    Compute the so(3) Lie algebra generators.

    Args:
        l (int): degree of generators.
        dtype (dtype): {float32, float64} data type of generators. Default: float32.

    Returns:
        Tensor, so(3) generators with the dtype is `dtype`.

    Raise:
        TypeError: If `l` is not int.
        ValueError: If matrices data are inconsistent.
    """
    if not isinstance(l, int):
        raise TypeError
    cdtype = {
        float32: complex64,
        float64: complex128,
    }[dtype]
    X = su2_generators(l, dtype=cdtype).asnumpy()
    Q = change_basis_real_to_complex(l, dtype=dtype).asnumpy()
    X = np.conj(Q.T) @ X @ Q

    if not np.all(np.abs(np.imag(X)) < 1e-5):
        raise ValueError
    X_real = np.real(X)
    return Tensor(X_real, dtype=dtype)


def wigner_D(l, alpha, beta, gamma):
    r"""
    Wigner D matrix representation of SO(3).

    It satisfies the following properties:
    * :math:`D(\text{identity rotation}) = \text{identity matrix}`
    * :math:`D(R_1 \circ R_2) = D(R_1) \circ D(R_2)`
    * :math:`D(R^{-1}) = D(R)^{-1} = D(R)^T`

    Args:
        l (int): degree of representation.
        alpha (Union[Tensor[float32], List[float], Tuple[float], ndarray[np.float32], float]): rotation :math:`\alpha` around Y axis, applied third.
            tensor of shape :math:`(...)`
        beta (Union[Tensor[float32], List[float], Tuple[float], ndarray[np.float32], float]): rotation :math:`\beta` around X axis, applied second.
            tensor of shape :math:`(...)`
        gamma (Union[Tensor[float32], List[float], Tuple[float], ndarray[np.float32], float]): rotation :math:`\gamma` around Y axis, applied first.
            tensor of shape :math:`(...)`

    Returns:
        Tensor, Wigner D matrix :math:`D^l(\alpha, \beta, \gamma)`.
            tensor of shape :math:`(2l+1, 2l+1)`
    """

    alpha, beta, gamma = broadcast_args(alpha, beta, gamma)
    alpha = _expand_last_dims(alpha) % (2 * PI)
    beta = _expand_last_dims(beta) % (2 * PI)
    gamma = _expand_last_dims(gamma) % (2 * PI)
    X = so3_generators(l)
    return ops.matmul(ops.matmul(_ndexpm(alpha * X[1]), _ndexpm(beta * X[0])), _ndexpm(gamma * X[1]))


def wigner_3j(l1, l2, l3, dtype=float32):
    r"""
    Wigner 3j symbols :math:`C_{lmn}`.

    It satisfies the following two properties:
        .. math::
            C_{lmn} = C_{ijk} D_{il}(g) D_{jm}(g) D_{kn}(g) \qquad \forall g \in SO(3)
        where :math:`D` are given by `wigner_D`.
        .. math::
            C_{ijk} C_{ijk} = 1

    Args:
        l1 (int): :math:`l_1`.
        l2 (int): :math:`l_2`.
        l3 (int): :math:`l_3`.

    Returns:
        Tensor, Wigner 3j symbols :math:`C_{lmn}`.
            tensor of shape :math:`(2l_1+1, 2l_2+1, 2l_3+1)`

    Raise:
        TypeError: If `l1`, `l2` or `l3` are not int.
        ValueError: If `l1`, `l2` and `l3` do not satisfy abs(l2 - l3) <= l1 <= l2 + l3.
    """
    if not isinstance(l1, int) and isinstance(l2, int) and isinstance(l3, int):
        raise TypeError
    if not abs(l2 - l3) <= l1 and l1 <= l2 + l3:
        raise ValueError(
            f"The inputs degree \"{l1}\" and \"{l2}\" do not match to output degree \"{l3}\". \nThe degrees should be |{l1} - {l2}| <= {l3} <= |{l1} + {l2}|.")
    C = _so3_clebsch_gordan(l1, l2, l3)

    return Tensor(C, dtype=dtype)


@functools.lru_cache(maxsize=None)
def _so3_clebsch_gordan(l1, l2, l3, dtype=float64):
    """Calculates the Clebsch-Gordon matrix for SO(3) coupling l1 and l2 to give l3."""
    Q1 = change_basis_real_to_complex(l1, dtype=dtype).asnumpy()
    Q2 = change_basis_real_to_complex(l2, dtype=dtype).asnumpy()
    Q3 = change_basis_real_to_complex(l3, dtype=dtype).asnumpy()
    C = _su2_clebsch_gordan(l1, l2, l3)

    C = np.einsum('ij,kl,mn,ikn->jlm', Q1, Q2, np.conj(Q3.T), C)

    if not np.all(np.abs(np.imag(C)) < 1e-5):
        raise ValueError
    C = np.real(C)

    C = C / np.linalg.norm(C)
    return C


@functools.lru_cache(maxsize=None)
def _su2_clebsch_gordan(j1, j2, j3):
    """Calculates the Clebsch-Gordon matrix for SU(2) coupling j1 and j2 to give j3."""
    if not (isinstance(j1, (int, float)) and isinstance(j2, (int, float)) and isinstance(j3, (int, float))):
        raise TypeError
    mat = np.zeros((int(2 * j1 + 1), int(2 * j2 + 1),
                    int(2 * j3 + 1)), np.float64)
    if int(2 * j3) in range(int(2 * abs(j1 - j2)), int(2 * (j1 + j2)) + 1, 2):
        for m1 in (x / 2 for x in range(-int(2 * j1), int(2 * j1) + 1, 2)):
            for m2 in (x / 2 for x in range(-int(2 * j2), int(2 * j2) + 1, 2)):
                if abs(m1 + m2) <= j3:
                    mat[int(j1 + m1), int(j2 + m2), int(j3 + m1 + m2)
                    ] = _su2_clebsch_gordan_coeff((j1, m1), (j2, m2), (j3, m1 + m2))

    return mat


def _su2_clebsch_gordan_coeff(idx1, idx2, idx3):
    """core function of the Clebsch-Gordon coefficient for SU(2) coupling (j1,m1) and (j2,m2) to give (j3,m3)."""

    j1, m1 = idx1
    j2, m2 = idx2
    j3, m3 = idx3

    if m3 != m1 + m2:
        return 0
    vmin = int(max([-j1 + j2 + m3, -j1 + m1, 0]))
    vmax = int(min([j2 + j3 + m1, j3 - j1 + j2, j3 + m3]))

    def f(n):
        if not n == round(n):
            raise ValueError
        return factorial(round(n))

    C = (
                (2.0 * j3 + 1.0) * Fraction(
            f(j3 + j1 - j2) * f(j3 - j1 + j2) *
            f(j1 + j2 - j3) * f(j3 + m3) * f(j3 - m3),
            f(j1 + j2 + j3 + 1) * f(j1 - m1) *
            f(j1 + m1) * f(j2 - m2) * f(j2 + m2)
        )
        ) ** 0.5

    S = 0
    for v in range(vmin, vmax + 1):
        S += (-1) ** int(v + j2 + m2) * Fraction(
            f(j2 + j3 + m1 - v) * f(j1 - m1 + v),
            f(v) * f(j3 - j1 + j2 - v) * f(j3 + m3 - v) * f(v + j1 - j2 - m3)
        )
    C = C * S
    return C
