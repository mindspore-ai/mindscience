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
"""SNO utils"""
import numpy as np
import scipy.special as scp
from scipy.interpolate import interp1d, interp2d

__all__ = ['poly_data', 'get_poly_transform',
           'interpolate_1d_dataset', 'interpolate_2d_dataset']

"""
    Functions for the transformation to the Chebyshev, Legendre, Jacobi polynomial space.
    These functions are applied for data interpolation and for calculation
    of direct and inverse transformation matrices for SNO, U-SNO models.
"""

poly_data = {
    "Legendre": lambda n, alpha: scp.roots_legendre(n)[0],
    "Chebyshev_u": lambda n, alpha: scp.roots_chebyu(n)[0],
    "Chebyshev_t": lambda n, alpha: scp.roots_chebyt(n)[0],
    "Jacobi": lambda n, alpha: scp.roots_jacobi(n, alpha[0], alpha[1])[0]
}


def get_analysis_data(n_spatial, gauss, polynomial):
    r"""
    Prepares data to calculate polynomial transform matrices.
    Args:
        n_spatial (int): the number of Gauss quadrature points.
        gauss (function): a function which calculates quadrature points and weights
        polynomial (function): a function which evaluates polynomials (Chebyshev, Legendre, etc.)

    Returns:
        The list of numpy arrays of shapes: :math:`(n, n), (n, n), n, n`.
    """

    points, weights = gauss(n_spatial)
    poly_vals = []
    diag_inv = []

    for n in range(n_spatial):
        poly = polynomial(n, points)
        prod = np.sum(weights * poly **2)
        poly_vals.append(poly)
        diag_inv.append(1 / prod)

    poly_vals = np.vstack(poly_vals).T
    diag_inv = np.diag(diag_inv)

    analysis_data = [poly_vals, diag_inv, weights, points]
    return analysis_data


def build_matrices(n_modes, analysis_data):
    r"""
    Calculates direct and inverse polynomial transform matrices.
    Args:
        n_modes (int): the number of modes of polynomial transform (expansion coefficients).
        analysis_data (list(numpy.array)): intermediate calculation results to compute transform matrices.
        The list of numpy arrays of shapes: :math:`(n, n), (n, n), n, n`.

    Returns:
        The list of numpy arrays of shapes: :math:`(n_modes, n), (n, n_modes)`.
    """
    transform = (analysis_data[1] @ analysis_data[0].T * analysis_data[2])[:n_modes, :]
    inv_transform = analysis_data[0][:, :n_modes] if len(analysis_data) == 4 else analysis_data[-1][:, :n_modes]
    return np.array(transform), np.array(inv_transform)


def get_transform_data(n_spatial, n_modes, gauss, polynomial):
    r"""
    Calculates direct and inverse polynomial transform matrices, weights and points of Gauss quadrature.
    Args:
        n_spatial (int): the number of Gauss quadrature points (spatial resolution).
        n_modes (int): the number of modes of polynomial transform (expansion coefficients).
        gauss (function): a function which calculates quadrature points and weights
        polynomial (function): a function which evaluates polynomials (Chebyshev, Legendre, etc.)

    Returns:
        Dict(str, numpy.array) with arrays of shapes:
        'weights': (n_spatial), 'points': (n_spatial), 'analysis' : (n_modes, n_spatial),
        'synthesis': (n_spatial, n_modes).
    """
    analysis_data = get_analysis_data(n_spatial, gauss, polynomial)
    transforms = build_matrices(n_modes, analysis_data)
    transform_data = {
        "weights": np.array(analysis_data[2]),
        "points": np.array(analysis_data[3]),
        "analysis": transforms[0],
        "synthesis": transforms[1]
    }
    return transform_data


def legendre(n_spatial, n_modes):
    gauss = scp.roots_legendre
    polynomial = scp.eval_legendre
    return get_transform_data(n_spatial, n_modes, gauss, polynomial)

def chebyshev_1(n_spatial, n_modes):
    gauss = scp.roots_chebyt
    polynomial = scp.eval_chebyt
    return get_transform_data(n_spatial, n_modes, gauss, polynomial)

def chebyshev_2(n_spatial, n_modes):
    gauss = scp.roots_chebyu
    polynomial = scp.eval_chebyu
    return get_transform_data(n_spatial, n_modes, gauss, polynomial)

def jacobi(n_spatial, n_modes, alpha, beta):
    gauss = lambda points: scp.roots_jacobi(points, alpha, beta)
    polynomial = lambda x, points: scp.eval_jacobi(x, alpha, beta, points)
    return get_transform_data(n_spatial, n_modes, gauss, polynomial)


def get_poly_transform(n_spatial, n_modes, poly_type='Chebyshev_t', alpha=None):
    r"""
    Chooses a function for polynomial transform.
    Args:
        n_spatial (int): the number of Gauss quadrature points (spatial resolution).
        n_modes (int): the number of modes of polynomial transform (expansion coefficients).
        poly_type (str): a type of polynomial transform (Gauss grid). Default: 'Chebyshev_t'.
        alpha (list): additional parameters for some transforms.

    Returns:
        function with two input arguments (n_spatial, n_modes)
    """

    if poly_type == 'Chebyshev_t':
        return chebyshev_1(n_spatial, n_modes)
    if poly_type == 'Chebyshev_u':
        return chebyshev_2(n_spatial, n_modes)
    if poly_type == 'Legendre':
        return legendre(n_spatial, n_modes)
    if poly_type == 'Jacobi':
        return jacobi(n_spatial, n_modes, alpha[0], alpha[1])
    return None


def interpolate_1d_dataset(data, poly_type='Chebyshev_t', kind='cubic', alpha=None):
    r"""
    Interpolates the 1D input data specified on regular grid,
    and returns the interpolated values on Gauss quadrature points.

    Args:
        data (numpy.array): input data on regular grid with shape of :math:`(n, length)`.
        poly_type (str): a type of polynomial transform (Gauss grid). Default: 'Chebyshev_t'
        kind (str): interpolation type. Default: 'cubic'.
        alpha (list): additional parameters for some transforms.

    Returns:
        The numpy array with shape of :math:`(n, length)`.
    """
    res = data.shape[1]
    x_unif = np.linspace(-1., 1.0, res)
    x_poly = poly_data[poly_type](res, alpha)

    interp_data = np.zeros(data.shape).astype(np.float32)
    for i in range(data.shape[0]):
        f = interp1d(x_unif, data[i], kind=kind)
        interp_data[i] = f(x_poly)

    return interp_data


def interpolate_2d_dataset(data, poly_type='Chebyshev_t', kind='cubic', alpha=None):
    r"""
    Interpolates the 2D input data specified on regular grid,
    and returns the interpolated values on Gauss quadrature points.

    Args:
        data (numpy.array): input data on regular grid with shape of :math:`(n, height, width)`.
        poly_type (str): a type of polynomial transform (Gauss grid). Default: 'Chebyshev_t'
        kind (str): interpolation type. Default: 'cubic'.
        alpha (list): additional parameters for some transforms.

    Returns:
        The numpy array with shape of :math:`(n, height, width)`.
    """
    res_x = data.shape[1]
    res_y = data.shape[2]
    x_unif = np.linspace(-1., 1.0, res_x)
    y_unif = np.linspace(-1., 1.0, res_y)
    x_poly = poly_data[poly_type](res_x, alpha)
    y_poly = poly_data[poly_type](res_y, alpha)

    interp_data = np.zeros(data.shape).astype(np.float32)
    for i in range(data.shape[0]):
        f = interp2d(x_unif, y_unif, data[i].flatten(), kind=kind)
        interp_data[i] = f(x_poly, y_poly).T
    return interp_data
