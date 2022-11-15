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
Utilities.
"""
import numpy as np
import mindspore as ms
from mindspore import ops
from .constants import c0


float_type = ms.float32
_oper_zeros = ops.Zeros()
_oper_ones = ops.Ones()
_oper_max = ops.Maximum()
_oper_min = ops.Minimum()
_oper_exp = ops.Exp()
_oper_sum = ops.ReduceSum()
_oper_cos = ops.Cos()
_oper_sin = ops.Sin()
zeros_like = ops.ZerosLike()
ones_like = ops.OnesLike()
hstack = ops.Stack(axis=0)
vstack = ops.Stack(axis=-1)


def tensor(data, dtype=float_type):
    """
    Convert data to mindspore.Tensor

    Args:
        data (Union[numpy.ndarray, float, int]): Raw data.
        dtype (type): Data type. Default: mindspore.float32.

    Returns:
        Tensor
    """
    return ms.Tensor(data, dtype=dtype)


def zeros(shape):
    """
    Create a tensor filled with zeros.

    Args:
        shape (tuple): Shape.

    Returns:
        Tensor.
    """
    return _oper_zeros(shape, float_type)


def ones(shape):
    """
    Create a tensor filled with ones.

    Args:
        shape (tuple): Shape.

    Returns:
        Tensor.
    """
    return _oper_ones(shape, float_type)


def estimate_time_interval(cell_lengths, cfl_number=1., epsr_min=1., mur_min=1.):
    """Estimates time interval based on the CFL condition.

    Args:
        cell_lengths (tuple): Lengths of Yee cells.
        cfl_number (float): CFL condition, should be no greater than 1.
        epsr_min: The minimum of relative permittivity in the problem.
        mur_min: The minimum of relative permeability in the problem.

    Returns:
        float
    """
    c = c0 / np.sqrt(epsr_min * mur_min)
    cell_lengths = np.array(cell_lengths)
    dt = (1. / c) / np.sqrt(np.sum(1. / cell_lengths**2))
    return dt * cfl_number


def estimate_frequency_resolution(cell_lengths, ncell):
    """Estimates the maximum frequency in terms of the space resolution.

    Args:
        cell_lengths (tuple): Lengths of Yee cells.
        ncell (int): Number of Yee cell per wavelength.

    Returns:
        fmax (float): Maximum frequency supported by the grid.
    """
    fmax = c0 / (ncell * np.max(np.array(cell_lengths)))
    return fmax


@ops.constexpr
def create_zero_tensor(shape):
    """Create zero tensor in the Cell.

    Args:
        shape (tuple): tensor shape

    Returns:
        Zero tensor
    """
    return zeros(shape)


def fcmpt(dt, epsr, sigma):
    """
    Compute FDTD updating coefficients.

    Args:
        dt (tensor): Scaled time interval.
        epsr (tensor): Relative permittivity or permeability.
        sigma (tensor): Electric or magnetic conductivity.
    """
    ctmp = ((0.5 * dt) * (sigma / epsr))
    c1 = (1 - ctmp) / (1 + ctmp)
    c2 = (dt) / (epsr * (1 + ctmp))
    return c1, c2


def sum_fields(fields, i_s, i_e, j_s, j_e, k_s, k_e):
    """
    Sum over recivers in 3D-FDTD.

    Args:
        fields (Tensor, shape=(ns, 1, ...)): Field tensor.
        i_s, i_e: index range, [Start, stop)
        j_s, j_e: index range, [Start, stop)
        k_s, k_e: index range, [Start, stop)

    Returns:
        rlt (Tensor, shape=(ns,)): result tensor.
    """
    rlt = _oper_sum(fields[:, 0, i_s:i_e, j_s:j_e, k_s:k_e], axis=(-3, -2, -1))
    return rlt


def compute_dft(ft, t, fs, dt):
    """
    Compute DFT coefficients.

    Args:
        ft (Tensor, shape=(nt, ns, ns)): Time-Domain signals.
        t (Tensor, shape=(nt,)): Time sampling points.
        fs (Tensor, shape=(nf,)): Frequency sampling points.
        dt (Tensor, shape=()): Time interval.
    """
    ws = tensor(2. * np.pi) * fs
    fr = _oper_sum(_oper_cos(ws[:, None] * t[None])[:, :, None, None] * ft[None] * dt, axis=(1))
    fi = _oper_sum(-_oper_sin(ws[:, None] * t[None])[:, :, None, None] * ft[None] * dt, axis=(1))
    return fr, fi
