# Copyright 2021-2023 @ Shenzhen Bay Laboratory &
#                       Peking University &
#                       Huawei Technologies Co., Ltd
#
# This code is a part of MindSPONGE:
# MindSpore Simulation Package tOwards Next Generation molecular modelling.
#
# MindSPONGE is open-source software based on the AI-framework:
# MindSpore (https://www.mindspore.cn/)
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
"""Metadynamics"""

from typing import Union

import mindspore as ms
import mindspore.numpy as msnp
from mindspore import Tensor, Parameter
from mindspore.ops import functional as F
from mindspore.nn import ReduceLogSumExp
try:
    # MindSpore 2.X
    from mindspore import jit
except ImportError:
    # MindSpore 1.X
    from mindspore import ms_function as jit

from ...potential.bias import Bias
from ...colvar import Colvar
from ...function import get_arguments
from ...function import get_integer, get_tensor, get_ms_array
from ...function import keepdims_sum


class GaussianKernel(Bias):
    r"""Gaussian type kernel

    Args:

        colvar (Colvar):        Collective variables (CVs) :math:`s(R)`.

        num_walker (int):       Number of multiple walkers. Default: None

        update_pace (int):      Frequency for hill addition. Default: 20

    Supported Platforms:

        ``Ascend`` ``GPU``

    """
    def __init__(self,
                 num_walker: int,
                 colvar: Colvar = None,
                 update_pace: int = 20,
                 bandwidth: Union[float, Tensor] = 0.1,
                 kernel_min: float = 2,
                 kernel_max: float = 4,
                 kernel_scale: float = 0.1,
                 **kwargs,
                 ):

        super().__init__(
            name='gaussian_kernel',
            colvar=colvar,
            update_pace=update_pace,
        )
        self._kwargs = get_arguments(locals(), kwargs)

        self.num_walker = get_integer(num_walker)

        self.kernel_min = get_ms_array(kernel_min, ms.float32)
        self.kernel_max = get_ms_array(kernel_max, ms.float32)
        self.kernel_mid = (self.kernel_max + self.kernel_min) / 2

        kernel_scale = get_ms_array(kernel_scale, ms.float32)
        self.kernel_scale = Parameter(kernel_scale, name='kernel_scale', requires_grad=False)

        bandwidth = get_tensor(bandwidth, ms.float32)
        self.bandwidth = Parameter(bandwidth, name='bandwidth', requires_grad=False)

        # (B, B)
        self.gaussian_exp = Parameter(F.zeros((self.num_walker, self.num_walker), ms.float32),
                                      name='gaussian_exp', requires_grad=False)

        self.eyes = msnp.eye(self.num_walker, self.num_walker, dtype=ms.bool_)

        self.m_min = self.kernel_min / (self.num_walker - 1)
        self.m_max = self.kernel_max / (self.num_walker - 1)

        self.logsumexp = ReduceLogSumExp(-1, True)

    def update(self, coordinate: Tensor, pbc_box: Tensor = None) -> Tensor:
        r"""update bandwidth"""
        bandwidth = self.get_bandwidth(self.gaussian_exp, self.bandwidth)
        return F.depend(bandwidth, F.assign(self.bandwidth, bandwidth))

    def initialize(self, system) -> Tensor:
        r"""initialize bandwidth"""
        colvar = self.calc_colvar(coordinate=system.coordinate, pbc_box=system.pbc_box)
        gaussian_exp = self.calc_gauss_exp(colvar)
        bandwidth = self.get_bandwidth(gaussian_exp)
        return F.assign(self.bandwidth, bandwidth)

    def get_bandwidth(self, gaussian_exp: Tensor, bandwidth: Tensor = None) -> Tensor:
        r"""get optimized bandwidth"""
        def _calc_mean(gauss_exp, bandwidth_):
            k = F.exp(gauss_exp * msnp.reciprocal((F.square(bandwidth_))))
            return F.reduce_mean(F.reduce_sum(k, -1))

        if bandwidth is not None:
            k_mean = _calc_mean(gaussian_exp, bandwidth)
            if self.kernel_min < k_mean < self.kernel_max:
                return self.identity(bandwidth)

        k_min = F.amin(gaussian_exp)
        k_max = F.amax(msnp.where(self.eyes, k_min, gaussian_exp))

        s_min = F.sqrt(k_max / F.log(self.m_min))
        s_max = F.sqrt(k_min / F.log(self.m_max))

        if bandwidth is not None:
            s_max = F.select(F.logical_and(k_mean > self.kernel_max, s_max > bandwidth), bandwidth, s_max)
            s_min = F.select(F.logical_and(k_mean < self.kernel_min, s_min < bandwidth), bandwidth, s_min)

        bandwidth = (s_min + s_max) / 2
        k_mean = _calc_mean(gaussian_exp, bandwidth)

        while k_mean > self.kernel_max or k_mean < self.kernel_min:
            s_max = F.select(k_mean > self.kernel_max, bandwidth, s_max)
            s_min = F.select(k_mean < self.kernel_min, bandwidth, s_min)

            bandwidth = (s_min + s_max) / 2
            k_mean = _calc_mean(gaussian_exp, bandwidth)

        return bandwidth

    def calc_gauss_exp(self, colvar: Tensor):
        r"""calcaulte gaussian exponent"""
        mw = colvar.shape[0]

        colvar_sg = F.stop_gradient(colvar)
        # (B, S) <- (B, s_1, s_2, ..., s_n)
        cv_a = F.reshape(colvar, (mw, -1))
        cv_b = F.reshape(colvar_sg, (mw, -1))

        dim = cv_a.shape[-1]

        # (S, B) <- (B, S)
        cv_b = F.transpose(cv_b, (1, 0))

        # (B, 1) <- (B, S)
        a2 = keepdims_sum(F.square(cv_a), -1)
        # (1, B) <- (S, B)
        b2 = keepdims_sum(F.square(cv_b), 0)

        # (B, B) = (B, S) X (S, B)
        ab = F.matmul(cv_a, cv_b)

        # (B, B) = (B, 1) + (B, B) + (1, B)
        # (s - s_0) ^ 2 = s^2 - 2 * s * s_0 + s_0^2
        dist2 = (a2 - 2 * ab + b2) / dim

        # (B, B)
        return -0.5 * dist2

    @jit
    def calc_colvar(self,
                    coordinate: Tensor,
                    neighbour_index: Tensor = None,
                    neighbour_mask: Tensor = None,
                    neighbour_vector: Tensor = None,
                    neighbour_distance: Tensor = None,
                    pbc_box: Tensor = None
                    ):
        """calculate the value of collective variables"""
        if self.colvar is None:
            return coordinate
        # (B, s_1, s_2, ..., s_n)
        return self.colvar(coordinate, pbc_box)

    def calc_bias(self, colvar: Tensor) -> Tensor:
        """calculate the value of bias potential as the function of collective variables"""
        # pylint: disable=unused-argument
        # (B, B)
        gauss_exp = self.calc_gauss_exp(colvar)
        gauss_exp = F.depend(gauss_exp, F.assign(self.gaussian_exp, gauss_exp))

        # (B, B)
        kernel_exp = gauss_exp * msnp.reciprocal(F.square(self.bandwidth))

        # (B, 1) <- (B, B)
        kernel = F.logsumexp(kernel_exp, -1, True)

        return kernel
