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

import mindspore as ms
import mindspore.numpy as msnp
from mindspore import Tensor, Parameter
from mindspore.ops import functional as F

from ...potential.bias import Bias
from ...colvar import Colvar
from ...function import get_integer, get_ms_array, keepdims_mean


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
                 colvar: Colvar,
                 num_walker: int,
                 update_pace: int = 20,
                 bandwidth: Parameter = None,
                 kernel_min: float = 2,
                 kernel_max: float = 4,
                 ):

        super().__init__(
            name='kernel_bt',
            colvar=colvar,
            update_pace=update_pace,
        )

        # S: dimension of the collective variables
        self.dim_colvar = self.colvar.shape[-1]
        self.num_walker = get_integer(num_walker)

        self.kernel_min = get_ms_array(kernel_min)
        self.kernel_max = get_ms_array(kernel_max)

        if isinstance(bandwidth, Parameter):
            self.bandwidth = bandwidth
        else:
            if bandwidth is None:
                bandwidth = F.ones((1,), ms.float32)
            else:
                bandwidth = get_ms_array(bandwidth)
            self.bandwidth = Parameter(bandwidth, name='bandwidth', requires_grad=False)

        self.kernels = Parameter(F.zeros((self.num_walker, self.num_walker), ms.float32),
                                 name='kernels', requires_grad=False)

    def update(self, coordinate: Tensor, pbc_box: Tensor = None) -> Tensor:
        kmean = F.reduce_sum(self.kernels, -1)
        if (kmean > self.kernel_max).any():
            pass
        elif(kmean < self.kernel_min).any():
            pass
        else:
            return

    def construct(self,
                  coordinate: Tensor,
                  neighbour_index: Tensor = None,
                  neighbour_mask: Tensor = None,
                  neighbour_vector: Tensor = None,
                  neighbour_distance: Tensor = None,
                  pbc_box: Tensor = None
                  ):
        r"""Calculate kernel.

        Args:
            coordinate (Tensor): Tensor of shape `(B, A, D)`. Data type is float.
                Position coordinate of atoms in system.
            neighbour_index (Tensor): Tensor of shape `(B, A, N)`. Data type is int.
                Index of neighbour atoms. Default: None
            neighbour_mask (Tensor): Tensor of shape `(B, A, N)`. Data type is bool.
                Mask for neighbour atoms. Default: None
            neighbour_vector (Tensor): Tensor of shape `(B, A, N)`. Data type is bool.
                Vectors from central atom to neighbouring atoms.
            neighbour_distance (Tensor): Tensor of shape `(B, A, N)`. Data type is float.
                Distance between neigh_shift atoms. Default: None
            pbc_box (Tensor): Tensor of shape `(B, D)`. Data type is float.
                Tensor of PBC box. Default: None

        Returns:
            kernel (Tensor): Tensor of shape `(B, 1)`. Data type is float.

        Symbols:
            B:  Batchsize, i.e. number of walkers in simulation
            A:  Number of atoms.
            N:  Maximum number of neighbour atoms.
            D:  Spatial dimension of the simulation system. Usually is 3.

        """

        # (B, s_1, s_2, ..., s_n)
        colvar = self.colvar(coordinate, pbc_box)
        colvar_sg = F.stop_gradient(colvar)

        # B
        mw = coordinate.shape[0]

        # (B, S) <- (B, s_1, s_2, ..., s_n)
        cv_a = F.reshape(colvar, (mw, -1))
        cv_b = F.reshape(colvar_sg, (mw, -1))

        # (S, B) <- (B, S)
        cv_b = F.transpose(cv_b, (1, 0))

        # (B, 1) <- (B, S)
        a2 = keepdims_mean(F.square(cv_a), -1)
        # (1, B) <- (S, B)
        b2 = keepdims_mean(F.square(cv_b), 0)

        # (B, B) = (B, S) X (S, B)
        ab = F.matmul(cv_a, cv_b) / mw

        # (B, B) = (B, 1) + (B, B) + (1, B)
        # (s - s_0) ^ 2 = s^2 - 2 * s * s_0 + s_0^2
        dist2 = a2 - 2 * ab + b2

        # (B, B)
        gauss_exp = -0.5 * dist2 * msnp.reciprocal(F.square(self.bandwidth))
        kernels = F.exp(gauss_exp)
        gauss_exp = F.depend(gauss_exp, F.assign(self.kernels, kernels))

        # (B, 1) <- (B, B)
        return F.logsumexp(gauss_exp, -1, True)
