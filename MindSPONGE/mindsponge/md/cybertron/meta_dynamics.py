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
"""meta_dynamics"""

import numpy as np
from mindspore import nn
import mindspore as ms
from mindspore import ops
from mindspore import Tensor
from mindspore import Parameter
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore import dtype as mstype

from .units import units

__all__ = [
    'Bias'
]


class Bias(nn.Cell):
    """Bias"""
    def __init__(self, hills, smin=0.15, smax=0.25, ds=0.0005,
                 omega=0.2, sigma=0.0002, dt=0.001,
                 t=300, alpha=0.5, gamma=6,
                 wall_potential=9e08,
                 kappa=4,
                 upper_bound=10,
                 lower_bound=190,
                 factor=1):
        super(Bias, self).__init__()
        self.smin = smin
        self.smax = smax
        self.ds = ds
        self.dt = dt
        self.alpha = alpha
        self.grid_num = 200
        self.wall_potential = Tensor([9e08], mstype.float32)
        self.wall = np.zeros(self.grid_num, dtype=np.float32)
        for i in range(20):
            self.wall[i] = wall_potential
            self.wall[-i - 1] = wall_potential
        self.wall_num = Tensor([5], mstype.float32)
        self.temperature = t
        self.kb = units.boltzmann()
        self.kbt = self.kb * self.temperature
        self.beta = 1.0 / self.kbt
        self.gamma = gamma
        self.wt_factor = -1.0 / (self.gamma - 1.0) * self.beta
        self.kappa = kappa
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.wall_factor = factor
        self.wall = Tensor(self.wall, mstype.float32)
        self.hill_matrix = Tensor([[abs(j - i) for j in range(self.grid_num)]
                                   for i in range(self.grid_num)], mstype.int32)
        self.sum = ops.ReduceSum()
        if hills is None:
            self.hills = Parameter(Tensor([0], mstype.float32))
        else:
            self.hills = Parameter(hills)
        self.sqrt2 = F.sqrt(Tensor(2.0, ms.float32))
        self.omega = omega
        self.sigma = Tensor(sigma, mstype.float32)
        self.exp = ops.Exp()
        self.add = ops.Add()
        self.norm = nn.Norm(axis=0)
        self.square = ops.Square()
        self.squeeze = ops.Squeeze()
        self.zeros = ops.Zeros()
        self.ones = ops.Ones()
        self.cast = ops.Cast()
        self.keep_sum = P.ReduceSum(keep_dims=True)
        self.cv_list = Tensor([i * ds + self.smin[0].asnumpy() for i in range(self.grid_num)], dtype=mstype.float32)

    def get_cv(self, r):
        """get_cv"""
        return self.norm(self.add(r[11], -r[14]))

    def construct(self, r):
        """construct"""
        cv = self.get_cv(r)
        cv_index = self.cast((cv - self.smin) / self.ds, mstype.int32)
        gaussian = self.omega * \
            self.exp(-self.square(cv - self.cv_list) / 2 / self.square(self.sigma))
        bias = self.sum(self.dt * self.hills * gaussian) * (cv_index <
                                                            self.upper_bound) * (cv_index >= self.lower_bound)
        # Upper Wall
        bias += self.kappa * ((cv - self.upper_bound * self.ds) /
                              self.wall_factor) ** 2 * (cv_index >= self.upper_bound)
        bias += self.kappa * ((self.upper_bound * self.ds - cv) /
                              self.wall_factor) ** 2 * (cv_index < self.lower_bound)
        return bias
