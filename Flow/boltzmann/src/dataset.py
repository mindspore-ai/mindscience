# ============================================================================
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
# ============================================================================
"""
dataset
"""
import numpy as np

import mindspore as ms
from mindspore import nn, ops
import mindspore.numpy as mnp

from src.cells import Maxwellian
from src.utils import mesh_nd


class Wave1DDataset(nn.Cell):
    """dataset for 1D wave problem"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        xmax = config["xtmesh"]["xmax"]
        xmin = config["xtmesh"]["xmin"]
        nv = config["vmesh"]["nv"]
        vmin = config["vmesh"]["vmin"]
        vmax = config["vmesh"]["vmax"]
        v, _ = mesh_nd(vmin, vmax, nv)
        self.xmax = xmax
        self.xmin = xmin
        self.vdis = ms.Tensor(v.astype(np.float32))
        self.maxwellian = Maxwellian(self.vdis)
        self.iv_points = self.config["dataset"]["iv_points"]
        self.bv_points = self.config["dataset"]["bv_points"]
        self.in_points = self.config["dataset"]["in_points"]
        self.uniform = ops.UniformReal(seed=0)

    def construct(self):
        """return the inner and boundary points"""
        # Initial value points
        iv_x = self.uniform(
            (self.iv_points, 1)) * (self.xmax - self.xmin) + self.xmin
        iv_t = mnp.zeros_like(iv_x)

        # boundary value points
        bv_x1 = -0.5 * mnp.ones(self.bv_points)[..., None]
        bv_t1 = self.uniform((self.bv_points, 1)) * 0.1
        bv_x2 = 0.5 * mnp.ones(self.bv_points)[..., None]
        bv_t2 = bv_t1

        # inner points
        in_x = self.uniform((self.in_points, 1)) - 0.5
        in_t = self.uniform((self.in_points, 1)) * 0.1

        return (ops.concat([in_x, in_t], axis=-1),
                ops.concat([iv_x, iv_t], axis=-1),
                (ops.concat([bv_x1, bv_t1], axis=-1),
                 ops.concat([bv_x2, bv_t2], axis=-1)))
