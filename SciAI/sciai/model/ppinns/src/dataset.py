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
"""dataset"""
import numpy as np


class Dataset:
    """Dataset"""

    def __init__(self, t_range, nt_train, n_train, n_bc):
        self.t_range = t_range
        self.nt = nt_train
        self.n_train = n_train
        self.n_bc = n_bc

    def build_data(self):
        """build data"""
        t0 = self.t_range[0]
        t1 = self.t_range[1]
        t_ = np.linspace(t0, t1, self.nt).reshape((-1, 1))
        x_id = np.random.choice(self.nt, self.n_train, replace=False)
        x = t_
        x_input = x[x_id]

        # initial/bcs
        t_0 = t_.min(0)
        t_0 = np.reshape(t_0, (-1, 1))
        t_1 = t_.max(0)
        t_1 = np.reshape(t_1, (-1, 1))

        x_min = t_0
        x_max = t_1

        t_bc = t_0
        u_bc = np.zeros((t_bc.shape[0], 1))

        x_bc_0_input = t_bc
        u_bc_0_input = u_bc

        return x_input, x_bc_0_input, u_bc_0_input, x_min, x_max
