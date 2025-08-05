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
# ==============================================================================
"""3 order runge kutta integrator scheme"""
from mindspore import jit_class

from .base import Integrator


@jit_class
class RungeKutta3(Integrator):
    """3rd-order TVD RK3 scheme"""

    def __init__(self):
        self.number_of_stages = 3

    def integrate(self, con_var, init_con_var, rhs, timestep, stage):
        res = None
        if stage == 0:
            res = con_var + timestep * rhs
        if stage == 1:
            res = 0.75 * init_con_var + 0.25 * con_var + 0.25 * timestep * rhs
        if stage == 2:
            res = 1.0 * init_con_var / 3.0 + 2.0 * con_var / 3.0 + 2.0 * timestep * rhs / 3.0
        return res
