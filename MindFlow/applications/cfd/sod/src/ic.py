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
"""initial condition sod tube flow"""
from mindspore import Tensor
from mindspore import numpy as mnp


def sod_ic_1d(mesh_x):
    large_x = mnp.greater(mesh_x, Tensor(0.5))
    small_x = mnp.less_equal(mesh_x, Tensor(0.5))
    rho = 1.0 * small_x + 0.125 * large_x
    u = mnp.zeros_like(rho)
    v = mnp.zeros_like(rho)
    w = mnp.zeros_like(rho)
    p = 1.0 * small_x + 0.1 * large_x

    return mnp.stack([rho, u, v, w, p], axis=0)
