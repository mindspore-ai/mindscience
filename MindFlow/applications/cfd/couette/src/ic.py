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
"""initial condition of couette flow"""
from mindspore import numpy as mnp


def couette_ic_2d(mesh_x, mesh_y):
    rho = mnp.ones_like(mesh_x)
    u = mnp.zeros_like(mesh_y)
    v = mnp.zeros_like(mesh_x)
    w = mnp.zeros_like(mesh_x)
    p = mnp.ones_like(mesh_x)

    return mnp.stack([rho, u, v, w, p], axis=0)
