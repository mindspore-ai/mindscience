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
"""process for pinns ntk"""
import os

import yaml
import numpy as np

from sciai.utils import to_tensor, parse_arg


def prepare():
    """prepare args"""
    abs_dir = os.path.abspath(os.path.dirname(__file__))
    with open(f"{abs_dir}/../config.yaml") as f:
        config_dict = yaml.safe_load(f)
    args_ = parse_arg(config_dict)
    return (args_,)


def u(x, a):
    """function u"""
    return np.sin(np.pi * a * x)


def u_xx(x, a):
    """function d2u/dx2"""
    return -(np.pi * a) ** 2 * np.sin(np.pi * a * x)


def generate_data(a, dom_coords, num, dtype):
    """generate data"""
    x_bc1 = dom_coords[0, 0] * np.ones((num // 2, 1))
    x_bc2 = dom_coords[1, 0] * np.ones((num // 2, 1))
    x_u = np.vstack([x_bc1, x_bc2])
    y_u = u(x_u, a)
    x_r = np.linspace(dom_coords[0, 0],
                      dom_coords[1, 0], num)[:, None]
    y_r = u_xx(x_r, a)
    x_u, y_u, x_r, y_r = to_tensor((x_u, y_u, x_r, y_r), dtype)
    return x_r, x_u, y_r, y_u
