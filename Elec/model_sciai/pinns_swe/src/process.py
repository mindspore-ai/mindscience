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
"""process for pinns swe"""
import os

import yaml
import numpy as np
from pyDOE import lhs

from sciai.utils import parse_arg


def prepare():
    abs_dir = os.path.abspath(os.path.dirname(__file__))
    with open(f"{abs_dir}/../config.yaml") as f:
        config_dict = yaml.safe_load(f)
    args_ = parse_arg(config_dict)
    return (args_,)


def collocation_points(args, t_bdry, x_bdry, y_bdry):
    """generate collocation points"""
    # Convert input to numpy array
    t_bdry, x_bdry, y_bdry = np.array(t_bdry), np.array(x_bdry), np.array(y_bdry)

    # Uniform random sampling on a sphere
    txy_min = np.array([t_bdry[0], x_bdry[0], 0])
    txy_max = np.array([t_bdry[1], x_bdry[1], 1])
    pde_points = txy_min + (txy_max - txy_min) * lhs(3, args.n_pde)
    t_pde = pde_points[:, 0]
    x_pde = pde_points[:, 1]
    y_pde = np.arccos(1 - 2 * pde_points[:, 2]) + y_bdry[0]

    # Stack all the PDE point data together
    pdes = np.column_stack([t_pde, x_pde, y_pde]).astype(np.float64)
    pdes = pdes[np.argsort(pdes[:, 0])]

    # Uniform random sampling on a sphere for initial values
    init_points = txy_min[1:] + (txy_max[1:] - txy_min[1:]) * lhs(2, args.n_iv)
    x_init = init_points[:, 0]
    y_init = np.arccos(1 - 2 * init_points[:, 1]) + y_bdry[0]
    t_init = t_bdry[0] + 0 * x_init

    # Stack all the ivp data together
    inits = np.column_stack([t_init, x_init, y_init]).astype(np.float64)

    return pdes.astype(np.float), inits.astype(np.float)
