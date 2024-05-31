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
# ==============================================================================
"""solve couette flow"""
import time
import os

import numpy as np
import pytest

from mindspore import numpy as mnp
from mindflow import load_yaml_config
from mindflow import cfd
from mindflow.cfd.runtime import RunTime
from mindflow.cfd.simulator import Simulator


def couette_ic_2d(mesh_x, mesh_y):
    rho = mnp.ones_like(mesh_x)
    u = mnp.zeros_like(mesh_y)
    v = mnp.zeros_like(mesh_x)
    w = mnp.zeros_like(mesh_x)
    p = mnp.ones_like(mesh_x)
    return mnp.stack([rho, u, v, w, p], axis=0)


def label_fun(y, t):
    nu = 0.1
    h = 1.0
    u_max = 0.1
    coe = 0.0
    for i in range(1, 100):
        coe += np.sin(i * np.pi * (1 - y / h)) * np.exp(-(i ** 2) * (np.pi ** 2) * nu * t / (h ** 2)) / i
    return u_max * y / h - (2 * u_max / np.pi) * coe


def train():
    '''train and evaluate the network'''
    config = load_yaml_config('{}/couette.yaml'.format(os.path.split(os.path.realpath(__file__))[0]))

    simulator = Simulator(config)
    runtime = RunTime(config['runtime'], simulator.mesh_info, simulator.material)

    mesh_x, mesh_y, _ = simulator.mesh_info.mesh_xyz()
    pri_var = couette_ic_2d(mesh_x, mesh_y)
    con_var = cfd.cal_con_var(pri_var, simulator.material)

    dy = 1 / config['mesh']['ny']
    cell_centers = np.linspace(dy / 2, 1 - dy / 2, config['mesh']['ny'])

    start = time.time()

    while runtime.time_loop(pri_var):
        runtime.compute_timestep(pri_var)
        con_var = simulator.integration_step(con_var, runtime.timestep)
        pri_var = cfd.cal_pri_var(con_var, simulator.material)
        runtime.advance()

    label_u = label_fun(cell_centers, runtime.current_time.asnumpy())
    sim_u = pri_var.asnumpy()[1, 0, :, 0]

    err = np.abs(label_u - sim_u).sum() / np.abs(label_u).sum()
    per_epoch_time = (time.time() - start) / 500

    print(f'l1 error: {err:.10f}')
    print(f'per epoch time: {per_epoch_time:.10f}')

    return err


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_couette_gpu():
    """
    Feature: cfd couette test in the gpu
    Description: None.
    Expectation: Success or throw error when error is larger than 1
    """
    err = train()
    assert err < 1.0
