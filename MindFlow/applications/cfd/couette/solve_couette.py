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
"""solve couette flow"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple

from mindspore import context
from mindflow import load_yaml_config
from mindflow import cfd
from mindflow.cfd.runtime import RunTime
from mindflow.cfd.simulator import Simulator

from src.ic import couette_ic_2d

context.set_context(device_target="GPU", device_id=0)

config = load_yaml_config('couette.yaml')

simulator = Simulator(config)
runtime = RunTime(config['runtime'], simulator.mesh_info, simulator.material)


def label_fun(y, t):
    nu = 0.1
    h = 1.0
    u_max = 0.1
    coe = 0.0
    for i in range(1, 100):
        coe += np.sin(i * np.pi * (1 - y / h)) * np.exp(-(i ** 2) * (np.pi ** 2) * nu * t / (h ** 2)) / i
    return u_max * y / h - (2 * u_max / np.pi) * coe


mesh_x, mesh_y, _ = simulator.mesh_info.mesh_xyz()
pri_var = couette_ic_2d(mesh_x, mesh_y)
con_var = cfd.cal_con_var(pri_var, simulator.material)

dy = 1 / config['mesh']['ny']
cell_centers = np.linspace(dy / 2, 1 - dy / 2, config['mesh']['ny'])
label_y = np.linspace(0, 1, 30, endpoint=True)
label_plot_list = []
simulation_plot_list = []
plot_step = 3

fig, ax = plt.subplots()

while runtime.time_loop(pri_var):
    runtime.compute_timestep(pri_var)
    con_var = simulator.integration_step(con_var, runtime.timestep)
    pri_var = cfd.cal_pri_var(con_var, simulator.material)
    runtime.advance()

    if np.abs(runtime.current_time.asnumpy() - 5.0 * 0.1 ** plot_step) < 0.1 * runtime.timestep:
        label_u = label_fun(label_y, runtime.current_time.asnumpy())
        simulation_plot_list.append(plt.plot(cell_centers, pri_var.asnumpy()[1, 0, :, 0], color='tab:blue')[0])
        label_plot_list.append(
            plt.plot(label_y, label_u, label='ground_truth', marker='o', linewidth=0, color='tab:orange')[0])
        plot_step -= 1

plt.legend(loc='best')
ax.legend([tuple(label_plot_list), tuple(simulation_plot_list)], ['ground_truth', 'mindflow_cfd'], numpoints=1,
          handler_map={tuple: HandlerTuple(ndivide=1)})
plt.xlabel('y')
plt.ylabel('velocity-x')
plt.savefig('couette.jpg')
